"""
h3 with every test week treated as its own run: the plan stops at the last hour of the week, so the battery is
emptied before each week ends. This scores lower over the whole run than h3 and higher on the last half of each week.

agent.py  --  BatteryAgent h3w. The week ends with the last hour of the calendar month (test layout: each week is
the last 7 days of a month); the signal forecaster returns nothing beyond that hour, exactly as it does for the
end of the run (2026-07-31 23:00, and 2026-08-31 23:00 if the run goes on). Everything else is h3 (hard mode:
specialist forecasters + planner that does not trust the signal forecast blindly):

Each hour the agent
  1. forecasts the score signal  w = co2 + KAPPA * price  beyond the 3 hours obs gives (SignalForecaster2:
     ridge regressions on the anomalies of the last 48 known hours; at the end of a week the forecast is
     tied to the stored train day that follows it; at night the level of hours +3..+8 is averaged with a
     per-hour-of-day model), and corrects it with the error of the hour revealed last,
  2. forecasts the campus load (LoadFC) and the solar output (SolarFC), each with its own error
     distribution, and combines them into Q equally likely net-load scenarios for the first hours (NetScen),
  3. values stored energy as a blend of its value on the signal forecast and its mean value over K error
     paths of the signal (SigAgent), with an exact dynamic programme on the marginal value of stored
     energy, and applies the first command. The plan never looks past the end of the run: fewer than 3
     prices in obs, or the calendar end of the run (2026-07-31 23:00; if the run goes on, 2026-08-31 23:00).

Everything fitted lives in h3w_state.npz next to this file (plain numpy arrays, written by build.py).

Safety net: four levels, each step is answered by the first that can
---------------------------------------------------------------------
  1. "h3"  the agent above.
  2. "h2"  the previous agent: same load / solar / scenarios, round-1 signal forecast, a plan that trusts it.
           Used when the new signal pieces cannot be built from the state file, or when level 1 raises or
           returns a non-finite value at a step.
  3. "h1"  table look-ups + the same planner, numpy only, tables embedded in this file as text. Used when the
           state file cannot be read, the specialists cannot be built, the observation is not usable by
           them, or level 2 fails too.
  4. 0.0 kW.
`BatteryAgent.answered` counts the steps per level; `fallbacks` = steps not answered by level 1.
* LightGBM is not needed on the server: the load model's trees are stored as arrays and evaluated with
  numpy (rt_trees). If those arrays fail their self-check, LoadFC runs in its rules-only mode with the error
  tables measured for that mode (`BatteryAgent.load_mode` = "rules" instead of "model").
* Every command is clamped to the power and capacity limits read from obs, 0.01 kW inside them (so a state
  of charge reported rounded to 2 decimals cannot cause an infraction); act() never raises and never
  returns NaN / inf.
"""
import json
import math
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:                                    # the specialists need pandas timestamps; h1 does not
    pd = None

from rt_h1 import H1Agent
from rt_trees import TreeModel
from rt_load import LoadFC
from rt_solar import SolarFC
from rt_signal import SignalFC
from rt_signal2 import SignalForecaster2
from rt_combined import NetScen, CombinedAgent
from rt_combined3 import EndCut
from rt_sigplan import Paths, SigAgent

ETA = 0.9 ** 0.5             # one-way efficiency
EPS = 0.01                   # kW, safety margin of the FINAL clamp only (the planner keeps its own 1e-4):
                             # covers a state of charge reported rounded to 2 decimals
CAP_DEFAULT = 6000.0         # kWh, used when obs has no "cap_max_kwh"
PMAX_DEFAULT = 1500.0        # kW, used when obs has no "power_max_kw"
STATE_NAME = "h3w_state.npz"
INF = float("inf")
NAN = float("nan")
LAST_KEYS = ("ultim_consum_kw", "ultima_solar_kw", "irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms")

try:
    H1_TABLES_JSON = _H1_TABLES_JSON                 # text put at the top of agent.py by bundle.py
except NameError:
    H1_TABLES_JSON = None


def _num(x):
    """float(x) if x is a finite number, else NaN. Never raises."""
    try:
        x = float(x)
    except Exception:
        return NAN
    return x if abs(x) < INF else NAN


def _vec(x):
    """1-D float array from a list-like (NaN where a value is not a number). Never raises."""
    try:
        return np.atleast_1d(np.asarray(x, float)).ravel()
    except Exception:
        pass
    try:
        return np.array([_num(v) for v in x], float)
    except Exception:
        return np.zeros(0)


# ----------------------------------------------------------------------------------------------
# State file -> objects
# ----------------------------------------------------------------------------------------------
def _text(a):
    return np.asarray(a, np.uint8).tobytes().decode("utf-8")


def _kv(z, name):
    return dict(zip(z[name + "_k"].tolist(), z[name + "_v"].tolist()))


def read_state(path):
    with np.load(str(path), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def build_load(z, mode):
    """mode "fc": with the tree model; "nm": rules only (and the residual tables of that mode)."""
    par = json.loads(_text(z["load_params"]))
    par["n_ref"] = tuple(par["n_ref"])
    vals, lens = z["load_res_" + mode], z["load_res_" + mode + "_len"].tolist()
    res, a, i = {}, 0, 0
    for h in range(24):
        for we in (0, 1):
            res[(h, we)] = vals[a:a + lens[i]]
            a += lens[i]
            i += 1
    st = {"params": par, "base": _kv(z, "load_base"), "tbase": _kv(z, "load_tbase"), "prof": z["load_prof"].tolist(),
          "res": res, "res_cold": z["load_res_cold_" + mode], "model": None}
    fc = LoadFC.from_state(st)
    if mode == "fc":
        model = TreeModel(z["tree_feat"], z["tree_thr"], z["tree_dtype"], z["tree_left"], z["tree_right"],
                          z["tree_leaf"], z["tree_nint"])
        if model.n_features > len(fc.ci):
            raise ValueError("the trees use features the forecaster does not have")
        got = model.predict(z["tree_check_x"])                   # self-check against what LightGBM returned at build time
        if not np.allclose(got, z["tree_check_y"], rtol=0.0, atol=1e-6):
            raise ValueError("tree model self-check failed")
        fc.model = model
        fc.use_model = True
    return fc


def build_solar(z):
    m = json.loads(_text(z["sol_meta"]))
    fc = SolarFC(None, H=int(m["Hfit"]), fit_quantiles=bool(m["fit_q"]), use_weather=bool(m["use_weather"]),
                 outage_rule=bool(m["outage_rule"]), q_scale=float(m["q_scale"]))
    fc.pv_fit, fc.dT_fit, fc.ta_fit = _kv(z, "sol_pv"), _kv(z, "sol_dT"), _kv(z, "sol_ta")
    fc.c_irr, fc.k_clim = float(m["c_irr"]), float(m["k_clim"])
    fc.env_tab = z["sol_env_tab"]
    fc.par = {"c1": z["sol_c1"], "c2": z["sol_c2"], "mu": z["sol_mu"]}
    fc.qtab = z["sol_qtab"] if "sol_qtab" in z else None
    fc._dcache = {}
    fc.reset()
    return fc


def round1_arrays(z):
    return {k[4:]: v for k, v in z.items() if k.startswith("sig_")}


def build_net(z, cfg):
    """(net-load scenario forecaster shared by levels 1 and 2, load mode "model" | "rules")."""
    try:
        load, mode = build_load(z, "fc"), "fc"
    except Exception:
        load, mode = build_load(z, "nm"), "nm"
    net = NetScen(load, build_solar(z), cfg["scen"], z.get("err_net_" + mode), z["err_load_" + mode],
                  dep=cfg["dep"], bias_tail=cfg["bias_tail"])
    return net, ("model" if mode == "fc" else "rules")


def build_h2(cfg, net, signal):
    """Level 2: the plan that trusts the round-1 signal forecast (agent h2)."""
    return CombinedAgent(signal, net, H=cfg["H"], Q=cfg["Q"], scen_hours=cfg["scen_hours"], spread=cfg["spread"],
                         v_end=cfg["v_end"], kappa=cfg["kappa"], truncate_end=cfg["truncate_end"],
                         end_ts=(run_ends(cfg) or [None])[0])


def build_h3(z, cfg, net):
    """Level 1: (SigAgent on the round-2 signal forecaster and the error paths, the round-2 forecaster)."""
    r1 = round1_arrays(z)
    d = {k: (z["sig2_" + k] if ("sig2_" + k) in z else r1[k]) for k in json.loads(_text(z["sig2_keys"]))}
    d["alt"] = {k: (z["alt_" + k] if ("alt_" + k) in z else d[k]) for k in json.loads(_text(z["alt_keys"]))}
    sig2 = SignalForecaster2.from_npz(d)
    K, m = int(cfg["K"]), int(cfg["m"])
    sets, beta = np.asarray(z["paths_sets"], float), np.asarray(z["paths_beta"], float)
    if sets.shape != (24, K, m - 3) or beta.shape != (m,) or not (np.isfinite(sets).all() and np.isfinite(beta).all()):
        raise ValueError("error paths do not fit the configuration")
    if sig2.bridge and not (np.isfinite(sig2.rho["c"]).all() and np.isfinite(sig2.rho["p"]).all()):
        raise ValueError("non-finite bridge weights")
    if sig2.night and sig2.alt is None:
        raise ValueError("night model missing")
    pool = Paths(sets, beta, K, m, bias=z.get("paths_bias"))
    w_fc = WeekRule(sig2, cfg.get("beyond_scale", 1.0), cfg.get("week_rule", "month_end"), cfg.get("week_cut", False))
    for e in sorted(run_ends(cfg), reverse=True):    # calendar ends of the run: the latest innermost, the earliest outermost,
        w_fc = EndCut(w_fc, e)                       # so the plan stops at the next end ahead and goes on normally after it
    agent = SigAgent(w_fc, net, pool=pool, lam=cfg["lam"], ar=cfg["ar"], H=cfg["H"], Q=cfg["Q"],
                     scen_hours=cfg["scen_hours"], J=int(cfg["J"]), exact=(cfg["lam"] == 0), v_end=cfg["v_end"],
                     kappa=cfg["kappa"], truncate_end=cfg["truncate_end"])
    return agent, sig2


def run_ends(cfg):
    """Calendar ends of the run as timestamps, earliest first (config "end_ts": None, one date or a list of dates)."""
    e = cfg.get("end_ts")
    if e is None:
        return []
    if isinstance(e, str):
        e = [e]
    return sorted(pd.Timestamp(x) for x in e)


def _rule(rule):
    if isinstance(rule, str):
        if rule != "month_end":
            raise ValueError("week rule")
        return rule
    return (int(rule[0]), int(rule[1]))


def week_end(ts, rule):
    """Last hour of the test week `ts` is in. rule "month_end": the last hour of the calendar month (test layout:
    each week is the last 7 days of a month); (lo, hi): day `hi` of the month (validation folds)."""
    day = int(ts.days_in_month) if rule == "month_end" else rule[1]
    return ts.normalize().replace(day=day) + pd.Timedelta(hours=23)


class WeekRule:
    """What happens to the signal forecast at the end of the current test week:
        cut=True     the forecast stops there ("each week is its own run"), the way EndCut stops it at the end of
                     the run: the plan ends with the week and energy left there is worth nothing;
        scale != 1   the rows after it are multiplied by `scale` (the cut probe);
        otherwise    nothing (h3)."""
    def __init__(self, fc, scale, rule, cut):
        self.fc, self.scale, self.rule, self.cut = fc, float(scale), _rule(rule), bool(cut)
        if not 0.0 < self.scale < 10.0:
            raise ValueError("scale")

    def predict(self, obs, hist, H):
        w = np.asarray(self.fc.predict(obs, hist, H), float)
        if not self.cut and self.scale == 1.0:
            return w
        ts = obs["timestamp"]
        r = int((week_end(ts, self.rule) - ts) / pd.Timedelta(hours=1)) + 1      # rows left in the week, this one included
        if self.cut:
            return w[:r] if r >= 1 else w
        w = w.copy()
        if 1 <= r < len(w):
            w[r:] = self.scale * w[r:]
        return w


class Round1View:
    """The round-1 signal forecast read off the round-2 forecaster's own store. Round 2 is round 1's model
    plus two corrections (bridge, night level) and both keep the same store and calendar bookkeeping, so
    switching the corrections off gives what h2's forecaster would say at this step with nothing to replay."""
    def __init__(self, fc):
        self.fc = fc

    def predict(self, obs, hist, H):
        fc = self.fc
        saved = fc.bridge
        fc.bridge = False                            # the bridge sits in _from(); the night level in the subclass's
        try:                                         # predict_parts(), which is not the one called here
            c, p = SignalFC.predict_parts(fc, obs, hist, H)
        finally:
            fc.bridge = saved
        return c + fc.kappa * p


def _default_path():
    try:
        return Path(__file__).parent / STATE_NAME
    except Exception:
        return Path(STATE_NAME)


# ----------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------
class BatteryAgent:
    state_path = None        # set on the class before construction to load another state file (validation)
    clamp_eps = EPS          # margin of the final clamp (validation sets 1e-4 to compare with the harness bit for bit)

    def __init__(self):
        self.steps = 0
        self.answered = {"h3": 0, "h2": 0, "h1": 0, "zero": 0}   # steps answered by each level
        self.fallbacks = 0               # steps not answered by level 1
        self.last_error = ""             # why the last fallback happened
        self.build_error = ""            # why NO specialist agent could be built ("" = at least level 2 exists)
        self.h3_error = ""               # why level 1 could not be built ("" = it was)
        self.h2_error = ""               # why level 2 could not be built ("" = it was)
        self.load_mode = None            # "model" | "rules" | None (no specialist agent)
        self.top = None                  # level 1: SigAgent
        self.mid = None                  # level 2: CombinedAgent
        self.h1 = None                   # level 3
        self.ends = []                   # calendar ends of the run (timestamps, earliest first)
        self.week_cut = False            # True: every week is its own run
        self.week_rule = "month_end"
        z = {}
        try:
            z = read_state(self.state_path if self.state_path is not None else _default_path())
        except Exception as e:
            self.build_error = "state: " + repr(e)[:200]
        try:                             # h1 first: it must be there whatever happens next
            tabs = json.loads(_text(z["h1_json"])) if "h1_json" in z else json.loads(H1_TABLES_JSON)
            self.h1 = H1Agent(tabs)
        except Exception:
            self.h1 = None
        if self.build_error:
            return
        try:
            if pd is None:
                raise ImportError("pandas")
            cfg = json.loads(_text(z["config"]))
            net, self.load_mode = build_net(z, cfg)
            self.ends = run_ends(cfg)
            self.week_cut = bool(cfg.get("week_cut", False))
            self.week_rule = _rule(cfg.get("week_rule", "month_end"))
        except Exception as e:
            self.load_mode = None
            self.build_error = "build: " + repr(e)[:200]
            return
        sig2 = None
        try:
            self.top, sig2 = build_h3(z, cfg, net)
        except Exception as e:
            self.top, sig2 = None, None
            self.h3_error = repr(e)[:200]
        try:                             # with level 1 in place, level 2 reads the round-1 forecast off its store
            self.mid = build_h2(cfg, net, Round1View(sig2) if self.top is not None else SignalFC.from_npz(round1_arrays(z)))
            if self.top is not None:
                self.mid.hist = self.top.hist            # one history of the run for both levels
        except Exception as e:
            self.mid = None
            self.h2_error = repr(e)[:200]
        if self.top is None and self.mid is None:
            self.load_mode = None
            self.build_error = "build: h3: " + self.h3_error + " | h2: " + self.h2_error

    # -- what the specialists are given --------------------------------------------------------
    def _clean(self, obs, soc, cap, pmax):
        """A well-formed copy of obs (pandas timestamp, float lists, every reading a float or NaN).
        Raises if the observation cannot be used by the specialists."""
        ts = obs["timestamp"]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(str(ts))
        if ts != ts:
            raise ValueError("timestamp")
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        co2 = _vec(obs.get("co2_properes_24h"))
        price = _vec(obs.get("preus_properes_24h"))
        n = min(len(co2), len(price))
        if n == 0 or not (np.isfinite(co2[:n]).all() and np.isfinite(price[:n]).all()):
            raise ValueError("price / co2 lists")
        last_in = obs.get("noves_dades_5min")
        if not isinstance(last_in, dict):
            last_in = {}
        last = {k: _num(last_in.get(k)) for k in LAST_KEYS}
        c0, p0 = _num(obs.get("co2_actual")), _num(obs.get("preu_actual"))
        return {"timestamp": ts, "hour": int(ts.hour), "day_of_week": int(ts.weekday()),
                "soc_kwh": min(max(soc, 0.0), cap), "cap_max_kwh": cap, "power_max_kw": pmax,
                "co2_actual": c0 if c0 == c0 else float(co2[0]), "preu_actual": p0 if p0 == p0 else float(price[0]),
                "co2_properes_24h": co2[:n].tolist(), "preus_properes_24h": price[:n].tolist(),
                "noves_dades_5min": last}

    def _end_for(self, ts):
        """The instant the plan must not look past at this step: the next calendar end of the run, or the end of
        the current week when every week is its own run (None = nothing of the kind ahead)."""
        ahead = [e for e in self.ends if e >= ts]
        end = ahead[0] if ahead else None
        if self.week_cut:
            wk = week_end(ts, self.week_rule)
            if wk >= ts and (end is None or wk < end):
                end = wk
        return end

    def _ask(self, agent, clean, name):
        """The level's command, or NaN if it raised or returned a non-finite value. A record the level added
        to the (shared) history before failing is taken out again: the next level adds its own."""
        n0 = len(agent.hist)
        try:
            p = float(agent.act(clean))
        except Exception as e:
            p = NAN
            self.last_error = name + ": " + repr(e)[:200]
        if p != p or abs(p) == INF:
            if p == p:
                self.last_error = name + ": non-finite command"
            elif not self.last_error.startswith(name + ":"):
                self.last_error = name + ": NaN command"
            del agent.hist[n0:]
            return NAN
        return p

    # -- API ---------------------------------------------------------------------------------
    def act(self, obs: dict) -> float:
        self.steps += 1
        try:
            cap = _num(obs.get("cap_max_kwh", CAP_DEFAULT))
            pmax = _num(obs.get("power_max_kw", PMAX_DEFAULT))
            if not cap > 0.0:
                cap = CAP_DEFAULT
            if not pmax > 0.0:
                pmax = PMAX_DEFAULT
            soc = _num(obs.get("soc_kwh"))
            if soc != soc:                           # fall back on the relative fields
                soc = _num(obs.get("soc_pct")) * cap
            if soc != soc:
                soc = _num(obs.get("soc_percent")) * cap / 100.0
        except Exception:
            self.fallbacks += 1
            self.answered["zero"] += 1
            self.last_error = "obs is not a dict"
            return 0.0
        try:                                         # h1 only remembers the month: keep it up to date
            self.h1._when(obs)
        except Exception:
            pass
        if soc != soc:
            self.fallbacks += 1
            self.answered["zero"] += 1
            self.last_error = "no state of charge"
            return 0.0                               # state unknown: 0 kW is the only command that is always legal
        p, level = NAN, None
        if self.top is not None or self.mid is not None:
            clean = None
            try:
                clean = self._clean(obs, soc, cap, pmax)
            except Exception as e:
                self.last_error = "obs: " + repr(e)[:200]
            if clean is not None:
                if self.top is not None:
                    p = self._ask(self.top, clean, "h3")
                    if p == p:
                        level = "h3"
                if level is None and self.mid is not None:
                    try:                             # level 2 has one end date: give it the one that applies now
                        self.mid.end_ts = self._end_for(clean["timestamp"])
                    except Exception:
                        pass
                    p = self._ask(self.mid, clean, "h2")
                    if p == p:
                        level = "h2"
        else:
            self.last_error = self.build_error
        if level is None:                            # no specialist answered: h1, then 0.0
            level = "h1"
            try:
                p = float(self.h1.act(obs))
            except Exception:
                p = NAN
            if p != p or abs(p) == INF:
                p, level = 0.0, "zero"
        self.answered[level] += 1
        if level != "h3":
            self.fallbacks += 1
        eps = self.clamp_eps
        max_c = max(0.0, min(pmax, (cap - soc) / ETA - eps))
        max_d = max(0.0, min(pmax, soc * ETA - eps))
        return float(min(max(p, -max_d), max_c)) + 0.0           # + 0.0: never -0.0
