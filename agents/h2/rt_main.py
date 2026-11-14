"""
agent.py  --  BatteryAgent h2 (hard mode: specialist forecasters + uncertainty-aware exact planner)

Each hour the agent
  1. forecasts the score signal  w = co2 + KAPPA * price  beyond the 3 hours obs gives (SignalFC: ridge
     regressions on the anomalies of the last 48 known hours, one per horizon),
  2. forecasts the campus load (LoadFC: meter regimes + a small LightGBM for the next hour + reference
     days with a holiday index for later hours) and the solar output (SolarFC: clear-sky envelope x
     predicted clearness), each with its own error distribution,
  3. combines them into Q equally likely net-load scenarios for the first hours (NetScen),
  4. finds the command that minimises the EXPECTED  sum w_k * grid_import_k  with an exact dynamic
     programme on the marginal value of stored energy (planner), and applies it. The plan never looks past
     the end of the run: fewer than 3 prices in obs, or the last hour of the test calendar (2026-08-31 23:00).

Everything fitted lives in h2_state.npz next to this file (plain numpy arrays, written by build.py).

Safety net
----------
* The file also contains agent h1 (table look-ups + the same planner, numpy only) with its tables
  embedded as text. If the state file cannot be read, a module cannot be built, the observation is not
  usable by the specialists, or anything raises during a step, h1 answers that step; if h1 fails too the
  answer is 0.0 kW. `BatteryAgent.fallbacks` counts those steps (0 on a normal run).
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
from rt_combined import NetScen, CombinedAgent

ETA = 0.9 ** 0.5             # one-way efficiency
EPS = 0.01                   # kW, safety margin of the FINAL clamp only (the planner and the embedded h1 keep their
                             # own 1e-4): covers a state of charge reported rounded to 2 decimals
CAP_DEFAULT = 6000.0         # kWh, used when obs has no "cap_max_kwh"
PMAX_DEFAULT = 1500.0        # kW, used when obs has no "power_max_kw"
STATE_NAME = "h2_state.npz"
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


def build_signal(z):
    return SignalFC.from_npz({k[4:]: v for k, v in z.items() if k.startswith("sig_")})


def build_combined(z):
    """(combined agent, load mode "model" | "rules")."""
    cfg = json.loads(_text(z["config"]))
    try:
        load, mode = build_load(z, "fc"), "fc"
    except Exception:
        load, mode = build_load(z, "nm"), "nm"
    net = NetScen(load, build_solar(z), cfg["scen"], z.get("err_net_" + mode), z["err_load_" + mode],
                  dep=cfg["dep"], bias_tail=cfg["bias_tail"])
    agent = CombinedAgent(build_signal(z), net, H=cfg["H"], Q=cfg["Q"], scen_hours=cfg["scen_hours"],
                          spread=cfg["spread"], v_end=cfg["v_end"], kappa=cfg["kappa"],
                          truncate_end=cfg["truncate_end"], end_ts=cfg.get("end_ts"))
    return agent, ("model" if mode == "fc" else "rules")


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
        self.fallbacks = 0               # steps not answered by the combined agent
        self.last_error = ""             # why the last fallback happened
        self.build_error = ""            # why the combined agent could not be built ("" = it was)
        self.load_mode = None            # "model" | "rules" | None (no combined agent)
        self.main = None
        self.h1 = None
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
        if not self.build_error:
            try:
                if pd is None:
                    raise ImportError("pandas")
                self.main, self.load_mode = build_combined(z)
            except Exception as e:
                self.main, self.load_mode = None, None
                self.build_error = "build: " + repr(e)[:200]

    # -- what the specialists are given --------------------------------------------------------
    def _clean(self, obs, soc, cap, pmax):
        """A well-formed copy of obs (pandas timestamp, float lists, every reading a float or NaN).
        Raises if the observation cannot be used by the combined agent."""
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
            self.last_error = "obs is not a dict"
            return 0.0
        try:                                         # h1 only remembers the month: keep it up to date
            self.h1._when(obs)
        except Exception:
            pass
        if soc != soc:
            self.fallbacks += 1
            self.last_error = "no state of charge"
            return 0.0                               # state unknown: 0 kW is the only command that is always legal
        p = NAN
        if self.main is not None:
            try:
                p = float(self.main.act(self._clean(obs, soc, cap, pmax)))
            except Exception as e:
                p = NAN
                self.last_error = repr(e)[:200]
        else:
            self.last_error = self.build_error
        if p != p or abs(p) == INF:                  # the combined agent did not answer: h1, then 0.0
            self.fallbacks += 1
            try:
                p = float(self.h1.act(obs))
            except Exception:
                p = 0.0
            if p != p or abs(p) == INF:
                p = 0.0
        eps = self.clamp_eps
        max_c = max(0.0, min(pmax, (cap - soc) / ETA - eps))
        max_d = max(0.0, min(pmax, soc * ETA - eps))
        return float(min(max(p, -max_d), max_c)) + 0.0           # + 0.0: never -0.0
