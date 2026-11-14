"""
experiments/29_stress_hard/lib.py
Stress harness for the hard-mode agents: packaged-agent loader, a simulator that records exceptions /
infractions / fallbacks / timings and lets a scenario rewrite obs, the perfect-foresight optimum on modified
data, and the scenario builders (every scenario is a transformation of a clean fold run).

A scenario = (df, obs_fn, soc0):
    df       hourly table of the run (what the organisers' table would hold; the score is computed from it)
    obs_fn   None or f(obs, t, ctx) -> obs   rewrites what the agent is shown (the physics is not touched)
    soc0     starting charge
"""
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from src.hard import CAP, ETA, METEO_COLS, PMAX, SOC0, get_folds, score_weights  # noqa: E402
from src.sim import score_from_grid, solve_lp  # noqa: E402

H1 = pd.Timedelta(hours=1)
_MODS = {}


# --------------------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------------------
def agent_module(kind, pkg=None):
    d = Path(pkg) if pkg is not None else HERE / "pkg" / kind
    key = str(d)
    if key not in _MODS:
        spec = importlib.util.spec_from_file_location(f"stress_{kind}_{len(_MODS)}", str(d / "agent.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODS[key] = mod
    return _MODS[key]


def load_agent(kind, state=None, pkg=None, **attrs):
    """kind = "h1" | "h2" (the extracted ZIPs under pkg/). state: tables / state file fitted elsewhere
    (None = the shipped ones, fitted on every train row). attrs: class attributes set before construction."""
    cls = agent_module(kind, pkg).BatteryAgent
    attr = "data_path" if hasattr(cls, "data_path") else "state_path"
    setattr(cls, attr, None if state is None else str(state))
    for k, v in attrs.items():
        setattr(cls, k, v)
    return cls()


def state_path(kind, fold, tag=""):
    return HERE / "states" / (f"{kind}_{fold}{tag}.json" if kind == "h1" else f"{kind}_{fold}{tag}.npz")


# --------------------------------------------------------------------------------------
# Simulator (physics of src.hard.simulate) with an obs hook and failure accounting
# --------------------------------------------------------------------------------------
def run(agent, df, obs_fn=None, soc0=SOC0, horizon=3, trace=False):
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_kw"].to_numpy(float)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    met = {c: df[c].to_numpy(float) for c in METEO_COLS if c in df.columns}
    ts = df.index
    n = len(df)
    price_l, co2_l = price.tolist(), co2.tolist()
    hours = ts.hour.to_numpy()
    dows = ts.day_of_week.to_numpy()
    ctx = {"price": price_l, "co2": co2_l, "n": n, "df": df}

    soc = float(soc0)
    grid = np.zeros(n)
    acts = np.zeros(n)
    reqs = np.zeros(n)
    socs = np.zeros(n)
    infr = np.zeros(n, dtype=int)
    exc = bad = 0
    first_exc = ""
    t_tot = t_max = 0.0
    for t in range(n):
        h_end = min(t + horizon, n)
        tl = t - 1 if t > 0 else t
        last = {"ultim_consum_kw": float(load[tl]), "ultima_solar_kw": float(pv[tl])}
        for c, a in met.items():
            last[c] = float(a[tl])
        obs = {
            "timestamp": ts[t], "hour": int(hours[t]), "day_of_week": int(dows[t]),
            "soc_kwh": float(soc), "soc_pct": float(soc / CAP), "soc_percent": float(soc / CAP * 100),
            "cap_max_kwh": CAP, "power_max_kw": PMAX,
            "preu_actual": price_l[t], "co2_actual": co2_l[t],
            "preus_properes_24h": price_l[t:h_end], "co2_properes_24h": co2_l[t:h_end],
            "noves_dades_5min": last,
        }
        if obs_fn is not None:
            obs = obs_fn(obs, t, ctx)
        t0 = time.perf_counter()
        try:
            p_req = agent.act(obs)
        except Exception as e:                       # a crash on the server = a failed run
            exc += 1
            first_exc = first_exc or f"t={t} {e!r}"[:160]
            p_req = 0.0
        dt = time.perf_counter() - t0
        t_tot += dt
        t_max = max(t_max, dt)
        try:
            p_req = float(p_req)
            if not np.isfinite(p_req):
                raise ValueError
        except Exception:
            bad += 1
            p_req = 0.0
        k = 0
        p = min(max(p_req, -PMAX), PMAX)
        if p != p_req:
            k += 1
        if p > 0:
            p_act = min(p, max(0.0, (CAP - soc) / ETA))
            if p_act < p:
                k += 1
            soc += p_act * ETA
        elif p < 0:
            p_act = -min(-p, max(0.0, soc * ETA))
            if p_act > p:
                k += 1
            soc += p_act / ETA
        else:
            p_act = 0.0
        if soc < -1e-6 or soc > CAP + 1e-6:
            k += 1
        soc = min(max(soc, 0.0), CAP)
        grid[t] = max(load[t] - pv[t] + p_act, 0.0)
        acts[t], reqs[t], socs[t], infr[t] = p_act, p_req, soc, k

    res = score_from_grid(grid, load, pv, price, co2, infr)
    res["score_raw"] = res["score"] + 0.2 * res["infractions"]            # before the infraction penalty
    net = load - pv
    spill = np.maximum(-(net + acts), 0.0) - np.maximum(-net, 0.0)        # exported because of the battery
    res.update(exceptions=exc, bad_returns=bad, first_exc=first_exc, secs=t_tot, max_step_ms=t_max * 1e3,
               fallbacks=getattr(agent, "fallbacks", np.nan), last_error=str(getattr(agent, "last_error", ""))[:120],
               load_mode=getattr(agent, "load_mode", ""),
               spill_kwh=float(np.maximum(spill, 0.0).sum()),             # battery energy sent to the grid for nothing
               lost_solar_kwh=float(np.maximum(-(net + acts), 0.0).sum()),  # everything exported (surplus not stored too)
               n=n, soc_end=float(soc))
    if trace:
        res["trace"] = pd.DataFrame({"p_kw": acts, "p_req_kw": reqs, "soc_kwh": socs, "grid_kw": grid, "infr": infr,
                                     "net_kw": net, "load_kw": load, "pv_kw": pv, "w": score_weights(df),
                                     "spill_kw": np.maximum(spill, 0.0)}, index=ts)
    return res


_ORC = {}


def oracle_score(df, soc0=SOC0):
    """Best achievable score on df with full foresight (cached on the content of the run)."""
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_kw"].to_numpy(float)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    key = (hash(load.tobytes()), hash(pv.tobytes()), hash(price.tobytes()), hash(co2.tobytes()), float(soc0))
    if key not in _ORC:
        w = score_weights(df)
        c, d, s, g, _ = solve_lp(w, load - pv, soc0=float(soc0), cap=CAP, pmax=PMAX)
        _ORC[key] = score_from_grid(g, load, pv, price, co2)["score"]
    return _ORC[key]


# --------------------------------------------------------------------------------------
# Data scenarios
# --------------------------------------------------------------------------------------
def weeks(df):
    """Week number (0 ..) of every row: one week per calendar month of the run."""
    return pd.factorize(df.index.year * 100 + df.index.month)[0]


def event_mask(df, dur, start_hour, day=2, which=None):
    """Rows inside one event per week: `dur` hours from start_hour on day number `day` (0 = first) of the week."""
    wk = weeks(df)
    m = np.zeros(len(df), bool)
    for i in np.unique(wk):
        if which is not None and i not in which:
            continue
        idx = df.index[wk == i]
        t0 = idx[0].normalize() + pd.Timedelta(days=day, hours=start_hour)
        m |= (df.index >= t0) & (df.index < t0 + pd.Timedelta(hours=dur)) & (wk == i)
    return m


def week_mask(df, which):
    return np.isin(weeks(df), list(which))


def night_freeze(df, m):
    """The meter is not read 22:00-06:00 (the 22:00 value is held): hours 23:00-05:00 take the fault state of
    the 22:00 reading before them."""
    m = m.copy()
    idx, hrs = df.index, df.index.hour
    held = None
    for t in range(len(df)):
        h = hrs[t]
        if t > 0 and idx[t] - idx[t - 1] != H1:
            held = None
        if h == 22:
            held = bool(m[t])
        elif h == 23 or h <= 5:
            if held is not None:
                m[t] = held
        else:
            held = None
    return m


def meter_fault(df, m, pv_too=False, freeze=True):
    """Load reads exactly 0 on rows m (the table holds 0, so the score is computed with 0 too)."""
    d = df.copy()
    ml = night_freeze(d, m) if freeze else m
    d.loc[ml, "load_kw"] = 0.0
    if pv_too:
        d.loc[m, "pv_kw"] = 0.0
    return d


def inverter_fault(df, m, irr_too=False):
    d = df.copy()
    d.loc[m, "pv_kw"] = 0.0
    if irr_too:
        d.loc[m, "irradiancia_wm2"] = 0.0
    return d


def scale_load(df, f, m=None):
    d = df.copy()
    if m is None:
        d["load_kw"] = d["load_kw"] * f
    else:
        d.loc[m, "load_kw"] = d.loc[m, "load_kw"] * f
    return d


def sunday_week(df, which):
    """Weekdays of the chosen weeks take the Sunday profile of the same week (a closed campus)."""
    d = df.copy()
    wk = weeks(d)
    for i in which:
        rows = d.index[wk == i]
        sun = d.loc[rows[rows.dayofweek == 6], "load_kw"]
        prof = sun.groupby(sun.index.hour).mean()
        if len(prof) < 24:
            continue
        tgt = rows[rows.dayofweek < 5]
        d.loc[tgt, "load_kw"] = prof.reindex(tgt.hour).to_numpy()
    return d


def const_week(df, which, col="load_kw"):
    """The pipeline filled a missing week by holding the last value before it."""
    d = df.copy()
    wk = weeks(d)
    for i in which:
        pos = np.flatnonzero(wk == i)
        d.iloc[pos, d.columns.get_loc(col)] = d[col].iloc[max(pos[0] - 1, 0)]
    return d


def drop_rows(df, m):
    return df[~m]


def gaps(df, seed=0, per_week=1, lo=2, hi=10, hour=None):
    """Remove `per_week` runs of lo..hi hours inside every week (the official table drops hours with no data)."""
    rng = np.random.default_rng(seed)
    wk = weeks(df)
    m = np.zeros(len(df), bool)
    for i in np.unique(wk):
        pos = np.flatnonzero(wk == i)
        for _ in range(per_week):
            L = int(rng.integers(lo, hi + 1))
            if hour is None:
                a = int(rng.integers(12, len(pos) - L - 12))
            else:
                cand = [k for k in range(12, len(pos) - L - 12) if df.index[pos[k]].hour == hour]
                a = int(rng.choice(cand))
            m[pos[a:a + L]] = True
    return df[~m]


def duplicate_rows(df, seed=0, n=12):
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(np.arange(5, len(df) - 5), n, replace=False))
    order = np.sort(np.concatenate([np.arange(len(df)), pos]))
    return df.iloc[order]


def relabel(df, mode="last7"):
    """New timestamps for every week of the run.
    "last7": the week ends on the last day of its month (the test calendar; weekdays no longer match the data).
    "plus7k": shifted by whole weeks so that it ends as close to the month end as possible (weekdays kept)."""
    wk = weeks(df)
    idx = df.index.to_numpy().copy()
    for i in np.unique(wk):
        rows = df.index[wk == i]
        last = rows[-1].normalize()
        dim = last.days_in_month
        shift = dim - last.day
        if mode == "plus7k":
            shift = (shift // 7) * 7
        idx[wk == i] = (rows + pd.Timedelta(days=shift)).to_numpy()
    d = df.copy()
    d.index = pd.DatetimeIndex(idx, name=df.index.name)
    return d


def add_week(df, src_week, month_shift_days, after=True):
    """Copy week `src_week` of the run shifted by a number of days (a week the train set does not have)."""
    wk = weeks(df)
    blk = df[wk == src_week].copy()
    blk.index = blk.index + pd.Timedelta(days=month_shift_days)
    return pd.concat([df, blk]).sort_index()


def dst(df):
    """Clock changes inside test weeks: 29 Mar 2026 has no 02:00, 26 Oct 2025 has 02:00 twice."""
    d = df[~((df.index.normalize() == pd.Timestamp("2026-03-29")) & (df.index.hour == 2))]
    pos = np.flatnonzero((d.index.normalize() == pd.Timestamp("2025-10-26")) & (d.index.hour == 2))
    if len(pos):
        order = np.sort(np.concatenate([np.arange(len(d)), pos[:1]]))
        d = d.iloc[order]
    return d


# --------------------------------------------------------------------------------------
# obs scenarios  (f(obs, t, ctx) -> obs)
# --------------------------------------------------------------------------------------
def chain(*fs):
    def f(obs, t, ctx):
        for g in fs:
            obs = g(obs, t, ctx)
        return obs
    return f


def o_ts(kind):
    def f(obs, t, ctx):
        ts = obs["timestamp"]
        if kind == "str":
            obs["timestamp"] = str(ts)
        elif kind == "iso":
            obs["timestamp"] = ts.isoformat()
        elif kind == "datetime":
            obs["timestamp"] = ts.to_pydatetime()
        elif kind == "np":
            obs["timestamp"] = ts.to_datetime64()
        elif kind == "tz":
            obs["timestamp"] = ts.tz_localize("Europe/Madrid", ambiguous=True, nonexistent="shift_forward")
        elif kind == "utc":
            obs["timestamp"] = ts.tz_localize("UTC")
        elif kind == "dmy":
            obs["timestamp"] = ts.strftime("%d/%m/%Y %H:%M")
        elif kind == "epoch":
            obs["timestamp"] = int(ts.value // 10 ** 9)
        elif kind == "none":
            obs["timestamp"] = None
        elif kind == "missing":
            del obs["timestamp"]
        elif kind == "const":
            obs["timestamp"] = ctx["df"].index[0]
        return obs
    return f


def o_lists(kind, every=50, seed=0):
    rng = np.random.default_rng(seed)
    short = {}

    def f(obs, t, ctx):
        if kind == "24":
            obs["preus_properes_24h"] = ctx["price"][t:t + 24]
            obs["co2_properes_24h"] = ctx["co2"][t:t + 24]
        elif kind in ("short_random", "short_weekend"):
            if kind == "short_random":
                if t not in short:
                    short[t] = int(rng.integers(1, 3)) if rng.random() < 1.0 / every else 0
                k = short[t]
            else:                                      # each week is served from its own table: 2 then 1 values
                idx = ctx["df"].index
                k = 0
                for j in (1, 2):
                    if t + j < ctx["n"] and idx[t + j] - idx[t] > pd.Timedelta(hours=j + 24) and k == 0:
                        k = j
            if k and len(obs["preus_properes_24h"]) > k:
                obs["preus_properes_24h"] = obs["preus_properes_24h"][:k]
                obs["co2_properes_24h"] = obs["co2_properes_24h"][:k]
        elif kind == "numpy":
            obs["preus_properes_24h"] = np.asarray(obs["preus_properes_24h"])
            obs["co2_properes_24h"] = np.asarray(obs["co2_properes_24h"])
        elif kind == "series":
            obs["preus_properes_24h"] = pd.Series(obs["preus_properes_24h"])
            obs["co2_properes_24h"] = pd.Series(obs["co2_properes_24h"])
        elif kind == "nan_random":
            if rng.random() < 1.0 / every:
                obs["preus_properes_24h"] = [obs["preus_properes_24h"][0]] + [float("nan")] * (len(obs["preus_properes_24h"]) - 1)
        elif kind == "none_random":
            if rng.random() < 1.0 / every:
                obs["co2_properes_24h"] = [None] * len(obs["co2_properes_24h"])
        elif kind == "empty_random":
            if rng.random() < 1.0 / every:
                obs["co2_properes_24h"] = []
                obs["preus_properes_24h"] = []
        elif kind == "missing":
            del obs["preus_properes_24h"]
            del obs["co2_properes_24h"]
        elif kind == "strings":
            obs["preus_properes_24h"] = [str(v) for v in obs["preus_properes_24h"]]
            obs["co2_properes_24h"] = [str(v) for v in obs["co2_properes_24h"]]
        return obs
    return f


def o_drop(*keys):
    def f(obs, t, ctx):
        for k in keys:
            obs.pop(k, None)
        return obs
    return f


def o_set(**kv):
    def f(obs, t, ctx):
        obs.update(kv)
        return obs
    return f


def o_soc(kind):
    def f(obs, t, ctx):
        s = obs["soc_kwh"]
        if kind == "out":                              # float noise outside the range at the limits
            if s >= CAP - 1e-9:
                s = CAP + 1e-7
            elif s <= 1e-9:
                s = -1e-7
        elif kind == "round2":
            s = round(s, 2)
        elif kind == "round0":
            s = float(round(s))
        elif kind == "str":
            s = str(s)
        elif kind == "pct_only":
            del obs["soc_kwh"]
            return obs
        elif kind == "none":
            del obs["soc_kwh"], obs["soc_pct"], obs["soc_percent"]
            return obs
        obs["soc_kwh"] = s
        return obs
    return f


def o_last(kind, seed=0, every=20):
    """What the hour that just ended looks like in obs["noves_dades_5min"]."""
    rng = np.random.default_rng(seed)
    wkeys = ("irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms")

    def f(obs, t, ctx):
        last = obs["noves_dades_5min"]
        if kind == "w_nan":
            for k in wkeys:
                last[k] = float("nan")
        elif kind == "w_missing":
            for k in wkeys:
                last.pop(k, None)
        elif kind == "w_const":
            last.update(irradiancia_wm2=300.0, temp_ambient_c=20.0, temp_panel_c=25.0, vent_ms=2.0)
        elif kind == "w_zero":
            for k in wkeys:
                last[k] = 0.0
        elif kind == "w_absurd":
            last.update(irradiancia_wm2=1e9, temp_ambient_c=-9999.0, temp_panel_c=1e6, vent_ms=-50.0)
        elif kind == "w_inf":
            last.update(irradiancia_wm2=float("inf"), temp_ambient_c=float("-inf"))
        elif kind == "w_none":
            for k in wkeys:
                last[k] = None
        elif kind == "w_str":
            for k in wkeys:
                last[k] = str(last.get(k))
        elif kind == "w_irr_kw":                       # irradiance in kW/m2 instead of W/m2
            last["irradiancia_wm2"] = last["irradiancia_wm2"] / 1000.0
        elif kind == "w_irr_x2":                       # another sensor / another aggregation
            last["irradiancia_wm2"] = last["irradiancia_wm2"] * 2.0
        elif kind == "w_irr_half":
            last["irradiancia_wm2"] = last["irradiancia_wm2"] * 0.5
        elif kind == "w_kelvin":
            last["temp_ambient_c"] += 273.15
            last["temp_panel_c"] += 273.15
        elif kind == "w_swapped":
            last["temp_ambient_c"], last["temp_panel_c"] = last["temp_panel_c"], last["temp_ambient_c"]
        elif kind == "w_nan_random":
            if rng.random() < 1.0 / every:
                for k in wkeys:
                    last[k] = float("nan")
        elif kind == "w_other_names":                  # the server uses other key names for the weather
            for k, k2 in zip(wkeys, ("irradiancia", "temp_ambient", "temp_panel", "vent")):
                last[k2] = last.pop(k)
        elif kind == "all_nan_random":
            if rng.random() < 1.0 / every:
                for k in list(last):
                    last[k] = float("nan")
        elif kind == "load_nan_random":
            if rng.random() < 1.0 / every:
                last["ultim_consum_kw"] = float("nan")
        elif kind == "pv_nan_random":
            if rng.random() < 1.0 / every:
                last["ultima_solar_kw"] = float("nan")
        elif kind == "dict_none":
            obs["noves_dades_5min"] = None
        elif kind == "dict_missing":
            del obs["noves_dades_5min"]
        elif kind == "dict_none_random":
            if rng.random() < 1.0 / every:
                obs["noves_dades_5min"] = None
        elif kind == "pv_not_x12":                     # the reading is the raw inverter value
            last["ultima_solar_kw"] = last["ultima_solar_kw"] / 12.0
        elif kind == "current_hour":                   # the server gives the hour being decided, not the last one
            d = ctx["df"]
            last["ultim_consum_kw"] = float(d["load_kw"].iloc[t])
            last["ultima_solar_kw"] = float(d["pv_kw"].iloc[t])
            for k in wkeys:
                last[k] = float(d[k].iloc[t])
        elif kind == "numpy_scalars":
            for k in list(last):
                last[k] = np.float32(last[k])
            obs["soc_kwh"] = np.float64(obs["soc_kwh"])
            obs["hour"] = np.int64(obs["hour"])
        elif kind == "ints":
            for k in list(last):
                v = last[k]
                last[k] = int(v) if v == v else v
        return obs
    return f


def clean_folds():
    f = get_folds()
    return {k: f[k] for k in ("A", "B", "C")}
