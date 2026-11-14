"""
experiments/02_stress/lib.py
Shared helpers for the robustness work: agent loading, the "test calendar" relabelling of the
validation folds, scenario builders (dropped weeks, zero weeks, outages, low load, NaN readings),
a cached oracle and a thin wrapper around src.sim.simulate that reports the *unclamped* score.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim import get_folds, oracle, simulate  # noqa: E402

CORE = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]
FOLD_LO = {"A": 1, "B": 8, "C": 15}
WEEK = 168
# position of each calendar month inside a 12-week run (Sep 2025 .. Aug 2026)
WEEK_OF = {"sep": 0, "oct": 1, "nov": 2, "dec": 3, "jan": 4, "feb": 5, "mar": 6, "apr": 7,
           "may": 8, "jun": 9, "jul": 10, "aug": 11}


# ----------------------------------------------------------------------------------------------
# agents
# ----------------------------------------------------------------------------------------------
def load_agent_module(path, name=None):
    """Import an agent.py by path (never touches sys.modules of other agents)."""
    path = Path(path).resolve()
    name = name or "agent_" + hashlib.md5(str(path).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_noguard(mod):
    """Same optimiser with the discharge cap and outage logic disabled (None if not applicable)."""
    cls = mod.BatteryAgent
    if not hasattr(cls, "_discharge_limits"):
        return None
    pmax = float(getattr(mod, "PMAX", 500.0))

    class NoGuard(cls):
        def _discharge_limits(self, obs, horizon):
            return pmax, 0

    return NoGuard


class ObsPatch:
    """Wraps an agent and replaces the last-hour readings it sees (the physics keep the truth).
    obs_load / obs_pv are per-row "what the sensor reported" arrays aligned with the run."""

    def __init__(self, agent, obs_load, obs_pv):
        self.agent, self.obs_load, self.obs_pv, self.t = agent, obs_load, obs_pv, 0

    def act(self, obs):
        tl = self.t - 1 if self.t > 0 else 0
        obs = dict(obs)
        obs["noves_dades_5min"] = {"ultim_consum_kw": float(self.obs_load[tl]),
                                   "ultima_solar_kw": float(self.obs_pv[tl])}
        self.t += 1
        return self.agent.act(obs)


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def test_calendar(df, lo):
    """Relabel a fold so that every 7-day block sits on the LAST 7 days of its month, like the
    hidden test (weeks end at 23:00 on the last day of the month). Data are untouched; only the
    timestamps move (by a whole number of days per month), so weekday labels no longer match the
    real weekday of the data. `lo` = first day-of-month of the fold (A: 1, B: 8, C: 15)."""
    idx = df.index
    shift = pd.to_timedelta(idx.days_in_month - 6 - lo, unit="D")
    out = df.copy()
    out.index = pd.DatetimeIndex(idx + shift, name=idx.name)
    return out


_FOLDS = {}


def folds(calendar="test"):
    """Clean 12-week folds A, B, C (2,016 h each). calendar='test' relabels them to month ends."""
    if calendar not in _FOLDS:
        raw = get_folds("filled")
        _FOLDS[calendar] = {k: (test_calendar(raw[k][CORE], FOLD_LO[k]) if calendar == "test" else raw[k][CORE].copy())
                            for k in ("A", "B", "C")}
    return _FOLDS[calendar]


def wk(i):
    return slice(WEEK * i, WEEK * (i + 1))


def drop_weeks(df, weeks):
    keep = np.ones(len(df), bool)
    for i in weeks:
        keep[wk(i)] = False
    return df[keep]


def zero_week(df, i, pv_too=True):
    out = df.copy()
    out.iloc[wk(i), out.columns.get_loc("load_kw")] = 0.0
    if pv_too:
        out.iloc[wk(i), out.columns.get_loc("pv_kw")] = 0.0
    return out


def zero_pv(df, weeks):
    out = df.copy()
    for i in weeks:
        out.iloc[wk(i), out.columns.get_loc("pv_kw")] = 0.0
    return out


def outages(df, dur, start_hour, weeks=range(12), day=2):
    """Meter outage (load AND pv read 0, as in the real one of 17-18 June 2026) of `dur` hours
    starting at `start_hour` on day `day` (0-based) of each week in `weeks`."""
    out = df.copy()
    jl, jp = out.columns.get_loc("load_kw"), out.columns.get_loc("pv_kw")
    for i in weeks:
        a = WEEK * i + 24 * day + start_hour
        b = min(a + dur, WEEK * (i + 1))          # never spill into the next week (a month later)
        out.iloc[a:b, jl] = 0.0
        out.iloc[a:b, jp] = 0.0
    return out


def glitches(df, dur_h, start_h, weeks=range(12), day=2, pv_too=True):
    """Logger glitch with sub-hour timing, like the real ones: starts at fractional hour
    `start_h` (e.g. 7.58 = 07:35) of day `day` and lasts `dur_h` hours (fractional). Every hourly
    mean is scaled by the share of the hour that was NOT covered, so the first and last hours
    are partial. pv_too=False leaves the inverters alive (meter-only glitch -> pv surplus)."""
    out = df.copy()
    jl, jp = out.columns.get_loc("load_kw"), out.columns.get_loc("pv_kw")
    for i in weeks:
        a = WEEK * i + 24 * day + start_h           # in hours from the start of the run
        b = min(a + dur_h, WEEK * (i + 1))
        for h in range(int(np.floor(a)), int(np.ceil(b))):
            cover = max(0.0, min(b, h + 1) - max(a, h))
            out.iloc[h, jl] = out.iloc[h, jl] * (1.0 - cover)
            if pv_too:
                out.iloc[h, jp] = out.iloc[h, jp] * (1.0 - cover)
    return out


def glitches_frozen(df, dur_h, start_h, weeks=range(12), day=2):
    """Like glitches() (load and pv), but honouring the meter's night freeze found by the EDA:
    it is not polled 22:00-06:00 and the hourly table holds the 22:00 value. So a glitch that is
    active at 22:00 reads 0 until 06:00 whatever its real end, and one that starts and ends
    during the night leaves no trace."""
    out = df.copy()
    jl, jp = out.columns.get_loc("load_kw"), out.columns.get_loc("pv_kw")
    for i in weeks:
        w0 = WEEK * i
        a = w0 + 24 * day + start_h
        b = min(a + dur_h, w0 + WEEK)
        cover = np.zeros(WEEK)
        for h in range(int(np.floor(a)), int(np.ceil(b))):
            cover[h - w0] = max(0.0, min(b, h + 1) - max(a, h))
        for d in range(7):
            t22 = w0 + 24 * d + 22
            lo, hi = 24 * d + 22, min(WEEK, 24 * d + 30)
            cover[lo:hi] = 1.0 if (a <= t22 < b) else 0.0
        for h in range(WEEK):
            if cover[h] > 0:
                out.iloc[w0 + h, jl] = out.iloc[w0 + h, jl] * (1.0 - cover[h])
                out.iloc[w0 + h, jp] = out.iloc[w0 + h, jp] * (1.0 - cover[h])
    return out


def scale_load(df, f, weeks=None):
    out = df.copy()
    j = out.columns.get_loc("load_kw")
    if weeks is None:
        out.iloc[:, j] = out.iloc[:, j] * f
    else:
        for i in weeks:
            out.iloc[wk(i), j] = out.iloc[wk(i), j] * f
    return out


def nan_blocks(n, n_blocks=12, length=4, seed=0):
    """Boolean mask with `n_blocks` runs of `length` hours (one per week, random start)."""
    rng = np.random.default_rng(seed)
    m = np.zeros(n, bool)
    for i in range(n_blocks):
        a = WEEK * i + int(rng.integers(0, WEEK - length))
        m[a:a + length] = True
    return m[:n]


# ----------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------
def raw_score(res):
    """Score in points WITHOUT the max(0, .) clamp of the official evaluator."""
    return 0.7 * res["co2_pct"] + 0.3 * res["cost_pct"] - 0.2 * res["infractions"]


def run(agent, df, obs_load=None, obs_pv=None, trace=False):
    if obs_load is not None or obs_pv is not None:
        ol = df["load_kw"].to_numpy(float) if obs_load is None else obs_load
        op = df["pv_kw"].to_numpy(float) if obs_pv is None else obs_pv
        agent = ObsPatch(agent, ol, op)
    res = simulate(agent, df, return_trace=trace)
    res["raw"] = raw_score(res)
    return res


_ORACLE_FILE = OUT / "oracle_cache.json"
_ORACLE = None


def oracle_score(df):
    """Perfect-foresight upper bound for this exact table (cached on disk by content hash)."""
    global _ORACLE
    if _ORACLE is None:
        try:
            _ORACLE = json.loads(_ORACLE_FILE.read_text())
        except Exception:
            _ORACLE = {}
    key = hashlib.sha1(np.ascontiguousarray(df[CORE].to_numpy(float)).tobytes()).hexdigest()
    if key not in _ORACLE:
        _ORACLE[key] = float(oracle(df)["score"])
        OUT.mkdir(exist_ok=True)
        _ORACLE_FILE.write_text(json.dumps(_ORACLE))
    return _ORACLE[key]


# ----------------------------------------------------------------------------------------------
# scenario suite
# ----------------------------------------------------------------------------------------------
def suite(calendar="test", fold_names=("A", "B", "C"), kind="harness"):
    """List of scenarios. Each is a dict:
        name, family, fold, df (the table the evaluator would hold), events (number of injected
        events, for per-event regret), obs_load / obs_pv (what the sensors report, if different).
    kind="harness": compact set for stress.py; kind="tune": more start hours / positions."""
    F = folds(calendar)
    S = []

    def add(name, family, fold, df, events=0, obs_load=None, obs_pv=None):
        S.append(dict(name=name, family=family, fold=fold, df=df, events=events,
                      obs_load=obs_load, obs_pv=obs_pv))

    tune = kind == "tune"
    for f in fold_names:
        base = F[f]
        n = len(base)
        add("clean", "a.clean", f, base)
        # b. weeks dropped
        add("drop Nov week", "b.dropped", f, drop_weeks(base, [2]))
        add("drop Aug week", "b.dropped", f, drop_weeks(base, [11]))
        add("drop Nov+Aug weeks", "b.dropped", f, drop_weeks(base, [2, 11]))
        # c. zero weeks
        for lab, i in (("first/Sep", 0), ("Nov", 2), ("Mar", 6), ("last/Aug", 11)):
            add("zero load+pv week: %s" % lab, "c.zero-week", f, zero_week(base, i, True), 1)
        if tune:
            for i in (1, 3, 4, 5, 7, 8, 9, 10):
                add("zero load+pv week: #%d" % i, "c.zero-week", f, zero_week(base, i, True), 1)
        add("zero load week (pv real): Nov", "c.zero-week", f, zero_week(base, 2, False), 1)
        add("zero pv week: Aug", "c.zero-pv", f, zero_pv(base, [11]))
        add("zero pv whole run", "c.zero-pv", f, zero_pv(base, range(12)))
        # d. outages: start at evening peak (20h), midday trough (14h), ramp (17h), morning peak (7h)
        hours = (20, 14, 17, 7) if tune else (20, 14, 17)
        for dur, weeks in ((1, range(12)), (3, range(12)), (14, range(12)), (72, (1, 4, 7, 10))):
            for h in hours:
                add("outage %2dh from %02d:00 (x%d)" % (dur, h, len(weeks)), "d.outage-%dh" % dur, f,
                    outages(base, dur, h, weeks), len(weeks))
        # g. realistic logger glitches: sub-hour timing (first / last hour partial), night freeze
        for dur, lab in ((25 / 60, "25min"), (110 / 60, "110min"), (13.9, "13.9h")):
            for st, sl in ((7.58, "07:35"), (19.4, "19:24")):
                add("glitch %s from %s (x12)" % (lab, sl), "g.glitch", f, glitches_frozen(base, dur, st), 12)
        # e. low load
        for s in (0.6, 0.45, 0.3):
            add("load x%.2f, Dec week" % s, "e.low-load-week", f, scale_load(base, s, [3]))
            add("load x%.2f, whole run" % s, "e.low-load-run", f, scale_load(base, s))
        # f. NaN readings: the sensors report NaN for 12 x 4 h; the truth is the real load ...
        m = nan_blocks(n, 12, 4, seed=7)
        ld, pv = base["load_kw"].to_numpy(float), base["pv_kw"].to_numpy(float)
        add("NaN load readings 12x4h (truth real)", "f.nan", f, base, 12, obs_load=np.where(m, np.nan, ld))
        add("NaN pv readings 12x4h (truth real)", "f.nan", f, base, 12, obs_pv=np.where(m, np.nan, pv))
        # ... or the truth is an outage (table holds 0) but the reading is NaN instead of 0
        dz = base.copy()
        dz.iloc[m, dz.columns.get_loc("load_kw")] = 0.0
        dz.iloc[m, dz.columns.get_loc("pv_kw")] = 0.0
        add("NaN readings 12x4h (truth = outage)", "f.nan", f, dz, 12,
            obs_load=np.where(m, np.nan, ld), obs_pv=np.where(m, np.nan, pv))
    return S
