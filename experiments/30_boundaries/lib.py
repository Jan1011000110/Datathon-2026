"""
experiments/30_boundaries/lib.py
Shared helpers for the week-boundary study: taped runs of h2 / h3 on folds A, B, C, the perfect-foresight
optimum of the whole run, week / position bookkeeping, masks of the public-half hypotheses.
Nothing outside this folder is modified; the combined agent is only wrapped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "experiments" / "27_combined"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import combined as cb  # noqa: E402
import combined3 as c3  # noqa: E402
from src.hard import get_folds, oracle, score_weights, simulate  # noqa: E402
from src.sim import score_from_grid  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
FOLDS = "ABC"
H2 = dict(signal="fc", planner="trust")
H3 = dict()                                    # make_agent3 defaults: round-2 signal + SigAgent lam=0.5 ar=1


def cal_end(fold):
    return pd.Timestamp(2026, 8, cb.FOLD_DAYS[fold][1], 23)


_T = {}


def tape(fold):
    if fold not in _T:
        z = np.load(cb.CACHE / f"tape_{fold}.npz")
        t = {k: z[k] for k in z.files}
        t["w_sig2"] = np.load(cb.CACHE / f"tape2_{fold}.npz")["w_sig2"]
        _T[fold] = t
    return _T[fold]


def agent(fold, df, kind="h2", **kw):
    base = dict(H2 if kind == "h2" else H3)
    base.update(kw)
    return c3.make_agent3(df, tape=tape(fold), end_ts=cal_end(fold), **base)


def run(ag, df):
    r = simulate(ag, df, return_trace=True)
    return r, r.pop("trace")


def positions(df, fold):
    """Per row: week label (YYYY-MM), hours since the week's first midnight (calendar), hours to the week's end."""
    lo = cb.FOLD_DAYS[fold][0]
    wk = df.index.strftime("%Y-%m").to_numpy()
    start = pd.to_datetime(pd.Series(wk) + f"-{lo:02d}").to_numpy()
    hs = ((df.index.to_numpy() - start) / np.timedelta64(1, "h")).astype(int)
    return wk, hs, 167 - hs


def regret(df, tr, otr):
    """Score points lost per hour against the optimum (sums to optimum score - agent score)."""
    return 100.0 * score_weights(df) * (tr["grid_kw"].to_numpy() - otr["grid_kw"].to_numpy())


def oracle_trace(fold, df):
    p = CACHE / f"oracle_{fold}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    tr = oracle(df, return_trace=True)["trace"]
    tr.to_pickle(p)
    return tr


def masked(df, grid, mask=None):
    return score_from_grid(np.asarray(grid, float), df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float),
                           df["precio_eur_per_kwh"].to_numpy(float), df["co2_kg_per_kwh"].to_numpy(float), None, mask)


def masks(df, fold, seeds=5):
    """{hypothesis: boolean mask of the PUBLIC hours}. The private half is the complement.
    Alternate weeks follow the calendar months (the test has all 12; the official folds have no November)."""
    wk, hs, _ = positions(df, fold)
    widx = pd.factorize(wk)[0]
    par = ((df.index.year * 12 + df.index.month) - (2025 * 12 + 9)).to_numpy() % 2
    out = {"first 84 h": hs < 84, "last 84 h": hs >= 84,
           "alt days (1,3,5,7)": (hs // 24) % 2 == 0, "alt days (2,4,6)": (hs // 24) % 2 == 1,
           "alt weeks (Sep, Nov, ...)": par == 0, "alt weeks (Oct, Dec, ...)": par == 1}
    for s in range(seeds):
        rng = np.random.default_rng(1000 * (ord(fold[0]) - 64) + s)
        m = np.zeros(len(df), bool)
        for w in np.unique(widx):                                   # half of the hours of each week, at random
            ii = np.flatnonzero(widx == w)
            m[rng.choice(ii, len(ii) // 2, replace=False)] = True
        out[f"random hours s{s}"] = m
    return out
