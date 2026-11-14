"""
experiments/21_load/cal_lib.py
Helpers for the holiday-calendar study: the h3 stack on folds A, B, C without August (taped signal and
solar forecasts; only the load forecast is replaced), per-day paired accounting. Not a runtime module.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "experiments" / "27_combined", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import combined as cb  # noqa: E402
import combined3 as c3  # noqa: E402
from src.hard import get_folds, score_weights, simulate  # noqa: E402
from src.sim import score_from_grid  # noqa: E402
from fit_table import load_fit_table  # noqa: E402
from load_fc import NS_H, LoadFC, day0, etype  # noqa: E402
from cal_fc import CALENDAR, CalLoadFC, d0_of  # noqa: E402

CUT = "2026-08-01"                      # the test run ends on 31 July 2026: folds are evaluated without August
LOW = pd.read_csv(HERE / "holidays.csv")
LOW_DATES = [pd.Timestamp(d) for d in LOW["date"]]
UNEXPLAINED = [pd.Timestamp("2026-02-12")]


def fold_full(f):
    return get_folds()[f]


def fold_cut(f):
    df = get_folds()[f]
    return df[df.index < CUT]


def end_ts(f):
    return pd.Timestamp(2026, 7, cb.FOLD_DAYS[f][1], 23)


_T = {}


def base_tape(f):
    if f not in _T:
        z = np.load(cb.CACHE / f"tape_{f}.npz")
        t = {k: z[k] for k in z.files}
        t["w_sig2"] = np.load(cb.CACHE / f"tape2_{f}.npz")["w_sig2"]
        _T[f] = t
    return _T[f]


def fit_table_for(f):
    return cb.fit_tables(fold_full(f))[1]


def record_load(fc, df):
    """(F (n, 36), Q (n, 45) float32): the load forecasts of a run, recorded exactly as the tapes were."""
    out, _ = cb.record(df, {"load_fc": fc}, quantiles=True)
    return out["load_fc"], out["load_fc_q"]


def with_load(tape, F, Q):
    t = dict(tape)
    t["load_fc"], t["load_fc_q"] = F, Q
    return t


def run_h3(f, df, tape):
    """h3 (make_agent3 defaults, as packaged) on df with the run ending at df's last July hour."""
    ag = c3.make_agent3(df, tape=tape, end_ts=end_ts(f))
    r = simulate(ag, df, return_trace=True)
    tr = r.pop("trace")
    return r, tr


def week_pos(df, f):
    """Hours since the week's first midnight, per row."""
    lo = cb.FOLD_DAYS[f][0]
    start = pd.to_datetime(pd.Series(df.index.strftime("%Y-%m")) + f"-{lo:02d}").to_numpy()
    return ((df.index.to_numpy() - start) / np.timedelta64(1, "h")).astype(int)


def masked_score(df, grid, mask=None):
    return score_from_grid(np.asarray(grid, float), df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float),
                           df["precio_eur_per_kwh"].to_numpy(float), df["co2_kg_per_kwh"].to_numpy(float), None, mask)


def deltas(df, grid_ref, grid_var):
    """Per-hour gain of `var` over `ref`: (fold-score points, kg CO2, EUR). Points sum to the score difference."""
    d = np.asarray(grid_ref, float) - np.asarray(grid_var, float)
    return (100.0 * score_weights(df) * d, df["co2_kg_per_kwh"].to_numpy(float) * d,
            df["precio_eur_per_kwh"].to_numpy(float) * d)


def window(df, start, days=1):
    a = pd.Timestamp(start)
    return np.asarray((df.index >= a) & (df.index < a + pd.Timedelta(days=days)))


def true_index(dates, n_after=0):
    """{meter-day start: true holiday index} for the given dates (and the n_after weekdays after each), from
    the whole train load table with the references a run would have that morning."""
    full = LoadFC(load_fit_table(), use_model=False, oof=False)
    out = {}
    todo = []
    for d in dates:
        todo.append(pd.Timestamp(d))
        x, k = pd.Timestamp(d), 0
        while k < n_after:
            x = x + pd.Timedelta(days=1)
            if x.dayofweek < 5:
                todo.append(x)
                k += 1
    for d in todo:
        d0 = d0_of(d)
        num, den, n = full._lam_parts(d0, d0 + 16)
        if n >= 8 and den > 1e4 * n:
            out[d0] = float(min(max(num / den, -1.5), 1.3))
    return out
