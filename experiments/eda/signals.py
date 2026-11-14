"""
experiments/eda/signals.py
Analysis helpers shared by the CO2 / price / combined-signal scripts (Q4-Q8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C


# ----------------------------------------------------------------------------- per-day statistics
def daily_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per train day (all 24 hours present) with the intraday spread of `col`.

    ratio          max / min of the 24 values, order ignored
    ratio_ordered  best ratio with the cheap hour BEFORE the dear hour (charge, then discharge,
                   inside the same calendar day)
    """
    m = C.day_matrix(df[col])
    x = m.values
    out = pd.DataFrame(index=m.index)
    out["mean"] = x.mean(axis=1)
    out["min"] = x.min(axis=1)
    out["max"] = x.max(axis=1)
    out["h_min"] = x.argmin(axis=1)
    out["h_max"] = x.argmax(axis=1)
    out["ratio"] = out["max"] / out["min"]
    run_min = np.minimum.accumulate(x, axis=1)
    out["ratio_ordered"] = (x / run_min).max(axis=1)
    out["spread"] = out["max"] - out["min"]
    out["std"] = x.std(axis=1)
    # mean of the 4 cheapest and the 4 dearest hours: the battery needs ~4 h to fill or empty
    xs = np.sort(x, axis=1)
    out["low4"] = xs[:, :4].mean(axis=1)
    out["high4"] = xs[:, -4:].mean(axis=1)
    out["ratio4"] = out["high4"] / out["low4"]
    out["month"] = m.index.month
    out["weekend"] = C.is_weekend(m.index)
    out["season"] = [C.SEASON_OF_MONTH[mm] for mm in m.index.month]
    return out


# ----------------------------------------------------------------------------- zigzag swings
def zigzag(x: np.ndarray, thr: float = C.BREAK_EVEN) -> list[tuple[int, str]]:
    """Significant turning points of a positive series: consecutive pivots differ by at least
    the factor `thr`. Returns [(index, "min" | "max"), ...] alternating."""
    n = len(x)
    piv: list[tuple[int, str]] = []
    lo = hi = 0
    trend = 0  # 0 unknown, +1 rising (looking for a max), -1 falling (looking for a min)
    for i in range(1, n):
        if trend == 0:
            if x[i] > x[hi]:
                hi = i
            if x[i] < x[lo]:
                lo = i
            if x[hi] >= thr * x[lo]:
                if lo < hi:
                    piv.append((lo, "min"))
                    trend = +1
                else:
                    piv.append((hi, "max"))
                    trend = -1
                    lo = i if x[i] < x[lo] else lo
                    lo = int(np.argmin(x[hi:i + 1])) + hi
                if trend == +1:
                    hi = int(np.argmax(x[lo:i + 1])) + lo
        elif trend == +1:
            if x[i] > x[hi]:
                hi = i
            elif x[hi] >= thr * x[i]:
                piv.append((hi, "max"))
                trend = -1
                lo = i
        else:
            if x[i] < x[lo]:
                lo = i
            elif x[i] >= thr * x[lo]:
                piv.append((lo, "min"))
                trend = +1
                hi = i
    # the last running extreme closes a swing if it is large enough
    if trend == +1 and piv and x[hi] >= thr * x[piv[-1][0]]:
        piv.append((hi, "max"))
    elif trend == -1 and piv and x[piv[-1][0]] >= thr * x[lo]:
        piv.append((lo, "min"))
    return piv


def upswings(x: np.ndarray, thr: float = C.BREAK_EVEN) -> list[tuple[int, int]]:
    """(trough index, peak index) of every profitable charge -> discharge swing."""
    piv = zigzag(x, thr)
    return [(a[0], b[0]) for a, b in zip(piv[:-1], piv[1:]) if a[1] == "min" and b[1] == "max"]


def swings_per_day(df: pd.DataFrame, col: str, thr: float = C.BREAK_EVEN) -> pd.DataFrame:
    """Run the zigzag on every contiguous train block (so that swings may cross midnight) and
    count the profitable upswings per day, attributed to the day of the peak (the discharge)."""
    s = df[col]
    days = pd.DatetimeIndex(sorted(s.index.normalize().unique()))
    blocks = C.block_id(days)
    rows = []
    for b in np.unique(blocks):
        bdays = days[blocks == b]
        seg = s[(s.index >= bdays[0]) & (s.index < bdays[-1] + pd.Timedelta(days=1))]
        ups = upswings(seg.values, thr)
        peak_days = pd.Series([seg.index[j].normalize() for _, j in ups], dtype="datetime64[ns]")
        cnt = peak_days.value_counts()
        for d in bdays:
            rows.append((d, int(cnt.get(d, 0)), b))
    out = pd.DataFrame(rows, columns=["date", "swings", "block"]).set_index("date")
    return out


def swing_table(df: pd.DataFrame, col: str, thr: float = C.BREAK_EVEN) -> pd.DataFrame:
    """Every profitable upswing: trough/peak timestamps, hours and ratio."""
    s = df[col]
    days = pd.DatetimeIndex(sorted(s.index.normalize().unique()))
    blocks = C.block_id(days)
    rows = []
    for b in np.unique(blocks):
        bdays = days[blocks == b]
        seg = s[(s.index >= bdays[0]) & (s.index < bdays[-1] + pd.Timedelta(days=1))]
        for i, j in upswings(seg.values, thr):
            rows.append((seg.index[i], seg.index[j], seg.values[i], seg.values[j]))
    out = pd.DataFrame(rows, columns=["t_trough", "t_peak", "v_trough", "v_peak"])
    out["ratio"] = out["v_peak"] / out["v_trough"]
    out["h_trough"] = out["t_trough"].dt.hour
    out["h_peak"] = out["t_peak"].dt.hour
    out["month"] = out["t_peak"].dt.month
    return out


# ----------------------------------------------------------------------------- lagged pairs
def lagged(s: pd.Series, lag_h: int) -> tuple[np.ndarray, np.ndarray]:
    """(x_t, x_{t-lag}) for every train hour whose calendar lag also exists in train.
    Works across the test-week holes because it aligns on real timestamps."""
    prev = s.reindex(s.index - pd.Timedelta(hours=lag_h))
    ok = ~np.isnan(prev.values)
    return s.values[ok], prev.values[ok]


def acf(s: pd.Series, lags) -> pd.Series:
    out = {}
    for L in lags:
        a, b = lagged(s, L)
        out[L] = np.corrcoef(a, b)[0, 1]
    return pd.Series(out)
