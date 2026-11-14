"""Shared helpers for experiments/23_signal: fit tables per fold and an offline replay."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.hard import get_folds, load_hard  # noqa: E402
from src.hard_mpc import KAPPA  # noqa: E402

FOLD_DAYS = {"A": (1, 7), "B": (8, 14), "C": (15, 21)}


def signal_fit_table(fold: str) -> pd.DataFrame:
    """Every train hour (price and CO2 are complete in the filled table) except the fold's days.
    Excluding by day of month also removes the few fold hours the official table dropped."""
    full = load_hard("filled")
    if fold is None:
        return full
    lo, hi = FOLD_DAYS[fold]
    d = full.index.day
    return full[~((d >= lo) & (d <= hi))]


def replay(fc, df, H=48, kappa=KAPPA, parts=True):
    """Run a forecaster over the rows of df as the agent would. Returns arrays (n, H) of the
    forecasts (NaN beyond the end of the run) for co2, price (if parts) and w."""
    n = len(df)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    ts = df.index
    hist = []
    W = np.full((n, H), np.nan)
    Cc = np.full((n, H), np.nan)
    Pp = np.full((n, H), np.nan)
    for t in range(n):
        obs = {"timestamp": ts[t], "hour": int(ts[t].hour), "day_of_week": int(ts[t].dayofweek),
               "preu_actual": price[t], "co2_actual": co2[t],
               "preus_properes_24h": price[t:t + 3].tolist(), "co2_properes_24h": co2[t:t + 3].tolist(),
               "noves_dades_5min": {}}
        hist.append({"ts": ts[t], "co2": co2[t], "price": price[t], "w": co2[t] + kappa * price[t]})
        m = min(H, n - t)
        if parts and hasattr(fc, "predict_parts"):
            c, p = fc.predict_parts(obs, hist, H)
            Cc[t, :m], Pp[t, :m] = c[:m], p[:m]
            W[t, :m] = (c + kappa * p)[:m]
        else:
            w = np.asarray(fc.predict(obs, hist, H), float)
            W[t, :m] = w[:m]
    return Cc, Pp, W


def truth(df, H=48, kappa=KAPPA):
    n = len(df)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    out = {}
    for name, x in (("c", co2), ("p", price), ("w", co2 + kappa * price)):
        T = np.full((n, H), np.nan)
        for k in range(H):
            T[:n - k, k] = x[k:]
        out[name] = T
    return out


def rank_corr_rows(F, T, lo=3, hi=27):
    """Spearman correlation between forecast and truth over horizons lo..hi-1, per origin."""
    from scipy.stats import rankdata
    out = np.full(len(F), np.nan)
    for i in range(len(F)):
        f, t = F[i, lo:hi], T[i, lo:hi]
        if np.isfinite(f).all() and np.isfinite(t).all():
            out[i] = np.corrcoef(rankdata(f), rankdata(t))[0, 1]
    return out
