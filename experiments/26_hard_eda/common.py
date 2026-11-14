"""Shared helpers for the hard-mode EDA figures (experiments/26_hard_eda).

Everything is computed on train rows only: the organisers' hourly table (5,993 h, solar x12) and the
test-like folds A / B / C of src/hard.py. The oracle takes about 0.2 s per fold, so nothing is cached.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hard import CAP, ETA, PMAX, SOC0, get_folds, load_hard, score_weights  # noqa: E402
from src.hard_mpc import KAPPA  # noqa: E402
from src.sim import solve_lp  # noqa: E402

FIGS = ROOT / "reports/15_hard_eda/figs"
HERE = Path(__file__).parent
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
MONTH_NUM = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
TOL = 1.0   # kW tolerance when classifying LP solutions


def signal(df: pd.DataFrame) -> pd.Series:
    """Score signal per kWh of grid import, w = co2 + KAPPA * price (kg CO2-equivalent per kWh)."""
    return df["co2_kg_per_kwh"] + KAPPA * df["precio_eur_per_kwh"]


def run_oracle(df: pd.DataFrame, net=None, cap=CAP, pmax=PMAX, soc0=SOC0) -> pd.DataFrame:
    """Perfect-foresight optimum on `df`. `net` overrides load - solar as the LP's net load (used to
    remove the free surplus); the score is always measured against the baseline max(net, 0).
    Returns the hourly trace with columns charge, discharge, p_kw, soc_kwh, grid_kw, net_kw, base_kw,
    w_score (score weight per kWh, fraction units) and pts (score points earned in that hour)."""
    if net is None:
        net = (df["load_kw"] - df["pv_kw"]).to_numpy(float)
    net = np.asarray(net, float)
    base = np.maximum(net, 0.0)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    w = 0.7 * co2 / np.sum(base * co2) + 0.3 * price / np.sum(base * price)
    c, d, s, g, _ = solve_lp(w, net, soc0=soc0, cap=cap, pmax=pmax)
    return pd.DataFrame({"charge": c, "discharge": d, "p_kw": c - d, "soc_kwh": s, "grid_kw": g,
                         "net_kw": net, "base_kw": base, "w_score": w,
                         "pts": 100.0 * w * (base - g)}, index=df.index)


def folds_abc() -> dict:
    f = get_folds()
    return {k: f[k] for k in ("A", "B", "C")}
