"""
src/sim.py
Evaluation harness: fast simulator (bit-for-bit port of starter_kit/environment.py),
test-like validation folds, and the perfect-foresight oracle (LP upper bound).

Typical use
-----------
    from src.sim import get_folds, simulate, oracle
    for name, df in get_folds("filled").items():
        res = simulate(MyAgent(), df)          # dict with score, co2/cost savings, infractions
        ub = oracle(df)                        # best achievable score with full foresight
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from src.data import CORE_COLS, TEST_MONTHS, load_hourly_filled, load_hourly_official

ROOT = Path(__file__).resolve().parents[1]

CAP = 2000.0
PMAX = 500.0
ETA = 0.9 ** 0.5          # one-way efficiency (charge and discharge)
SOC0 = 1000.0
ALPHA, BETA = 0.7, 0.3    # score weights: CO2, cost
PENALTY = 0.002           # per infraction, in score fraction (= 0.20 points)


# --------------------------------------------------------------------------------------
# Validation folds
# --------------------------------------------------------------------------------------
def get_folds(table: str = "filled") -> dict:
    """Test-like folds built from the train set.

    The hidden test = last 7 days of each month, 12 weeks concatenated (SOC carries over the
    month-long gaps). We mimic it with three folds made of train days:
        A = days 1-7 of every month, B = days 8-14, C = days 15-21.
    Each fold is the 12 weeks concatenated in calendar order (2,016 h with table="filled").

    table = "filled"   : all train hours, missing load/pv imputed (recommended for comparing
                         policies: uses all the price/CO2 data).
            "official" : organisers' recipe with NaN rows dropped (Nov 2025 etc. missing).
    Also returns "ALL": every train day concatenated (lowest-noise comparisons).
    """
    df = load_hourly_filled() if table == "filled" else load_hourly_official()
    df = df[CORE_COLS]
    day = df.index.day
    ym = list(zip(df.index.year, df.index.month))
    in_test_months = np.array([k in TEST_MONTHS for k in ym])
    folds = {}
    for name, (lo, hi) in {"A": (1, 7), "B": (8, 14), "C": (15, 21)}.items():
        m = (day >= lo) & (day <= hi) & in_test_months
        folds[name] = df[m]
    folds["ALL"] = df[in_test_months]
    return folds


def train_complement(fold_df: pd.DataFrame, table: str = "full") -> pd.DataFrame:
    """Train rows NOT in the given fold (what an agent may be fitted on when evaluated on it)."""
    from src.data import load_hourly_full
    src = {"full": load_hourly_full, "filled": load_hourly_filled, "official": load_hourly_official}[table]()
    return src[~src.index.isin(fold_df.index)]


# --------------------------------------------------------------------------------------
# Fast simulator (same logic, same numbers as starter_kit/environment.py)
# --------------------------------------------------------------------------------------
def simulate(agent, df: pd.DataFrame, mask=None, return_trace: bool = False) -> dict:
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_kw"].to_numpy(float)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    ts = df.index
    n = len(df)
    price_l, co2_l = price.tolist(), co2.tolist()
    hours = ts.hour.to_numpy()
    dows = ts.day_of_week.to_numpy()

    soc = SOC0
    grid = np.zeros(n)
    acts = np.zeros(n)
    socs = np.zeros(n)
    infr = np.zeros(n, dtype=int)
    for t in range(n):
        h_end = min(t + 24, n)
        tl = t - 1 if t > 0 else t
        obs = {
            "timestamp": ts[t],
            "hour": int(hours[t]),
            "day_of_week": int(dows[t]),
            "soc_kwh": float(soc),
            "soc_pct": float(soc / CAP),
            "soc_percent": float(soc / CAP * 100),
            "cap_max_kwh": CAP,       # added to the official obs on 2026-11-14 (hard-mode update)
            "power_max_kw": PMAX,
            "preu_actual": price_l[t],
            "co2_actual": co2_l[t],
            "preus_properes_24h": price_l[t:h_end],
            "co2_properes_24h": co2_l[t:h_end],
            "noves_dades_5min": {"ultim_consum_kw": float(load[tl]), "ultima_solar_kw": float(pv[tl])},
        }
        p_req = float(agent.act(obs))
        k = 0
        p = min(max(p_req, -PMAX), PMAX)
        if p != p_req:
            k += 1
        if p > 0:
            max_c = (CAP - soc) / ETA
            p_act = min(p, max(0.0, max_c))
            if p_act < p:
                k += 1
            soc += p_act * ETA
        elif p < 0:
            max_d = soc * ETA
            p_act = -min(-p, max(0.0, max_d))
            if p_act > p:
                k += 1
            soc += p_act / ETA
        else:
            p_act = 0.0
        if soc < -1e-6 or soc > CAP + 1e-6:
            k += 1
        soc = min(max(soc, 0.0), CAP)
        grid[t] = max(load[t] - pv[t] + p_act, 0.0)
        acts[t] = p_act
        socs[t] = soc
        infr[t] = k

    res = score_from_grid(grid, load, pv, price, co2, infr, mask)
    if return_trace:
        res["trace"] = pd.DataFrame({"p_kw": acts, "soc_kwh": socs, "grid_kw": grid, "infr": infr}, index=ts)
    return res


def score_from_grid(grid, load, pv, price, co2, infr=None, mask=None) -> dict:
    base = np.maximum(load - pv, 0.0)
    infr = np.zeros(len(grid), dtype=int) if infr is None else infr
    if mask is not None:
        m = np.asarray(mask, bool)
        grid, base, price, co2, infr = grid[m], base[m], price[m], co2[m], infr[m]
    co2_b, cost_b = float(np.sum(base * co2)), float(np.sum(base * price))
    co2_a, cost_a = float(np.sum(grid * co2)), float(np.sum(grid * price))
    d_co2 = (co2_b - co2_a) / co2_b if co2_b > 0 else 0.0
    d_cost = (cost_b - cost_a) / cost_b if cost_b > 0 else 0.0
    n_inf = int(np.sum(infr))
    score = ALPHA * d_co2 + BETA * d_cost - PENALTY * n_inf
    return {
        "score": max(0.0, score * 100),          # unrounded (the platform rounds to 2 decimals)
        "co2_pct": d_co2 * 100,
        "cost_pct": d_cost * 100,
        "co2_saved_kg": co2_b - co2_a,
        "cost_saved_eur": cost_b - cost_a,
        "co2_base_kg": co2_b,
        "cost_base_eur": cost_b,
        "infractions": n_inf,
    }


def simulate_official(agent, df: pd.DataFrame) -> dict:
    """Run the organisers' own environment (slow path, used to cross-check `simulate`)."""
    sk = str(ROOT / "starter_kit")
    if sk not in sys.path:
        sys.path.insert(0, sk)
    from environment import run_agent_simulation
    return run_agent_simulation(agent, df[CORE_COLS])


# --------------------------------------------------------------------------------------
# Perfect-foresight oracle
# --------------------------------------------------------------------------------------
def score_weights(df: pd.DataFrame) -> np.ndarray:
    """Per-kWh weight of grid import in the score (fraction units): score = 1 - sum(w * grid)."""
    base = np.maximum(df["load_kw"].to_numpy(float) - df["pv_kw"].to_numpy(float), 0.0)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    return ALPHA * co2 / np.sum(base * co2) + BETA * price / np.sum(base * price)


def solve_lp(w, net, soc0=SOC0, soc_end_value=0.0, cap=CAP, pmax=PMAX):
    """min sum_t w_t * g_t - soc_end_value * s_T
       s.t. s_t = s_{t-1} + ETA*c_t - d_t/ETA,  0<=s<=CAP,  0<=c,d<=PMAX,  g_t >= net_t + c_t - d_t,  g>=0
    Returns (c, d, s, g, objective)."""
    T = len(w)
    I = sparse.identity(T, format="csr")
    L = sparse.diags([np.ones(T), -np.ones(T - 1)], [0, -1], format="csr")   # s_t - s_{t-1}
    Z = sparse.csr_matrix((T, T))
    # x = [c, d, s, g]
    A_eq = sparse.hstack([-ETA * I, I / ETA, L, Z], format="csr")
    b_eq = np.zeros(T); b_eq[0] = soc0
    A_ub = sparse.hstack([I, -I, Z, -I], format="csr")                         # c - d - g <= -net
    b_ub = -np.asarray(net, float)
    # NOTE: score weights are ~1e-7 per kWh, below HiGHS' tolerances -> rescale the objective
    scale = 1.0 / float(np.mean(np.abs(w)))
    cost = np.concatenate([np.zeros(2 * T), np.zeros(T), np.asarray(w, float) * scale])
    cost[3 * T - 1] -= soc_end_value * scale
    bounds = [(0, pmax)] * (2 * T) + [(0, cap)] * T + [(0, None)] * T
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    x = r.x
    return x[:T], x[T:2 * T], x[2 * T:3 * T], x[3 * T:], r.fun / scale


def oracle(df: pd.DataFrame, return_trace: bool = False) -> dict:
    """Best achievable score on `df` knowing everything in advance (true upper bound)."""
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_kw"].to_numpy(float)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    w = score_weights(df)
    c, d, s, g, _ = solve_lp(w, load - pv)
    res = score_from_grid(g, load, pv, price, co2)
    if return_trace:
        res["trace"] = pd.DataFrame({"p_kw": c - d, "soc_kwh": s, "grid_kw": g}, index=df.index)
    return res


class ReplayAgent:
    """Plays back a fixed power schedule (used to push oracle plans through the simulator)."""
    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.i = 0

    def act(self, obs):
        p = self.schedule[self.i]
        self.i += 1
        soc = obs["soc_kwh"]
        return float(min(max(p, -max(0.0, soc * ETA - 1e-6)), max(0.0, (CAP - soc) / ETA - 1e-6)))
