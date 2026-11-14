"""
src/hard.py
Hard mode ("Mode Dificil") evaluation harness: data table, test-like folds, fast simulator and
perfect-foresight oracle. Rules: docs/NEW_RULES.md.

    battery 6,000 kWh / +-1,500 kW, SOC starts at 3,000 kWh, solar x12,
    obs carries only 3 values of price / CO2 (current hour + next 2),
    obs["noves_dades_5min"] also carries last hour's weather readings.

Typical use
-----------
    from src.hard import get_folds, simulate, oracle
    for name, df in get_folds().items():
        res = simulate(MyAgent(), df)     # dict: score, co2_pct, cost_pct, infractions, ...
        ub = oracle(df)                   # best achievable with full foresight
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import CORE_COLS, TEST_MONTHS, load_hourly_filled, load_hourly_full
from src.sim import score_from_grid, solve_lp

ROOT = Path(__file__).resolve().parents[1]

CAP = 6000.0
PMAX = 1500.0
ETA = 0.9 ** 0.5
SOC0 = 3000.0
HORIZON = 3               # price / CO2 values in obs: current hour + next 2
PV_SCALE = 12.0
ALPHA, BETA = 0.7, 0.3

# our column names in data.load_hourly_full()  ->  key names in obs["noves_dades_5min"]
METEO_MAP = {"irradiance_wm2": "irradiancia_wm2", "temp_amb_c": "temp_ambient_c",
             "temp_panel_c": "temp_panel_c", "wind_ms": "vent_ms"}
METEO_COLS = list(METEO_MAP.values())

SPEC_HARD = {"capacity_kwh": CAP, "power_kw": PMAX, "efficiency": 0.90, "soc_init": 0.50,
             "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0, "forecast_horizon": HORIZON}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_hard(table: str = "official") -> pd.DataFrame:
    """Hourly hard-mode table: CORE_COLS with pv_kw already x12, plus the 4 weather columns
    named as in obs. Extra REE columns (pct_emitting, gen_*) are kept for analysis only: they
    are NOT available to an agent at run time.

    table = "official": rows where load or pv is missing are dropped (5,993 h), as the
                        organisers' pipeline does. Recommended: in hard mode load and solar
                        drive the result, so imputed rows would flatter any forecaster.
            "filled"  : every train hour (6,768), missing load / pv imputed (column `imputed`).
    """
    full = load_hourly_full().rename(columns=METEO_MAP)
    if table == "official":
        df = full.dropna(subset=CORE_COLS).copy()
        df["imputed"] = False
    else:
        fil = load_hourly_filled()
        df = full.copy()
        df[["load_kw", "pv_kw"]] = fil[["load_kw", "pv_kw"]]
        df["imputed"] = fil["imputed"]
    df["pv_kw"] = df["pv_kw"] * PV_SCALE
    return df


def get_folds(table: str = "official") -> dict:
    """Test-like folds from train days: A = days 1-7 of every month, B = 8-14, C = 15-21,
    each concatenated in calendar order like the hidden test (last 7 days of every month).
    "ALL" = every train day. With table="official" November is missing and August stops on
    the 16th. B and C are the closest to the real test: like the test weeks, they come right
    after train days the agent may have stored (A comes right after hidden test days)."""
    df = load_hard(table)
    day = df.index.day
    ym = list(zip(df.index.year, df.index.month))
    in_test_months = np.array([k in TEST_MONTHS for k in ym])
    folds = {}
    for name, (lo, hi) in {"A": (1, 7), "B": (8, 14), "C": (15, 21)}.items():
        folds[name] = df[(day >= lo) & (day <= hi) & in_test_months]
    folds["ALL"] = df[in_test_months]
    return folds


def train_complement(fold_df: pd.DataFrame, table: str = "official") -> pd.DataFrame:
    """Train rows NOT in the fold: the only rows a model may be fitted on when it is evaluated
    on that fold."""
    src = load_hard(table)
    return src[~src.index.isin(fold_df.index)]


# --------------------------------------------------------------------------------------
# Fast simulator (same logic and numbers as starter_kit/environment.py with SPEC_HARD)
# --------------------------------------------------------------------------------------
def simulate(agent, df: pd.DataFrame, mask=None, return_trace: bool = False, horizon: int = HORIZON) -> dict:
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

    soc = SOC0
    grid = np.zeros(n)
    acts = np.zeros(n)
    reqs = np.zeros(n)
    socs = np.zeros(n)
    infr = np.zeros(n, dtype=int)
    for t in range(n):
        h_end = min(t + horizon, n)
        tl = t - 1 if t > 0 else t
        last = {"ultim_consum_kw": float(load[tl]), "ultima_solar_kw": float(pv[tl])}
        for c, a in met.items():
            last[c] = float(a[tl])
        obs = {
            "timestamp": ts[t],
            "hour": int(hours[t]),
            "day_of_week": int(dows[t]),
            "soc_kwh": float(soc),
            "soc_pct": float(soc / CAP),
            "soc_percent": float(soc / CAP * 100),
            "cap_max_kwh": CAP,
            "power_max_kw": PMAX,
            "preu_actual": price_l[t],
            "co2_actual": co2_l[t],
            "preus_properes_24h": price_l[t:h_end],
            "co2_properes_24h": co2_l[t:h_end],
            "noves_dades_5min": last,
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
        reqs[t] = p_req
        socs[t] = soc
        infr[t] = k

    res = score_from_grid(grid, load, pv, price, co2, infr, mask)
    if return_trace:
        net = load - pv
        res["trace"] = pd.DataFrame({
            "p_kw": acts, "p_req_kw": reqs, "soc_kwh": socs, "grid_kw": grid, "infr": infr,
            "net_kw": net,
            "spill_kw": np.maximum(-(net + acts), 0.0),          # energy sent to the grid for nothing
        }, index=ts)
    return res


def simulate_official(agent, df: pd.DataFrame) -> dict:
    """Run the organisers' own environment in hard mode (slow path, for cross-checks)."""
    sk = str(ROOT / "starter_kit")
    if sk not in sys.path:
        sys.path.insert(0, sk)
    from environment import run_agent_simulation
    cols = CORE_COLS + [c for c in METEO_COLS if c in df.columns]
    return run_agent_simulation(agent, df[cols], spec=dict(SPEC_HARD))


# --------------------------------------------------------------------------------------
# Perfect-foresight oracle
# --------------------------------------------------------------------------------------
def score_weights(df: pd.DataFrame) -> np.ndarray:
    """Per-kWh weight of grid import in the score (fraction units): score = 1 - sum(w * grid)."""
    base = np.maximum(df["load_kw"].to_numpy(float) - df["pv_kw"].to_numpy(float), 0.0)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    return ALPHA * co2 / np.sum(base * co2) + BETA * price / np.sum(base * price)


def rho(df: pd.DataFrame) -> float:
    """kg CO2 per EUR of baseline import: the only thing the CO2-vs-price weighting depends on."""
    base = np.maximum(df["load_kw"].to_numpy(float) - df["pv_kw"].to_numpy(float), 0.0)
    return float(np.sum(base * df["co2_kg_per_kwh"].to_numpy(float)) /
                 np.sum(base * df["precio_eur_per_kwh"].to_numpy(float)))


def oracle(df: pd.DataFrame, return_trace: bool = False) -> dict:
    """Best achievable score on `df` knowing everything in advance (true upper bound)."""
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_kw"].to_numpy(float)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    w = score_weights(df)
    c, d, s, g, _ = solve_lp(w, load - pv, soc0=SOC0, cap=CAP, pmax=PMAX)
    res = score_from_grid(g, load, pv, price, co2)
    if return_trace:
        res["trace"] = pd.DataFrame({"p_kw": c - d, "soc_kwh": s, "grid_kw": g, "net_kw": load - pv}, index=df.index)
    return res


class ReplayAgent:
    """Plays back a fixed power schedule (pushes oracle plans through the simulator)."""
    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.i = 0

    def act(self, obs):
        p = self.schedule[self.i]
        self.i += 1
        soc = obs["soc_kwh"]
        return float(min(max(p, -max(0.0, soc * ETA - 1e-6)), max(0.0, (CAP - soc) / ETA - 1e-6)))


def safe_clamp(p: float, soc: float, eps: float = 1e-4) -> float:
    max_c = max(0.0, min(PMAX, (CAP - soc) / ETA - eps))
    max_d = max(0.0, min(PMAX, soc * ETA - eps))
    return float(min(max(p, -max_d), max_c))
