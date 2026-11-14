"""
experiments/eda/battery_lp.py
Linear-programming benchmarks for the battery, used by the EDA to measure how much value the
signals contain and how much of it a longer lookahead would unlock. Not an agent.

Physics as in starter_kit/environment.py: 2,000 kWh, +-500 kW, eta_c = eta_d = sqrt(0.9),
SOC starts at 50 %. Grid import = max(0, net + charge - discharge); discharge is bounded by the
net load so that nothing is exported for free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

ETA = 0.9 ** 0.5
CAP = 2000.0
PMAX = 500.0
SOC0 = 1000.0


def solve_lp(w: np.ndarray, net: np.ndarray, soc0: float = SOC0, terminal_value: float = 0.0):
    """Perfect-foresight optimum over len(w) hours.
    maximise  sum_t w_t (d_t - c_t) + terminal_value * soc_T
    Returns (charge, discharge, soc_after) arrays."""
    n = len(w)
    # variables: c (n), d (n), s (n)
    cost = np.concatenate([w, -w, np.zeros(n)])
    cost[-1] -= terminal_value
    eye = sp.identity(n, format="csr")
    shift = sp.diags([-np.ones(n - 1)], [-1], shape=(n, n), format="csr")
    a_eq = sp.hstack([-ETA * eye, eye / ETA, eye + shift], format="csr")
    b_eq = np.zeros(n)
    b_eq[0] = soc0
    d_max = np.clip(net, 0.0, PMAX)
    bounds = [(0.0, PMAX)] * n + [(0.0, float(v)) for v in d_max] + [(0.0, CAP)] * n
    res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    x = res.x
    return x[:n], x[n:2 * n], x[2 * n:]


class Mpc:
    """Receding-horizon controller: at every hour solve the LP on the next `horizon` hours of
    the signal it is given, apply the first action, move on."""

    def __init__(self, horizon: int):
        self.h = horizon
        self._cache: dict[int, tuple] = {}

    def _mats(self, n: int):
        if n not in self._cache:
            tri = np.tril(np.ones((n, n)))
            a = np.hstack([ETA * tri, -tri / ETA])          # soc_k - soc0
            self._cache[n] = (np.vstack([a, -a]),)
        return self._cache[n]

    def first_action(self, w: np.ndarray, net0: float, soc: float) -> float:
        n = len(w)
        (a_ub,) = self._mats(n)
        b_ub = np.concatenate([np.full(n, CAP - soc), np.full(n, soc)])
        cost = np.concatenate([w, -w])
        ub_d = np.full(n, PMAX)
        ub_d[0] = min(PMAX, max(net0, 0.0))
        bounds = np.column_stack([np.zeros(2 * n), np.concatenate([np.full(n, PMAX), ub_d])])
        res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(res.message)
        return float(res.x[0] - res.x[n])


def run_mpc(w_known: np.ndarray, net: np.ndarray, horizon: int, known: int = 24,
            w_forecast: np.ndarray | None = None, soc0: float = SOC0):
    """Simulate the receding-horizon controller on the whole series.

    w_known     true signal; the controller sees hours t .. t+known-1 exactly (as `obs` does)
    w_forecast  (T, horizon-known) array: row t holds the forecast for hours t+known .. t+horizon-1
                made at time t. None -> the true values (perfect foresight over the horizon).
    The load is observed with a one-hour delay in the real game; here the current hour's net load
    is used only as the discharge cap, which binds in ~0.5 % of hours.
    """
    t_max = len(w_known)
    mpc = Mpc(horizon)
    soc = soc0
    p = np.zeros(t_max)
    for t in range(t_max):
        end = min(t + horizon, t_max)
        if w_forecast is None or horizon <= known:
            w = w_known[t:end]
        else:
            k_end = min(t + known, t_max)
            w = np.concatenate([w_known[t:k_end], w_forecast[t, : max(end - k_end, 0)]])
        a = mpc.first_action(w, net[t], soc)
        # physical clamp, identical to the simulator's
        if a > 0:
            a = min(a, (CAP - soc) / ETA)
            soc += a * ETA
        elif a < 0:
            a = -min(-a, soc * ETA)
            soc += a / ETA
        soc = min(max(soc, 0.0), CAP)
        p[t] = a
    return p


def evaluate(p: np.ndarray, df: pd.DataFrame) -> dict:
    """Score a power schedule p (kW, + charge) exactly as the simulator does."""
    net = (df["load_kw"] - df["pv_kw"]).values
    base = np.maximum(net, 0.0)
    grid = np.maximum(net + p, 0.0)
    co2, price = df["co2_kg_per_kwh"].values, df["precio_eur_per_kwh"].values
    co2_b, cost_b = float((base * co2).sum()), float((base * price).sum())
    d_co2 = 1 - float((grid * co2).sum()) / co2_b
    d_cost = 1 - float((grid * price).sum()) / cost_b
    dis = np.clip(-p, 0, None)
    return {"score": 100 * (0.7 * d_co2 + 0.3 * d_cost), "co2_red_pct": 100 * d_co2, "cost_red_pct": 100 * d_cost,
            "discharged_kwh": float(dis.sum()), "charged_kwh": float(np.clip(p, 0, None).sum()),
            "cycles_per_day": float(dis.sum() / (CAP * ETA) / (len(p) / 24)),
            "exported_kwh": float(np.clip(-(net + p), 0, None).sum())}


def weights(df: pd.DataFrame, a_co2: float = 0.7, a_cost: float = 0.3) -> np.ndarray:
    """Score points gained per kWh of import avoided in each hour (x 100 / 1 handled by evaluate)."""
    net = (df["load_kw"] - df["pv_kw"]).clip(lower=0).values
    co2_b = float((net * df["co2_kg_per_kwh"].values).sum())
    cost_b = float((net * df["precio_eur_per_kwh"].values).sum())
    return a_co2 * df["co2_kg_per_kwh"].values / co2_b + a_cost * df["precio_eur_per_kwh"].values / cost_b
