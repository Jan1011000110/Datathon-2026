# Usage: .venv/Scripts/python.exe experiments/24_planner/test_exact.py
"""Exactness of planner.plan_det / plan_stoch against scipy LPs, and timing per call.
Writes exact.csv (one row per instance family)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import CAP, ETA, PMAX, get_folds  # noqa: E402
from src.hard_mpc import KAPPA, plan_lp  # noqa: E402
import planner as P  # noqa: E402


def obj_det(w, n, soc, p, lam=0.0):
    s = soc + np.sum(np.where(p > 0, p * ETA, p / ETA))
    return float(np.sum(w * np.maximum(n + p, 0.0)) - lam * s)


def feasible(soc, p):
    s = soc + np.cumsum(np.where(p > 0, p * ETA, p / ETA))
    return bool(np.all(np.abs(p) <= PMAX + 1e-6) and s.min() >= -1e-6 and s.max() <= CAP + 1e-6)


def lp_stoch(w, scen, probs, soc, lam=0.0):
    """Reference LP for the expected-cost problem: x = [c (H), d (H), g (H*Q)]."""
    H, Q = scen.shape
    tri = np.tril(np.ones((H, H)))
    S = np.hstack([ETA * tri, -tri / ETA, np.zeros((H, H * Q))])
    rows = []
    for k in range(H):
        for j in range(Q):
            r = np.zeros(2 * H + H * Q)
            r[k], r[H + k], r[2 * H + k * Q + j] = 1.0, -1.0, -1.0
            rows.append(r)
    A = np.vstack([S, -S, np.array(rows)])
    b = np.concatenate([np.full(H, CAP - soc), np.full(H, soc), -scen.ravel()])
    cost = np.concatenate([np.full(H, -lam * ETA), np.full(H, lam / ETA), (w[:, None] * probs).ravel()])
    r = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, PMAX)] * (2 * H) + [(0, None)] * (H * Q), method="highs")
    return r.fun - lam * soc


def random_instance(rng, H):
    w = rng.uniform(0.02, 0.4, H)
    kind = rng.integers(0, 3)
    if kind == 0:
        n = rng.normal(900, 400, H)
    elif kind == 1:
        n = rng.normal(300, 900, H)                 # lots of surplus hours
    else:
        n = rng.uniform(-2500, 2500, H)             # beyond the power limits on both sides
    return w, n, float(rng.uniform(0, CAP))


def main():
    rng = np.random.default_rng(0)
    rows = []

    # ---- deterministic: random instances -------------------------------------------------
    for label, lam_on in (("random, v_end=0", False), ("random, v_end>0", True)):
        gaps, rel, bad = [], [], 0
        for i in range(1500):
            H = int(rng.choice([6, 12, 24, 36, 48]))
            w, n, soc = random_instance(rng, H)
            lam = float(rng.uniform(0.02, 0.3)) if lam_on else 0.0
            c, d, g = plan_lp(w, n, soc, lam=lam)
            ref = float(w @ g) - lam * (soc + np.sum(ETA * c - d / ETA))
            p = P.plan_det(w, n, soc, v_end=lam if lam_on else None, full=True)
            bad += not feasible(soc, p)
            mine = obj_det(w, n, soc, p, lam)
            gaps.append(mine - ref)
            rel.append((mine - ref) / max(abs(ref), 1e-9))
            assert abs(P.plan_det(w, n, soc, v_end=lam if lam_on else None) - p[0]) < 1e-9
        rows.append({"family": "det " + label, "n": len(gaps), "max_gap": np.max(gaps), "mean_gap": np.mean(gaps),
                     "min_gap": np.min(gaps), "max_rel_gap": np.max(np.abs(rel)), "infeasible": bad})

    # ---- deterministic: real instances (true w and net load of fold B, random SOC) ---------
    df = get_folds()["B"]
    wt = (df["co2_kg_per_kwh"] + KAPPA * df["precio_eur_per_kwh"]).to_numpy(float)
    nt = (df["load_kw"] - df["pv_kw"]).to_numpy(float)
    gaps, rel, bad = [], [], 0
    for t in rng.choice(len(df) - 36, 1500, replace=False):
        w, n, soc = wt[t:t + 36], nt[t:t + 36], float(rng.uniform(0, CAP))
        c, d, g = plan_lp(w, n, soc)
        ref = float(w @ g)
        p = P.plan_det(w, n, soc, full=True)
        bad += not feasible(soc, p)
        gaps.append(obj_det(w, n, soc, p) - ref)
        rel.append(gaps[-1] / max(abs(ref), 1e-9))
    rows.append({"family": "det real (fold B, H=36)", "n": len(gaps), "max_gap": np.max(gaps),
                 "mean_gap": np.mean(gaps), "min_gap": np.min(gaps), "max_rel_gap": np.max(np.abs(rel)), "infeasible": bad})

    # ---- stochastic: random instances against the scenario LP ------------------------------
    gaps, rel, bad = [], [], 0
    for i in range(400):
        H = int(rng.choice([6, 12, 24]))
        Q = int(rng.choice([3, 5, 9]))
        w, n, soc = random_instance(rng, H)
        scen = n[:, None] + rng.normal(0, 250, (H, Q))
        probs = rng.dirichlet(np.ones(Q))
        lam = float(rng.uniform(0, 0.2)) if i % 2 else 0.0
        ref = lp_stoch(w, scen, probs, soc, lam)
        p = P.plan_stoch(w, scen, probs, soc, v_end=lam if lam else None, full=True)
        bad += not feasible(soc, p)
        s_end = soc + np.sum(np.where(p > 0, p * ETA, p / ETA))
        mine = P.expected_cost(w, scen, probs, p) - lam * s_end
        gaps.append(mine - ref)
        rel.append(gaps[-1] / max(abs(ref), 1e-9))
    rows.append({"family": "stoch random (Q=3/5/9)", "n": len(gaps), "max_gap": np.max(gaps),
                 "mean_gap": np.mean(gaps), "min_gap": np.min(gaps), "max_rel_gap": np.max(np.abs(rel)), "infeasible": bad})

    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).with_name("exact.csv"), index=False)
    print(out.to_string())

    # ---- timing ----------------------------------------------------------------------------
    trows = []
    insts = [(wt[t:t + 48], nt[t:t + 48], float(rng.uniform(0, CAP))) for t in rng.choice(len(df) - 48, 300, replace=False)]
    for H in (24, 36, 48):
        t0 = time.perf_counter()
        for w, n, soc in insts:
            plan_lp(w[:H], n[:H], soc)
        t_lp = (time.perf_counter() - t0) / len(insts) * 1e3
        t0 = time.perf_counter()
        for _ in range(5):
            for w, n, soc in insts:
                P.plan_det(w[:H], n[:H], soc)
        t_det = (time.perf_counter() - t0) / (5 * len(insts)) * 1e3
        row = {"H": H, "plan_lp_ms": t_lp, "plan_det_ms": t_det}
        for Q in (3, 5, 9, 20):
            off = rng.normal(0, 200, (48, Q))
            pr = np.full(Q, 1.0 / Q)
            for name, K in (("h0", 1), ("h0-5", 6), ("all", H)):
                t0 = time.perf_counter()
                for w, n, soc in insts:
                    P.plan_stoch(w[:H], n[:K, None] + off[:K], pr, soc, n_tail=n[K:H])
                row[f"stoch_Q{Q}_{name}_ms"] = (time.perf_counter() - t0) / len(insts) * 1e3
        trows.append(row)
    tim = pd.DataFrame(trows)
    tim.to_csv(Path(__file__).with_name("timing.csv"), index=False)
    print(tim.T.round(3).to_string())


if __name__ == "__main__":
    main()
