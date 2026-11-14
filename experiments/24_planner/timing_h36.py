# Usage: .venv/Scripts/python.exe experiments/24_planner/timing_h36.py
"""Planner time per call at H=36 on the real inputs of a run (fold B, ClimW + QuantileN, SOC
uniform), forecasters excluded. Two rounds, best and median of 9 passes over 300 hours.
Writes timing_h36.csv. Timings move a lot with what else the machine is doing: compare with the
LP timed in the same round."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import CAP, get_folds, train_complement  # noqa: E402
from src.hard_mpc import ClimW, plan_lp  # noqa: E402
import planner as P  # noqa: E402

df = get_folds()["B"]
fit = train_complement(df)
cw, qn = ClimW(fit), P.QuantileN(fit, H=36)
rng = np.random.default_rng(0)
load, pv = df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float)
co2, price = df["co2_kg_per_kwh"].to_numpy(float), df["precio_eur_per_kwh"].to_numpy(float)
inst = []
for t in np.sort(rng.choice(np.arange(1, len(df)), 300, replace=False)):
    obs = {"timestamp": df.index[t], "co2_properes_24h": co2[t:t + 3], "preus_properes_24h": price[t:t + 3],
           "noves_dades_5min": {"ultim_consum_kw": load[t - 1], "ultima_solar_kw": pv[t - 1]}}
    w = np.maximum(np.asarray(cw.predict(obs, None, 36), float), 1e-9)
    n = np.asarray(qn.predict(obs, None, 36), float)
    inst.append((w, n, {Q: qn.predict_scen(obs, None, 36, Q)[0] for Q in (9, 20)}, float(rng.uniform(0, CAP))))


def clock(fn, reps=9):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for it in inst:
            fn(*it)
        ts.append((time.perf_counter() - t0) / len(inst) * 1e3)
    return min(ts), float(np.median(ts))


p9, p20 = np.full(9, 1 / 9), np.full(20, 1 / 20)
tests = [("scipy LP (plan_lp)", lambda w, n, sc, s: plan_lp(w, n, s), 2),
         ("plan_det", lambda w, n, sc, s: P.plan_det(w, n, s), 9),
         ("plan_stoch Q=9, hour 0", lambda w, n, sc, s: P.plan_stoch(w, sc[9][:1], p9, s, n_tail=n[1:]), 9),
         ("plan_stoch Q=9, hours 0-11", lambda w, n, sc, s: P.plan_stoch(w, sc[9][:12], p9, s, n_tail=n[12:]), 9),
         ("plan_stoch Q=9, all 36 hours", lambda w, n, sc, s: P.plan_stoch(w, sc[9], p9, s), 9),
         ("plan_stoch Q=20, all 36 hours", lambda w, n, sc, s: P.plan_stoch(w, sc[20], p20, s), 9)]
rows = []
for rnd in range(2):
    for name, fn, reps in tests:
        b, m = clock(fn, reps)
        rows.append({"round": rnd, "planner": name, "best_ms": b, "median_ms": m, "run_2016_calls_s": b * 2.016})
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("timing_h36.csv"), index=False)
print(out.round(3).to_string())
