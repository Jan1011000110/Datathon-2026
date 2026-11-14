# Usage: .venv/Scripts/python.exe experiments/24_planner/timing.py
"""Time per planner call on the real inputs of a run (fold B, ClimW + QuantileN forecasts, SOC
drawn uniformly), planner only (forecasters excluded). Writes timing_real.csv."""
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
cw, qn = ClimW(fit), P.QuantileN(fit, H=48)
rng = np.random.default_rng(0)
load, pv = df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float)
co2, price = df["co2_kg_per_kwh"].to_numpy(float), df["precio_eur_per_kwh"].to_numpy(float)
steps = np.sort(rng.choice(np.arange(1, len(df)), 200, replace=False))
inst = []
for t in steps:
    obs = {"timestamp": df.index[t], "co2_properes_24h": co2[t:t + 3], "preus_properes_24h": price[t:t + 3],
           "noves_dades_5min": {"ultim_consum_kw": load[t - 1], "ultima_solar_kw": pv[t - 1]}}
    w = np.maximum(np.asarray(cw.predict(obs, None, 48), float), 1e-9)
    n = np.asarray(qn.predict(obs, None, 48), float)
    sc = {Q: qn.predict_scen(obs, None, 48, Q)[0] for Q in (3, 5, 9, 20)}
    inst.append((w, n, sc, float(rng.uniform(0, CAP))))


def clock(fn, reps=7):
    """ms per call: (best of reps, median of reps) of the mean over the instances."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for it in inst:
            fn(*it)
        ts.append((time.perf_counter() - t0) / len(inst) * 1e3)
    return min(ts), float(np.median(ts))


rows = []
for H in (24, 36, 48):
    rows.append({"H": H, "planner": "scipy LP (plan_lp)", **dict(zip(("best_ms", "median_ms"), clock(lambda w, n, sc, s: plan_lp(w[:H], n[:H], s), 1)))})
    rows.append({"H": H, "planner": "plan_det", **dict(zip(("best_ms", "median_ms"), clock(lambda w, n, sc, s: P.plan_det(w[:H], n[:H], s))))})
    for Q in (3, 5, 9, 20):
        pr = np.full(Q, 1.0 / Q)
        for name, K in (("hour 0", 1), ("hours 0-5", 6), ("all hours", H)):
            b, m = clock(lambda w, n, sc, s: P.plan_stoch(w[:H], sc[Q][:K], pr, s, n_tail=n[K:H]))
            rows.append({"H": H, "planner": f"plan_stoch Q={Q}, scenarios for {name}", "best_ms": b, "median_ms": m})
out = pd.DataFrame(rows)
out["run_2016_calls_s"] = out["median_ms"] * 2.016
out.to_csv(Path(__file__).with_name("timing_real.csv"), index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string())

# number of segments describing the marginal value after hour 0
for label, Q in (("det", None), ("stoch Q=9 all hours", 9), ("stoch Q=20 all hours", 20)):
    ns = []
    for w, n, sc, s in inst:
        if Q is None:
            sv, sl = P.stage_segments_det(w[:36], n[:36])
        else:
            sv, sl = P.stage_segments(w[:36], sc[Q][:36], np.full(Q, 1.0 / Q))
        vv, vl = P.backward(sv, sl)
        ns.append(len(vv))
    print(f"segments in v after hour 0, H=36, {label}: mean {np.mean(ns):.0f}, max {np.max(ns)}")
