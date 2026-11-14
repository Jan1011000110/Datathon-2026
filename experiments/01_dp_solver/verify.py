"""Verify the lattice DP in agents/v1/agent.py against the LP (HiGHS) and time it."""
import sys, time, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.mpc import plan_lp, MPCAgent
from src.sim import get_folds, simulate, simulate_official, oracle

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
ETA, CAP, PMAX = v1.ETA, v1.CAP, v1.PMAX

def dp_rollout(w, s0):
    """Open-loop optimum via DP: returns objective sum w*p and final soc."""
    s, obj = s0, 0.0
    for k in range(len(w)):
        v = v1.marginal_values(w[k:], np.zeros(v1.NI))
        p = v1.first_step(s, v, w[k], PMAX)
        p = min(max(p, -min(PMAX, s * ETA)), min(PMAX, (CAP - s) / ETA))
        obj += w[k] * p
        s += p * ETA if p > 0 else p / ETA
        assert -1e-6 <= s <= CAP + 1e-6
    return obj, s

rng = np.random.default_rng(0)
worst = 0.0
real = get_folds("filled")["ALL"]
wr = (real["co2_kg_per_kwh"] + v1.KAPPA * real["precio_eur_per_kwh"]).to_numpy()
t_lp = t_dp = 0.0
for it in range(300):
    H = int(rng.integers(2, 49))
    if it % 2 == 0:
        i = int(rng.integers(0, len(wr) - H)); w = wr[i:i + H].copy()
    else:
        w = rng.uniform(0.01, 0.3, H)
    s0 = float(rng.choice([0.0, CAP, 1000.0, rng.uniform(0, CAP)]))
    t0 = time.perf_counter(); c, d = plan_lp(w, s0); t_lp += time.perf_counter() - t0
    lp_obj = float(np.sum(w * (c - d)))
    dp_obj, _ = dp_rollout(w, s0)
    worst = max(worst, abs(dp_obj - lp_obj) / max(1e-9, abs(lp_obj)))
print("DP vs LP open-loop objective, 300 random instances: worst relative diff = %.2e" % worst)

# closed loop on the folds
folds = get_folds("filled")
for f in ("A", "B", "C", "ALL"):
    df = folds[f]
    t0 = time.perf_counter(); r = simulate(v1.BatteryAgent(), df); dt = time.perf_counter() - t0
    o = oracle(df)
    print("fold %-3s  v1 %.4f  (co2 %.3f%%, cost %.3f%%, infr %d)   oracle %.4f   gap %.4f   %.2f s (%.3f ms/step)" % (
        f, r["score"], r["co2_pct"], r["cost_pct"], r["infractions"], o["score"], o["score"] - r["score"], dt, 1e3 * dt / len(df)))
off = simulate_official(v1.BatteryAgent(), folds["A"])
print("official env, fold A:", off)
off2 = simulate_official(v1.BatteryAgent(), get_folds("official")["C"])
print("official env, official-table fold C (has the June outage):", off2)
