"""Independent check of the ceiling, with no linear-programming theory involved.

Exhaustive search by dynamic programming over EVERY battery trajectory whose state of charge sits
on a grid (2 kWh, 1 kWh, 0.5 kWh steps), for a whole 2,016-hour run, using the true load, solar,
price and CO2 of every hour. It evaluates each move with the simulator's own formula
(grid import = max(0, load - pv + P)), so it also covers discharging into low load.
If a better schedule than the LP oracle existed, a fine enough grid would find it.
Run from the repo root:  .venv/Scripts/python.exe experiments/12_ceiling_check/bruteforce.py"""
import sys, time, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate, oracle, score_weights, ReplayAgent

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
CAP, PMAX, ETA = 2000.0, 500.0, 0.9 ** 0.5

def exhaustive(df, step):
    """Best schedule among all SOC trajectories on a `step`-kWh grid. Returns the power schedule."""
    w = score_weights(df); net = (df.load_kw - df.pv_kw).to_numpy(float)
    n = int(round(CAP / step)) + 1
    up = int(np.floor(PMAX * ETA / step + 1e-9)); dn = int(np.floor(PMAX / ETA / step + 1e-9))
    offs = np.arange(-dn, up + 1)                                   # change in SOC, in grid steps
    power = np.where(offs >= 0, offs * step / ETA, offs * step * ETA)   # grid-side kW for each move
    T = len(df)
    V = np.zeros(n)                                                 # value of ending the run at each SOC: 0
    best = np.empty((T, n), dtype=np.int16)
    idx = np.arange(n)
    for t in range(T - 1, -1, -1):
        base = max(net[t], 0.0)
        gain = w[t] * (base - np.maximum(net[t] + power, 0.0))      # score gained by each move this hour
        cand = np.full((len(offs), n), -np.inf)
        for k, o in enumerate(offs):
            lo, hi = max(0, -o), min(n, n - o)                      # states from which the move stays in [0, CAP]
            cand[k, lo:hi] = gain[k] + V[lo + o:hi + o]
        best[t] = np.argmax(cand, axis=0)
        V = cand[best[t], idx]
    s = int(round(1000.0 / step)); sched = np.empty(T)
    for t in range(T):
        k = best[t, s]; sched[t] = power[k]; s += offs[k]
    return sched, 100 * V[int(round(1000.0 / step))]

rows = []
for fold in ("A", "B", "C"):
    df = get_folds("filled")[fold]
    o = oracle(df)["score"]; a = simulate(v1.BatteryAgent(), df)["score"]
    row = {"fold": fold, "agent v1": a, "LP oracle": o}
    for step in (2.0, 1.0, 0.5):
        t0 = time.time(); sched, val = exhaustive(df, step)
        r = simulate(ReplayAgent(sched), df)                        # replay through the simulator port
        row[f"exhaustive, {step} kWh grid"] = r["score"]
        print(f"fold {fold} grid {step} kWh: exhaustive-search score {r['score']:.5f} (infractions {r['infractions']}), "
              f"LP oracle {o:.5f}, difference {o - r['score']:+.5f}   [{time.time() - t0:.0f} s]", flush=True)
    rows.append(row)
out = pd.DataFrame(rows); out.to_csv(Path(__file__).parent / "results.csv", index=False)
print(); print(out.round(5).to_string(index=False))
