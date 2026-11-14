"""First pass: check the fast simulator against the official one, then compare
starter heuristic vs MPC variants vs the perfect-foresight oracle on the test-like folds."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "starter_kit"))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate, simulate_official, oracle, ReplayAgent
from src.mpc import MPCAgent
from agent import BatteryAgent as StarterAgent

folds = get_folds("filled")

# 1. simulator cross-check (starter agent + an MPC agent, fold A)
for mk in (StarterAgent, lambda: MPCAgent(n_ext=24)):
    a = simulate(mk(), folds["A"]); b = simulate_official(mk(), folds["A"])
    print("cross-check  fast: %.4f  official: %.2f  | co2 %.4f vs %.2f | cost %.4f vs %.2f | infr %d vs %d" % (
        a["score"], b["score"], a["co2_pct"], b["co2_estalviat_pct"], a["cost_pct"], b["cost_estalviat_pct"],
        a["infractions"], b["infraccions"]))

# 2. comparison table
rows = []
def add(name, fn):
    t0 = time.time(); r = {}
    for f in ("A", "B", "C", "ALL"):
        res = fn(folds[f]); r[f] = res["score"]; r[f + "_infr"] = res["infractions"]
        if f == "A": r["co2%_A"], r["cost%_A"] = res["co2_pct"], res["cost_pct"]
    r["secs"] = time.time() - t0
    rows.append(pd.Series(r, name=name)); print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)

add("oracle (full foresight)", oracle)
add("oracle replayed in sim", lambda df: simulate(ReplayAgent(oracle(df, True)["trace"]["p_kw"]), df))
add("starter heuristic", lambda df: simulate(StarterAgent(), df))
add("MPC 24h, no terminal", lambda df: simulate(MPCAgent(n_ext=0), df))
add("MPC 24h + 12h periodic ext", lambda df: simulate(MPCAgent(n_ext=12), df))
add("MPC 24h + 24h periodic ext", lambda df: simulate(MPCAgent(n_ext=24), df))
add("MPC 24h + 48h periodic ext", lambda df: simulate(MPCAgent(n_ext=48), df))
add("MPC 24h + 24h ext, no load guard", lambda df: simulate(MPCAgent(n_ext=24, load_guard=False), df))
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).parent / "results.csv")
print(out.round(3).to_string())
