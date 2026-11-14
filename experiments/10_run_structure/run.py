"""If the public leaderboard is a SELF-CONTAINED run over half of each test week (1,008 h),
what does that run look like, and how far is a plain 24 h optimiser from perfect foresight on it?
We build candidate public runs from train weeks and measure v1 vs the oracle on each.
Run from the repo root:  .venv/Scripts/python.exe experiments/10_run_structure/run.py"""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate, oracle

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

rng = np.random.default_rng(7)
def structures(df):
    n = len(df); hw = np.arange(n) % 168; day = hw // 24
    return {
        "full weeks (168 h blocks)": np.ones(n, bool),
        "first 84 h of each week": hw < 84,
        "last 84 h of each week": hw >= 84,
        "days 1-3 + day 4 morning... (first 3 days, 72 h)": day < 3,
        "odd days 1,3,5,7 (96 h)": day % 2 == 0,
        "even days 2,4,6 (72 h)": day % 2 == 1,
        "alternating 12 h blocks": (hw // 12) % 2 == 0,
        "every other hour": np.arange(n) % 2 == 0,
        "random half of the hours": rng.random(n) < 0.5,
        "random half of the days": np.repeat(rng.random(n // 24 + 1) < 0.5, 24)[:n],
    }

rows = []
folds = get_folds("filled")
for f in "AB":                                   # C contains the June meter outage; keep runs clean
    full = folds[f]
    for name, m in structures(full).items():
        df = full[m]
        r = simulate(v1.BatteryAgent(), df); o = oracle(df)
        rows.append({"fold": f, "structure": name, "hours": len(df), "v1": r["score"], "oracle": o["score"],
                     "gap": o["score"] - r["score"], "co2_%": r["co2_pct"], "cost_%": r["cost_pct"],
                     "kg_saved_per_h": r["co2_saved_kg"] / len(df), "eur_saved_per_h": r["cost_saved_eur"] / len(df),
                     "base_kg_per_h": r["co2_base_kg"] / len(df), "infr": r["infractions"]})
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).parent / "results.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
print("\nPublic leaderboard, for comparison (v1): score 6.432, CO2 5.96%, cost 7.52%; if 1,008 h: 4.249 kg/h and 8.864 EUR/h saved, baseline 71.2 kg/h")
print("Lead of the 6.44 team over v1: +0.0066 points (about +0.006 on the scale of these folds)")
