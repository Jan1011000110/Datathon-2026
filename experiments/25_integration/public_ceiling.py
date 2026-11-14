# Usage: .venv/Scripts/python.exe experiments/25_integration/public_ceiling.py [leaderboard.csv]
"""Estimate the perfect-foresight ceiling on the PUBLIC hard-mode test, which we cannot compute
directly, from two things we can measure:
  (1) on train folds, how much more CO2 and EUR the hard-mode optimum saves than the
      standard-mode optimum on the same hours (a ratio);
  (2) what the standard-mode optimum saves on the public test (ten teams sit on it: 4,289.4 kg, 8,907.87 EUR).
Then express each leaderboard row as a share of that ceiling."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import hard, sim  # noqa: E402

STD_PUBLIC_OPT = (4289.4, 8907.87)        # kg, EUR saved by the plain optimum on the public test (standard mode)
HARD_PUBLIC_BASE = (60245.0, 95000.0)     # kg, EUR: no-battery baseline of the public test in hard mode (from the leaderboard)

rows = []
hf = hard.get_folds()
for f in ("A", "B", "C", "ALL"):
    dh = hf[f]
    ds = dh.copy()
    ds["pv_kw"] = ds["pv_kw"] / hard.PV_SCALE
    o_s, o_h = sim.oracle(ds), hard.oracle(dh)
    rows.append({"fold": f, "std_score": o_s["score"], "hard_score": o_h["score"],
                 "std_kg": o_s["co2_saved_kg"], "hard_kg": o_h["co2_saved_kg"], "std_eur": o_s["cost_saved_eur"], "hard_eur": o_h["cost_saved_eur"],
                 "r_kg": o_h["co2_saved_kg"] / o_s["co2_saved_kg"], "r_eur": o_h["cost_saved_eur"] / o_s["cost_saved_eur"],
                 "r_score": o_h["score"] / o_s["score"]})
t = pd.DataFrame(rows).set_index("fold")
t["est_public_ceiling"] = 100 * (0.7 * STD_PUBLIC_OPT[0] * t.r_kg / HARD_PUBLIC_BASE[0] + 0.3 * STD_PUBLIC_OPT[1] * t.r_eur / HARD_PUBLIC_BASE[1])
t["est_by_score_ratio"] = 6.432 * t.r_score
print(t.round(3).to_string())
lo, hi, mid = t.est_public_ceiling[["A", "B", "C"]].min(), t.est_public_ceiling[["A", "B", "C"]].max(), t.est_public_ceiling[["A", "B", "C"]].mean()
print(f"\nestimated public ceiling: {mid:.2f} (range over folds {lo:.2f} to {hi:.2f})")
t.to_csv(Path(__file__).with_name("public_ceiling.csv"))

if len(sys.argv) > 1:
    lb = pd.read_csv(sys.argv[1])
    lb.columns = ["time_utc", "team", "score", "co2_pct", "cost_pct", "kg", "eur", "infr"]
    lb["team_key"] = lb.team.str.lower()
    best = lb.sort_values("score", ascending=False).groupby("team_key", as_index=False).first().sort_values("score", ascending=False)
    best["n_subs"] = best.team_key.map(lb.team_key.value_counts())
    best["pct_of_ceiling"] = 100 * best.score / mid
    best["kg_vs_std_opt"] = best.kg / STD_PUBLIC_OPT[0]
    best["eur_vs_std_opt"] = best.eur / STD_PUBLIC_OPT[1]
    print(best[["team", "score", "co2_pct", "cost_pct", "kg", "eur", "n_subs", "time_utc", "pct_of_ceiling", "kg_vs_std_opt", "eur_vs_std_opt"]].round(2).to_string(index=False))
    print("implied baselines (kg, EUR):", (lb.kg / lb.co2_pct * 100).median().round(0), (lb.eur / lb.cost_pct * 100).median().round(0))
