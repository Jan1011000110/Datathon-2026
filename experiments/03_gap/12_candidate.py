"""Validate a candidate agent file against v1 and the oracle on A, B, C, ALL.

    python experiments/03_gap/12_candidate.py [candidate_dir_name]      (default: v1_tv)

Prints score, gain in points and as % of the fold's oracle gap, per-week paired gains (carry-
corrected), hours whose action differs from v1, infractions, run time, and a cross-check with
the organisers' environment.
"""
import sys
import time
from lib import *
from src.sim import simulate_official

name = sys.argv[1] if len(sys.argv) > 1 else "v1_tv"
cand = load_agent_module(Path(__file__).parent / "candidates" / name / "agent.py", "cand_" + name)
folds = get_folds("filled")
pd.set_option("display.width", 220)
rows = []
for f in FOLDS:
    df = folds[f]
    a = run(v1.BatteryAgent(), df)
    t0 = time.perf_counter(); b = run(cand.BatteryAgent(), df); dt = time.perf_counter() - t0
    o = oracle(df)["score"]
    g = paired(df, b, a)
    st = summarize(g)
    rows.append({"fold": f, "v1": a["score"], name: b["score"], "oracle": o, "gain_pts": b["score"] - a["score"],
                 "gain_%gap": 100 * (b["score"] - a["score"]) / (o - a["score"]),
                 "hours_changed": int((np.abs(b["trace"].p_kw - a["trace"].p_kw) > 1e-6).sum()),
                 "weeks": st["n"], "weeks+": st["pos"], "weeks-": st["neg"], "worst_week_pts": st["min"], "best_week_pts": st["max"],
                 "infr": b["infractions"], "secs": dt, "ms/step": 1e3 * dt / len(df)})
    if f != "ALL":
        print("fold %s weekly gains (1e-4 pts):" % f, np.round(g * 1e4, 2))
print(pd.DataFrame(rows).set_index("fold").round(5).to_string())
off = simulate_official(cand.BatteryAgent(), folds["B"])
print("\norganisers' environment, fold B:", off["score"], "infractions", off["infraccions"])
