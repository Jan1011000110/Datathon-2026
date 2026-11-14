# Usage: .venv/Scripts/python.exe experiments/28_signal_planning/timing_live.py [K] [lam]
"""Whole-agent time per step, live (no tapes): the combined agent's forecasters fitted outside fold B
+ the path planner. Prints score, total seconds, ms per step and the planner's share; appends to
timing_live.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "experiments" / "27_combined"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import combined as cb  # noqa: E402
import sigplan as sp  # noqa: E402
from src.hard import get_folds, simulate  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
fold = "B"
df = get_folds()[fold]
t0 = time.perf_counter()
b = cb.make_agent(df, signal="fc", load="fc", solar="fc", scen="conv")
t_fit = time.perf_counter() - t0
rows = []
for label, pool, lam_ in (("certainty-equivalent (exact planner)", None, 0.0), (f"paths K={K}, lam={lam} + AR correction x1.0", sp.Paths(fold, K=K), lam)):
    for fc in (b.w_fc, b.net.load_fc, b.net.pv_fc):
        if hasattr(fc, "reset"):
            fc.reset()
    ag = sp.SigAgent(b.w_fc, b.net, pool=pool, lam=lam_, ar=1.0 if pool is not None else 0.0, H=b.H, Q=b.Q,
                     scen_hours=b.scen_hours, exact=True,
                     v_end=b.v_end, truncate_end=b.truncate_end)
    t0 = time.perf_counter()
    r = simulate(ag, df)
    tot = time.perf_counter() - t0
    rows.append({"config": label, "fold": fold, "score": r["score"], "infractions": int(r["infractions"]),
                 "run_s": tot, "ms_per_step": tot / len(df) * 1e3, "planner_ms_per_step": ag.sec / len(df) * 1e3,
                 "fit_s": t_fit})
out = pd.DataFrame(rows)
p = HERE / "timing_live.csv"
out.to_csv(p, mode="a", header=not p.exists(), index=False)
print(out.round(3).to_string())
