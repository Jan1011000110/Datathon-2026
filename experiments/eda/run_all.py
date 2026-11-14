"""
Regenerate every EDA figure and stats file under reports/01_eda/.

    .venv/Scripts/python.exe experiments/eda/run_all.py

q9_foresight.py caches its LP simulations in reports/01_eda/stats/q9_runs.json (about 6 minutes
the first time); delete that file to recompute them. q10_extras.py reads that cache, so it runs last.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ["q1_coverage", "q2_load", "q3_pv", "q4_q5_signals", "q6_agreement", "q7_predictability",
           "q8_best_hours", "q9_foresight", "q10_extras"]

for name in SCRIPTS:
    t0 = time.time()
    res = subprocess.run([sys.executable, str(HERE / f"{name}.py")], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    figs = [ln for ln in res.stdout.splitlines() if ln.startswith("[fig]")]
    status = "ok" if res.returncode == 0 else "FAILED"
    print(f"{name:20s} {status:7s} {time.time() - t0:5.0f}s  {len(figs)} figures")
    if res.returncode != 0:
        print(res.stderr[-2000:])
        sys.exit(1)
