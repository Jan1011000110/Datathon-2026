"""
experiments/04_whatif/run_all.py
Regenerate every figure and stats file of reports/04_whatif/ (about 3 minutes, one process).

    .venv/Scripts/python.exe experiments/04_whatif/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["a1_annual.py", "a2_pareto.py", "a3_sizing.py", "a4_efficiency.py", "a5_pv_export.py",
           "a6_value_sources.py", "a7_peak_import.py"]

if __name__ == "__main__":
    for name in SCRIPTS:
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE.parents[1])
        if r.returncode != 0:
            raise SystemExit(f"{name} failed with exit code {r.returncode}")
        print(f"[{name}: {time.time() - t0:.0f} s]", flush=True)
