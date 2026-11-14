"""Validate the files in submissions/: official checker, infractions, and score vs v1 on the folds."""
import sys, importlib.util, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate
folds = get_folds("filled"); A = folds["A"]; hw = np.arange(len(A)) % 168
runs = {"A": A, "B": folds["B"], "C": folds["C"], "A last 84 h": A[hw >= 84], "B first 84 h": folds["B"][np.arange(len(folds["B"])) % 168 < 84]}
def load(p):
    spec = importlib.util.spec_from_file_location("m_" + p.parent.name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
base = {k: simulate(load(ROOT / "submissions/v1/agent.py").BatteryAgent(), df) for k, df in runs.items()}
rows = []
for d in sorted((ROOT / "submissions").iterdir()):
    f = d / "agent.py"
    chk = subprocess.run([sys.executable, str(ROOT / "starter_kit/check_submission.py"), str(f)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = "VÀLIDA" in chk.stdout or "V�LIDA" in chk.stdout or chk.returncode == 0
    r = {"file": f"submissions/{d.name}/agent.py", "checker": "pass" if ok else "FAIL"}
    infr = 0
    for k, df in runs.items():
        res = simulate(load(f).BatteryAgent(), df); infr += res["infractions"]
        r[f"{k}: vs v1"] = res["score"] - base[k]["score"]
    r["infractions"] = infr; rows.append(r)
out = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
