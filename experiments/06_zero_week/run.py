"""What does a week of zero load (possible in the hidden test: Nov 2025 has no meter data) cost
an optimiser with and without v1's outage guard?  Fold A, with the November week's load set to 0."""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate, oracle

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

class NoGuard(v1.BatteryAgent):
    def _discharge_limits(self, obs, horizon):
        return v1.PMAX, 0

rows = []
for fold in ("A", "B", "C"):
    clean = get_folds("filled")[fold]
    for label, month in (("clean", None), ("zero-load week (Nov)", 11), ("zero-load week (Mar)", 3)):
        df = clean.copy()
        if month is not None:
            df.loc[df.index.month == month, "load_kw"] = 0.0
        rows.append({"fold": fold, "scenario": label,
                     "v1 (guard)": simulate(v1.BatteryAgent(), df)["score"],
                     "no guard": simulate(NoGuard(), df)["score"],
                     "oracle": oracle(df)["score"]})
out = pd.DataFrame(rows)
out["guard gain"] = out["v1 (guard)"] - out["no guard"]
out["v1 gap to oracle"] = out["oracle"] - out["v1 (guard)"]
out.to_csv(Path(__file__).parent / "results.csv", index=False)
print(out.round(4).to_string(index=False))
