# Usage: .venv/Scripts/python.exe experiments/24_planner/check_official.py
"""Cross-check: the recommended configuration through the organisers' environment.py (hard-mode
spec) against our fast simulator, fold B. Prints both results and the time of the run."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, simulate, simulate_official, train_complement  # noqa: E402
from src.hard_mpc import ClimW  # noqa: E402
import planner as P  # noqa: E402

df = get_folds()["B"]
fit = train_complement(df)
cw, qn = ClimW(fit), P.QuantileN(fit, H=36)
kw = dict(H=36, Q=9, scen_hours=None)
t0 = time.perf_counter()
a = simulate(P.MPCStoch(cw, qn, **kw), df)
t1 = time.perf_counter()
b = simulate_official(P.MPCStoch(cw, qn, **kw), df)
t2 = time.perf_counter()
print("fast sim :", round(a["score"], 4), "infractions", a["infractions"], f"{t1 - t0:.1f} s (forecasters + planner, uncached)")
print("official :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items() if not hasattr(v, "__len__")}, f"{t2 - t1:.1f} s")
