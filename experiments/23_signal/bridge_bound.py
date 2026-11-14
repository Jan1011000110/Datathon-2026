# Usage: .venv/Scripts/python.exe experiments/23_signal/bridge_bound.py
"""Upper bound of the "bridge" idea: the day after every test week is a train day, so the level of the
last hours of a week could be interpolated instead of extrapolated. Bound: give the truth for every
target hour that falls in the last D hours of its week (targets +3..+11 only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402


def run(D, f):
    df = fast.fold(f)
    b = fast.base_tape(f)
    F = fast.w_of(b["Cc"], b["Pp"]); T = fast.w_of(b["Tc"], b["Tp"])
    lo = {"A": 1, "B": 8, "C": 15}[f]
    hrs_in = (df.index.day.values - lo) * 24 + df.index.hour.values           # hour within the week, 0..167
    W = F.copy()
    for k in range(3, 12):
        tgt = hrs_in + k
        m = (tgt < 168) & (tgt >= 168 - D) & np.isfinite(T[:, k])
        W[m, k] = T[m, k]
    pn, cm = fast.both(W, f)
    return {"last_hours": D, "fold": f, "pn": pn, "comb": cm}


if __name__ == "__main__":
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(D, f) for D in (0, 12, 24, 48) for f in "ABC"))
    out.to_csv(Path(__file__).with_name("bridge_bound.csv"), index=False)
    for col in ("pn", "comb"):
        piv = out.pivot(index="last_hours", columns="fold", values=col); piv["mean"] = piv.mean(axis=1)
        print(f"== {col} =="); print(piv.round(3).to_string())
