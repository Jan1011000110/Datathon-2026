# Usage: .venv/Scripts/python.exe experiments/23_signal/min_mix.py
"""Counterpart of level_mix.py for day forecasts (04 h to 21 h), where the minimum of hours +3..+8 carries
most of the value: shift the window so that its minimum is the average of the two models' minima
(offline the average has a 7 % lower error on that minimum in every fold). Closed loop per fold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402

NAMES = ["single model (round 1)", "day: minimum averaged with the per-hour model", "day: level averaged with the per-hour model"]


def run(name, f):
    b = fast.base_tape(f)
    F = fast.w_of(b["Cc"], b["Pp"])
    G = np.load(fast.CACHE / f"hs_AH12_per_hour_{f}.npy")
    h = fast.fold(f).index.hour.values
    day = (h > 3) & (h < 22)
    s = slice(3, 9)
    W = F.copy()
    if name == NAMES[1]:
        d = 0.5 * (G[:, s].min(1) - F[:, s].min(1))[:, None]
        W[day, 3:9] = (F[:, s] + d)[day]
    elif name == NAMES[2]:
        d = 0.5 * (G[:, s].mean(1) - F[:, s].mean(1))[:, None]
        W[day, 3:9] = (F[:, s] + d)[day]
    pn, cm = fast.both(W, f)
    return {"cand": name, "fold": f, "pn": pn, "comb": cm}


if __name__ == "__main__":
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(n, f) for n in NAMES for f in "ABC"))
    out.to_csv(Path(__file__).with_name("min_mix.csv"), index=False)
    pd.set_option("display.width", 200)
    for col in ("pn", "comb"):
        piv = out.pivot(index="cand", columns="fold", values=col); piv["mean"] = piv.mean(axis=1)
        print(f"== {col} =="); print(piv.loc[NAMES].round(3).to_string())
