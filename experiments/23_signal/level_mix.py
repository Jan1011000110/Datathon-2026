# Usage: .venv/Scripts/python.exe experiments/23_signal/level_mix.py
"""Hypothesis from need_night.py: at night only the level of hours +3..+8 matters, and the per-hour-of-day
model predicts that level 10 % better. Test: shape of the single model, level (mean of +3..+8) of the
per-hour model, for night forecasts only / for all forecasts. Closed loop per fold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402

NAMES = ["single model (round 1)", "per-hour level at night (22-03 h)", "per-hour level at all hours",
         "per-hour model at night (22-03 h)", "mean of both levels at night"]


def run(name, f):
    b = fast.base_tape(f)
    F = fast.w_of(b["Cc"], b["Pp"]); T = fast.w_of(b["Tc"], b["Tp"])
    G = np.load(fast.CACHE / f"hs_AH12_per_hour_{f}.npy")
    h = fast.fold(f).index.hour.values
    night = (h >= 22) | (h <= 3)
    s = slice(3, 9)
    W = F.copy()
    d = (G[:, s].mean(1) - F[:, s].mean(1))[:, None]
    if name == NAMES[1]:
        W[night, 3:9] = (F[:, s] + d)[night]
    elif name == NAMES[2]:
        W[:, s] = F[:, s] + d
    elif name == NAMES[3]:
        W[night, 3:12] = G[night, 3:12]
    elif name == NAMES[4]:
        W[night, 3:9] = (F[:, s] + 0.5 * d)[night]
    pn, cm = fast.both(W, f)
    return {"cand": name, "fold": f, "pn": pn, "comb": cm,
            "lvl_night": np.nanmean(np.abs(np.nanmean((W - T)[night][:, s], axis=1))) * 1e3}


if __name__ == "__main__":
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(n, f) for n in NAMES for f in "ABC"))
    out.to_csv(Path(__file__).with_name("level_mix.csv"), index=False)
    pd.set_option("display.width", 200)
    for col in ("pn", "comb", "lvl_night"):
        piv = out.pivot(index="cand", columns="fold", values=col); piv["mean"] = piv.mean(axis=1)
        print(f"== {col} =="); print(piv.loc[NAMES].round(3).to_string())
