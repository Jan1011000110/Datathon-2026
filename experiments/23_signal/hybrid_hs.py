# Usage: .venv/Scripts/python.exe experiments/23_signal/hybrid_hs.py   (after hourspec.py)
"""Per-hour-of-day models only where they are better offline (near horizons), the single model beyond;
and the plain average of the two near forecasts. Closed loop per fold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402

AH = "hs_AH12_per_hour"


def run(name, f):
    b = fast.base_tape(f)
    F = fast.w_of(b["Cc"], b["Pp"]); T = fast.w_of(b["Tc"], b["Tp"])
    G = np.load(fast.CACHE / f"{AH}_{f}.npy")
    W = F.copy()
    if name == "per-hour model in +3..+11":
        W[:, 3:12] = G[:, 3:12]
    elif name == "per-hour model in +3..+8":
        W[:, 3:9] = G[:, 3:9]
    elif name == "average of both in +3..+11":
        W[:, 3:12] = 0.5 * (F[:, 3:12] + G[:, 3:12])
    elif name == "per-hour model everywhere":
        W = G
    pn, cm = fast.both(W, f)
    return {"cand": name, "fold": f, "pn": pn, "comb": cm,
            "lvl": np.nanmean(np.abs(np.nanmean((W - T)[:, 3:9], axis=1))) * 1e3,
            "mae_3_11": np.nanmean(np.abs(W - T)[:, 3:12]) * 1e3,
            "revision": np.nanmean(np.abs(W[1:, 3:8] - W[:-1, 4:9])) * 1e3}


if __name__ == "__main__":
    names = ["single model (round 1)", "per-hour model everywhere", "per-hour model in +3..+11", "per-hour model in +3..+8",
             "average of both in +3..+11"]
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(n, f) for n in names for f in "ABC"))
    out.to_csv(Path(__file__).with_name("hybrid_hs.csv"), index=False)
    pd.set_option("display.width", 220)
    for col in ("pn", "comb", "lvl", "mae_3_11", "revision"):
        piv = out.pivot(index="cand", columns="fold", values=col); piv["mean"] = piv.mean(axis=1)
        print(f"== {col} =="); print(piv.loc[names].round(3).to_string())
