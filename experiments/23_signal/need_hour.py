# Usage: .venv/Scripts/python.exe experiments/23_signal/need_hour.py
"""At which hours of the day does the planner need the near forecast? Give the truth for hours +3..+8
only to the forecasts made in one band of hours of the day; closed loop per fold. Writes need_hour.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402

BANDS = [(0, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18), (19, 21), (22, 23)]


def run(band, f):
    b = fast.base_tape(f)
    F = fast.w_of(b["Cc"], b["Pp"]); T = fast.w_of(b["Tc"], b["Tp"])
    T = np.where(np.isfinite(T), T, F)
    W = F.copy()
    h = fast.fold(f).index.hour.values
    lvl = np.abs((F - T)[:, 3:9].mean(1))
    if band is not None:
        m = (h >= band[0]) & (h <= band[1])
        W[m, 3:9] = T[m, 3:9]
    else:
        m = np.ones(len(h), bool)
    pn, cm = fast.both(W, f)
    return {"band": "none" if band is None else f"{band[0]:02d}-{band[1]:02d} h", "hours": int(m.sum()) if band else 0,
            "fold": f, "pn": pn, "comb": cm, "lvl_mae_g": lvl[m].mean() * 1e3}


if __name__ == "__main__":
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(b, f) for b in [None] + BANDS for f in "ABC"))
    out.to_csv(Path(__file__).with_name("need_hour.csv"), index=False)
    base = out[out.band == "none"].set_index("fold")
    out["gain_pn"] = out.pn - out.fold.map(base.pn)
    out["gain_comb"] = out.comb - out.fold.map(base.comb)
    g = out[out.band != "none"].groupby("band").agg(hours_per_fold=("hours", "mean"), level_mae_g=("lvl_mae_g", "mean"),
                                                   gain_pn=("gain_pn", "mean"), gain_comb=("gain_comb", "mean"),
                                                   min_fold_gain_comb=("gain_comb", "min"))
    g["gain_comb_per_100h"] = g.gain_comb / g.hours_per_fold * 100
    pd.set_option("display.width", 200)
    print(g.round(3).to_string())
    print("sum of bands:", g.gain_pn.sum().round(3), g.gain_comb.sum().round(3))
