# Usage: .venv/Scripts/python.exe experiments/23_signal/need_night.py
"""Night forecasts (made 22 h to 3 h) against the rest: which property of hours +3..+8 is worth what?
Also the level error of the per-hour-of-day model in the same bands. Writes need_night.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402

KINDS = ["none", "truth", "true level", "true min and max", "true max only", "true min only"]


def run(band, kind, f):
    b = fast.base_tape(f)
    F = fast.w_of(b["Cc"], b["Pp"]); T = fast.w_of(b["Tc"], b["Tp"])
    T = np.where(np.isfinite(T), T, F)
    h = fast.fold(f).index.hour.values
    m = ((h >= 22) | (h <= 3)) if band == "night" else ((h > 3) & (h < 22))
    s = slice(3, 9)
    W = F.copy()
    Fs, Ts = F[m][:, s], T[m][:, s]
    if kind == "truth":
        new = Ts
    elif kind == "true level":
        new = Fs - Fs.mean(1, keepdims=True) + Ts.mean(1, keepdims=True)
    elif kind == "true min and max":
        fmin, fmax = Fs.min(1, keepdims=True), Fs.max(1, keepdims=True)
        new = Ts.min(1, keepdims=True) + (Fs - fmin) / np.maximum(fmax - fmin, 1e-9) * (Ts.max(1, keepdims=True) - Ts.min(1, keepdims=True))
    elif kind == "true max only":          # shift so that the maximum is right
        new = Fs + (Ts.max(1, keepdims=True) - Fs.max(1, keepdims=True))
    elif kind == "true min only":
        new = Fs + (Ts.min(1, keepdims=True) - Fs.min(1, keepdims=True))
    else:
        new = Fs
    tmp = W[m]; tmp[:, s] = new; W[m] = tmp
    G = np.load(fast.CACHE / f"hs_AH12_per_hour_{f}.npy")
    pn, cm = fast.both(W, f)
    return {"band": band, "kind": kind, "fold": f, "pn": pn, "comb": cm,
            "lvl_single": np.abs((F - T)[m][:, s].mean(1)).mean() * 1e3, "lvl_per_hour": np.abs((G - T)[m][:, s].mean(1)).mean() * 1e3,
            "max_err_single": np.abs(Fs.max(1) - Ts.max(1)).mean() * 1e3, "min_err_single": np.abs(Fs.min(1) - Ts.min(1)).mean() * 1e3}


if __name__ == "__main__":
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(b, k, f) for b in ("night", "day") for k in KINDS for f in "ABC"))
    out.to_csv(Path(__file__).with_name("need_night.csv"), index=False)
    base = out[out.kind == "none"].groupby("fold")[["pn", "comb"]].first()
    out["gain_pn"] = out.pn - out.fold.map(base.pn); out["gain_comb"] = out.comb - out.fold.map(base.comb)
    g = out.groupby(["band", "kind"]).agg(gain_pn=("gain_pn", "mean"), gain_comb=("gain_comb", "mean"), min_fold=("gain_comb", "min"),
                                          lvl_single=("lvl_single", "mean"), lvl_per_hour=("lvl_per_hour", "mean"),
                                          max_err=("max_err_single", "mean"), min_err=("min_err_single", "mean"))
    pd.set_option("display.width", 200)
    print(g.round(3).to_string())
