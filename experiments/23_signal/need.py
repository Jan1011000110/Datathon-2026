# Usage: .venv/Scripts/python.exe experiments/23_signal/need.py
"""What does the planner need from the signal forecast in hours +3..+8 (and +3..+11)?
Truth-substitution experiments on the round-1 forecast tape: replace one property of the forecast by
the truth and score the result in closed loop (net load known, and inside the combined agent).
Writes need.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402


def variants(b):
    F = fast.w_of(b["Cc"], b["Pp"])
    T = fast.w_of(b["Tc"], b["Tp"])
    T = np.where(np.isfinite(T), T, F)
    out = {"forecast (round 1)": F, "truth, all hours": T}

    def sub(lo, hi):
        W = F.copy()
        W[:, lo:hi] = T[:, lo:hi]
        return W
    for lo, hi in ((3, 4), (3, 5), (3, 6), (3, 9), (3, 12), (4, 12), (6, 12), (9, 12), (12, 36)):
        out[f"truth in +{lo}..+{hi - 1}"] = sub(lo, hi)
    for lo, hi in ((3, 9), (3, 12)):
        s = slice(lo, hi)
        mf, mt = F[:, s].mean(1, keepdims=True), T[:, s].mean(1, keepdims=True)
        # level: mean of the window; shape: deviations around it
        W = F.copy(); W[:, s] = F[:, s] - mf + mt
        out[f"true level, forecast shape, +{lo}..+{hi - 1}"] = W
        W = F.copy(); W[:, s] = T[:, s] - mt + mf
        out[f"forecast level, true shape, +{lo}..+{hi - 1}"] = W
        # ordering: the forecast's own values, put in the true order
        W = F.copy()
        srt = np.sort(F[:, s], axis=1)
        rk = np.argsort(np.argsort(T[:, s], axis=1), axis=1)
        W[:, s] = np.take_along_axis(srt, rk, axis=1)
        out[f"forecast values in the true order, +{lo}..+{hi - 1}"] = W
        # true values in the forecast's order
        W = F.copy()
        srt = np.sort(T[:, s], axis=1)
        rk = np.argsort(np.argsort(F[:, s], axis=1), axis=1)
        W[:, s] = np.take_along_axis(srt, rk, axis=1)
        out[f"true values in the forecast's order, +{lo}..+{hi - 1}"] = W
        # first step: remove the error of the first forecast hour from the whole window (a level shift)
        W = F.copy(); W[:, s] = F[:, s] + (T[:, lo:lo + 1] - F[:, lo:lo + 1])
        out[f"shift window by the +{lo} error, +{lo}..+{hi - 1}"] = W
        # extremes: true minimum and maximum of the window, forecast in between (affine map)
        fmin, fmax = F[:, s].min(1, keepdims=True), F[:, s].max(1, keepdims=True)
        tmin, tmax = T[:, s].min(1, keepdims=True), T[:, s].max(1, keepdims=True)
        W = F.copy(); W[:, s] = tmin + (F[:, s] - fmin) / np.maximum(fmax - fmin, 1e-9) * (tmax - tmin)
        out[f"true min and max, forecast in between, +{lo}..+{hi - 1}"] = W
    return out


def run(name, f):
    b = fast.base_tape(f)
    W = variants(b)[name]
    pn, cm = fast.both(W, f)
    return {"variant": name, "fold": f, "pn": pn, "comb": cm}


if __name__ == "__main__":
    for f in "ABC":
        fast.base_tape(f)
    names = list(variants(fast.base_tape("A")))
    rows = Parallel(n_jobs=3)(delayed(run)(n, f) for n in names for f in "ABC")
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).with_name("need.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    for col in ("pn", "comb"):
        piv = out.pivot(index="variant", columns="fold", values=col)
        piv["mean"] = piv.mean(axis=1)
        print(f"\n== {col} ==")
        print(piv.loc[names].round(3).to_string())
