# Usage: .venv/Scripts/python.exe experiments/23_signal/calib.py [tape-key]
"""Level and spread calibration of a forecast tape, per fold: multiply the forecast hours by (1 + s),
or stretch them around the mean of hours +3..+11 by g. Closed loop, both harnesses. Writes calib.csv.
Also prints the measured level bias of the tape (forecast - truth, hours +3..+8) per fold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else None


def tape(f):
    b = fast.base_tape(f)
    F = fast.w_of(b["Cc"], b["Pp"])
    T = fast.w_of(b["Tc"], b["Tp"])
    if KEY:
        F = np.load(fast.CACHE / f"near_{f}.npz")[KEY]
    return F, T


def run(kind, v, f):
    F, T = tape(f)
    W = F.copy()
    if kind == "shift":
        W[:, 3:] = F[:, 3:] * (1.0 + v)
    elif kind == "shift near":                     # only the hours that matter
        W[:, 3:12] = F[:, 3:12] * (1.0 + v)
    else:
        m = F[:, 3:12].mean(1, keepdims=True)
        W[:, 3:12] = np.maximum(m + v * (F[:, 3:12] - m), 1e-3)
    pn, cm = fast.both(W, f)
    return {"kind": kind, "value": v, "fold": f, "pn": pn, "comb": cm}


if __name__ == "__main__":
    for f in "ABC":
        F, T = tape(f)
        e = (F - T)[:, 3:9]
        print(f"fold {f}: level bias +3..+8 {np.nanmean(e) * 1e3:+.2f} g ({100 * np.nanmean(e) / np.nanmean(T[:, 3:9]):+.1f} %), "
              f"MAE of the window mean {np.nanmean(np.abs(np.nanmean(e, axis=1))) * 1e3:.2f} g, MAE per hour {np.nanmean(np.abs(e)) * 1e3:.2f} g")
    jobs = [("shift", v) for v in (-0.06, -0.03, 0.0, 0.03, 0.06, 0.09)] + \
           [("shift near", v) for v in (-0.03, 0.03, 0.06)] + [("spread", v) for v in (0.85, 1.15, 1.3)]
    rows = Parallel(n_jobs=3)(delayed(run)(k, v, f) for k, v in jobs for f in "ABC")
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).with_name(f"calib{'_' + KEY if KEY else ''}.csv"), index=False)
    for col in ("pn", "comb"):
        piv = out.pivot(index=["kind", "value"], columns="fold", values=col)
        piv["mean"] = piv.mean(axis=1)
        print(f"\n== {col} ==")
        print(piv.round(3).to_string())
