# Usage: .venv/Scripts/python.exe experiments/28_signal_planning/diag_errors.py
"""Structure of the cross-fitted signal-forecast errors (errors_<F>.npz): size by horizon, scaling
with the level, correlation along the horizon, and whether the last revealed errors predict the
next ones. Prints a table per fit set and writes diag_errors.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rows = []
for F in ("A", "B", "C", "ALL"):
    z = np.load(HERE / f"errors_{F}.npz")
    E, Fc, hod, pos, blk = z["E"], z["F"], z["hod"], z["pos"], z["block"]
    ok = np.isfinite(E[:, 3:18]).all(1)
    r = {"fit set": F, "origins": int(ok.sum()), "mean w": float(np.nanmean(Fc[:, 0]))}
    for k in (3, 4, 5, 8, 11, 17):
        r[f"rmse k={k}"] = float(np.sqrt(np.nanmean(E[ok, k] ** 2)))
    r["bias k=3..11"] = float(np.nanmean(E[ok, 3:12]))
    # scaling with the level: correlation of |error| with the forecast, and rmse of low / high forecast halves
    a, f = np.abs(E[ok, 3:12]).ravel(), Fc[ok, 3:12].ravel()
    r["corr(|e|, forecast)"] = float(np.corrcoef(a, f)[0, 1])
    med = np.median(f)
    r["rmse low half"] = float(np.sqrt(np.mean(E[ok, 3:12].ravel()[f < med] ** 2)))
    r["rmse high half"] = float(np.sqrt(np.mean(E[ok, 3:12].ravel()[f >= med] ** 2)))
    # along the horizon
    c = np.corrcoef(E[ok, 3:18].T)
    r["corr e3,e4"], r["corr e3,e6"], r["corr e3,e12"] = float(c[0, 1]), float(c[0, 3]), float(c[0, 9])
    # do the last revealed errors predict the next ones? at origin t the value of hour t+2 is new:
    # its error against the forecast made one hour earlier is E[t-1, 3]
    same = (pos[1:] == pos[:-1] + 1) & (blk[1:] == blk[:-1])
    prev1 = np.full(len(E), np.nan)
    prev1[1:] = np.where(same, E[:-1, 3], np.nan)
    prev3 = np.full(len(E), np.nan)                      # mean of the last 3 one-step-revealed errors
    s3 = same[2:] & same[1:-1] & same[:-2]
    prev3[3:] = np.where(s3, (E[2:-1, 3] + E[1:-2, 3] + E[:-3, 3]) / 3.0, np.nan)
    nxt1, nxt3 = E[:, 3], E[:, 3:6].mean(1)
    for name, x, y in (("corr(last revealed e, next e3)", prev1, nxt1), ("corr(mean last 3 revealed, mean next 3)", prev3, nxt3)):
        m = np.isfinite(x) & np.isfinite(y)
        r[name] = float(np.corrcoef(x[m], y[m])[0, 1])
    m = np.isfinite(prev3) & np.isfinite(nxt3)
    r["slope next3 ~ last3"] = float(np.polyfit(prev3[m], nxt3[m], 1)[0])
    # bias by hour of day at k=3..5 relative to rmse
    bh = np.array([np.nanmean(E[ok & (hod == h), 3:6]) for h in range(24)])
    r["max |bias by hour| k=3..5"] = float(np.abs(bh).max())
    rows.append(r)
out = pd.DataFrame(rows).set_index("fit set").T
out.to_csv(HERE / "diag_errors.csv")
pd.set_option("display.width", 200)
print(out.round(4).to_string())
