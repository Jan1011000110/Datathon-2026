# Usage: .venv/Scripts/python.exe experiments/23_signal/errstruct.py
"""Error structure of the round-1 forecast of w on folds A, B, C (for the planner):
 1. by horizon: bias, MAE, standard deviation, and the share of the error that is a common shift of the
    window +3..+8;
 2. correlation of the errors between horizons;
 3. by hour of day of the origin: MAE of the window level;
 4. peak and trough: is the height of the evening peak / depth of the midday trough predicted?
Writes errstruct_horizon.csv, errstruct_corr.csv, errstruct_hour.csv, errstruct_extremes.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    E, hours, FF, TT, days, dows = [], [], [], [], [], []
    for f in "ABC":
        b = fast.base_tape(f)
        F = fast.w_of(b["Cc"], b["Pp"]); T = fast.w_of(b["Tc"], b["Tp"])
        # rows whose next 35 rows are the next 35 hours (no week boundary inside)
        idx = fast.fold(f).index
        ok = np.zeros(len(idx), bool)
        hrs = idx.values.astype("datetime64[h]").astype(np.int64)
        ok[:-35] = (hrs[35:] - hrs[:-35]) == 35
        E.append((F - T)[ok]); hours.append(idx.hour.values[ok]); FF.append(F[ok]); TT.append(T[ok])
        days.append(idx.normalize().values[ok]); dows.append(idx.dayofweek.values[ok])
    E, hours, FF, TT = np.vstack(E), np.concatenate(hours), np.vstack(FF), np.vstack(TT)
    wm = float(np.mean(TT))
    print(f"{len(E)} forecast origins, mean w {wm * 1e3:.1f} g")

    # 1. by horizon
    lvl = E[:, 3:9].mean(1)
    rows = []
    for k in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24, 35):
        e = E[:, k]
        rows.append({"k": k, "bias_g": e.mean() * 1e3, "mae_g": np.abs(e).mean() * 1e3, "sd_g": e.std() * 1e3,
                     "sd_pct_of_mean_w": 100 * e.std() / wm, "corr_with_window_level": np.corrcoef(e, lvl)[0, 1]})
    h = pd.DataFrame(rows)
    h.to_csv(HERE / "errstruct_horizon.csv", index=False)
    print(h.round(2).to_string(index=False))
    W = E[:, 3:9]
    tot = (W ** 2).mean()
    common = (lvl ** 2).mean()
    print(f"\nwindow +3..+8: mean square error {tot * 1e6:.1f} g2, of which common shift (window mean) {common * 1e6:.1f} g2 "
          f"= {100 * common / tot:.0f} %; MAE of the window level {np.abs(lvl).mean() * 1e3:.2f} g, sd {lvl.std() * 1e3:.2f} g")

    # 2. correlation between horizons
    ks = [3, 4, 5, 6, 8, 11, 17, 23]
    C = pd.DataFrame(np.corrcoef(E[:, ks].T), index=ks, columns=ks)
    C.to_csv(HERE / "errstruct_corr.csv")
    print("\ncorrelation of errors between horizons"); print(C.round(2).to_string())

    # 3. by hour of day of the origin
    g = pd.DataFrame({"hour": hours, "abs_lvl": np.abs(lvl), "lvl": lvl, "abs_k3": np.abs(E[:, 3])})
    hh = g.groupby("hour").agg(mae_level_g=("abs_lvl", lambda x: x.mean() * 1e3), bias_level_g=("lvl", lambda x: x.mean() * 1e3),
                               mae_k3_g=("abs_k3", lambda x: x.mean() * 1e3))
    hh.to_csv(HERE / "errstruct_hour.csv")
    print("\nby hour of day of the origin"); print(hh.round(1).T.to_string())

    # 4. extremes of the window +3..+11: predicted vs actual, and against a no-skill reference
    rows = []
    for name, fn in (("max of +3..+11", np.max), ("min of +3..+11", np.min), ("range of +3..+11", np.ptp), ("mean of +3..+8", None)):
        a = TT[:, 3:9].mean(1) if fn is None else fn(TT[:, 3:12], axis=1)
        p = FF[:, 3:9].mean(1) if fn is None else fn(FF[:, 3:12], axis=1)
        known = TT[:, :3].mean(1)
        slope, icpt = np.polyfit(p, a, 1)
        rows.append({"quantity": name, "mean_actual_g": a.mean() * 1e3, "mean_forecast_g": p.mean() * 1e3,
                     "slope_actual_on_forecast": slope, "corr": np.corrcoef(a, p)[0, 1],
                     "mae_g": np.abs(a - p).mean() * 1e3,
                     "corr_of_change_vs_known": np.corrcoef(a - known, p - known)[0, 1]})
    x = pd.DataFrame(rows)
    x.to_csv(HERE / "errstruct_extremes.csv", index=False)
    print("\nextremes and level of the window: forecast against actual"); print(x.round(3).to_string(index=False))
    # evening peak seen from noon, midday trough seen from 06:00
    for label, h0, lo, hi, fn in (("evening peak (max of 18-22 h) forecast at 13:00", 13, 5, 10, np.max),
                                  ("midday trough (min of 11-16 h) forecast at 06:00", 6, 5, 11, np.min)):
        m = hours == h0
        a, p = fn(TT[m, lo:hi], axis=1), fn(FF[m, lo:hi], axis=1)
        known = TT[m, :3].mean(1)
        slope, _ = np.polyfit(p, a, 1)
        print(f"{label}: n={m.sum()}, corr {np.corrcoef(a, p)[0, 1]:.2f}, slope {slope:.2f}, bias {1e3 * (p - a).mean():+.1f} g, "
              f"MAE {1e3 * np.abs(p - a).mean():.1f} g ({100 * np.abs(p - a).mean() / a.mean():.0f} % of its mean); "
              f"ratio to the known hours: corr {np.corrcoef(a / known, p / known)[0, 1]:.2f}")
