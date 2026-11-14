# Usage: .venv/Scripts/python.exe experiments/27_combined/calibration.py
"""How well the net-load scenarios describe reality on folds A, B, C (tables fitted outside each fold):
  1. share of actual hour-0 net load below the 10/25/50/75/90 % scenario quantiles, by block of hours;
  2. pinball loss and width of the hour-0 distribution per method;
  3. correlation between the hour-0 errors of LoadFC and SolarFC (the independence assumption of "conv");
  4. the same coverage for later horizons.
Writes calib_h0.csv, calib_horizon.csv, corr.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined as cb  # noqa: E402
from src.hard import get_folds  # noqa: E402

HERE = Path(__file__).resolve().parent
QS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
BLOCKS = {"23-05 (meter frozen, no sun)": (23, 0, 1, 2, 3, 4, 5), "06-08": (6, 7, 8), "09-12": (9, 10, 11, 12),
          "13-16": (13, 14, 15, 16), "17-22": (17, 18, 19, 20, 21, 22)}


def dist(tp, tabs, method, load, solar, k):
    """(n, M) equally likely values of the net load of hour t+k as seen at t; (n,) point forecast."""
    lk, pk = cb.LOAD_KEYS[load], cb.PV_KEYS[solar]
    table, load_table = tabs[:2]
    hod = (tp["hrs"] + k) % 24
    point = tp[lk][:, k] - tp[pk][:, k]
    if method == "emp" or (method == "hybrid" and k > 0):
        return point[:, None] + table[hod, k], point
    lq = tp[lk + "_q"].astype(float) if k == 0 else tp[lk][:, k][:, None] + load_table[hod, k]
    sq = tp[pk + "_q"][:, :, k].astype(float)
    return (lq[:, :, None] - sq[:, None, :]).reshape(len(lq), -1), point


def pinball(q, y, lev):
    d = y[:, None] - q
    return np.mean(np.maximum(lev * d, (lev - 1) * d), axis=1)


rows, hrows, crows = [], [], []
for fold in "ABC":
    df = get_folds()[fold]
    z = np.load(cb.CACHE / f"tape_{fold}.npz")
    tp = {k: z[k] for k in z.files}
    n = len(df)
    net = tp["load"] - tp["pv"]
    hrs = tp["hrs"]
    for load, solar, method in (("fc", "fc", "emp"), ("fc", "fc", "conv"), ("nomodel", "fc", "conv"), ("naive", "naive", "emp")):
        tabs = cb.error_tables_for(df, load, solar)
        for k in (0, 1, 3, 6, 11):
            V, point = dist(tp, tabs, method, load, solar, k)
            ok = np.zeros(n, bool)
            ok[:n - k] = (hrs[k:] - hrs[:n - k]) == k
            y = np.full(n, np.nan)
            y[:n - k] = net[k:]
            q = np.quantile(V, QS, axis=1).T
            flat = (q[:, 4] - q[:, 0]) < 5.0                     # "known" hours: 10-90 % band narrower than 5 kW
            hod = (hrs + k) % 24
            rec = dict(fold=fold, load=load, solar=solar, method=method, k=k)
            m = ok & ~flat
            hrows.append({**rec, "n": int(m.sum()), **{f"below_{int(a * 100)}": float(np.mean(y[m] < q[m, j])) for j, a in enumerate(QS)},
                          "pinball_kw": float(pinball(q[m], y[m], QS).mean()), "width_10_90_kw": float(np.mean(q[m, 4] - q[m, 0])),
                          "mae_point_kw": float(np.mean(np.abs(y[m] - point[m]))), "n_flat": int((ok & flat).sum()),
                          "flat_exact": float(np.mean(np.abs(y[ok & flat] - point[ok & flat]) < 5.0)) if (ok & flat).any() else np.nan})
            if k:
                continue
            for bname, hh in BLOCKS.items():
                mb = ok & np.isin(hod, hh)
                mw = mb & ~flat
                r = {**rec, "block": bname, "n": int(mb.sum()), "n_zero_width": int((mb & flat).sum()),
                     "zero_width_exact": float(np.mean(np.abs(y[mb & flat] - point[mb & flat]) < 5.0)) if (mb & flat).any() else np.nan,
                     "zero_width_mae_kw": float(np.mean(np.abs(y[mb & flat] - point[mb & flat]))) if (mb & flat).any() else np.nan}
                if mw.sum() >= 20:
                    r.update({f"below_{int(a * 100)}": float(np.mean(y[mw] < q[mw, j])) for j, a in enumerate(QS)})
                    r.update(pinball_kw=float(pinball(q[mw], y[mw], QS).mean()), width_10_90_kw=float(np.mean(q[mw, 4] - q[mw, 0])),
                             mae_point_kw=float(np.mean(np.abs(y[mw] - point[mw]))))
                rows.append(r)
    # independence of the two hour-0 errors (sunlit, meter alive)
    el = tp["load"] - tp["load_fc"][:, 0]
    es = tp["pv"] - tp["pv_fc"][:, 0]
    hod = hrs % 24
    for bname, hh in list(BLOCKS.items())[1:] + [("06-22", tuple(range(6, 23)))]:
        m = np.isin(hod, hh) & (tp["pv_fc"][:, 0] > 20.0)
        if m.sum() > 30:
            rk = lambda x: np.argsort(np.argsort(x))
            crows.append(dict(fold=fold, source="evaluation fold", block=bname, n=int(m.sum()), pearson=float(np.corrcoef(el[m], es[m])[0, 1]),
                              spearman=float(np.corrcoef(rk(el[m]), rk(es[m]))[0, 1]),
                              sd_load_kw=float(el[m].std()), sd_solar_kw=float(es[m].std())))

cal = pd.DataFrame(rows)
cal.to_csv(HERE / "calib_h0.csv", index=False)
hor = pd.DataFrame(hrows)
hor.to_csv(HERE / "calib_horizon.csv", index=False)
cor = pd.DataFrame(crows)
cor.to_csv(HERE / "corr.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 500)
cols = [f"below_{int(a * 100)}" for a in QS] + ["pinball_kw", "width_10_90_kw", "mae_point_kw"]
print("\nHour 0, pooled over folds (rows weighted equally per fold), hours with a non-degenerate distribution")
g = cal.groupby(["load", "solar", "method", "block"], sort=False)
print(pd.concat([g[["n", "n_zero_width"]].sum(), g[["zero_width_exact", "zero_width_mae_kw"] + cols].mean()], axis=1).round(3).to_string())
print("\nBy horizon, mean of folds")
print(hor.groupby(["load", "solar", "method", "k"], sort=False)[cols + ["flat_exact"]].mean().round(3).to_string())
print("\nBy fold, hour 0, all non-degenerate hours")
print(hor[hor.k == 0].set_index(["load", "solar", "method", "fold"])[["n"] + cols].round(3).to_string())
print("\nCorrelation of the hour-0 errors of LoadFC and SolarFC")
print(cor.round(3).to_string())
