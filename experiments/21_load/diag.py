# Usage: .venv/Scripts/python.exe experiments/21_load/diag.py v2
"""Where is the score still lost after the load model? Uses the traces of the
'perfect signal, perfect solar' runs (so load is the only unknown): traces/<variant>_<fold>.parquet.
  * discharge spill  = energy discharged beyond the true net load (thrown away), by hour of day,
                       in kWh and in score points (kWh x score weight of that hour);
  * under-cover      = grid import in hours where the battery was discharging below its limits
                       (the forecast was too low; the energy stays in the battery, so the loss is smaller);
  * score on holiday stretches (1-7 Jan, 2-6 Apr, 6-8 Dec, other low weekdays) against normal days.
Writes diag_<variant>_by_hour.csv and diag_holiday_scores.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hard import PMAX, ETA, get_folds, score_weights
from src.sim import score_from_grid

here = Path(__file__).parent
variants = sys.argv[1:] or ["v2"]
folds = get_folds()
hol = pd.to_datetime(pd.read_csv(here / "holidays.csv")["date"])
out_h, out_s = [], []
for v in variants + ["ProfileN", "perfect"]:
    for f in ("A", "B", "C"):
        p = here / "traces" / f"{v}_{f}.parquet"
        if not p.exists():
            continue
        tr = pd.read_parquet(p); df = folds[f]
        w = score_weights(df) * 100                       # points per kWh of grid import
        net = tr["net_kw"].to_numpy(); pk = tr["p_kw"].to_numpy(); soc = tr["soc_kwh"].to_numpy()
        dspill = np.where(pk < 0, np.maximum(-pk - np.maximum(net, 0), 0), 0.0)
        soc_before = np.r_[3000.0, soc[:-1]]
        free = (pk < 0) & (-pk < PMAX - 1) & (soc > 1.0)   # discharging, not limited by power or by an empty battery
        under = np.where(free, tr["grid_kw"].to_numpy(), 0.0)
        g = pd.DataFrame({"variant": v, "fold": f, "hour": tr.index.hour, "dspill_kwh": dspill, "dspill_pts": dspill * w,
                          "under_kwh": under, "under_pts_max": under * w, "discharge_kwh": np.maximum(-pk, 0)})
        out_h.append(g)
        md = (tr.index - pd.Timedelta(hours=6)).normalize()
        masks = {"all": np.ones(len(df), bool), "holiday weekdays": np.asarray(md.isin(hol)),
                 "normal days": ~np.asarray(md.isin(hol)),
                 "1-7 Jan": (df.index.month == 1) & (df.index.day <= 7), "1-7 Apr (Easter 2-6)": (df.index.month == 4) & (df.index.day <= 7),
                 "6-8 Dec": (df.index.month == 12) & (df.index.day >= 6) & (df.index.day <= 8)}
        for lab, m in masks.items():
            if m.sum() < 24:
                continue
            r = score_from_grid(tr["grid_kw"].to_numpy(), df["load_kw"].to_numpy(), df["pv_kw"].to_numpy(),
                                df["precio_eur_per_kwh"].to_numpy(), df["co2_kg_per_kwh"].to_numpy(), None, m)
            out_s.append({"variant": v, "fold": f, "subset": lab, "hours": int(m.sum()), "score": r["score"],
                          "dspill_kwh": float(dspill[m].sum())})
H = pd.concat(out_h); S = pd.DataFrame(out_s)
S.to_csv(here / "diag_holiday_scores.csv", index=False)
print("== score on subsets of hours (0.7 dCO2% + 0.3 dCost% computed on those hours only) ==")
print(S.pivot_table(index=["subset", "fold", "hours"], columns="variant", values="score").round(2).to_string())
print("== discharge spill on subsets, kWh ==")
print(S.pivot_table(index=["subset", "fold"], columns="variant", values="dspill_kwh").round(0).to_string())
for v in variants + ["ProfileN", "perfect"]:
    x = H[H.variant == v]
    if x.empty:
        continue
    t = x.groupby("hour")[["dspill_kwh", "dspill_pts", "under_kwh", "under_pts_max", "discharge_kwh"]].sum() / 3  # mean per fold
    t.to_csv(here / f"diag_{v}_by_hour.csv")
    print(f"\n== {v}: per fold (mean of A, B, C), by hour of day ==")
    print(t.round(2).T.to_string())
    print("totals:", t.sum().round(2).to_dict())
