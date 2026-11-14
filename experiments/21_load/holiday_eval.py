# Usage: .venv/Scripts/python.exe experiments/21_load/holiday_eval.py [tag]
"""Horizon-0 (and later) error on the low-load weekdays found by holidays.py, against normal
days, from offline_<tag>_errors.csv. Holiday days are taken as meter days (06:00 -> 05:59)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hard import get_folds

here = Path(__file__).parent
tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
e = pd.read_csv(here / f"offline_{tag}_errors.csv")
hol = pd.to_datetime(pd.read_csv(here / "holidays.csv")["date"]).dt.normalize()
folds = get_folds()
e["ts"] = np.concatenate([folds[f].index[g["t"].to_numpy()] for f, g in e.groupby("fold", sort=False)]) if False else None
parts = []
for f, g in e.groupby("fold", sort=False):
    g = g.copy(); g["ts"] = folds[f].index[g["t"].to_numpy()]; parts.append(g)
e = pd.concat(parts)
e["target"] = e["ts"] + pd.to_timedelta(e["k"], unit="h")
mday = (e["target"] - pd.Timedelta(hours=6)).dt.normalize()
e["holiday"] = mday.isin(hol)
first = hol[~(hol - pd.Timedelta(days=1)).isin(hol) & ~((hol.dt.dayofweek == 0) & (hol - pd.Timedelta(days=3)).isin(hol))]
e["hol_first"] = mday.isin(first)
e["ae"] = e["err"].abs()
day = e[(e.target.dt.hour >= 6) & (e.target.dt.hour <= 22)]
print("holiday meter-days in folds:", sorted(set(mday[e.holiday].dt.date)))
print("first days of a stretch:", sorted(set(first.dt.date)))
for lab, m in (("holiday weekdays (all)", day.holiday), ("  first day of a stretch", day.hol_first), ("  later days of a stretch", day.holiday & ~day.hol_first),
               ("normal days", ~day.holiday)):
    x = day[m]
    print(f"\n== {lab}: MAE by horizon, target hours 06-22, folds A+B+C (n at k=0: {int((x.k == 0).groupby(x.model).sum().max())}) ==")
    print(x.pivot_table(index="model", columns="k", values="ae").round(0).to_string())
    print(f"== {lab}: bias ==")
    print(x.pivot_table(index="model", columns="k", values="err").round(0).to_string())
x = day[day.holiday & (day.k == 0)]
print("\n== holiday weekdays, horizon 0, MAE by hour ==")
print(x.pivot_table(index="model", columns=x.target.dt.hour, values="ae").round(0).astype(int).to_string())
day[day.holiday | True].groupby(["model", "holiday", "hol_first", "k"])["ae"].mean().round(1).to_csv(here / f"holiday_{tag}_mae.csv")
