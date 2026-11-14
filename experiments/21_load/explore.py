# Usage: .venv/Scripts/python.exe experiments/21_load/explore.py
"""Quick checks of the meter's hourly behaviour (night freeze, 06:00 release, 22:00 freeze value)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hard import load_hard, get_folds, train_complement

df = load_hard()
print(df.shape, df.index.min(), df.index.max())
L = df["load_kw"]
full = L.asfreq("h")
d = full.diff()
h = full.index.hour
print("abs diff to previous hour by hour (median, p90, max, share exactly<0.5kW):")
for hh in range(24):
    x = d[h == hh].dropna().abs()
    print(hh, round(x.median(), 1), round(x.quantile(.9), 1), round(x.max(), 1), round((x < 0.5).mean(), 3), len(x))
# gaps in the official table
gap = df.index.to_series().diff().dt.total_seconds().div(3600)
print("gaps >1h:", (gap > 1).sum()); print(gap[gap > 1].head(40))
# hour 22 vs hour 21 and 20
w = pd.DataFrame({"l22": full[h == 22].values}, index=full.index[h == 22].normalize())
for k in (21, 20, 19):
    w[f"l{k}"] = full[h == k].values[:len(w)] if (h == k).sum() == len(w) else np.nan
w = w.dropna()
print(len(w), "corr", w.corr().round(3).to_string())
X = np.c_[np.ones(len(w)), w.l21, w.l20]
b = np.linalg.lstsq(X, w.l22, rcond=None)[0]; print("l22 ~ 1,l21,l20:", b.round(3), "MAE", np.abs(X @ b - w.l22).mean().round(1), "persist MAE", (w.l22 - w.l21).abs().mean().round(1))
# hour 6 vs frozen value and yesterday's hour 6
w6 = pd.DataFrame({"l6": full[h == 6]})
w6["l5"] = full.shift(1)[h == 6]; w6["l6y"] = full.shift(24)[h == 6]; w6["l7y"] = full.shift(23)[h == 6]; w6["l6w"] = full.shift(168)[h == 6]
w6["dow"] = w6.index.dayofweek
print(w6.dropna().describe().round(0).to_string())
print(w6.dropna().corr().round(3).to_string())
print("drop at 06 (l6-l5) by dow:", (w6.l6 - w6.l5).groupby(w6.dow).agg(["mean", "std", "count"]).round(0).to_string())
print("ratio l6/l5 by dow:", (w6.l6 / w6.l5).groupby(w6.dow).agg(["mean", "std"]).round(3).to_string())
