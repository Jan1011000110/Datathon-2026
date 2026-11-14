# Usage: .venv/Scripts/python.exe experiments/21_load/holidays.py
"""Find low-load weekdays (holidays, closures) from the load itself: weekdays whose 08-19h mean
is below 80 % of the median of the other weekdays within +-21 days. Writes holidays.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_hourly_full

L = load_hourly_full()["load_kw"].dropna()
L = L[L >= 50]
d = L[(L.index.hour >= 8) & (L.index.hour <= 19)]
day = d.groupby(d.index.normalize()).agg(["mean", "count"])
day = day[day["count"] >= 10]["mean"]
wd = day[day.index.dayofweek < 5]
we = day[day.index.dayofweek >= 5]
rows = []
for t, v in wd.items():
    near = wd[(abs((wd.index - t).days) <= 21) & (wd.index != t)]
    wnear = we[abs((we.index - t).days) <= 21]
    rows.append((t.date(), t.day_name()[:3], round(v), round(near.median()), round(wnear.median()) if len(wnear) else np.nan, round(v / near.median(), 2)))
h = pd.DataFrame(rows, columns=["date", "dow", "mean_08_19_kw", "median_nearby_weekdays", "median_nearby_weekends", "ratio"])
low = h[h.ratio < 0.8]
low.to_csv(Path(__file__).with_name("holidays.csv"), index=False)
print(low.to_string(index=False))
print("weekday means Dec 15-24:"); print(h[(h.date >= pd.Timestamp("2025-12-15").date()) & (h.date <= pd.Timestamp("2026-01-12").date())].to_string(index=False))
print("Jul/Aug:"); print(h[(h.date >= pd.Timestamp("2026-07-13").date())].to_string(index=False))
