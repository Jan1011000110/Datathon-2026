import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.hard import load_hard
from src.data import load_hourly_full
pd.set_option("display.width", 250)
full = load_hourly_full()
print(full.columns.tolist())
print("full pv notna by month:"); print(full.pv_kw.notna().groupby([full.index.year, full.index.month]).sum().to_string())
print("full irr notna by month:"); print(full.irradiance_wm2.notna().groupby([full.index.year, full.index.month]).sum().to_string())
print("full load notna by month:"); print(full.load_kw.notna().groupby([full.index.year, full.index.month]).sum().to_string())
# days present per month
print(full.index.normalize().unique().to_series().groupby(lambda d: (d.year, d.month)).agg(["min","max","count"]).to_string())
# centroid of irradiance by day (local hour)
g = full.dropna(subset=["irradiance_wm2"])
cen = (g.irradiance_wm2 * (g.index.hour + 0.5)).groupby(g.index.normalize()).sum() / g.irradiance_wm2.groupby(g.index.normalize()).sum()
print(cen["2025-10-15":"2025-11-08"].round(2).to_string())
print(cen["2026-03-15":"2026-04-08"].round(2).to_string())
# duplicated / missing timestamps
print("dup idx:", full.index.duplicated().sum())
aug = full["2026-08-10":"2026-08-24"]
print(aug[["pv_kw","irradiance_wm2","load_kw"]].between_time("12:00","13:00").to_string())
