import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.hard import load_hard, get_folds, train_complement
pd.set_option("display.width", 250)
df = load_hard()
full = load_hard("filled")
print(df.shape, full.shape)
print(df[["pv_kw","irradiancia_wm2","temp_ambient_c","temp_panel_c","vent_ms"]].describe().round(1))
print("NaN meteo in official:", df[["irradiancia_wm2","temp_ambient_c","temp_panel_c","vent_ms"]].isna().sum().to_dict())
# hourly mean pv by month x hour
print((df.pv_kw.groupby([df.index.month, df.index.hour]).mean().unstack().round(0)).to_string())
print((df.pv_kw.groupby([df.index.month, df.index.hour]).quantile(0.95).unstack().round(0)).to_string())
# irradiance relation
d = df[df.irradiancia_wm2 > 20]
print("corr pv-irr same hour:", d[["pv_kw","irradiancia_wm2"]].corr().iloc[0,1])
print((d.pv_kw/d.irradiancia_wm2).groupby([d.index.month]).describe().round(2))
print((d.pv_kw/d.irradiancia_wm2).groupby([d.index.hour]).median().round(2).to_dict())
# zeros at midday
mid = df[(df.index.hour>=11)&(df.index.hour<=14)]
print("midday pv==0:", (mid.pv_kw==0).sum(), "of", len(mid), "; pv<5:", (mid.pv_kw<5).sum())
print(mid[mid.pv_kw<5][["pv_kw","irradiancia_wm2"]].head(40))
# gaps
gaps = df.index.to_series().diff().dt.total_seconds().div(3600)
print("gaps>1h:", (gaps>1).sum()); print(gaps[gaps>1].sort_values().tail(15))
for k,f in get_folds().items():
    print(k, len(f), f.index.min(), f.index.max(), "pv mean", f.pv_kw.mean().round(1))
