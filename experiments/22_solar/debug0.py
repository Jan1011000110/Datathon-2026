# NOTE: historical development script. It ran against the FIRST version of solar_fc.py (pooled LightGBM with a
# per-horizon ridge fallback, 22 features) and does not run against the current file. Its printed results are
# quoted in the report (reports/12_solar_forecast). The current model is cross-validated by cv_final.py.
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, train_complement
import solar_fc as S
df = get_folds()["C"]; fit = train_complement(df)
fc = S.SolarFC(model="linear", fit_quantiles=False)
t0=time.perf_counter()
df2 = fit[fit["pv_kw"].notna()].sort_index()
keys = df2.index.as_unit("ns").asi8 // S.HNS
pv = df2["pv_kw"].to_numpy(float)
fc.pv_fit = dict(zip(keys.tolist(), pv.tolist()))
fc.env_tab = fc._build_envelope(keys, pv); fc._dcache = {}; fc.c_irr = 1.65
print("env time", time.perf_counter()-t0)
np.set_printoptions(linewidth=250, suppress=True)
for doy in (14, 105, 196, 288, 319):
    print(doy, fc.env_tab[doy].round(0))
t0=time.perf_counter()
X, y, E, D = fc._replay(df2, 36)
print("replay time", time.perf_counter()-t0, X.shape)
F = pd.DataFrame(X, columns=S.FEATS); F["y"]=y; F["E"]=E; F["r"]=y/E
print(F.describe().T.round(3).to_string())
h0 = F[F.h==0]
print("h=0 corr r vs k_last:", h0[["r","k_last","k_today","k_yday","k_same"]].corr().round(3).to_string())
print("h=0 MAE of E*k_last:", (h0.E*h0.k_last - h0.y).abs().mean(), "n", h0.k_last.notna().sum(), len(h0))
print("r>1.2 share", (F.r>1.2).mean(), " by Et bucket:"); print(F.groupby(pd.cut(F.E,[0,60,150,400,800,2000])).r.describe().round(2).to_string())
