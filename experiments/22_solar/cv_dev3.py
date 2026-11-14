# NOTE: historical development script. It ran against the FIRST version of solar_fc.py (pooled LightGBM with a
# per-horizon ridge fallback, 22 features) and does not run against the current file. Its printed results are
# quoted in the report (reports/12_solar_forecast). The current model is cross-validated by cv_final.py.
# Usage: .venv/Scripts/python.exe experiments/22_solar/cv_dev3.py
"""Blocked CV: does the panel-minus-ambient temperature (a night-time cloud sensor) help, and in which form?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.hard import load_hard
from cv_dev import table, cv, HS
from cv_dev2 import cell_ridge
import solar_fc as S

if __name__ == "__main__":
    fc, F = table()
    kc = float(F.y.sum() / F.E.sum())
    df = load_hard()
    off = F.D.map({d: S._doy_off(d)[1] for d in F.D.unique()})
    F["key"] = (F.D * 24 + F.hs_now + off).astype(np.int64)
    s = pd.DataFrame({"dT": (df.temp_panel_c - df.temp_ambient_c).to_numpy(), "ta": df.temp_ambient_c.to_numpy()},
                     index=df.index.as_unit("ns").asi8 // S.HNS)
    full = s.reindex(np.arange(s.index.min(), s.index.max() + 1))
    for n in (3, 6):
        m = full.dT.rolling(n, min_periods=max(1, n // 2)).mean()
        F[f"dT{n}"] = (F.key - 1).map(m).where(F.dT.notna())     # mean over the last n hours ending at the last hour
    F["dA24"] = ((F.key - 1).map(full.ta) - (F.key - 25).map(full.ta)).where(F.dT.notna())
    F["dTmin6"] = (F.key - 1).map(full.dT.rolling(6, min_periods=3).min()).where(F.dT.notna())
    base = ["k_last", "k_today", "k_yday", "k_same"]
    cv(F, cell_ridge(base, lam=0.3, kc=kc), "cells lam=0.3 base")
    for name, extra in [("+dT", ["dT"]), ("+dT3", ["dT3"]), ("+dT6", ["dT6"]), ("+dT,dT6", ["dT", "dT6"]), ("+dT,dA24", ["dT", "dA24"]),
                        ("+dT,dTmin6", ["dT", "dTmin6"]), ("+dT,tamb", ["dT", "tamb"]), ("+dT,dT6,dA24,tamb", ["dT", "dT6", "dA24", "tamb"])]:
        for lam in (0.1, 0.3):
            cv(F, cell_ridge(base + extra, lam=lam, kc=kc), f"cells lam={lam} {name}")
    # where does dT help? MAE by current solar hour for h in 1..12
    p0 = cv(F, cell_ridge(base, lam=0.3, kc=kc), "base again")
    p1 = cv(F, cell_ridge(base + ["dT"], lam=0.3, kc=kc), "+dT again")
    ok = ~np.isnan(p0)
    d = F[ok].assign(e0=np.abs(p0[ok] - F.y[ok]), e1=np.abs(p1[ok] - F.y[ok]))
    d = d[(d.h >= 1) & (d.h <= 14)]
    print(d.groupby("hs_now")[["e0", "e1"]].mean().round(0).T.to_string())
    print("corr(dT at night, next-day clearness):")
    n = F[(F.hs_now <= 3) & (F.hs_now >= 0) & (F.h >= 6) & (F.h <= 12)]
    print(np.corrcoef(n.dT.fillna(0), n.y / n.E)[0, 1], " vs k_yday:", np.corrcoef(n.k_yday.fillna(kc), n.y / n.E)[0, 1])
