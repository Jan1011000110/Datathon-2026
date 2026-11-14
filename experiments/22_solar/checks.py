# Usage: .venv/Scripts/python.exe experiments/22_solar/checks.py
"""Robustness checks, offline: (1) cold start (fold A), (2) a synthetic inverter outage week,
(3) calibration of the horizon-0 quantiles and the scale that would fix it."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, train_complement
from src.hard_mpc import ProfileN
from eval_offline import replay
from solar_fc import SolarFC

def mae_table(df, fcs, hs=(0, 1, 3, 6, 24), sel=None):
    tmap = dict(zip(df.index, df["pv_kw"].to_numpy()))
    err = {n: {h: [] for h in hs} for n in fcs}
    for t, obs, hist in replay(None, df):
        ts = obs["timestamp"]
        if sel is not None and not sel(ts):
            for f in fcs.values():
                f.predict_pv(obs, hist, 36)
            continue
        P = {n: f.predict_pv(obs, hist, 36) for n, f in fcs.items()}
        for h in hs:
            tt = ts + pd.Timedelta(hours=h)
            y = tmap.get(tt)
            if y is None or not (8 <= tt.hour <= 19):
                continue
            for n in fcs:
                err[n][h].append(P[n][h] - y)
    return pd.DataFrame({n: {f"MAE h{h}": np.mean(np.abs(e)) for h, e in d.items()} | {f"bias h{h}": np.mean(e) for h, e in d.items() if h in (0, 6)}
                         for n, d in err.items()}).round(1)

# 1. cold start
A = get_folds()["A"]; fitA = train_complement(A)
print("fold A (days before each week are hidden), daytime targets:")
print(mae_table(A, {"SolarFC": SolarFC(fitA, fit_quantiles=False), "profile": ProfileN(fitA)}).to_string())

# 2. synthetic outage: the inverter reports 0 during the July week of fold B, the weather station keeps working
B = get_folds()["B"]; fitB = train_complement(B)
Bz = B.copy(); wk = (Bz.index >= "2026-07-08") & (Bz.index < "2026-07-15")
Bz.loc[wk, "pv_kw"] = 0.0
sel = lambda ts: pd.Timestamp("2026-07-08") <= ts < pd.Timestamp("2026-07-15")
print("\nsynthetic outage week (output = 0, irradiance normal), daytime targets inside the week:")
print(mae_table(Bz, {"SolarFC": SolarFC(fitB, fit_quantiles=False), "SolarFC no outage rule": SolarFC(fitB, fit_quantiles=False, outage_rule=False),
                     "profile": ProfileN(fitB)}, sel=sel).to_string())
Bn = Bz.copy(); Bn.loc[wk, "irradiancia_wm2"] = 0.0
print("same, irradiance also reported as 0:")
print(mae_table(Bn, {"SolarFC": SolarFC(fitB, fit_quantiles=False), "profile": ProfileN(fitB)}, sel=sel).to_string())
sel2 = lambda ts: pd.Timestamp("2026-08-08") <= ts < pd.Timestamp("2026-08-15")
print("week after the outage (recovery), real data:")
print(mae_table(Bz, {"SolarFC": SolarFC(fitB, fit_quantiles=False), "profile": ProfileN(fitB)}, sel=sel2).to_string())

# 3. quantile calibration at horizon 0
o = pd.read_csv(Path(__file__).with_name("offline_final.csv"))
h0 = o[(o.h == 0) & (o.hour_t >= 8) & (o.hour_t <= 19)]
for s in (1.0, 1.1, 1.2, 1.3, 1.4):
    cov = {q: round(float((h0.y <= h0.q50 + s * (h0[q] - h0.q50)).mean()), 3) for q in ("q10", "q25", "q75", "q90")}
    print("scale", s, cov)
