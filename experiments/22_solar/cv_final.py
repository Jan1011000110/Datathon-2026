# Usage: .venv/Scripts/python.exe experiments/22_solar/cv_final.py
"""Blocked CV (the out-of-block predictions SolarFC.fit makes for its quantile tables) on all train rows:
variants of the final model. MAE in kW over sunlit target hours (envelope >= 60 kW)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import load_hard
import solar_fc as S

df = load_hard()
HS = [0, 1, 2, 3, 6, 12, 24]

def report(name, fc):
    o = fc.oof
    e = np.abs(o["pred"] - o["y"]); big = o["E"] >= S.E_MIN
    res = {f"h{h}": round(float(e[big & (o["h"] == h)].mean()), 1) for h in HS}
    res["all"] = round(float(e[big].mean()), 1)
    print(f"{name:42s}", res, flush=True)

class Hinge(S.SolarFC):
    knots = (0.4, 0.85)
    def _design1(self, X):
        Z = super()._design1(X)
        kl = np.nan_to_num(X[:, S.IX["k_last"]], nan=self.k_clim)
        return np.column_stack([Z] + [np.maximum(kl - a, 0.0) if a > 0.6 else np.minimum(kl - a, 0.0) for a in self.knots])

class Trend(S.SolarFC):
    pass

report("final (as shipped)", S.SolarFC(df))
report("no weather stage", S.SolarFC(df, use_weather=False))
report("hinges on k_last (0.4, 0.85)", Hinge(df))
for lam2 in (0.05, 1.0, 3.0):
    report(f"lam2={lam2}", S.SolarFC(df, lam2=lam2))
for q, w in ((0.95, 12), (0.8, 12), (0.9, 8), (0.9, 20)):
    report(f"envelope q={q} window=+-{w}d", S.SolarFC(df, env_q=q, env_win=w))
