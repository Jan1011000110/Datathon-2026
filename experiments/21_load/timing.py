# Usage: .venv/Scripts/python.exe experiments/21_load/timing.py
"""Per-call cost of LoadFC fitted on the full load table, on the fold-B obs sequence, and a
round-trip check of to_state / from_state. ProfileN.predict_load is timed as a yardstick."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import METEO_COLS, get_folds, train_complement
from src.hard_mpc import ProfileN
from load_fc import LoadFC
from fit_table import load_fit_table

df = get_folds()["B"]
full = load_fit_table()
fit = full[~full.index.isin(df.index)]
t0 = time.perf_counter(); fc = LoadFC(fit); t_fit = time.perf_counter() - t0
t0 = time.perf_counter(); fc_fast = LoadFC(fit, oof=False); t_fit_fast = time.perf_counter() - t0
st = fc.to_state(); fc2 = LoadFC.from_state(st)
P = ProfileN(train_complement(df))
load = df["load_kw"].to_numpy(float); ts = df.index
met = {c: df[c].to_numpy(float) for c in METEO_COLS}
T = {"LoadFC.predict_load(H=36)": [], "LoadFC quantiles (9)": [], "ProfileN.predict_load(H=36)": []}
qs = np.arange(0.1, 0.91, 0.1); hist = []; maxdiff = 0.0
for t in range(len(df)):
    tl = max(t - 1, 0)
    last = {"ultim_consum_kw": float(load[tl]), "ultima_solar_kw": 0.0, **{c: float(met[c][tl]) for c in METEO_COLS}}
    obs = {"timestamp": ts[t], "noves_dades_5min": last}; hist.append({"ts": ts[t]})
    a = time.perf_counter(); p = fc.predict_load(obs, hist, 36); b = time.perf_counter()
    q = fc.predict_load_quantiles(obs, hist, qs); c = time.perf_counter()
    P.predict_load(obs, hist, 36); d = time.perf_counter()
    T["LoadFC.predict_load(H=36)"].append(b - a); T["LoadFC quantiles (9)"].append(c - b); T["ProfileN.predict_load(H=36)"].append(d - c)
    maxdiff = max(maxdiff, float(np.abs(fc2.predict_load(obs, hist, 36) - p).max()))
print(f"fit on {fit['load_kw'].notna().sum()} load hours: {t_fit:.1f} s with out-of-fold residuals, {t_fit_fast:.1f} s without")
for k, v in T.items():
    v = np.array(v) * 1000
    print(f"{k:30s} mean {v.mean():.3f} ms  median {np.median(v):.3f}  p95 {np.quantile(v, .95):.3f}  max {v.max():.2f}   total for 2016 calls {v.mean() * 2.016:.2f} s")
print("from_state round trip, max abs difference (kW):", maxdiff)
import pickle; print("state size (pickle, kB):", round(len(pickle.dumps(st)) / 1024))
