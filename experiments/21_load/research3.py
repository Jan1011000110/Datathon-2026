# Usage: .venv/Scripts/python.exe experiments/21_load/research3.py
"""Are direct quantile models sharper than 'point forecast + residual quantile by (hour, weekend)'?
Blocked CV on the official train table, pinball loss (kW) at q = 0.15, 0.25, 0.35, hours 06-22."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import load_hard
from load_fc import LoadFC, FEATS, COLS_V2, etype

fc = LoadFC(load_hard(), use_model=False)
X, y, ii = fc._Xall, fc._y, fc._ii
ci = [FEATS.index(c) for c in COLS_V2]
hours = ii % 24; we = np.array([etype(int(i)) > 0 for i in ii]); wk = (ii // 168) % 5
P0 = dict(fc.gbm_params); P0["num_threads"] = 3

def cvfit(params, n=300):
    out = np.zeros(len(y))
    for k in range(5):
        tr = wk != k
        m = lgb.train({**P0, **params}, lgb.Dataset(X[tr][:, ci], y[tr]), num_boost_round=n)
        out[~tr] = m.predict(X[~tr][:, ci])
    return out

def pinball(pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))

point = cvfit({})
res = y - point
for q in (0.15, 0.25, 0.35):
    off = np.zeros(len(y))
    for k in range(5):                       # offsets from the other blocks' out-of-fold residuals
        te = wk == k
        for h in range(6, 23):
            for w in (False, True):
                m = (hours == h) & (we == w)
                if (m & ~te).sum() > 10:
                    off[m & te] = np.quantile(res[m & ~te], q)
    a = pinball(point + off, q)
    b = pinball(cvfit(dict(objective="quantile", alpha=q)), q)
    cover_a = float(np.mean(y < point + off)); 
    print(f"q={q}: point+offset pinball {a:.2f} (coverage {cover_a:.3f}) | direct quantile GBM pinball {b:.2f} | gain {100 * (1 - b / a):.1f} %", flush=True)
