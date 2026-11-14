# Usage: .venv/Scripts/python.exe experiments/21_load/research.py
"""Feature-set / hyper-parameter comparison for the horizon-0 model by blocked CV (weeks mod 5)
on the whole official train table. Prints MAE over hours 06-22 and by hour."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import load_hard
from load_fc import LoadFC, FEATS, COLS_V1

fc = LoadFC(load_hard(), use_model=False)
X, y, ii = fc._Xall, fc._y, fc._ii
hours = ii % 24
wk = (ii // 168) % 5
print("rows", len(y))
P0 = dict(objective="l1", learning_rate=0.04, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3, seed=0)

def cv(cols, params=None, n=300):
    p = dict(P0); p.update(params or {})
    ci = [FEATS.index(c) for c in cols]
    res = np.zeros(len(y))
    for k in range(5):
        tr = wk != k
        m = lgb.train(p, lgb.Dataset(X[tr][:, ci], y[tr]), num_boost_round=n)
        res[~tr] = y[~tr] - m.predict(X[~tr][:, ci])
    return res

def show(name, res):
    byh = pd.Series(np.abs(res)).groupby(hours).mean().round(0).astype(int)
    print(f"{name:34s} MAE {np.abs(res).mean():6.2f} RMSE {np.sqrt((res**2).mean()):6.1f} | " + " ".join(f"{v:3d}" for v in byh.values), flush=True)

print(" " * 58 + " ".join(f"{h:3d}" for h in range(6, 23)))
show("reference-profile change (no model)", y - X[:, FEATS.index("dS")])
show("persistence", y)
R = {}
R["v1"] = cv(COLS_V1); show("v1", R["v1"])
gaps = COLS_V1 + ["yg", "wg", "sg"]
R["v1+gaps"] = cv(gaps); show("v1+gaps", R["v1+gaps"])
for extra in (["a3"], ["a6"], ["doy"], ["d48"], ["rS"], ["yS"], ["dtemp1", "dtemp24"], ["dS_next"]):
    show("gaps+" + ",".join(extra), cv(gaps + extra))
show("all", cv(FEATS))
show("all - temp,irr,dtemp", cv([c for c in FEATS if c not in ("temp", "irr", "dtemp1", "dtemp24")]))
show("all - doy", cv([c for c in FEATS if c != "doy"]))
for nm, pp, n in (("all l2", dict(objective="l2"), 300), ("all huber", dict(objective="huber", alpha=60.0), 300),
                  ("all lr.02 n800", dict(learning_rate=0.02), 800), ("all leaves31", dict(num_leaves=31), 300),
                  ("all leaves7 n600", dict(num_leaves=7), 600), ("all mdl10", dict(min_data_in_leaf=10), 300),
                  ("all ff0.5 n500", dict(feature_fraction=0.5), 500)):
    show(nm, cv(FEATS, pp, n))
