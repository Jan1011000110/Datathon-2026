# Usage: .venv/Scripts/python.exe experiments/21_load/research2.py
"""Does a holiday-aware base (target = change minus the blended-reference change) help the
hour-0 model? Blocked CV on the official train table; MAE on normal days / first holiday day /
later holiday days, all hours 06-22 and the morning ramp 06-10."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import load_hard
from load_fc import LoadFC, FEATS, COLS_V2, COLS_V3

fc = LoadFC(load_hard(), use_model=False)
X, y, ii = fc._Xall, fc._y, fc._ii
hours = ii % 24
wk = (ii // 168) % 5
ts = pd.to_datetime(ii * 3600, unit="s")
mday = (ts - pd.Timedelta(hours=6)).normalize()
hol = pd.to_datetime(pd.read_csv(Path(__file__).with_name("holidays.csv"))["date"])
first = hol[~(hol - pd.Timedelta(days=1)).isin(hol) & ~((hol.dt.dayofweek == 0) & (hol - pd.Timedelta(days=3)).isin(hol))]
is_h = np.asarray(mday.isin(hol)); is_f = np.asarray(mday.isin(first))
P0 = dict(fc.gbm_params); P0["num_threads"] = 3
dsm = X[:, FEATS.index("dS_mix")]

def cv(cols, base=None, params=None, n=300):
    p = dict(P0); p.update(params or {})
    ci = [FEATS.index(c) for c in cols]
    yy = y - (base if base is not None else 0.0)
    res = np.zeros(len(y))
    for k in range(5):
        tr = wk != k
        m = lgb.train(p, lgb.Dataset(X[tr][:, ci], yy[tr]), num_boost_round=n)
        res[~tr] = yy[~tr] - m.predict(X[~tr][:, ci])
    return res

def show(name, res):
    a = np.abs(res); am = (hours <= 10)
    f = lambda m: f"{a[m].mean():6.1f}"
    print(f"{name:30s} all {f(slice(None))} | normal {f(~is_h)} first {f(is_f)} later {f(is_h & ~is_f)} | 06-10h: normal {f(~is_h & am)} first {f(is_f & am)} later {f(is_h & ~is_f & am)}"
          f" | bias hol {res[is_h].mean():6.1f}", flush=True)

print("n", len(y), "holiday rows", is_h.sum(), "first", is_f.sum())
show("persistence", y)
show("reference change dS", y - X[:, FEATS.index("dS")])
show("blended reference change", y - dsm)
show("v2", cv(COLS_V2))
show("v2 + mix feats", cv(COLS_V3))
show("v3: base=blend, v3 feats", cv(COLS_V3, base=dsm))
show("v3: base=blend, v2 feats", cv(COLS_V2, base=dsm))

# --- run-time blend towards the holiday-aware reference when the holiday index is high -------
print("\nblend: pred = gbm + g * (alt - gbm), g = clip(c * lam_n, 0, 1); alt = blended-reference change (level at 06:00)")
r2 = cv(COLS_V2)
pg = y - r2
lam = X[:, FEATS.index("lam_n")]
alt = np.where(hours == 6, X[:, FEATS.index("sg_mix")], dsm)
for c in (0.0, 0.5, 1.0, 1.5):
    for thr in (0.0, 0.3):
        g = np.clip(c * np.where(lam > thr, lam, 0.0), 0, 1)
        show(f"blend c={c} thr={thr}", y - (pg + g * (alt - pg)))
