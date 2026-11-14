# NOTE: historical development script. It ran against the FIRST version of solar_fc.py (pooled LightGBM with a
# per-horizon ridge fallback, 22 features) and does not run against the current file. Its printed results are
# quoted in the report (reports/12_solar_forecast). The current model is cross-validated by cv_final.py.
# Usage: .venv/Scripts/python.exe experiments/22_solar/cv_dev.py
"""Fast development loop: blocked cross-validation (7-day blocks, like the test weeks) on ALL train rows,
on the feature table produced by SolarFC._replay. Rows whose target day is in the held-out block are
removed from training, so a model cannot memorise a target day. Prints MAE by horizon for model variants."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import load_hard
import solar_fc as S

HS = [0, 1, 2, 3, 6, 12, 24]

def table():
    df = load_hard()
    fc = S.SolarFC(model="linear", fit_quantiles=False)
    df2 = df[df["pv_kw"].notna()].sort_index()
    keys = df2.index.as_unit("ns").asi8 // S.HNS
    pv = df2["pv_kw"].to_numpy(float)
    fc.pv_fit = dict(zip(keys.tolist(), pv.tolist()))
    fc.env_tab = fc._build_envelope(keys, pv); fc._dcache = {}; fc.c_irr = 1.65
    X, y, E, D = fc._replay(df2, 36)
    F = pd.DataFrame(X, columns=S.FEATS)
    F["y"], F["E"], F["D"] = y, E, D
    F["Dt"] = (D * 24 + F.hs_now + F.h + 2) // 24          # approx target day (offset 1 or 2 -> same day in daylight)
    return fc, F

def cv(F, fit_pred, name, nb=3):
    pred = np.full(len(F), np.nan)
    bd, bt = (F.D // 7) % nb, (F.Dt // 7) % nb
    for b in range(nb):
        tr = ((bd != b) & (bt != b)).to_numpy(); te = ((bd == b) & (bt == b)).to_numpy()
        pred[te] = fit_pred(F[tr], F[te])
    ok = ~np.isnan(pred)
    d = F[ok]; e = (pred[ok] - d.y)
    res = {"model": name}
    for h in HS:
        m = (d.h == h).to_numpy()
        res[f"h{h}"] = round(float(np.abs(e[m]).mean()), 1)
    res["all"] = round(float(np.abs(e).mean()), 1)
    res["rmse0"] = round(float(np.sqrt((e[(d.h == 0).to_numpy()] ** 2).mean())), 1)
    print(res, flush=True)
    return pred

def lgb_model(feats, params=None, n=300, target="ratio", aug=False):
    p = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0, num_threads=3)
    p.update(params or {})
    def f(tr, te):
        Xtr = tr[feats].to_numpy(); r = np.clip(tr.y / tr.E, 0, 1.6).to_numpy(); w = ((tr.E / 1000) ** 2).to_numpy()
        bst = lgb.train(p, lgb.Dataset(Xtr, label=r, weight=w), num_boost_round=n)
        return np.clip(bst.predict(te[feats].to_numpy()), 0, 1.3) * te.E.to_numpy()
    return f

def lin_model(fc):
    def f(tr, te):
        coef = fc._solve_linear(tr[S.FEATS].to_numpy(), np.clip(tr.y / tr.E, 0, 1.6).to_numpy(), ((tr.E / 1000) ** 2).to_numpy())
        return np.clip(fc._apply_linear(te[S.FEATS].to_numpy(), coef), 0, 1.3) * te.E.to_numpy()
    return f

if __name__ == "__main__":
    fc, F = table()
    F["Erel"] = (F.Et / F.emax).round(2)
    F["season"] = (F.emax / 150).round()
    F["k_ref"] = F.k_last.fillna(F.k_today).fillna(F.k_yday)
    print(len(F), "rows")
    kc = float(F.y.sum() / F.E.sum())
    cv(F, lambda tr, te: (te.E * te.k_last.fillna(te.k_today).fillna(te.k_yday).fillna(kc)).to_numpy(), "E * k_last (fallbacks)")
    cv(F, lambda tr, te: (te.E * kc).to_numpy(), "E * mean clearness")
    cv(F, lin_model(fc), "ridge per horizon")
    full = S.FEATS
    cv(F, lgb_model(full, dict(num_leaves=31, min_data_in_leaf=60, lambda_l2=1.0), n=400), "lgb all feats (v0)")
    safe = ["h", "hs_t", "hs_now", "same_day", "k_last", "k_prev", "dk", "k_today", "sd_today", "n_today",
            "k_yday", "sd_yday", "k_yday2", "k_same", "r_inv", "Erel"]
    cv(F, lgb_model(safe), "lgb safe feats")
    cv(F, lgb_model(safe + ["season"]), "lgb safe + season")
    cv(F, lgb_model(safe + ["dT"]), "lgb safe + dT")
    cv(F, lgb_model(safe + ["tamb", "dT"]), "lgb safe + tamb,dT")
    cv(F, lgb_model(safe, dict(num_leaves=7, min_data_in_leaf=400), n=300), "lgb safe small")
    cv(F, lgb_model(safe, dict(num_leaves=31, min_data_in_leaf=300), n=200), "lgb safe 31 leaves")
