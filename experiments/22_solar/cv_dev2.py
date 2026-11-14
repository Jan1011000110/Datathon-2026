# NOTE: historical development script. It ran against the FIRST version of solar_fc.py (pooled LightGBM with a
# per-horizon ridge fallback, 22 features) and does not run against the current file. Its printed results are
# quoted in the report (reports/12_solar_forecast). The current model is cross-validated by cv_final.py.
# Usage: .venv/Scripts/python.exe experiments/22_solar/cv_dev2.py
"""Blocked CV of linear clearness models by cell = (horizon, current solar hour), and LightGBM on their residuals."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, str(Path(__file__).parent))
from cv_dev import table, cv, HS
import solar_fc as S

def design(F, feats, kc):
    cols = [np.ones(len(F))]
    for f in feats:
        if "*" in f:
            a, b = f.split("*"); v = (F[a] * F[b]).to_numpy()
        else:
            v = F[f].to_numpy()
        miss = np.isnan(v)
        cols.append(np.where(miss, 0.0, v - kc if f.startswith("k_") and "*" not in f else v))
        if f in ("k_last", "k_today", "k_yday", "k_same"):
            cols.append(miss.astype(float))
    return np.column_stack(cols)

def cell_ridge(feats, lam=1.0, by_now=True, kc=0.74, ret_fit=False):
    def f(tr, te):
        Ztr, Zte = design(tr, feats, kc), design(te, feats, kc)
        ctr = (tr.h * 30 + (tr.hs_now + 3) * by_now).astype(int).to_numpy()
        cte = (te.h * 30 + (te.hs_now + 3) * by_now).astype(int).to_numpy()
        r = np.clip(tr.y / tr.E, 0, 1.6).to_numpy() - kc; w = ((tr.E / 1000) ** 2).to_numpy()
        out = np.full(len(te), kc)
        order = np.argsort(ctr, kind="stable"); cs = ctr[order]
        starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); ends = np.r_[starts[1:], len(cs)]
        coefs = {}
        for s, e in zip(starts, ends):
            idx = order[s:e]
            A = Ztr[idx] * np.sqrt(w[idx])[:, None]; b = r[idx] * np.sqrt(w[idx])
            P = lam * np.eye(Ztr.shape[1]) * w[idx].mean()
            coefs[cs[s]] = np.linalg.solve(A.T @ A + P, A.T @ b)
        for c in np.unique(cte):
            if c in coefs:
                m = cte == c
                out[m] = kc + Zte[m] @ coefs[c]
        return np.clip(out, 0, 1.3) * te.E.to_numpy()
    return f

if __name__ == "__main__":
    fc, F = table()
    kc = float(F.y.sum() / F.E.sum())
    base = ["k_last", "k_today", "k_yday", "k_same"]
    cv(F, cell_ridge(base, by_now=False, kc=kc), "ridge per h (4 feats)")
    for lam in (0.3, 1.0, 3.0, 10.0):
        cv(F, cell_ridge(base, lam=lam, kc=kc), f"ridge per (h, hour) lam={lam}")
    F["k_last2"] = (F.k_last - kc) ** 2
    F["kl_sd"] = (F.k_last - kc) * F.sd_today
    for name, extra in [("+k_prev", ["k_prev"]), ("+dk", ["dk"]), ("+sd_today", ["sd_today"]), ("+k_yday2", ["k_yday2"]),
                        ("+sd_yday", ["sd_yday"]), ("+k_last2", ["k_last2"]), ("+r_inv", ["r_inv"]), ("+dT", ["dT"]), ("+tamb", ["tamb"]),
                        ("+kl_sd", ["kl_sd"]),
                        ("+k_prev,sd_today,k_yday2,k_last2", ["k_prev", "sd_today", "k_yday2", "k_last2"])]:
        cv(F, cell_ridge(base + extra, lam=3.0, kc=kc), "ridge cell lam=3 " + name)
