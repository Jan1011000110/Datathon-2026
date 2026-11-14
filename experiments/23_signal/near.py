# Usage: .venv/Scripts/python.exe experiments/23_signal/near.py [strict]
"""Round 2, near horizons (+3..+11): a second-stage model that corrects the round-1 forecast of w with
everything else the agent knows at run time (slope of the known hours, the forecast's own recently
revealed errors, last hour's local weather, campus load and solar, hour of day).
Offline: MAE of w by horizon before/after, per fold. Closed loop: both scores per fold. Writes near.csv
and the corrected tapes under cache2/."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402
from common import KAPPA, signal_fit_table  # noqa: E402
from signal_fc import BACK, HOUR_NS, SignalForecaster  # noqa: E402
from src.hard import load_hard  # noqa: E402

KS = np.arange(3, 12)                     # horizons corrected
WX = ["irradiancia_wm2", "vent_ms", "temp_ambient_c", "load_kw", "pv_kw"]
GROUPS = {
    "cal": ["hour", "dow", "dt0", "doy_s", "doy_c", "k", "th", "dtk"],
    "base": ["fa_c", "fa_p", "fw", "clw", "fw_rel", "fmin_rel", "fmax_rel"],
    "known": ["ac0", "ac1", "ac2", "ap0", "ap1", "ap2", "sc1", "sc2", "sp1", "sp2", "m24c", "m24p"],
    "err": ["e3c0", "e3c1", "e3c2", "e3p0", "e3p1", "e3p2", "e6c", "e6p", "e3w_mean"],
    "wx": ["irr", "irr_ratio", "wind", "wind6", "temp", "load_an", "pv", "pv_ratio"],
}


def world(fc, tab):
    """Hourly arrays on the forecaster's calendar from the rows of `tab` (NaN elsewhere)."""
    pos = ((tab.index.values.astype("datetime64[ns]").astype("int64") - fc.t0_ns) // HOUR_NS).astype(int)
    W = {}
    for name, col in (("c", "co2_kg_per_kwh"), ("p", "precio_eur_per_kwh")):
        a = np.full(fc.n, np.nan); a[pos] = tab[col].to_numpy(float); W[name] = a
    for col in WX:
        a = np.full(fc.n, np.nan); a[pos] = tab[col].to_numpy(float); W[col] = a
    return W


def base_all(fc, S):
    """Round-1 forecasts (co2, price) for k in KS at every calendar origin with a complete window."""
    n = fc.n
    O = np.arange(BACK + 2, n - fc.kmax - 2)
    out = {}
    for name in "cp":
        o2 = "p" if name == "c" else "c"
        g, Pm = fc._feats(S[name], fc.CL[name], fc.CL[name], O, False, S[o2], fc.CL[o2])
        A, B, c0, _ = fc.coef[(name, False)]
        Y = c0[0][None, :] + g @ A[0].T + (Pm * B[0][None]).sum(2) + fc.CL[name][O[:, None] + fc.ks[None, :]]
        F = np.full((n, len(KS)), np.nan)
        F[O] = np.maximum(Y[:, :len(KS)], fc._floor(name))
        out[name] = F
    return out


def features(fc, S, F, O, clim_x, grid):
    """Long-format feature table: one row per (origin in O, k in KS). Everything is known at the origin."""
    O = np.asarray(O)
    hour = fc.hour_grid[O]
    d = {}
    ts = grid[O]
    d["hour"] = hour
    d["dow"] = ts.dayofweek.values
    d["dt0"] = fc.dt_grid[O]
    doy = ts.dayofyear.values
    d["doy_s"], d["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for nm in "cp":
        an = S[nm] - fc.CL[nm]
        for j in range(3):
            d[f"a{nm}{j}"] = an[O + j]
        d[f"s{nm}1"] = S[nm][O + 1] - S[nm][O]
        d[f"s{nm}2"] = S[nm][O + 2] - S[nm][O + 1]
        d[f"m24{nm}"] = np.nanmean(an[O[:, None] + np.arange(-21, 3)[None, :]], axis=1)
        for j in range(3):                      # revealed errors of the +3 forecast, newest first
            d[f"e3{nm}{j}"] = S[nm][O + 2 - j] - F[nm][O - 1 - j, 0]
        d[f"e6{nm}"] = S[nm][O + 2] - F[nm][O - 4, 3]
    d["e3w_mean"] = np.nanmean([d[f"e3c{j}"] + KAPPA * d[f"e3p{j}"] for j in range(3)], axis=0)
    cell = (grid.month.values - 1) * 24 + grid.hour.values
    d["irr"] = S["irradiancia_wm2"][O - 1]
    d["irr_ratio"] = d["irr"] / (clim_x["irr"][cell[O - 1]] + 30.0)
    d["wind"] = S["vent_ms"][O - 1]
    d["wind6"] = np.nanmean(S["vent_ms"][O[:, None] + np.arange(-6, 0)[None, :]], axis=1)
    d["temp"] = S["temp_ambient_c"][O - 1]
    d["load_an"] = S["load_kw"][O - 1] - clim_x["load"][fc.dt_grid[O - 1] * 24 + fc.hour_grid[O - 1]]
    d["pv"] = S["pv_kw"][O - 1]
    d["pv_ratio"] = d["pv"] / (clim_x["pv"][cell[O - 1]] + 30.0)
    wk = (S["c"][O[:, None] + np.arange(3)[None, :]] + KAPPA * S["p"][O[:, None] + np.arange(3)[None, :]]).mean(1)
    fw_all = F["c"][O] + KAPPA * F["p"][O]
    base = pd.DataFrame(d)
    rows = []
    for j, k in enumerate(KS):
        r = base.copy()
        r["k"] = k
        r["th"] = (hour + k) % 24
        r["dtk"] = fc.dt_grid[O + k]
        r["fa_c"] = F["c"][O, j] - fc.CL["c"][O + k]
        r["fa_p"] = F["p"][O, j] - fc.CL["p"][O + k]
        r["fw"] = fw_all[:, j]
        r["clw"] = fc.CL["c"][O + k] + KAPPA * fc.CL["p"][O + k]
        r["fw_rel"] = fw_all[:, j] / wk
        r["fmin_rel"] = fw_all.min(1) / wk
        r["fmax_rel"] = fw_all.max(1) / wk
        r["origin"] = O
        r["y"] = S["c"][O + k] + KAPPA * S["p"][O + k] - fw_all[:, j]        # residual of w
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def prepare(f, strict=False):
    df = fast.fold(f)
    fit = signal_fit_table(f)
    if strict:
        fit = fit[fit.index.day <= 14]
    fc = SignalForecaster(fit, KAPPA)
    grid = pd.date_range(pd.Timestamp(fc.t0_ns), periods=fc.n, freq="h")
    hol = np.array([(m, dd) in __import__("signal_fc").HOLIDAYS for m, dd in zip(grid.month, grid.day)])
    fc.dt_grid = ((grid.dayofweek.values >= 5) | hol).astype(int)
    full = load_hard("filled")
    fit_x = full[full.index.isin(fit.index)]
    Sf = world(fc, fit_x)                                   # what may be used for fitting
    cell = (grid.month.values - 1) * 24 + grid.hour.values
    cx = {}
    for key, col in (("irr", "irradiancia_wm2"), ("pv", "pv_kw")):
        ok = np.isfinite(Sf[col])
        cx[key] = np.bincount(cell[ok], weights=Sf[col][ok], minlength=288) / np.maximum(np.bincount(cell[ok], minlength=288), 1)
    ok = np.isfinite(Sf["load_kw"])
    c2 = fc.dt_grid * 24 + fc.hour_grid
    cx["load"] = np.bincount(c2[ok], weights=Sf["load_kw"][ok], minlength=48) / np.maximum(np.bincount(c2[ok], minlength=48), 1)
    Ff = base_all(fc, Sf)
    O = np.arange(BACK + 8, fc.n - fc.kmax - 2)
    O = O[np.isfinite(Ff["c"][O, 0]) & np.isfinite(Sf["c"][O + 2])]
    tr = features(fc, Sf, Ff, O, cx, grid)
    tr = tr[np.isfinite(tr.y)]
    # evaluation world: fit rows + the fold's own rows; base forecasts = the replayed tape
    Se = world(fc, full[full.index.isin(fit.index) | full.index.isin(df.index)])
    b = fast.base_tape(f, strict)
    pos = ((df.index.values.astype("datetime64[ns]").astype("int64") - fc.t0_ns) // HOUR_NS).astype(int)
    Fe = {nm: np.full((fc.n, len(KS)), np.nan) for nm in "cp"}
    Fe["c"][pos] = b["Cc"][:, 3:12]
    Fe["p"][pos] = b["Pp"][:, 3:12]
    te = features(fc, Se, Fe, pos, cx, grid)
    return tr, te, b, pos


def fit_predict(tr, te, groups, kind="gbm", seed=0):
    cols = [c for g in groups for c in GROUPS[g]]
    if kind == "gbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=150,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                              n_jobs=1, random_state=seed, verbose=-1)
        m.fit(tr[cols], tr.y)
        return m.predict(te[cols]), m
    from sklearn.linear_model import Ridge
    mu, sd = tr[cols].mean(), tr[cols].std() + 1e-9
    X = ((tr[cols] - mu) / sd).fillna(0.0)
    m = Ridge(alpha=300.0).fit(X, tr.y)
    return m.predict(((te[cols] - mu) / sd).fillna(0.0)), m


SETS = {
    "gbm all": (("cal", "base", "known", "err", "wx"), "gbm"),
    "gbm no weather": (("cal", "base", "known", "err"), "gbm"),
    "gbm no errors": (("cal", "base", "known", "wx"), "gbm"),
    "gbm cal+base only": (("cal", "base"), "gbm"),
    "ridge all": (("cal", "base", "known", "err", "wx"), "ridge"),
}


def one(f, strict=False):
    t0 = time.perf_counter()
    tr, te, b, pos = prepare(f, strict)
    F = fast.w_of(b["Cc"], b["Pp"])
    T = fast.w_of(b["Tc"], b["Tp"])
    rows, tapes = [], {}
    n = len(pos)
    for name, (groups, kind) in SETS.items():
        pred, _ = fit_predict(tr, te, groups, kind)
        corr = pred.reshape(len(KS), n).T                       # (n, 9)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        W = F.copy()
        W[:, 3:12] = np.maximum(F[:, 3:12] + corr, 1e-3)
        tapes[name] = W
        e0, e1 = np.abs(F - T)[:, 3:12], np.abs(W - T)[:, 3:12]
        rows.append({"fold": f, "model": name, "n_train": len(tr),
                     **{f"base_{k}": np.nanmean(e0[:, j]) * 1e3 for j, k in enumerate(KS)},
                     **{f"new_{k}": np.nanmean(e1[:, j]) * 1e3 for j, k in enumerate(KS)},
                     "base_3_5": np.nanmean(e0[:, :3]) * 1e3, "new_3_5": np.nanmean(e1[:, :3]) * 1e3,
                     "lvl_base": np.nanmean(np.abs((F - T)[:, 3:9].mean(1))) * 1e3,
                     "lvl_new": np.nanmean(np.abs((W - T)[:, 3:9].mean(1))) * 1e3,
                     "base_3_11": np.nanmean(e0) * 1e3, "new_3_11": np.nanmean(e1) * 1e3})
    tag = f"{f}{'_strict' if strict else ''}"
    np.savez_compressed(fast.CACHE / f"near_{tag}.npz", **{k.replace(" ", "_"): v for k, v in tapes.items()})
    print(f"fold {f}: prepared and fitted in {time.perf_counter() - t0:.1f} s", flush=True)
    return rows


def score(name, f, strict=False):
    tag = f"{f}{'_strict' if strict else ''}"
    W = np.load(fast.CACHE / f"near_{tag}.npz")[name.replace(" ", "_")]
    pn, cm = fast.both(W, f)
    return {"fold": f, "model": name, "pn": pn, "comb": cm}


if __name__ == "__main__":
    strict = "strict" in sys.argv
    folds = ["C"] if strict else list("ABC")
    rows = sum(Parallel(n_jobs=3)(delayed(one)(f, strict) for f in folds), [])
    off = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(off[["fold", "model", "n_train", "lvl_base", "lvl_new", "base_3_5", "new_3_5", "base_3_11", "new_3_11", "base_3", "new_3", "base_6", "new_6", "base_9", "new_9"]].round(2).to_string(index=False))
    sc = pd.DataFrame(Parallel(n_jobs=3)(delayed(score)(m, f, strict) for m in SETS for f in folds))
    out = off.merge(sc, on=["fold", "model"])
    out.to_csv(Path(__file__).with_name("near_strict.csv" if strict else "near.csv"), index=False)
    for col in ("pn", "comb"):
        piv = sc.pivot(index="model", columns="fold", values=col)
        piv["mean"] = piv.mean(axis=1)
        print(f"\n== {col} ==")
        print(piv.loc[list(SETS)].round(3).to_string())
