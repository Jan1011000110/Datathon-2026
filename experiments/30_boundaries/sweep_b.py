# Usage: .venv/Scripts/python.exe experiments/30_boundaries/sweep_b.py <set> [h2|h3]
"""Closed-loop scores of boundary variants on folds A, B, C (taped forecasts, whole-run score).
Sets: diag (truth beyond the jump: what is there to gain and from which input), net, sig, combo.
Writes results_<set>_<agent>.csv and cache/grid_<set>_<agent>.npz (hourly grid import and SOC of every run)."""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L
import rules as R
from lib import cb


def configs(which):
    out = [("base", {})]
    if which == "diag":
        out += [("true signal beyond the jump", dict(w="truth")),
                ("true net load beyond the jump", dict(net="truth")),
                ("true signal + net beyond the jump", dict(w="truth", net="truth")),
                ("true signal + net beyond the jump, horizon +24 h there", dict(w="truth", net="truth", extend=24)),
                ("true signal beyond the jump, day-profile net", dict(w="truth", net="profile")),
                ("day-profile net beyond the jump", dict(net="profile"))]
    if which == "days":
        for d in range(1, 8):
            out.append((f"true signal for the hours of day {d}", dict(tw=d)))
            out.append((f"true net load for the hours of day {d}", dict(tn=d)))
            out.append((f"true signal + net load for the hours of day {d}", dict(tw=d, tn=d)))
    if which == "net":
        for nl in (1, 2, 3, 5, 8):
            for npv in (3, 7, 14):
                out.append((f"profile net: load {nl} same-type days, solar {npv} days", dict(net="profile", n_load=nl, n_pv=npv)))
        for wd in (0.0, 0.5, 1.5, 2.0, 3.0):
            out.append((f"profile net (3, 7), scenario width beyond the jump x{wd}", dict(net="profile", width=wd)))
    if which == "sig":
        for sc in (0.9, 0.95, 1.03, 1.06, 1.1, 1.15, 1.2):
            out.append((f"signal beyond the jump x{sc}", dict(w="blend", a=0.0, scale=sc)))
            out.append((f"profile net + signal beyond the jump x{sc}", dict(net="profile", w="blend", a=0.0, scale=sc)))
        for a in (0.25, 0.5, 0.75, 1.0):
            out.append((f"signal beyond the jump: {a} on the stored day before the block", dict(w="blend", a=a)))
            out.append((f"profile net + signal: {a} on the stored day before the block", dict(net="profile", w="blend", a=a)))
    if which == "value":
        out.append(("split at the block end, same value (check)", dict(v="same")))
        for lam in (0.25, 0.5, 0.75, 1.0):
            out.append((f"value at the week end: {lam} on the train value of the month", dict(v="clim", lam=lam)))
        for g in (0.8, 1.2):
            out.append((f"value at the week end: 0.5 on the train value x{g}", dict(v="clim", lam=0.5, gain=g)))
        for fl in (1500.0, 3000.0, 4500.0):
            for g in (0.8, 1.0, 1.2):
                out.append((f"floor: the first {fl:.0f} kWh worth at least {g} x train value", dict(v="floor", floor_kwh=fl, gain=g)))
        for g in (0.8, 1.0):
            out.append((f"value at the week end: max(forecast, {g} x train value)", dict(v="max", gain=g)))
    if which == "hedge":
        for g in (0.0, 1.5, 2.0, 3.0):
            out.append((f"signal error paths beyond the jump x{g}", dict(gamma=g)))
    if which == "week":
        out += [("signal beyond the week end x1.2 (the probe)", dict(w="blend", a=0.0, scale=1.2)),
                ("signal beyond the week end x0.75", dict(w="blend", a=0.0, scale=0.75)),
                ("signal beyond the week end x0.5", dict(w="blend", a=0.0, scale=0.5)),
                ("signal beyond the week end x0.25", dict(w="blend", a=0.0, scale=0.25)),
                ("signal beyond the week end x0 (multiplied, floored at 1e-9)", dict(w="blend", a=0.0, scale=0.0)),
                ("plan cut at the week end (each week its own run)", dict(cut=True))]
    if which in ("short", "hedge"):
        out += [("profile net beyond the jump", dict(net="profile")),
                ("signal beyond the jump x1.06", dict(w="blend", a=0.0, scale=1.06)),
                ("signal beyond the jump x1.15", dict(w="blend", a=0.0, scale=1.15)),
                ("signal beyond the jump x0.95", dict(w="blend", a=0.0, scale=0.95)),
                ("signal beyond the jump: 0.5 on the stored day before the block", dict(w="blend", a=0.5)),
                ("true signal beyond the jump", dict(w="truth")),
                ("true signal + net beyond the jump", dict(w="truth", net="truth"))]
    return out


def build(fold, df, kind, cfg):
    ag = L.agent(fold, df, kind)
    cfg = dict(cfg)
    w, net = cfg.pop("w", None), cfg.pop("net", None)
    extend = cfg.pop("extend", 0)
    tw, tn = cfg.pop("tw", None), cfg.pop("tn", None)
    if cfg.pop("cut", False):
        import rules3 as R3
        if hasattr(ag.w_fc, "fc") and hasattr(ag.w_fc, "end"):       # h3: inside combined3.EndCut
            ag.w_fc.fc = R3.WeekCut(ag.w_fc.fc)
        else:
            ag.w_fc = R3.WeekCut(ag.w_fc)
        return ag
    if "v" in cfg:
        import rules2 as R2
        mode = cfg.pop("v")
        mv = None if mode == "same" else R2.MidnightValue(cb.fit_tables(df)[0])
        return R2.BoundaryAgent(ag, mv, mode=mode, **cfg)
    if "gamma" in cfg:
        import rules2 as R2
        ag.pool = R2.JumpPaths(ag.pool, cfg.pop("gamma"))
        ag.w_fc.fc = R2.NoteTs(ag.w_fc.fc, ag.pool)                 # inside combined3.EndCut
        return ag
    if tw or tn:
        _, hs, _ = L.positions(df, fold)
        wt = (df["co2_kg_per_kwh"] + cb.KAPPA * df["precio_eur_per_kwh"]).to_numpy(float)
        nt = (df["load_kw"] - df["pv_kw"]).to_numpy(float)
        if tw:
            ag.w_fc = R.TargetW(ag.w_fc, wt, hs // 24 == tw - 1)
        if tn:
            ag.net = R.TargetNet(ag.net, nt, hs // 24 == tn - 1)
        return ag
    wk = nk = None
    if w == "truth":
        wt = (df["co2_kg_per_kwh"] + cb.KAPPA * df["precio_eur_per_kwh"]).to_numpy(float)
        wk = dict(fn=R.truth_fn(wt), extend=extend)
    elif w == "blend":
        st = R.SignalStore(cb.fit_tables(df)[2])
        wk = dict(fn=st.blend_fn(a=cfg.pop("a", 0.0), scale=cfg.pop("scale", 1.0)))
    if net == "truth":
        nk = dict(fn=R.truth_fn((df["load_kw"] - df["pv_kw"]).to_numpy(float)), zero_width=True, extend=extend)
    elif net == "profile":
        prof = R.DayProfile(cb.fit_tables(df)[0], n_load=cfg.pop("n_load", 3), n_pv=cfg.pop("n_pv", 7))
        nk = dict(fn=prof.fn(), width=cfg.pop("width", 1.0))
    if extend:
        ag.H = ag.H + extend
    assert not cfg, cfg
    return R.install(ag, wk, nk)


def run(i, label, cfg, fold, kind):
    df = L.get_folds()[fold]
    t0 = time.perf_counter()
    ag = build(fold, df, kind, cfg)
    r, tr = L.run(ag, df)
    return dict(i=i, config=label, fold=fold, score=r["score"], infractions=int(r["infractions"]),
                spill_mwh=float(tr["spill_kw"].sum()) / 1000.0, seconds=time.perf_counter() - t0), \
        (f"{label}|{fold}", tr["grid_kw"].to_numpy(), tr["soc_kwh"].to_numpy())


def main(which, kind="h2"):
    cfgs = configs(which)
    res = Parallel(n_jobs=4)(delayed(run)(i, lab, c, f, kind) for i, (lab, c) in enumerate(cfgs) for f in L.FOLDS)
    out = pd.DataFrame([r for r, _ in res])
    out.to_csv(L.HERE / f"results_{which}_{kind}.csv", index=False)
    np.savez_compressed(L.CACHE / f"grid_{which}_{kind}.npz", **{k: g for _, (k, g, s) in res},
                        **{"soc|" + k: s for _, (k, g, s) in res})
    show(which, kind)


def show(which, kind="h2"):
    out = pd.read_csv(L.HERE / f"results_{which}_{kind}.csv")
    piv = out.pivot(index="config", columns="fold", values="score")
    order = out.drop_duplicates("config").sort_values("i")["config"]
    piv = piv.loc[order]
    piv["mean"] = piv.mean(axis=1)
    for f in L.FOLDS:
        piv["d" + f] = piv[f] - piv.loc["base", f]
    piv["gain"] = piv["mean"] - piv.loc["base", "mean"]
    piv["infr"] = out.groupby("config")["infractions"].sum()
    piv["spill"] = out.groupby("config")["spill_mwh"].mean()
    pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 500)
    print(f"=== {which} on {kind}")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "h2")
