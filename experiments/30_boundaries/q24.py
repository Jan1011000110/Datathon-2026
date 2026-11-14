# Usage: .venv/Scripts/python.exe experiments/30_boundaries/q24.py
"""Questions 2 and 4 from the saved runs of sweep_b.py (h2: sets diag, net, sig, value; h3: set hedge):
  - whole-run gain of every legal candidate, per fold; best candidate chosen on two folds and scored on the third
  - paired gain by boundary (rows of the last 36 h of a week and of the next week up to its last 36 h)
  - change of the public and private halves under each hypothesis about the cut, and of the arrival charge
Writes q2_nested.csv, q2_paired.csv, q4_halves.csv, q3_cuts.csv."""
from math import comb

import numpy as np
import pandas as pd

import lib as L
from q3 import HYP, halves

SETS = {"h2": ("diag", "net", "sig", "value"), "h3": ("hedge",)}
SHOW = {"h2": ["true signal + net beyond the jump", "day-profile net beyond the jump", "signal beyond the jump x0.95",
               "signal beyond the jump x1.06", "signal beyond the jump x1.15", "signal beyond the jump x1.2",
               "signal beyond the jump: 0.5 on the stored day before the block",
               "value at the week end: 0.5 on the train value of the month",
               "value at the week end: 1.0 on the train value of the month",
               "floor: the first 1500 kWh worth at least 1.0 x train value",
               "floor: the first 3000 kWh worth at least 1.0 x train value",
               "floor: the first 4500 kWh worth at least 1.2 x train value",
               "value at the week end: max(forecast, 1.0 x train value)"],
        "h3": ["true signal + net beyond the jump", "profile net beyond the jump", "signal beyond the jump x1.06",
               "signal beyond the jump x1.15", "signal error paths beyond the jump x0.0", "signal error paths beyond the jump x2.0",
               "signal error paths beyond the jump x3.0"]}


def load(kind):
    res, grid = [], {}
    for s in SETS[kind]:
        r = pd.read_csv(L.HERE / f"results_{s}_{kind}.csv")
        r["set"] = s
        res.append(r)
        z = np.load(L.CACHE / f"grid_{s}_{kind}.npz")
        for k in z.files:
            grid.setdefault(k, z[k])
    r = pd.concat(res).drop_duplicates(["config", "fold"])
    return r, grid


def boundary_ids(df, fold):
    """Row -> index of the boundary it is charged to (-1: before the first boundary's window)."""
    wk, hs, he = L.positions(df, fold)
    widx = pd.factorize(wk)[0]
    return np.where(he < 36, widx, widx - 1)


def main():
    folds = L.get_folds()
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)
    nested, paired, hrow, crow = [], [], [], []
    for kind in ("h2", "h3"):
        res, grid = load(kind)
        piv = res.pivot(index="config", columns="fold", values="score")
        gain = piv - piv.loc["base"]
        legal = [c for c in gain.index if "true " not in c and c != "base" and "check" not in c]
        # chosen on two folds, scored on the third
        for f in L.FOLDS:
            others = [x for x in L.FOLDS if x != f]
            best = gain.loc[legal, others].mean(axis=1).idxmax()
            nested.append(dict(agent=kind, test_fold=f, chosen=best, gain_on_two=float(gain.loc[best, others].mean()),
                               gain_on_third=float(gain.loc[best, f]), n_candidates=len(legal)))
        print(f"\n=== {kind}: {len(legal)} legal candidates; best mean gain {gain.loc[legal].mean(axis=1).max():+.4f} "
              f"({gain.loc[legal].mean(axis=1).idxmax()}); best on B and C {gain.loc[legal, ['B', 'C']].mean(axis=1).max():+.4f} "
              f"({gain.loc[legal, ['B', 'C']].mean(axis=1).idxmax()})")
        rng = np.random.default_rng(0)
        for c in SHOW[kind]:
            per = []
            for f in L.FOLDS:
                df = folds[f]
                w = L.score_weights(df) * 100.0
                g0, g1 = grid[f"base|{f}"], grid[f"{c}|{f}"]
                s0, s1 = grid[f"soc|base|{f}"], grid[f"soc|{c}|{f}"]
                d = w * (g0 - g1)
                b = boundary_ids(df, f)
                nb = b.max()                                             # last index = the last week's own tail, no boundary
                per += [dict(fold=f, b=i, gain=float(d[b == i].sum())) for i in range(nb)]
                rest = float(d[(b < 0) | (b == nb)].sum())
                _, hs, he = L.positions(df, f)
                ends = [e for e in np.flatnonzero(he == 0) if e + 1 < len(df)]
                hb, hx = halves(df, f, g0), halves(df, f, g1)
                for h in HYP:
                    hrow.append(dict(agent=kind, config=c, fold=f, hyp=h, d_whole=float(d.sum()), d_public=hx[h][0] - hb[h][0],
                                     d_private=hx[h][1] - hb[h][1], d_arrival_mwh=float(np.mean(s1[ends] - s0[ends])) / 1000.0,
                                     outside=rest))
            x = np.array([p["gain"] for p in per])
            n, pos, neg = len(x), int((x > 1e-9).sum()), int((x < -1e-9).sum())
            m = pos + neg
            k = max(pos, neg)
            p_sign = min(1.0, 2.0 * sum(comb(m, i) for i in range(k, m + 1)) / 2.0 ** m) if m else 1.0
            by = [np.array([p["gain"] for p in per if p["fold"] == f]) for f in L.FOLDS]
            boot = np.array([np.mean([rng.choice(v, len(v)).sum() for v in by]) for _ in range(5000)])
            paired.append(dict(agent=kind, config=c, boundaries=n, better=pos, worse=neg, unchanged=n - m, sign_p=p_sign,
                               gain_mean_of_folds=float(np.mean([v.sum() for v in by])), ci_lo=float(np.quantile(boot, 0.025)),
                               ci_hi=float(np.quantile(boot, 0.975)), sd_per_boundary=float(x.std()), worst=float(x.min()), best=float(x.max())))
    n_ = pd.DataFrame(nested)
    n_.to_csv(L.HERE / "q2_nested.csv", index=False)
    print("\n--- best legal candidate on two folds, scored on the third")
    print(n_.round(4).to_string())
    p_ = pd.DataFrame(paired)
    p_.to_csv(L.HERE / "q2_paired.csv", index=False)
    print("\n--- paired by boundary (30 boundaries)")
    print(p_.round(4).to_string())
    h_ = pd.DataFrame(hrow)
    h_.to_csv(L.HERE / "q4_halves.csv", index=False)
    print("\n--- Q4: change of the whole run and of each half (points, mean of folds A, B, C)")
    for kind in ("h2", "h3"):
        x = h_[h_.agent == kind]
        t = x.groupby(["config", "hyp"])[["d_public", "d_private"]].mean().unstack("hyp")
        t = t.reorder_levels([1, 0], axis=1)[[(h, v) for h in HYP for v in ("d_public", "d_private")]]
        t.insert(0, "arrival MWh", x.groupby("config")["d_arrival_mwh"].mean())
        t.insert(0, "whole", x.drop_duplicates(["config", "fold"]).groupby("config")["d_whole"].mean())
        print(kind)
        print(t.loc[SHOW[kind]].round(3).to_string())
    # state of charge at the cut instants: what each agent brings into the first 84 hours of a week
    for f in L.FOLDS:
        df = folds[f]
        wk, hs, he = L.positions(df, f)
        for kind in ("optimum", "h2", "h3"):
            tr = L.oracle_trace(f, df) if kind == "optimum" else pd.read_pickle(L.CACHE / f"trace_{kind}_{f}.pkl")
            soc = tr["soc_kwh"].to_numpy()
            ends = [e for e in np.flatnonzero(he == 0) if e + 1 < len(df)]
            mids = np.flatnonzero(hs == 83)
            mid_d = {d: soc[(hs % 24 == 23) & (hs // 24 == d)].mean() for d in range(7)}
            crow.append(dict(agent=kind, fold=f, soc_week_end=float(soc[ends].mean()), soc_h84=float(soc[mids].mean()),
                             net_into_first_half_mwh=float((soc[ends].mean() - soc[mids].mean()) / 1000.0),
                             soc_midnights_inside=float(np.mean([mid_d[d] for d in range(6)]))))
    c_ = pd.DataFrame(crow)
    c_.to_csv(L.HERE / "q3_cuts.csv", index=False)
    print("\n--- state of charge at the cut instants (kWh): end of a week (23:00), end of hour 84 (noon of day 4), other midnights")
    print(c_.round(1).to_string())


if __name__ == "__main__":
    main()
