# Usage: .venv/Scripts/python.exe experiments/30_boundaries/q5.py
"""Follow-up (20:42): the family "signal beyond the week end x s" on h3, s = 1.2, 1 (h3), 0.75, 0.5, 0.25, 0, and the
plan cut at the week end, measured on the whole run, on the first 84 hours of each week (public) and on the last 84
(private). Needs sweep_b.py week h3. Writes q5_family.csv, q5_paired.csv, q5_weeks.csv, q5_ceiling.csv."""
from math import comb

import numpy as np
import pandas as pd

import lib as L
from src.hard import CAP, PMAX, SOC0
from src.sim import solve_lp

ORDER = ["signal beyond the week end x1.2 (the probe)", "base", "signal beyond the week end x0.75",
         "signal beyond the week end x0.5", "signal beyond the week end x0.25",
         "signal beyond the week end x0 (multiplied, floored at 1e-9)", "plan cut at the week end (each week its own run)"]
SHORT = {ORDER[0]: "x1.2 (probe)", "base": "x1.0 (h3)", ORDER[2]: "x0.75", ORDER[3]: "x0.5", ORDER[4]: "x0.25",
         ORDER[5]: "x0 (multiplied)", ORDER[6]: "plan cut at the week end"}


def mask_weights(df, m):
    """Per-hour weight of grid import in the score computed on the hours of mask m (points per kWh; 0 outside m)."""
    base = np.maximum(df["load_kw"].to_numpy(float) - df["pv_kw"].to_numpy(float), 0.0)
    co2, pr = df["co2_kg_per_kwh"].to_numpy(float), df["precio_eur_per_kwh"].to_numpy(float)
    w = 100.0 * (0.7 * co2 / np.sum((base * co2)[m]) + 0.3 * pr / np.sum((base * pr)[m]))
    return np.where(m, w, 0.0)


def main():
    folds = L.get_folds()
    res = pd.read_csv(L.HERE / "results_week_h3.csv")
    z = np.load(L.CACHE / "grid_week_h3.npz")
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60)
    rows, wrows = [], []
    info = {}
    for f in L.FOLDS:
        df = folds[f]
        wk, hs, he = L.positions(df, f)
        first = hs < 84
        ends = [e for e in np.flatnonzero(he == 0) if e + 1 < len(df)]
        mids = np.flatnonzero(hs == 83)
        info[f] = (df, wk, hs, he, first)
        ww = {"whole": mask_weights(df, np.ones(len(df), bool)), "first": mask_weights(df, first), "last": mask_weights(df, ~first)}
        g0 = z[f"base|{f}"]
        for c in ORDER:
            g, soc = z[f"{c}|{f}"], z[f"soc|{c}|{f}"]
            r = res[(res.config == c) & (res.fold == f)].iloc[0]
            rows.append(dict(config=SHORT[c], fold=f, whole=L.masked(df, g)["score"], first84=L.masked(df, g, first)["score"],
                             last84=L.masked(df, g, ~first)["score"], soc_week_end=float(soc[ends].mean()),
                             soc_h84=float(soc[mids].mean()), spill_mwh=float(r.spill_mwh), infr=int(r.infractions)))
            if c != "base":
                for half, w in ww.items():
                    d = w * (g0 - g)
                    s = pd.Series(d).groupby(wk).sum()
                    for k, v in s.items():
                        wrows.append(dict(config=SHORT[c], fold=f, week=k, half=half, gain=float(v)))
    fam = pd.DataFrame(rows)
    fam.to_csv(L.HERE / "q5_family.csv", index=False)
    order = [SHORT[c] for c in ORDER]
    print("--- per fold")
    for col in ("whole", "first84", "last84", "soc_week_end", "soc_h84", "spill_mwh"):
        t = fam.pivot(index="config", columns="fold", values=col).loc[order]
        t["mean"] = t.mean(axis=1)
        t["vs h3"] = t["mean"] - t.loc["x1.0 (h3)", "mean"]
        print(f"\n{col}")
        print(t.round(3 if col not in ("soc_week_end", "soc_h84") else 0).to_string())
    print("\ninfractions:", int(fam.infr.sum()))
    m = fam.groupby("config")[["whole", "first84", "last84", "soc_week_end", "soc_h84", "spill_mwh"]].mean().loc[order]
    print("\n--- mean of folds")
    print(m.round(3).to_string())

    wk_ = pd.DataFrame(wrows)
    wk_.to_csv(L.HERE / "q5_weeks.csv", index=False)
    rng = np.random.default_rng(0)
    prow = []
    for c in order:
        if c == "x1.0 (h3)":
            continue
        for half in ("last", "whole", "first"):
            x = wk_[(wk_.config == c) & (wk_.half == half)]
            v = x.gain.to_numpy()
            pos, neg = int((v > 1e-9).sum()), int((v < -1e-9).sum())
            n = pos + neg
            k = max(pos, neg)
            p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n) if n else 1.0
            by = [x[x.fold == f].gain.to_numpy() for f in L.FOLDS]
            boot = np.array([np.mean([rng.choice(b, len(b)).sum() for b in by]) for _ in range(10000)])
            prow.append(dict(config=c, half=half, weeks=len(v), better=pos, worse=neg, same=len(v) - n, sign_p=p,
                             gain_mean_of_folds=float(np.mean([b.sum() for b in by])), ci_lo=float(np.quantile(boot, 0.025)),
                             ci_hi=float(np.quantile(boot, 0.975)), worst_fold=float(min(b.sum() for b in by)),
                             best_fold=float(max(b.sum() for b in by)), worst_week=float(v.min()), best_week=float(v.max()),
                             **{f"fold_{f}": float(b.sum()) for f, b in zip(L.FOLDS, by)}))
    pr = pd.DataFrame(prow)
    pr.to_csv(L.HERE / "q5_paired.csv", index=False)
    print("\n--- paired by week against h3 (33 weeks; gains sum to the change of that score, mean of folds)")
    print(pr.round(3).to_string())

    # the rule in the first and the last week
    print("\n--- hours whose command differs from h3, plan cut at the week end")
    for f in L.FOLDS:
        df, wk, hs, he, first = info[f]
        weeks = pd.unique(wk)
        d = np.abs(z[f"{ORDER[6]}|{f}"] - z[f"base|{f}"]) > 1e-6
        ds = np.abs(z[f"soc|{ORDER[6]}|{f}"] - z[f"soc|base|{f}"]) > 1e-6
        w0, w1 = wk == weeks[0], wk == weeks[-1]
        print(f, "first week: differing hours", int((d | ds)[w0].sum()), "first differing hour of the week:",
              int(hs[w0][(d | ds)[w0]].min()) if (d | ds)[w0].any() else None,
              "| last week: differing hours", int((d | ds)[w1].sum()), "last differing hour of that week:",
              int(hs[w1][(d | ds)[w1]].max()) if (d | ds)[w1].any() else None, "of", int(w1.sum()))

    # ceiling: perfect foresight for the last-half score alone (first-half import uncounted, tiny tie-break weight)
    crow = []
    for f in L.FOLDS:
        df, wk, hs, he, first = info[f]
        net = (df["load_kw"] - df["pv_kw"]).to_numpy(float)
        wl = mask_weights(df, ~first)
        ww = mask_weights(df, np.ones(len(df), bool))
        for name, w in (("optimum of the last-half score", wl + 1e-4 * ww), ("optimum of the first-half score", mask_weights(df, first) + 1e-4 * ww)):
            c, d, s, g, _ = solve_lp(w, net, soc0=SOC0, cap=CAP, pmax=PMAX)
            ends = [e for e in np.flatnonzero(he == 0) if e + 1 < len(df)]
            crow.append(dict(agent=name, fold=f, whole=L.masked(df, g)["score"], first84=L.masked(df, g, first)["score"],
                             last84=L.masked(df, g, ~first)["score"], soc_week_end=float(s[ends].mean()),
                             soc_h84=float(s[hs == 83].mean())))
        otr = L.oracle_trace(f, df)
        g, s = otr["grid_kw"].to_numpy(), otr["soc_kwh"].to_numpy()
        ends = [e for e in np.flatnonzero(he == 0) if e + 1 < len(df)]
        crow.append(dict(agent="optimum of the whole run", fold=f, whole=L.masked(df, g)["score"], first84=L.masked(df, g, first)["score"],
                         last84=L.masked(df, g, ~first)["score"], soc_week_end=float(s[ends].mean()), soc_h84=float(s[hs == 83].mean())))
    ce = pd.DataFrame(crow)
    ce.to_csv(L.HERE / "q5_ceiling.csv", index=False)
    print("\n--- ceilings (perfect foresight)")
    for col in ("last84", "first84", "whole", "soc_week_end", "soc_h84"):
        t = ce.pivot(index="agent", columns="fold", values=col)
        t["mean"] = t.mean(axis=1)
        print(col)
        print(t.round(2 if col not in ("soc_week_end", "soc_h84") else 0).to_string())


if __name__ == "__main__":
    main()
