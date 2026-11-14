# Usage: .venv/Scripts/python.exe experiments/30_boundaries/q1.py
"""Question 1: what do h2 / h3 carry across week boundaries, against the perfect-foresight optimum of the whole run?
Writes cache/trace_<agent>_<fold>.pkl, q1_boundaries.csv (one row per week), q1_daypos.csv, q1_hours.csv,
q1_forecast.csv (signal and net-load forecast across the jump against within-week forecasts)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L
from lib import cb


def fold_job(fold):
    df = L.get_folds()[fold]
    otr = L.oracle_trace(fold, df)
    out = {"oracle": (L.masked(df, otr["grid_kw"])["score"], otr)}
    for kind in ("h2", "h3"):
        r, tr = L.run(L.agent(fold, df, kind), df)
        tr.to_pickle(L.CACHE / f"trace_{kind}_{fold}.pkl")
        out[kind] = (r["score"], tr)
        assert r["infractions"] == 0
    return fold, out


def main():
    res = dict(Parallel(n_jobs=3)(delayed(fold_job)(f) for f in L.FOLDS))
    brow, drow, hrow, frow = [], [], [], []
    for fold in L.FOLDS:
        df = L.get_folds()[fold]
        wk, hs, he = L.positions(df, fold)
        o_score, otr = res[fold]["oracle"]
        tp = L.tape(fold)
        w_true = (df["co2_kg_per_kwh"] + cb.KAPPA * df["precio_eur_per_kwh"]).to_numpy(float)
        net_true = (df["load_kw"] - df["pv_kw"]).to_numpy(float)
        weeks = pd.unique(wk)
        print(f"fold {fold}: optimum {o_score:.3f}  h2 {res[fold]['h2'][0]:.3f}  h3 {res[fold]['h3'][0]:.3f}  weeks {len(weeks)}")
        for kind in ("h2", "h3"):
            score, tr = res[fold][kind]
            reg = L.regret(df, tr, otr)
            for d in range(7):
                drow.append(dict(agent=kind, fold=fold, day=d + 1, regret=float(reg[hs // 24 == d].sum()), total=float(reg.sum())))
            for lab, m in (("first 6 h", hs < 6), ("first 12 h", hs < 12), ("first 24 h", hs < 24), ("last 24 h", he < 24),
                           ("last 12 h", he < 12), ("middle (days 2-6)", (hs >= 24) & (he >= 24))):
                hrow.append(dict(agent=kind, fold=fold, span=lab, regret=float(reg[m].sum()), hours=int(m.sum()), total=float(reg.sum())))
            for i, w in enumerate(weeks):
                ii = np.flatnonzero(wk == w)
                last = ii[-1]
                complete_end = he[last] == 0
                nxt = last + 1 if last + 1 < len(df) else None
                row = dict(agent=kind, fold=fold, week=w, n_rows=len(ii), ends_2300=bool(complete_end), is_last=nxt is None,
                           soc_end=float(tr["soc_kwh"].iloc[last]), soc_end_opt=float(otr["soc_kwh"].iloc[last]),
                           soc_m12=float(tr["soc_kwh"].iloc[max(last - 12, 0)]), soc_m12_opt=float(otr["soc_kwh"].iloc[max(last - 12, 0)]),
                           reg_day1=float(reg[ii][hs[ii] < 24].sum()), reg_day7=float(reg[ii][he[ii] < 24].sum()),
                           reg_week=float(reg[ii].sum()),
                           dis_last12=float(-np.minimum(tr["p_kw"].to_numpy()[ii][he[ii] < 12], 0).sum()),
                           dis_last12_opt=float(-np.minimum(otr["p_kw"].to_numpy()[ii][he[ii] < 12], 0).sum()),
                           chg_last12=float(np.maximum(tr["p_kw"].to_numpy()[ii][he[ii] < 12], 0).sum()),
                           chg_last12_opt=float(np.maximum(otr["p_kw"].to_numpy()[ii][he[ii] < 12], 0).sum()),
                           dis_first12=float(-np.minimum(tr["p_kw"].to_numpy()[ii][hs[ii] < 12], 0).sum()),
                           dis_first12_opt=float(-np.minimum(otr["p_kw"].to_numpy()[ii][hs[ii] < 12], 0).sum()),
                           chg_first12=float(np.maximum(tr["p_kw"].to_numpy()[ii][hs[ii] < 12], 0).sum()),
                           chg_first12_opt=float(np.maximum(otr["p_kw"].to_numpy()[ii][hs[ii] < 12], 0).sum()))
                if nxt is not None:
                    jj = np.flatnonzero(wk == weeks[i + 1])
                    row["reg_next_day1"] = float(reg[jj][hs[jj] < 24].sum())
                    row["soc_p12"] = float(tr["soc_kwh"].iloc[min(nxt + 11, len(df) - 1)])
                    row["soc_p12_opt"] = float(otr["soc_kwh"].iloc[min(nxt + 11, len(df) - 1)])
                brow.append(row)
        # forecast across the jump vs within the week: origins at 23:00 (last row of a day), horizons 1..24
        for key, truth, name in (("w_sig", w_true, "signal round 1"), ("w_sig2", w_true, "signal round 2"),
                                 ("net", net_true, "net load")):
            F = tp[key] if key != "net" else tp["load_fc"] - tp["pv_fc"]
            for t in np.flatnonzero((df.index.hour == 23)):
                if t + 25 > len(df):
                    continue
                kk = np.arange(1, 25)
                # rows t+1..t+24 must be one full calendar day (same block, or the first day of the next block)
                tgt = df.index[t + kk]
                if not ((tgt.hour == np.arange(24)).all() and len(set(tgt.date)) == 1):
                    continue
                cross = he[t] == 0
                if not cross and (df.index[t + 1] - df.index[t]) != pd.Timedelta(hours=1):
                    continue
                f, a = F[t, 1:25], truth[t + 1:t + 25]
                kk2 = kk >= 3 if key != "net" else kk >= 1                    # hours 1, 2 of the signal are known
                frow.append(dict(fold=fold, what=name, cross=bool(cross), ts=df.index[t],
                                 mean_fc=float(f[kk2].mean()), mean_true=float(a[kk2].mean()),
                                 bias=float((f - a)[kk2].mean()), mae=float(np.abs(f - a)[kk2].mean()),
                                 sd_fc=float(f[kk2].std()), sd_true=float(a[kk2].std()),
                                 corr=float(np.corrcoef(f[kk2], a[kk2])[0, 1]),
                                 # night (hours 0-6 of the target day) against evening (18-22): what decides the carry
                                 night_fc=float(f[2:7].mean()), night_true=float(a[2:7].mean()),
                                 eve_fc=float(f[18:23].mean()), eve_true=float(a[18:23].mean()),
                                 max_fc=float(f[kk2].max()), max_true=float(a[kk2].max())))
    b = pd.DataFrame(brow)
    b.to_csv(L.HERE / "q1_boundaries.csv", index=False)
    d = pd.DataFrame(drow)
    d.to_csv(L.HERE / "q1_daypos.csv", index=False)
    h = pd.DataFrame(hrow)
    h.to_csv(L.HERE / "q1_hours.csv", index=False)
    f = pd.DataFrame(frow)
    f.to_csv(L.HERE / "q1_forecast.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    print("\n--- regret by day of the week (points), share of the fold's regret")
    piv = d.pivot_table(index=["agent", "fold"], columns="day", values="regret")
    piv["total"] = piv.sum(axis=1)
    print(piv.round(3).to_string())
    print((piv.div(piv["total"], axis=0) * 100).round(0).to_string())
    print("\n--- regret by span")
    print(h.pivot_table(index=["agent", "fold"], columns="span", values="regret").round(3).to_string())
    print("\n--- boundaries (h2), weeks that are followed by another week")
    bb = b[(b.agent == "h2") & (~b.is_last)]
    print(bb[["fold", "week", "n_rows", "ends_2300", "soc_end", "soc_end_opt", "soc_m12", "soc_m12_opt", "dis_last12",
              "dis_last12_opt", "chg_last12", "chg_last12_opt", "reg_day7", "reg_next_day1", "soc_p12", "soc_p12_opt"]].round(0).to_string())
    for kind in ("h2", "h3"):
        bb = b[(b.agent == kind) & (~b.is_last)]
        g = bb.assign(diff=bb.soc_end - bb.soc_end_opt).groupby("fold").agg(
            n=("diff", "size"), soc_end=("soc_end", "mean"), soc_end_opt=("soc_end_opt", "mean"), diff_mean=("diff", "mean"),
            diff_median=("diff", "median"), lower_500=("diff", lambda x: int((x < -500).sum())),
            higher_500=("diff", lambda x: int((x > 500).sum())), reg_next_day1=("reg_next_day1", "sum"), reg_day7=("reg_day7", "sum"))
        print(f"\n--- arrival charge at the end of a week, {kind}")
        print(g.round(1).to_string())
        print("corr(arrival deficit, regret on the next first day):",
              bb.assign(diff=bb.soc_end - bb.soc_end_opt).groupby("fold").apply(lambda x: np.corrcoef(x["diff"], x["reg_next_day1"])[0, 1]).round(2).to_dict())
    print("\n--- forecasts made at 23:00 for the next day: across the jump against inside the week")
    g = f.groupby(["what", "fold", "cross"]).agg(n=("bias", "size"), mean_fc=("mean_fc", "mean"), mean_true=("mean_true", "mean"),
                                                 bias=("bias", "mean"), mae=("mae", "mean"), sd_fc=("sd_fc", "mean"), sd_true=("sd_true", "mean"),
                                                 corr=("corr", "mean"), night_fc=("night_fc", "mean"), night_true=("night_true", "mean"),
                                                 eve_fc=("eve_fc", "mean"), eve_true=("eve_true", "mean"), max_fc=("max_fc", "mean"),
                                                 max_true=("max_true", "mean"))
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
