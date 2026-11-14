# Usage: .venv/Scripts/python.exe experiments/30_boundaries/q1c.py
"""Question 1, detail: (1) aggregate of what h2 and the optimum do in the last 12 hours of a week and the first 12 of
the next; (2) three boundaries traced hour by hour (q1_trace_<fold>_<week>.csv); (3) the 84-hour cut seen from the
middle of the week: h2 forced to hold 1,000 kWh more at the end of hour 84 (noon of day 4)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L
import rules as R
from lib import cb
from q3 import HYP, halves

P = cb.P
TRACE = [("A", "2025-12"), ("C", "2025-10"), ("B", "2026-07")]


class ForceAt:
    """The wrapped agent plus `extra` kWh of charge at the row with `at_r` rows left in the week (85 = hour 84)."""
    def __init__(self, agent, extra, at_r):
        self.a, self.extra, self.at_r, self.cal = agent, extra, at_r, R.Calendar()

    def act(self, obs):
        p = self.a.act(obs)
        r, _ = self.cal.step(obs["timestamp"])
        if r == self.at_r:
            p = P.safe_clamp(p + self.extra / P.ETA, float(obs["soc_kwh"]))
        return p


def mid_job(fold, extra):
    df = L.get_folds()[fold]
    ag = L.agent(fold, df, "h2")
    if extra:
        ag = ForceAt(ag, extra, 85)
    r, tr = L.run(ag, df)
    return fold, extra, r["score"], tr["grid_kw"].to_numpy(), tr["soc_kwh"].to_numpy()


def main():
    pd.set_option("display.width", 250, "display.max_rows", 500)
    b = pd.read_csv(L.HERE / "q1_boundaries.csv")
    b = b[(b.agent == "h2") & (~b.is_last)]
    nxt = pd.read_csv(L.HERE / "q1_boundaries.csv")
    nxt = nxt[nxt.agent == "h2"]
    print("--- h2 against the optimum around a boundary, mean of 10 boundaries per fold (kWh)")
    rows = []
    for f in L.FOLDS:
        x = b[b.fold == f]
        y = nxt[nxt.fold == f].iloc[1:]                                  # the weeks after a boundary
        rows.append(dict(fold=f, discharged_last12_h2=x.dis_last12.mean(), discharged_last12_opt=x.dis_last12_opt.mean(),
                         charged_last12_h2=x.chg_last12.mean(), charged_last12_opt=x.chg_last12_opt.mean(),
                         soc_end_h2=x.soc_end.mean(), soc_end_opt=x.soc_end_opt.mean(),
                         discharged_first12_h2=y.dis_first12.mean(), discharged_first12_opt=y.dis_first12_opt.mean(),
                         charged_first12_h2=y.chg_first12.mean(), charged_first12_opt=y.chg_first12_opt.mean(),
                         soc_p12_h2=x.soc_p12.mean(), soc_p12_opt=x.soc_p12_opt.mean()))
    agg = pd.DataFrame(rows)
    agg.to_csv(L.HERE / "q1_around.csv", index=False)
    print(agg.round(0).T.to_string())

    folds = L.get_folds()
    for fold, week in TRACE:
        df = folds[fold]
        wk, hs, he = L.positions(df, fold)
        last = np.flatnonzero(wk == week)[-1]
        tr = pd.read_pickle(L.CACHE / f"trace_h2_{fold}.pkl")
        otr = L.oracle_trace(fold, df)
        tp = L.tape(fold)
        reg = L.regret(df, tr, otr)
        w = (df["co2_kg_per_kwh"] + cb.KAPPA * df["precio_eur_per_kwh"]).to_numpy(float)
        sl = slice(last - 11, last + 25)
        k = np.arange(sl.start, sl.stop) - last                          # hours from the last row of the week
        fc_w = np.where(k >= 1, tp["w_sig"][last, np.clip(k, 0, 35)], np.nan)
        fc_n = np.where(k >= 1, (tp["load_fc"] - tp["pv_fc"])[last, np.clip(k, 0, 35)], np.nan)
        out = pd.DataFrame({"k": k, "w": w[sl], "w_fc_at_23h": fc_w, "net": tr["net_kw"].to_numpy()[sl], "net_fc_at_23h": fc_n,
                            "h2_p": tr["p_kw"].to_numpy()[sl], "h2_soc": tr["soc_kwh"].to_numpy()[sl],
                            "opt_p": otr["p_kw"].to_numpy()[sl], "opt_soc": otr["soc_kwh"].to_numpy()[sl],
                            "regret": reg[sl], "cum_regret": np.cumsum(reg[sl])}, index=df.index[sl])
        out.to_csv(L.HERE / f"q1_trace_{fold}_{week}.csv")
        print(f"\n=== fold {fold}, week {week} -> next (weekday of the first day after: {df.index[last + 1].day_name()})")
        print(out.round({"w": 4, "w_fc_at_23h": 4, "net": 0, "net_fc_at_23h": 0, "h2_p": 0, "h2_soc": 0, "opt_p": 0, "opt_soc": 0,
                         "regret": 3, "cum_regret": 3}).to_string())

    res = Parallel(n_jobs=4)(delayed(mid_job)(f, x) for f in L.FOLDS for x in (0, 1000))
    base = {f: (s, g, soc) for f, x, s, g, soc in res if x == 0}
    rows = []
    for f, x, s, g, soc in res:
        if not x:
            continue
        df = folds[f]
        _, hs, he = L.positions(df, f)
        mids = np.flatnonzero(hs == 83)
        s0, g0, soc0 = base[f]
        hb, hx = halves(df, f, g0), halves(df, f, g)
        for h in HYP:
            rows.append(dict(fold=f, hyp=h, held_mwh=float(np.mean(soc[mids] - soc0[mids])) / 1000.0, d_whole=s - s0,
                             d_public=hx[h][0] - hb[h][0], d_private=hx[h][1] - hb[h][1]))
    m = pd.DataFrame(rows)
    m.to_csv(L.HERE / "q3_mid.csv", index=False)
    g = m.groupby("hyp")[["held_mwh", "d_whole", "d_public", "d_private"]].mean().loc[HYP]
    g["public_per_mwh"] = g.d_public / g.held_mwh
    g["private_per_mwh"] = g.d_private / g.held_mwh
    print("\n--- h2 forced to hold 1,000 kWh more at the end of hour 84 of every week (mean of folds)")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
