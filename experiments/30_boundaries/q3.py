# Usage: .venv/Scripts/python.exe experiments/30_boundaries/q3.py
"""Question 3: how much does the cut of the public half matter?
1. Score of h2, h3 and the perfect-foresight optimum on the public half and on the private half under each
   hypothesis about the cut (one run, metrics restricted to the masked hours, as evaluate_results(mask=...)).
2. Sensitivity: the same agent forced to carry X more kWh across every week boundary (extra charge in the last
   hour of each week, on top of its own command): change of the whole-run, public and private scores.
Writes q3_halves.csv, q3_carry.csv."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L
import rules as R
from lib import cb

P = cb.P
SEEDS = 200
HYP = ["first 84 h", "last 84 h", "alt days (1,3,5,7)", "alt days (2,4,6)", "alt weeks (Sep, Nov, ...)", "alt weeks (Oct, Dec, ...)", "random hours"]


class ForceCarry:
    """The wrapped agent, plus `extra` kWh of charge spread over the last `hours` rows of every week (not the last week
    of the run when `end_ts` is given). Diagnostic of the split, not a rule."""
    def __init__(self, agent, extra, hours=1, end_ts=None):
        self.a, self.extra, self.hours, self.cal, self.end = agent, extra, hours, R.Calendar(), end_ts

    def act(self, obs):
        p = self.a.act(obs)
        ts = obs["timestamp"]
        r, _ = self.cal.step(ts)
        if r <= self.hours and (self.end is None or ts + pd.Timedelta(hours=r) <= self.end):
            p = P.safe_clamp(p + self.extra / self.hours / P.ETA, float(obs["soc_kwh"]))
        return p


def halves(df, fold, grid, mk=None):
    """{hypothesis: (public score, private score)}; random hours = mean over the seeds."""
    mk = mk or L.masks(df, fold, seeds=SEEDS)
    out = {}
    for h in HYP:
        keys = [k for k in mk if k.startswith(h)]
        pub = np.mean([L.masked(df, grid, mk[k])["score"] for k in keys])
        prv = np.mean([L.masked(df, grid, ~mk[k])["score"] for k in keys])
        out[h] = (float(pub), float(prv))
    return out


def carry_job(fold, kind, extra, hours):
    df = L.get_folds()[fold]
    ag = L.agent(fold, df, kind)
    if extra:
        ag = ForceCarry(ag, extra, hours, L.cal_end(fold) - pd.Timedelta(days=1))
    r, tr = L.run(ag, df)
    return fold, kind, extra, r["score"], int(r["infractions"]), tr["grid_kw"].to_numpy(), tr["soc_kwh"].to_numpy()


def main():
    folds = L.get_folds()
    rows, noise, parts = [], [], []
    for fold in L.FOLDS:
        df = folds[fold]
        mk = L.masks(df, fold, seeds=SEEDS)
        for kind in ("optimum", "h2", "h3"):
            tr = L.oracle_trace(fold, df) if kind == "optimum" else pd.read_pickle(L.CACHE / f"trace_{kind}_{fold}.pkl")
            whole = L.masked(df, tr["grid_kw"])["score"]
            rs = [L.masked(df, tr["grid_kw"], mk[k])["score"] for k in mk if k.startswith("random")]
            noise.append(dict(agent=kind, fold=fold, sd_public_random=float(np.std(rs)), mean_public_random=float(np.mean(rs)),
                              whole=whole))
            for k in ("first 84 h", "last 84 h"):
                r_ = L.masked(df, tr["grid_kw"], mk[k])
                parts.append(dict(agent=kind, fold=fold, hyp=k, co2_pct=r_["co2_pct"], cost_pct=r_["cost_pct"],
                                  ratio=r_["cost_pct"] / r_["co2_pct"], base_kg_per_eur=r_["co2_base_kg"] / r_["cost_base_eur"]))
            r_ = L.masked(df, tr["grid_kw"])
            parts.append(dict(agent=kind, fold=fold, hyp="whole run", co2_pct=r_["co2_pct"], cost_pct=r_["cost_pct"],
                              ratio=r_["cost_pct"] / r_["co2_pct"], base_kg_per_eur=r_["co2_base_kg"] / r_["cost_base_eur"]))
            for h, (pub, prv) in halves(df, fold, tr["grid_kw"].to_numpy(), mk).items():
                rows.append(dict(agent=kind, fold=fold, hyp=h, whole=whole, public=pub, private=prv))
    hv = pd.DataFrame(rows)
    hv.to_csv(L.HERE / "q3_halves.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print("--- public / private half scores by hypothesis (per fold)")
    for kind in ("optimum", "h2", "h3"):
        d = hv[hv.agent == kind]
        pub = d.pivot(index="hyp", columns="fold", values="public").loc[HYP]
        prv = d.pivot(index="hyp", columns="fold", values="private").loc[HYP]
        t = pd.concat({"public": pub, "private": prv}, axis=1)
        t[("mean", "public")] = pub.mean(axis=1)
        t[("mean", "private")] = prv.mean(axis=1)
        t[("mean", "pub-priv")] = t[("mean", "public")] - t[("mean", "private")]
        whole = d.drop_duplicates("fold").set_index("fold")["whole"]
        t[("mean", "pub/whole")] = (pub / whole).mean(axis=1)
        print(f"\n{kind}: whole run {whole.round(3).to_dict()} mean {whole.mean():.3f}")
        print(t.round(3).to_string())

    print("--- random half of the hours: spread of the public score over the seeds")
    print(pd.DataFrame(noise).round(3).to_string())
    print("--- CO2 and cost parts (public score on the board: CO2 21.80, cost 26.00, ratio 1.193, 0.634 kg per EUR)")
    print(pd.DataFrame(parts).round(3).to_string())
    pd.DataFrame(parts).to_csv(L.HERE / "q3_parts.csv", index=False)
    # sensitivity to energy carried across the boundaries
    jobs = [(f, k, x, 1 if abs(x) <= 1000 else 2) for f in L.FOLDS for k in ("h2",) for x in (0, 500, 1000, 2000)]
    res = Parallel(n_jobs=4)(delayed(carry_job)(*j) for j in jobs)
    base = {(f, k): (s, g, soc) for f, k, x, s, inf, g, soc in res if x == 0}
    crow = []
    for fold, kind, extra, score, inf, grid, soc in res:
        if extra == 0:
            continue
        df = folds[fold]
        wk, hs, he = L.positions(df, fold)
        s0, g0, soc0 = base[(fold, kind)]
        ends = np.flatnonzero(he == 0)[:-1] if (he == 0).any() else []
        ends = [e for e in ends if e + 1 < len(df)]
        carried = float(np.mean(soc[ends] - soc0[ends])) / 1000.0          # MWh actually added at a week end, mean
        hb = halves(df, fold, g0)
        hx = halves(df, fold, grid)
        for h in HYP:
            crow.append(dict(agent=kind, fold=fold, extra=extra, infr=inf, carried_mwh=carried, d_whole=score - s0, hyp=h,
                             d_public=hx[h][0] - hb[h][0], d_private=hx[h][1] - hb[h][1]))
    c = pd.DataFrame(crow)
    c.to_csv(L.HERE / "q3_carry.csv", index=False)
    print("\n--- forced extra carry across every boundary: change of the scores (points), mean of folds")
    g = c.groupby(["extra", "hyp"]).agg(carried_mwh=("carried_mwh", "mean"), d_whole=("d_whole", "mean"), d_public=("d_public", "mean"),
                                        d_private=("d_private", "mean"), infr=("infr", "sum")).reset_index()
    g["public_per_mwh"] = g.d_public / g.carried_mwh
    g["private_per_mwh"] = g.d_private / g.carried_mwh
    g["whole_per_mwh"] = g.d_whole / g.carried_mwh
    for x in sorted(g.extra.unique()):
        print(g[g.extra == x].set_index("hyp").loc[HYP].round(3).to_string())
    print("\nper fold, +1000 kWh:")
    print(c[c.extra == 1000].pivot_table(index="hyp", columns="fold", values=["d_public", "d_private"]).loc[HYP].round(3).to_string())
    print(c[c.extra == 1000].drop_duplicates("fold")[["fold", "carried_mwh", "d_whole", "infr"]].round(3).to_string())


if __name__ == "__main__":
    main()
