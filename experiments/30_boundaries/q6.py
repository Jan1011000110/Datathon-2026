# Usage: .venv/Scripts/python.exe experiments/30_boundaries/q6.py
"""Follow-up (21:00): the week-boundary question in STANDARD mode (2,000 kWh, +-500 kW, 24 known hours).
Agents are loaded read-only from agents/v1 and agents/v2 and only wrapped here:

    WeekWindow(agent, last_day, scale)   the part of the 24-hour window that lies beyond the last hour of the week is
                                         cut off (scale 0: "each week is its own run") or multiplied by `scale`
    ForceCarry(agent, last_day, extra)   diagnostic: `extra` kWh of charge added in the last hour of every week

Folds A, B, C of src.sim (days 1-7 / 8-14 / 15-21), August dropped because the test run ends on 2026-07-31.
Tables: "filled" (every hour, 11 weeks, missing load / solar imputed) and "official" (rows with missing load or solar
dropped, no November). Writes q6_scores.csv, q6_weeks.csv, q6_paired.csv, q6_ceiling.csv."""
import importlib.util
from math import comb

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib as L
from q5 import mask_weights
from src import sim as S

ETA = S.ETA
DIM = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
VARIANTS = [("v1", None), ("v2", None), ("v2, beyond the week end x1.2", ("scale", 1.2)), ("v2, beyond the week end x0.75", ("scale", 0.75)),
            ("v2, beyond the week end x0.5", ("scale", 0.5)), ("v2, beyond the week end x0.25", ("scale", 0.25)),
            ("v2, window cut at the week end", ("scale", 0.0)), ("v1, window cut at the week end", ("scale", 0.0)),
            ("v2 + 200 kWh in the last hour of each week", ("carry", 200.0)), ("v2 + 400 kWh in the last hour of each week", ("carry", 400.0))]


def load_agent(kind):
    spec = importlib.util.spec_from_file_location(f"std_{kind}", L.ROOT / "agents" / kind / "agent.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.BatteryAgent()


def hours_left(ts, last_day=None):
    """Hours left in the week, this one included. last_day None = the test layout (the week ends with the month)."""
    if last_day is None:
        leap = ts.year % 4 == 0 and (ts.year % 100 != 0 or ts.year % 400 == 0)
        last_day = 29 if (ts.month == 2 and leap) else DIM[ts.month - 1]
    return (last_day - ts.day) * 24 + (24 - ts.hour)


class WeekWindow:
    def __init__(self, agent, last_day=None, scale=0.0):
        self.a, self.last_day, self.scale = agent, last_day, scale

    def act(self, obs):
        left = hours_left(obs["timestamp"], self.last_day)
        co2, pr = list(obs["co2_properes_24h"]), list(obs["preus_properes_24h"])
        if 1 <= left < len(co2):
            o = dict(obs)
            if self.scale == 0.0:
                o["co2_properes_24h"], o["preus_properes_24h"] = co2[:left], pr[:left]
            else:
                o["co2_properes_24h"] = co2[:left] + [c * self.scale for c in co2[left:]]
                o["preus_properes_24h"] = pr[:left] + [p * self.scale for p in pr[left:]]
            return self.a.act(o)
        return self.a.act(obs)


class ForceCarry:
    def __init__(self, agent, last_day, extra, end_ts):
        self.a, self.last_day, self.extra, self.end = agent, last_day, extra, end_ts

    def act(self, obs):
        p = self.a.act(obs)
        ts = obs["timestamp"]
        if hours_left(ts, self.last_day) == 1 and ts < self.end:
            soc = float(obs["soc_kwh"])
            p = min(p + self.extra / ETA, max(0.0, min(S.PMAX, (S.CAP - soc) / ETA - 1e-4)))
        return p


def fold_df(table, fold):
    df = S.get_folds(table)[fold]
    return df[df.index < "2026-08-01"]


def job(table, fold):
    df = fold_df(table, fold)
    last_day = L.cb.FOLD_DAYS[fold][1]
    out = {}
    for name, opt in VARIANTS:
        ag = load_agent(name[:2])
        if opt is not None and opt[0] == "scale":
            ag = WeekWindow(ag, last_day, opt[1])
        elif opt is not None:
            ag = ForceCarry(ag, last_day, opt[1], df.index[-1])
        r = S.simulate(ag, df, return_trace=True)
        out[name] = (r["trace"]["grid_kw"].to_numpy(), r["trace"]["soc_kwh"].to_numpy(), int(r["infractions"]))
    wk, hs, he = L.positions(df, fold)
    first = hs < 84
    net = (df["load_kw"] - df["pv_kw"]).to_numpy(float)
    ww = mask_weights(df, np.ones(len(df), bool))
    for name, w in (("optimum of the whole run", ww), ("optimum of the last-half score", mask_weights(df, ~first) + 1e-4 * ww),
                    ("optimum of the first-half score", mask_weights(df, first) + 1e-4 * ww)):
        c, d, s, g, _ = S.solve_lp(w, net, soc0=S.SOC0, cap=S.CAP, pmax=S.PMAX)
        out[name] = (g, s, 0)
    return table, fold, out


def main():
    res = Parallel(n_jobs=4)(delayed(job)(t, f) for t in ("filled", "official") for f in L.FOLDS)
    rows, wrows = [], []
    for table, fold, out in res:
        df = fold_df(table, fold)
        wk, hs, he = L.positions(df, fold)
        first = hs < 84
        ends = [e for e in np.flatnonzero(he == 0) if e + 1 < len(df)]
        mids = np.flatnonzero(hs == 83)
        wts = {"whole": mask_weights(df, np.ones(len(df), bool)), "first": mask_weights(df, first), "last": mask_weights(df, ~first)}
        g0 = out["v2"][0]
        for name, (g, soc, infr) in out.items():
            rows.append(dict(table=table, fold=fold, agent=name, whole=L.masked(df, g)["score"], first84=L.masked(df, g, first)["score"],
                             last84=L.masked(df, g, ~first)["score"], soc_week_end=float(soc[ends].mean()), soc_h84=float(soc[mids].mean()),
                             infr=infr, weeks=len(pd.unique(wk)), hours=len(df)))
            if name.startswith("v") and name != "v2":
                for half, w in wts.items():
                    s = pd.Series(w * (g0 - g)).groupby(wk).sum()
                    wrows += [dict(table=table, fold=fold, agent=name, week=k, half=half, gain=float(v)) for k, v in s.items()]
    sc = pd.DataFrame(rows)
    sc.to_csv(L.HERE / "q6_scores.csv", index=False)
    wk_ = pd.DataFrame(wrows)
    wk_.to_csv(L.HERE / "q6_weeks.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60)
    order = [n for n, _ in VARIANTS] + ["optimum of the whole run", "optimum of the last-half score", "optimum of the first-half score"]
    prow = []
    for table in ("filled", "official"):
        x = sc[sc.table == table]
        print(f"\n================ table {table}: hours per fold {x.drop_duplicates('fold').set_index('fold')['hours'].to_dict()}, "
              f"weeks {x.drop_duplicates('fold').set_index('fold')['weeks'].to_dict()}, infractions {int(x.infr.sum())}")
        m = x.groupby("agent")[["whole", "first84", "last84", "soc_week_end", "soc_h84"]].mean().loc[order]
        for c in ("whole", "first84", "last84", "soc_week_end"):
            m["d_" + c] = m[c] - m.loc["v2", c]
        print(m.round(3).to_string())
        for col in ("whole", "first84", "last84", "soc_week_end", "soc_h84"):
            t = x.pivot(index="agent", columns="fold", values=col).loc[order]
            print(f"\n{col} per fold")
            print(t.round(3 if "soc" not in col else 0).to_string())
        # sensitivity per MWh at the week end
        v = m.loc[[n for n, o in VARIANTS if o is not None and n.startswith("v2")]]
        xs = v.d_soc_week_end.to_numpy() / 1000.0
        print("\nper MWh of charge at the week end (each variant against v2):")
        print(pd.DataFrame({"d_MWh": xs, "first84_per_MWh": v.d_first84 / xs, "last84_per_MWh": v.d_last84 / xs,
                            "whole_per_MWh": v.d_whole / xs}).round(3).to_string())
        sl = {c: float(xs @ v["d_" + c].to_numpy() / (xs @ xs)) for c in ("first84", "last84", "whole")}
        print("least squares through the origin:", {k: round(a, 3) for k, a in sl.items()},
              "| kWh at every week end for +0.01 on the first half: %.0f" % (10.0 / sl["first84"]))
        rng = np.random.default_rng(0)
        for name in ("v2, window cut at the week end", "v2, beyond the week end x0.5", "v2, beyond the week end x0.75", "v1"):
            for half in ("last", "whole", "first"):
                y = wk_[(wk_.table == table) & (wk_.agent == name) & (wk_.half == half)]
                g = y.gain.to_numpy()
                pos, neg = int((g > 1e-9).sum()), int((g < -1e-9).sum())
                n = pos + neg
                k = max(pos, neg)
                p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n) if n else 1.0
                by = [y[y.fold == f].gain.to_numpy() for f in L.FOLDS]
                boot = np.array([np.mean([rng.choice(b, len(b)).sum() for b in by]) for _ in range(10000)])
                prow.append(dict(table=table, agent=name, half=half, weeks=len(g), better=pos, worse=neg, same=len(g) - n, sign_p=p,
                                 gain=float(np.mean([b.sum() for b in by])), ci_lo=float(np.quantile(boot, 0.025)),
                                 ci_hi=float(np.quantile(boot, 0.975)), worst_week=float(g.min()), best_week=float(g.max()),
                                 **{f"fold_{f}": float(b.sum()) for f, b in zip(L.FOLDS, by)}))
    pr = pd.DataFrame(prow)
    pr.to_csv(L.HERE / "q6_paired.csv", index=False)
    print("\n--- paired by week against v2")
    print(pr.round(3).to_string())
    ce = sc[sc.agent.str.startswith("optimum") | sc.agent.isin(["v2", "v2, window cut at the week end"])]
    ce.to_csv(L.HERE / "q6_ceiling.csv", index=False)


if __name__ == "__main__":
    main()
