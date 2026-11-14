"""
experiments/29_stress_hard/lowload_days.py
A low-load week (Christmas, late July): what is lost on its first day and on the later days.
For each fold (B, C), agent (h1, h2) and scenario, per day of the modified week:
    extra regret = regret on the modified run - regret of the same agent on the clean run, same day
    (regret = 100 * sum w * (grid_agent - grid_oracle), each run against its own perfect-foresight plan),
    battery energy exported for nothing, and the solar surplus the battery could not take.
    lowload_days.py   -> out/lowload_days.csv
"""
import numpy as np
import pandas as pd

import lib as L
import suite as S
from src.hard import oracle


def day_table(agent_fn, df, m):
    r = L.run(agent_fn(), df, trace=True)
    tr = r["trace"]
    otr = oracle(df, return_trace=True)["trace"]
    reg = 100 * tr["w"].to_numpy() * (tr["grid_kw"].to_numpy() - otr["grid_kw"].to_numpy())
    d = pd.DataFrame({"regret": reg, "spill": tr["spill_kw"].to_numpy(),
                      "unused": np.maximum(tr["net_kw"].to_numpy() + tr["p_kw"].to_numpy(), 0.0) * 0.0}, index=df.index)[m]
    g = d.groupby(d.index.date).sum()
    g["soc_end"] = tr["soc_kwh"][m].groupby(tr.index[m].date).last()
    return g, r["score"]


if __name__ == "__main__":
    folds = L.clean_folds()
    rows = []
    for fold in ("B", "C"):
        df = folds[fold]
        for mname, mth in (("Dec", 12), ("Jul", 7)):
            i = S.wk_of(df, mth)
            m = L.week_mask(df, [i])
            scen = {"load x0.6": L.scale_load(df, 0.6, m), "weekdays take the Sunday profile": L.sunday_week(df, [i])}
            for kind in ("h1", "h2"):
                fn = (lambda k=kind: L.load_agent(k, L.state_path(k, fold)))
                base, s0 = day_table(fn, df, m)
                for name, d in scen.items():
                    mod, s1 = day_table(fn, d, m)
                    x = (mod - base)
                    for k, (day, v) in enumerate(x.iterrows()):
                        rows.append(dict(fold=fold, week=mname, agent=kind, scenario=name, day=k + 1, date=str(day),
                                         dow=pd.Timestamp(day).dayofweek, extra_regret=v["regret"],
                                         spill_kwh=mod["spill"].iloc[k], spill_clean=base["spill"].iloc[k]))
                    print(fold, mname, kind, name, "extra regret by day:", np.round(x["regret"].to_numpy(), 3),
                          "spill by day:", np.round(mod["spill"].to_numpy()), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(L.HERE / "out" / "lowload_days.csv", index=False)
    d["part"] = np.where(d.day == 1, "day 1", np.where(d.day == 2, "day 2", "days 3-7 (sum)"))
    print(d.groupby(["agent", "scenario", "part"])[["extra_regret", "spill_kwh", "spill_clean"]].sum().div(4).round(3))
