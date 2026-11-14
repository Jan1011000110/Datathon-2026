"""
experiments/29_stress_hard/holiday.py
Real low-load days in the train folds: what each packaged agent loses on them, day by day.

Per calendar day of a fold run:  regret = 100 * sum_h w_h * (grid_agent - grid_oracle)   (score points; the
oracle is the perfect-foresight optimum of the whole run, so energy carried over midnight blurs single days a
little), battery energy exported for nothing (kWh), mean load 08-20 h and its ratio to the same weekday type.
    holiday.py [pkg_h1_dir] [pkg_h2_dir]      -> out/holiday_days.csv, out/holiday_summary.csv
"""
import sys

import numpy as np
import pandas as pd

import lib as L
from src.hard import oracle

# weekdays the load report found low (reports/11_load_forecast/report.md, section 5)
LOW = ["2025-09-11", "2025-12-08", "2025-12-24", "2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06",
       "2026-02-12", "2026-04-02", "2026-04-03", "2026-04-06", "2026-05-01"]
LOW = set(pd.to_datetime(LOW).date)


def per_day(tr, otr):
    w = tr["w"].to_numpy()
    d = pd.DataFrame({"regret": 100 * w * (tr["grid_kw"].to_numpy() - otr["grid_kw"].to_numpy()),
                      "spill_kwh": tr["spill_kw"].to_numpy(), "load": tr["load_kw"].to_numpy(),
                      "hour": tr.index.hour}, index=tr.index)
    g = d.groupby(d.index.date)
    out = pd.DataFrame({"regret": g["regret"].sum(), "spill_kwh": g["spill_kwh"].sum(),
                        "load_day": d[(d.hour >= 8) & (d.hour < 20)].groupby(d[(d.hour >= 8) & (d.hour < 20)].index.date)["load"].mean(),
                        "hours": g["regret"].size()})
    # the same split for the first working hours of the day (06-11) and the rest
    am = d[(d.hour >= 6) & (d.hour < 12)]
    out["regret_06_11"] = am.groupby(am.index.date)["regret"].sum()
    out["spill_06_11"] = am.groupby(am.index.date)["spill_kwh"].sum()
    return out


def main(pkgs=None, tag=""):
    folds = L.clean_folds()
    rows = []
    for fold, df in folds.items():
        otr = oracle(df, return_trace=True)["trace"]
        for kind in ("h1", "h2"):
            pkg = None if pkgs is None else pkgs.get(kind)
            a = L.load_agent(kind, L.state_path(kind, fold), pkg=pkg)
            r = L.run(a, df, trace=True)
            pdx = per_day(r["trace"], otr)
            pdx["agent"], pdx["fold"], pdx["score"] = kind, fold, r["score"]
            pdx["date"] = pdx.index
            rows.append(pdx)
            print(kind, fold, "score %.3f infr %d fb %s" % (r["score"], r["infractions"], r["fallbacks"]), flush=True)
    d = pd.concat(rows, ignore_index=True)
    dt = pd.to_datetime(d["date"])
    d["dow"] = dt.dt.dayofweek
    d["first_day"] = dt.dt.day.isin([1, 8, 15])
    d["kind"] = np.where(d["date"].isin(LOW), "low weekday", np.where(d["dow"] >= 5, "weekend", "normal weekday"))
    d.to_csv(L.HERE / "out" / f"holiday_days{tag}.csv", index=False)
    s = d.groupby(["agent", "kind"]).agg(days=("regret", "size"), regret=("regret", "mean"), regret_06_11=("regret_06_11", "mean"),
                                         spill_kwh=("spill_kwh", "mean"), load_day=("load_day", "mean")).round(4)
    print(s)
    s.to_csv(L.HERE / "out" / f"holiday_summary{tag}.csv")
    print(d[d.kind == "low weekday"].pivot_table(index="date", columns="agent", values=["regret", "regret_06_11", "spill_kwh", "load_day"]).round(3))
    print(d.groupby(["agent", "fold"])["score"].first().unstack())
    return d


if __name__ == "__main__":
    pk = None
    if len(sys.argv) > 2:
        pk = {"h1": sys.argv[1], "h2": sys.argv[2]}
    main(pk, tag="" if pk is None else "_fix")
