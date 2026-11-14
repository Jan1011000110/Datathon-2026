"""
experiments/29_stress_hard/firstday.py
Hour by hour on the real low-load weekdays that open a fold week (1 Jan, 1 May in fold A; 8 Dec in fold B):
what the packaged agent did against the perfect-foresight plan.   firstday.py [pkg_dir_of_h2]
Writes out/firstday_<date>[_fix].csv and prints a digest.
"""
import sys

import numpy as np
import pandas as pd

import lib as L
from src.hard import oracle

DAYS = {"A": ["2026-01-01", "2026-05-01", "2026-04-03"], "B": ["2025-12-08", "2025-09-11"]}

if __name__ == "__main__":
    pkg = sys.argv[1] if len(sys.argv) > 1 else None
    tag = "" if pkg is None else "_fix"
    folds = L.clean_folds()
    pd.set_option("display.width", 250)
    for fold, days in DAYS.items():
        df = folds[fold]
        otr = oracle(df, return_trace=True)["trace"]
        tr = {}
        for kind in ("h1", "h2"):
            a = L.load_agent(kind, L.state_path(kind, fold), pkg=pkg if kind == "h2" else None)
            tr[kind] = L.run(a, df, trace=True)["trace"]
        for day in days:
            d0 = pd.Timestamp(day)
            m = (df.index >= d0) & (df.index < d0 + pd.Timedelta(hours=48))
            t = tr["h2"][m]
            out = pd.DataFrame({"load": t["load_kw"], "pv": t["pv_kw"], "net": t["net_kw"], "w_e6": t["w"] * 1e6,
                                "h2_p": t["p_kw"], "h2_soc": t["soc_kwh"], "h1_p": tr["h1"]["p_kw"][m], "h1_soc": tr["h1"]["soc_kwh"][m],
                                "or_p": otr["p_kw"][m], "or_soc": otr["soc_kwh"][m],
                                "h2_reg": 100 * t["w"] * (t["grid_kw"] - otr["grid_kw"][m]),
                                "h1_reg": 100 * tr["h1"]["w"][m] * (tr["h1"]["grid_kw"][m] - otr["grid_kw"][m]),
                                "h2_spill": t["spill_kw"]}).round(1)
            out[["h2_reg", "h1_reg"]] = (100 * np.c_[t["w"] * (t["grid_kw"] - otr["grid_kw"][m]),
                                                     tr["h1"]["w"][m] * (tr["h1"]["grid_kw"][m] - otr["grid_kw"][m])]).round(4)
            out.to_csv(L.HERE / "out" / f"firstday_{day}{tag}.csv")
            day1 = out[out.index < d0 + pd.Timedelta(hours=24)]
            print(f"\n=== {day} fold {fold}: regret day 1  h2 {day1.h2_reg.sum():+.3f}  h1 {day1.h1_reg.sum():+.3f};  "
                  f"48 h  h2 {out.h2_reg.sum():+.3f}  h1 {out.h1_reg.sum():+.3f}")
            print(day1[["load", "pv", "net", "w_e6", "h2_p", "h2_soc", "or_p", "or_soc", "h2_reg", "h1_p", "h1_reg"]].to_string())
