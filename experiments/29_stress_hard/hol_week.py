"""
experiments/29_stress_hard/hol_week.py
Would a calendar prior for the holiday index (hol_patch.py) help h2 in a closed week (Christmas)?
The December week of folds B and C is turned into a closed week (weekdays take the Sunday profile, or load x0.6)
and its weekdays are given to the patched agent as holiday dates. Also the cost of a WRONG prior: the same dates
flagged while the week is in fact normal.
    hol_week.py    -> out/hol_week.csv
"""
import pandas as pd

import lib as L
import suite as S
from hol_patch import write_patch

if __name__ == "__main__":
    folds = L.clean_folds()
    rows = []
    for fold in ("B", "C"):
        df = folds[fold]
        i = S.wk_of(df, 12)
        m = L.week_mask(df, [i])
        days = sorted({str(d.date()) for d in df.index[m] if d.dayofweek < 5})
        pkg = write_patch(f"h2_hol_week_{fold}", days, 1.0)
        cases = {"clean (the prior is wrong)": df, "Dec week: weekdays take the Sunday profile": L.sunday_week(df, [i]),
                 "Dec week load x0.6": L.scale_load(df, 0.6, m)}
        for name, d in cases.items():
            r0 = L.run(L.load_agent("h2", L.state_path("h2", fold)), d)
            r1 = L.run(L.load_agent("h2", L.state_path("h2", fold), pkg=pkg), d)
            rows.append(dict(fold=fold, case=name, packaged=r0["score"], with_prior=r1["score"], gain=r1["score"] - r0["score"],
                             spill_packaged=r0["spill_kwh"], spill_prior=r1["spill_kwh"], infr=r1["infractions"],
                             fallbacks=r1["fallbacks"]))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(L.HERE / "out" / "hol_week.csv", index=False)
