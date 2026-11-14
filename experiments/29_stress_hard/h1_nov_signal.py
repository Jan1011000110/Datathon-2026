"""
experiments/29_stress_hard/h1_nov_signal.py
Candidate fix for h1 (and for the h1 embedded in h2) in the November week.

h1's signal profile w_tab[month, weekend, hour] is fitted on the official table (5,993 h), which has no November
rows because the METER has no November data. But the train set does have November prices and CO2 (the filled
table, 6,768 h, also has 8 more August days). Fitting w_tab / w_any on every train hour that has a price gives
h1 a real November signal profile instead of one interpolated from October and December.

Test: (1) clean folds A, B, C, tables fitted outside the fold; (2) the November case replayed on a month that
does exist: month M removed from the load / solar fit rows (as November is) and
    a) also from the signal rows, filled from the neighbours     = what the shipped nov_fill tables do
    b) signal fitted with month M's real prices                  = the proposal
    h1_nov_signal.py     -> out/h1_nov_signal.csv
"""
import json

import pandas as pd

import lib as L
from hidden_month import _load, week_score
from src.hard import load_hard, train_complement

if __name__ == "__main__":
    h1b = _load("h1_build_ns", L.ROOT / "agents" / "h1" / "build.py")
    folds = L.clean_folds()
    filled = load_hard("filled")
    tmp = L.HERE / "states"
    rows = []

    def tables(fit_off, fit_sig, path):
        t = h1b.fit_tables(fit_off, fill_missing="neighbours")
        if fit_sig is not None:
            p = h1b.fit_profiles(fit_sig, fill_missing="neighbours")
            t["w_tab"], t["w_any"] = p["w_tab"].tolist(), p["w_any"].tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(t, f)
        return path

    def run(path, df, mth=None):
        r = L.run(L.load_agent("h1", path), df, trace=True)
        tr = r["trace"]
        tr["price"], tr["co2"] = df["precio_eur_per_kwh"].to_numpy(), df["co2_kg_per_kwh"].to_numpy()
        return r["score"], (week_score(tr, df.index.month == mth) if mth else float("nan")), r["infractions"]

    for fold, df in folds.items():
        lo, hi = int(df.index.day.min()), int(df.index.day.max())
        off = train_complement(df)
        sig = filled[~((filled.index.day >= lo) & (filled.index.day <= hi))]
        s0, _, i0 = run(tables(off, None, tmp / f"h1_{fold}_sigofficial.json"), df)
        s1, _, i1 = run(tables(off, sig, tmp / f"h1_{fold}_sigfilled.json"), df)
        rows.append(dict(fold=fold, case="clean", month=0, shipped=s0, proposal=s1, gain=s1 - s0, infr=i0 + i1))
        print(rows[-1], flush=True)
        if fold == "A":
            continue
        for mth in (10, 12, 3, 6):
            offm = off[off.index.month != mth]
            a = run(tables(offm, None, tmp / f"h1_{fold}_no{mth}_a.json"), df, mth)
            b = run(tables(offm, sig, tmp / f"h1_{fold}_no{mth}_b.json"), df, mth)
            rows.append(dict(fold=fold, case="load/solar month hidden: whole run", month=mth, shipped=a[0], proposal=b[0],
                             gain=b[0] - a[0], infr=a[2] + b[2]))
            rows.append(dict(fold=fold, case="load/solar month hidden: that week", month=mth, shipped=a[1], proposal=b[1],
                             gain=b[1] - a[1], infr=a[2] + b[2]))
            print(rows[-2], rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(L.HERE / "out" / "h1_nov_signal.csv", index=False)
    print(d.round(3).to_string())
