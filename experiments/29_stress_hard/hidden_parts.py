"""
experiments/29_stress_hard/hidden_parts.py
Which specialist of h2 suffers when a month is missing from its fit rows? (November in the real train set has no
load and no solar rows, but it does have prices and CO2.)
    hidden_parts.py B 10 12      -> out/hidden_parts_<fold>.csv
For each month: h2 states with the month removed from ONE fit table at a time
    load    (LoadFC's table)        solar  (the official rows: SolarFC)        signal (price / CO2 table)
    load+solar = the real November case
and the score of the fold run and of that month's week alone.
"""
import importlib.util
import sys

import pandas as pd

import lib as L
from hidden_month import _load, week_score

if __name__ == "__main__":
    fold = sys.argv[1]
    months = [int(x) for x in sys.argv[2:]]
    df = L.clean_folds()[fold]
    h2b = _load("h2_build_hp", L.ROOT / "agents" / "h2" / "build.py")
    cb = h2b.combined()
    orig = cb.fit_tables
    PARTS = {"load": (1,), "solar": (0,), "signal": (2,), "load+solar": (0, 1)}
    rows = []

    def one(state, mth):
        r = L.run(L.load_agent("h2", state), df, trace=True)
        tr = r["trace"]
        tr["price"], tr["co2"] = df["precio_eur_per_kwh"].to_numpy(), df["co2_kg_per_kwh"].to_numpy()
        return r, week_score(tr, df.index.month == mth)

    for mth in months:
        rb, wb = one(L.state_path("h2", fold), mth)
        for part, which in PARTS.items():
            def fit_tables(fold_df=None, drop_days=None, _m=mth, _w=which):
                t = list(orig(fold_df, drop_days))
                for k in _w:
                    t[k] = t[k][t[k].index.month != _m]
                return tuple(t)
            cb.fit_tables = fit_tables
            try:
                arrays, _, _ = h2b.build_state(df)
            finally:
                cb.fit_tables = orig
            p = L.state_path("h2", fold, f"_no{mth:02d}_{part.replace('+', '')}")
            h2b.write_state(arrays, p)
            r, w = one(p, mth)
            rows.append(dict(fold=fold, month=mth, hidden_from=part, run_seen=rb["score"], run_hidden=r["score"],
                             d_run=r["score"] - rb["score"], week_seen=wb, week_hidden=w, d_week=w - wb,
                             infractions=r["infractions"], exceptions=r["exceptions"], fallbacks=r["fallbacks"],
                             spill_kwh=r["spill_kwh"], spill_seen=rb["spill_kwh"]))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
            pd.DataFrame(rows).to_csv(L.HERE / "out" / f"hidden_parts_{fold}.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
