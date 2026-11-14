"""
experiments/29_stress_hard/hidden_month.py
A week whose month the forecasters have never seen (the November case): hide a whole month from the fit and
run its week. For each fold and hidden month M: states of h1 and h2 fitted on the fold-out rows minus month M;
the fold is run with them and compared with the normal fold-out state, on the whole run and on the hours of
month M alone (the score of those hours against their own baseline).
    hidden_month.py B 10 12 3 6     -> out/hidden_month_<fold>.csv
h2's state is built by agents/h2/build.py with experiments/27_combined/combined.fit_tables wrapped so that the
month is removed (the cross-fitted error tables are the fold's usual ones).
"""
import importlib.util
import sys

import numpy as np
import pandas as pd

import lib as L
from src.hard import score_weights, train_complement
from src.sim import score_from_grid


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def week_score(tr, m):
    """Score of the hours m alone, against the no-battery import of those hours."""
    t = tr[m]
    return score_from_grid(t["grid_kw"].to_numpy(), t["load_kw"].to_numpy(), t["pv_kw"].to_numpy(),
                           t["price"].to_numpy(), t["co2"].to_numpy())["score"]


if __name__ == "__main__":
    fold = sys.argv[1]
    months = [int(x) for x in sys.argv[2:]]
    df = L.clean_folds()[fold]
    h1b = _load("h1_build_hm", L.ROOT / "agents" / "h1" / "build.py")
    h2b = _load("h2_build_hm", L.ROOT / "agents" / "h2" / "build.py")
    cb = h2b.combined()
    orig = cb.fit_tables
    rows = []

    def runs(tag, mth):
        out = {}
        for kind in ("h1", "h2"):
            a = L.load_agent(kind, L.state_path(kind, fold, tag))
            r = L.run(a, df, trace=True)
            tr = r["trace"]
            tr["price"], tr["co2"] = df["precio_eur_per_kwh"].to_numpy(), df["co2_kg_per_kwh"].to_numpy()
            out[kind] = (r, week_score(tr, df.index.month == mth))
        return out

    for mth in months:
        tag = f"_no{mth:02d}"
        fit = train_complement(df)
        h1b.write_tables(fit[fit.index.month != mth], L.state_path("h1", fold, tag), fill_missing="neighbours")
        # the same tables with the missing month filled by the all-month average (the plain h1 variant)
        h1b.write_tables(fit[fit.index.month != mth], L.state_path("h1", fold, tag + "_mean"), fill_missing="mean")

        def fit_tables(fold_df=None, drop_days=None, _m=mth):
            return tuple(t[t.index.month != _m] for t in orig(fold_df, drop_days))
        cb.fit_tables = fit_tables
        try:
            arrays, _, _ = h2b.build_state(df)
        finally:
            cb.fit_tables = orig
        h2b.write_state(arrays, L.state_path("h2", fold, tag))
        base, hid = runs("", mth), runs(tag, mth)
        a = L.load_agent("h1", L.state_path("h1", fold, tag + "_mean"))
        r = L.run(a, df, trace=True)
        tr = r["trace"]
        tr["price"], tr["co2"] = df["precio_eur_per_kwh"].to_numpy(), df["co2_kg_per_kwh"].to_numpy()
        hid["h1 (all-month fill)"] = (r, week_score(tr, df.index.month == mth))
        base["h1 (all-month fill)"] = base["h1"]
        for kind in hid:
            rb, wb = base[kind]
            rh, wh = hid[kind]
            rows.append(dict(fold=fold, month=mth, agent=kind, run_seen=rb["score"], run_hidden=rh["score"],
                             week_seen=wb, week_hidden=wh, d_run=rh["score"] - rb["score"], d_week=wh - wb,
                             infractions=rh["infractions"], exceptions=rh["exceptions"], fallbacks=rh["fallbacks"]))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(L.HERE / "out" / f"hidden_month_{fold}.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
