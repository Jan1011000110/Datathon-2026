# Usage: .venv/Scripts/python.exe experiments/30_boundaries/q3_std.py
"""Standard mode, for the calibration question of Q3: public-half / whole-run score of the plain optimum (agents/v1,
24 known values) under the same hypotheses about the cut, on the standard folds. Writes q3_std.csv."""
import importlib.util

import numpy as np
import pandas as pd

import lib as L
from src import sim as S

HYP = ["first 84 h", "last 84 h", "alt days (1,3,5,7)", "alt days (2,4,6)", "alt weeks (Sep, Nov, ...)", "alt weeks (Oct, Dec, ...)", "random hours"]


def main():
    spec = importlib.util.spec_from_file_location("agent_v1", L.ROOT / "agents" / "v1" / "agent.py")
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    rows = []
    for table in ("official", "filled"):
        folds = S.get_folds(table)
        for fold in "ABC":
            df = folds[fold]
            r = S.simulate(v1.BatteryAgent(), df, return_trace=True)
            g = r["trace"]["grid_kw"].to_numpy()
            mk = L.masks(df, fold, seeds=200)
            sc = lambda m=None: S.score_from_grid(g, df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float),
                                                  df["precio_eur_per_kwh"].to_numpy(float), df["co2_kg_per_kwh"].to_numpy(float), None, m)["score"]
            whole = sc()
            for h in HYP:
                keys = [k for k in mk if k.startswith(h)]
                rows.append(dict(table=table, fold=fold, hyp=h, whole=whole, public=float(np.mean([sc(mk[k]) for k in keys])),
                                 private=float(np.mean([sc(~mk[k]) for k in keys])),
                                 sd=float(np.std([sc(mk[k]) for k in keys])) if len(keys) > 1 else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(L.HERE / "q3_std.csv", index=False)
    pd.set_option("display.width", 220)
    for table in ("official", "filled"):
        x = d[d.table == table]
        print(table, x.drop_duplicates("fold").set_index("fold")["whole"].round(3).to_dict())
        t = x.assign(ratio=x.public / x.whole).pivot(index="hyp", columns="fold", values="ratio").loc[HYP]
        t["mean"] = t.mean(axis=1)
        t["sd_random"] = x.groupby("hyp")["sd"].mean()
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()
