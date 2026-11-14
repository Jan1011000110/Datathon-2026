# Usage: .venv/Scripts/python.exe experiments/30_boundaries/q3_base.py
"""A model-free check of the public cut: the leaderboards give the no-battery baseline of the public hours in both
modes (standard: about 71,800 kg and 118,800 EUR; hard, solar x12: about 60,250 kg and 95,000 EUR). Their ratio
depends on how much solar the public hours hold, and the first 84 hours of a week hold hours 0-11 four times and
hours 12-23 three times (the last 84 the reverse). Same ratio on the folds under each hypothesis. Writes q3_base.csv."""
import numpy as np
import pandas as pd

import lib as L
from src.hard import load_hard

OBS = {"co2": 60250.0 / 71800.0, "cost": 95000.0 / 118800.0}
HYP = ["whole run", "first 84 h", "last 84 h", "alt days (1,3,5,7)", "alt days (2,4,6)", "alt weeks (Sep, Nov, ...)", "alt weeks (Oct, Dec, ...)"]


def main():
    rows = []
    for table in ("official", "filled"):
        src = load_hard(table)
        for fold, (lo, hi) in L.cb.FOLD_DAYS.items():
            for lab, (a, b) in {fold: (lo, hi), fold + "+": (lo + 21 if fold == "A" else lo, hi + 21 if fold == "A" else hi)}.items():
                df = src[(src.index.day >= a) & (src.index.day <= b)]
                if lab.endswith("+") and fold != "A":
                    continue
                mk = dict(L.masks(df, "A" if a == 1 else {8: "B", 15: "C"}.get(a, "A"), seeds=0)) if a in (1, 8, 15) else None
                if mk is None:                                     # days 22-28: positions from day 22
                    wk = df.index.strftime("%Y-%m").to_numpy()
                    hs = ((df.index.day.to_numpy() - a) * 24 + df.index.hour.to_numpy())
                    par = ((df.index.year * 12 + df.index.month) - (2025 * 12 + 9)).to_numpy() % 2
                    mk = {"first 84 h": hs < 84, "last 84 h": hs >= 84, "alt days (1,3,5,7)": (hs // 24) % 2 == 0,
                          "alt days (2,4,6)": (hs // 24) % 2 == 1, "alt weeks (Sep, Nov, ...)": par == 0, "alt weeks (Oct, Dec, ...)": par == 1}
                mk["whole run"] = np.ones(len(df), bool)
                load, pv12 = df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float)
                co2, pr = df["co2_kg_per_kwh"].to_numpy(float), df["precio_eur_per_kwh"].to_numpy(float)
                bh, bs = np.maximum(load - pv12, 0), np.maximum(load - pv12 / 12.0, 0)
                for h in HYP:
                    m = mk[h]
                    rows.append(dict(table=table, days=f"{a}-{b}", hyp=h, hours=int(m.sum()),
                                     co2=float((bh * co2)[m].sum() / (bs * co2)[m].sum()),
                                     cost=float((bh * pr)[m].sum() / (bs * pr)[m].sum()),
                                     kg_std=float((bs * co2)[m].sum()), kg_hard=float((bh * co2)[m].sum()),
                                     kg_per_eur_hard=float((bh * co2)[m].sum() / (bh * pr)[m].sum()),
                                     kg_per_eur_std=float((bs * co2)[m].sum() / (bs * pr)[m].sum())))
    d = pd.DataFrame(rows)
    d.to_csv(L.HERE / "q3_base.csv", index=False)
    pd.set_option("display.width", 220)
    print("observed on the boards: hard / standard public baseline  CO2 %.4f  cost %.4f;  kg per EUR hard 0.634, standard 0.604" % (OBS["co2"], OBS["cost"]))
    for table in ("official", "filled"):
        x = d[d.table == table]
        for col in ("co2", "cost", "kg_per_eur_hard", "kg_per_eur_std"):
            t = x.pivot(index="hyp", columns="days", values=col).loc[HYP]
            t["mean"] = t.mean(axis=1)
            t["sd"] = t.drop(columns="mean").std(axis=1)
            print(f"--- {table}: {col}")
            print(t.round(4).to_string())


if __name__ == "__main__":
    main()
