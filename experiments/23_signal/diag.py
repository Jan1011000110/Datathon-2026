# Usage: .venv/Scripts/python.exe experiments/23_signal/diag.py [config_stem]
"""Where does the signal forecast cost score? Per-day closed-loop loss against perfect w
(both with perfect net load, H=36), from the traces saved by closed_loop.py, grouped by kind of
day. Writes diag_days.csv (one row per fold day) and prints the group tables."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAPPA, get_folds  # noqa: E402
from signal_fc import HOLIDAYS  # noqa: E402
from src.hard import score_weights  # noqa: E402

HERE = Path(__file__).resolve().parent
STEMS = {"fc": sys.argv[1] if len(sys.argv) > 1 else "FINAL_perfectN_H36", "clim": "clim_perfectN_H36"}
REF = "perfect_perfectN_H36"
# holiday stretches present in train (low national demand)
STRETCH = [("2026-01-01", "2026-01-07"), ("2026-04-02", "2026-04-06"), ("2025-12-06", "2025-12-08")]


def day_table():
    rows = []
    for f, df in get_folds().items():
        if f == "ALL":
            continue
        ws = score_weights(df) * 100.0                      # points per kWh of grid import
        ref = pd.read_pickle(HERE / "traces" / f"{REF}_{f}.pkl")
        d = pd.DataFrame({"day": df.index.normalize(), "fold": f}, index=df.index)
        # value of a kWh left in the battery after each hour: sqrt(0.9) x mean weight of the next 24 rows
        cs = np.concatenate([[0.0], np.cumsum(ws)])
        n = len(ws)
        hi = np.minimum(np.arange(n) + 25, n)
        lo = np.minimum(np.arange(n) + 1, n)
        val = 0.9 ** 0.5 * np.where(hi > lo, (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1), 0.0)
        for k, stem in STEMS.items():
            tr = pd.read_pickle(HERE / "traces" / f"{stem}_{f}.pkl")
            raw = (tr["grid_kw"].to_numpy() - ref["grid_kw"].to_numpy()) * ws
            # credit the charge carried over: a run that ends an hour with more energy than the reference
            # has not lost that energy yet (without this, losses are booked on the wrong day)
            carry = (tr["soc_kwh"].to_numpy() - ref["soc_kwh"].to_numpy()) * val
            d[f"loss_{k}"] = raw - np.diff(np.concatenate([[0.0], carry]))
            d[f"raw_{k}"] = raw
        w = (df["co2_kg_per_kwh"] + KAPPA * df["precio_eur_per_kwh"])
        d["co2"] = df["co2_kg_per_kwh"]
        d["w"] = w
        g = d.groupby(["fold", "day"])
        out = g[[c for c in d.columns if c.startswith("loss_") or c.startswith("raw_")]].sum()
        out["n_h"] = g.size()
        hh = d.index.hour
        mid = d[(hh >= 10) & (hh < 16)].groupby(["fold", "day"])["co2"].mean()
        night = d[(hh >= 20) | (hh < 8)].groupby(["fold", "day"])["co2"].mean()
        out["co2_mid_over_night"] = mid / night
        out["w_max_over_min"] = g["w"].max() / g["w"].min()
        rows.append(out.reset_index())
    t = pd.concat(rows, ignore_index=True)
    t = t[t.n_h == 24].copy()
    dow = t.day.dt.dayofweek
    hol = np.array([(a.month, a.day) in HOLIDAYS for a in t.day])
    t["kind"] = np.select([hol, dow == 0, dow == 5, dow == 6], ["holiday", "Monday", "Saturday", "Sunday"], "Tue-Fri")
    t["season"] = t.day.dt.month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                                      6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"})
    t["co2_shape"] = np.where(t.co2_mid_over_night > 0.9, "flat or inverted (midday >= 0.9 x night)",
                              np.where(t.co2_mid_over_night > 0.6, "moderate trough (0.6-0.9)", "deep trough (< 0.6)"))
    t["block_day"] = np.where(t.day.dt.day.isin([1, 8, 15]), "first day of a week", "other days")
    t["stretch"] = "ordinary"
    for a, b in STRETCH:
        t.loc[(t.day >= a) & (t.day <= b), "stretch"] = "holiday stretch (1-7 Jan, 2-6 Apr, 6-8 Dec)"
    return t


if __name__ == "__main__":
    t = day_table()
    t.to_csv(HERE / "diag_days.csv", index=False)
    pd.set_option("display.width", 200)
    for by in ("kind", "season", "co2_shape", "stretch", "block_day", "fold"):
        g = t.groupby(by).agg(days=("loss_fc", "size"), loss_fc_per_day=("loss_fc", "mean"),
                              loss_clim_per_day=("loss_clim", "mean"), loss_fc_total=("loss_fc", "sum"),
                              loss_clim_total=("loss_clim", "sum"))
        print(f"\n== by {by} (points of fold score; totals summed over the 3 folds) ==")
        print(g.round(4).to_string())
    print("\nworst 12 days for the forecaster:")
    print(t.sort_values("loss_fc", ascending=False).head(12)[["fold", "day", "kind", "co2_shape", "loss_fc", "loss_clim"]].round(4).to_string())
