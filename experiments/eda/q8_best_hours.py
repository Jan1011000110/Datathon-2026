"""
Q8 - When are the best hours?  Hour of the daily minimum (charge) and maximum (discharge) of CO2,
price and the combined signal w, by month.
  fig q8_best_hours.png
  stats q8_best_hours.json
All 282 train days.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
import signals as S

C.set_style()
df = C.signals_frame()
KEYS = ("co2", "price", "w")
stats: dict = {}
mats: dict = {}

for k in KEYS:
    cfg = C.SIGNALS[k]
    ds = S.daily_stats(df, cfg["col"])
    m = C.day_matrix(df[cfg["col"]])
    out = {}
    for kind, hcol in (("min", "h_min"), ("max", "h_max")):
        tab = pd.crosstab(ds["month"], ds[hcol]).reindex(index=C.MONTH_ORDER, columns=range(24)).fillna(0)
        share = tab.div(tab.sum(axis=1), axis=0) * 100
        mats[(k, kind)] = share
        overall = ds[hcol].value_counts(normalize=True).sort_index()
        out[f"{kind}_hour_distribution_pct"] = {int(h): float(v * 100) for h, v in overall.items()}
        out[f"{kind}_modal_hour"] = int(overall.idxmax())
        out[f"{kind}_modal_hour_share"] = float(overall.max())
        out[f"{kind}_modal_hour_by_month"] = {C.MONTH_ABBR[mm]: int(share.loc[mm].idxmax()) for mm in C.MONTH_ORDER}
        out[f"{kind}_by_season_top3"] = {
            s: {int(h): float(v) for h, v in ds.loc[ds["season"] == s, hcol].value_counts(normalize=True).head(3).items()}
            for s in C.SEASON_ORDER}
    h_min, h_max = ds["h_min"], ds["h_max"]
    out["share_min_between_10_and_17"] = float(h_min.between(10, 17).mean())
    out["share_min_between_13_and_16"] = float(h_min.between(13, 16).mean())
    out["share_min_at_night_22_to_07"] = float(((h_min >= 22) | (h_min <= 7)).mean())
    out["share_max_between_18_and_22"] = float(h_max.between(18, 22).mean())
    out["share_max_between_05_and_09"] = float(h_max.between(5, 9).mean())
    out["share_max_between_23_and_04"] = float(((h_max >= 23) | (h_max <= 4)).mean())
    out["share_min_before_max"] = float((h_min < h_max).mean())
    winter = ds["season"] == "Winter"
    out["winter_share_min_at_night_22_to_07"] = float(((h_min[winter] >= 22) | (h_min[winter] <= 7)).mean())
    summer = ds["season"] == "Summer"
    out["summer_share_max_between_05_and_09"] = float(h_max[summer].between(5, 9).mean())
    out["summer_share_max_between_18_and_22"] = float(h_max[summer].between(18, 22).mean())
    # membership of the 4 cheapest / 4 dearest hours (the battery needs about 4 h to fill or empty)
    order = np.argsort(m.values, axis=1)
    low4 = np.zeros(24)
    high4 = np.zeros(24)
    for row in order:
        low4[row[:4]] += 1
        high4[row[-4:]] += 1
    out["share_of_days_hour_in_4_cheapest_pct"] = {h: float(low4[h] / len(m) * 100) for h in range(24)}
    out["share_of_days_hour_in_4_dearest_pct"] = {h: float(high4[h] / len(m) * 100) for h in range(24)}
    stats[k] = out
C.write_stats("q8_best_hours", stats)

# ================================================================== FIGURE
w = stats["w"]
fig = plt.figure(figsize=(13.5, 7.4))
C.titled(fig, f"The best hour to charge is 14:00 on {w['min_modal_hour_share']:.0%} of days; the best hour to discharge "
              "is the evening peak, or dawn in summer",
         "Share of days in each month whose daily minimum (top row) or maximum (bottom row) falls in each hour. "
         "Darker = more days.", y=0.978)
gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.835, bottom=0.10, wspace=0.13, hspace=0.36)
for j, k in enumerate(KEYS):
    cfg = C.SIGNALS[k]
    for i, kind in enumerate(("min", "max")):
        ax = fig.add_subplot(gs[i, j])
        share = mats[(k, kind)]
        im = ax.imshow(share.values, aspect="auto", cmap=cfg["cmap"], vmin=0, vmax=70)
        ax.set_yticks(range(12))
        ax.set_yticklabels([C.MONTH_ABBR[m] for m in C.MONTH_ORDER] if j == 0 else [], fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
        ax.grid(which="minor", color=C.SURFACE, linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        what = "lowest hour (charge)" if kind == "min" else "highest hour (discharge)"
        name = {"co2": "CO₂", "price": "Price", "w": "Combined signal w"}[k]
        ax.set_title(f"{name}: {what}", fontsize=12)
        if i == 1:
            ax.set_xlabel("Hour of day (local time)")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045, ticks=[0, 35, 70])
        cb.ax.set_yticklabels(["0", "35", "70 %"], fontsize=9.5)
        cb.outline.set_visible(False)
C.footnote(fig, f"Sample: all {df['date'].nunique()} train days (21–24 days per month). For w: the minimum falls between "
                f"10:00 and 17:59 on {w['share_min_between_10_and_17']:.0%} of days, the maximum between 18:00 and 22:59 on "
                f"{w['share_max_between_18_and_22']:.0%} and between 05:00 and 09:59 on {w['share_max_between_05_and_09']:.0%}.",
           y=0.01)
C.save(fig, "q8_best_hours")

for k in KEYS:
    print(k, {a: b for a, b in stats[k].items() if not isinstance(b, dict)})
    print("   min modal by month", stats[k]["min_modal_hour_by_month"])
    print("   max modal by month", stats[k]["max_modal_hour_by_month"])
