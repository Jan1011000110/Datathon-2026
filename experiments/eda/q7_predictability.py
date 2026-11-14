"""
Q7 - Predictability beyond the 24 h lookahead (statistical part; the operational part is q9).
  fig q7_autocorrelation.png : ACF of CO2 and price + error of naive forecasts by lag
  fig q7_shape_vs_level.png  : how stable the daily level and the daily shape are
  stats q7_predictability.json
All pairs are aligned on real timestamps, so a lag never jumps over a hidden test week.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

import common as C
import signals as S

C.set_style()
df = C.signals_frame()
KEYS = ("co2", "price")
LAGS = (24, 48, 72, 168)
stats: dict = {}

acf_lags = list(range(1, 193))
acfs = {k: S.acf(df[C.SIGNALS[k]["col"]], acf_lags) for k in KEYS + ("w",)}

err: dict = {}
for k in KEYS + ("w",):
    col = C.SIGNALS[k]["col"]
    s = df[col]
    e = {}
    for L in LAGS:
        a, b = S.lagged(s, L)
        d = a - b
        e[f"lag_{L}h"] = {"n": int(len(a)), "mae_pct": float(np.abs(d).mean() / a.mean() * 100),
                          "rmse_pct": float(np.sqrt((d ** 2).mean()) / a.mean() * 100),
                          "bias_pct": float(d.mean() / a.mean() * 100)}
    # reference: leave-one-day-out mean profile by (month, weekday/weekend, hour)
    g = df.groupby(["month", "weekend", "hour"])[col]
    clim = (g.transform("sum") - s) / (g.transform("count") - 1)
    d = s - clim
    e["profile"] = {"n": int(len(s)), "mae_pct": float(d.abs().mean() / s.mean() * 100),
                    "rmse_pct": float(np.sqrt((d ** 2).mean()) / s.mean() * 100), "bias_pct": float(d.mean() / s.mean() * 100)}
    err[k] = e
    stats[k] = {"acf": {f"{L}h": float(acfs[k][L]) for L in (1, 12, 24, 48, 72, 168)}, "naive_forecast_error": e}

# ------------------------------------------------------------------ level vs shape by daily lag
day_lags = list(range(1, 8))
lvl_corr, lvl_rel, shp_rho, shp_p10, n_pairs, mse_level = ({k: [] for k in KEYS} for _ in range(6))
overlap = {}
for k in KEYS:
    m = C.day_matrix(df[C.SIGNALS[k]["col"]])
    days = m.index
    level = m.mean(axis=1)
    for lag in day_lags:
        prev = days - pd.Timedelta(days=lag)
        ok = prev.isin(days)
        d1, d0 = days[ok], prev[ok]
        l1, l0 = level[d1].values, level[d0].values
        x1, x0 = m.loc[d1].values, m.loc[d0].values
        rho = np.array([st.spearmanr(a, b)[0] for a, b in zip(x1, x0)])
        lvl_corr[k].append(float(np.corrcoef(l1, l0)[0, 1]))
        lvl_rel[k].append(float(np.median(np.abs(l1 - l0) / l0) * 100))
        shp_rho[k].append(float(np.median(rho)))
        shp_p10[k].append(float(np.quantile(rho, 0.1)))
        n_pairs[k].append(int(ok.sum()))
        mse = ((x1 - x0) ** 2).mean()
        mse_level[k].append(float(((l1 - l0) ** 2).mean() / mse))
        if lag == 1:
            same = C.is_weekend(d1) == C.is_weekend(d0)
            stats[k]["shape_rank_corr_lag1_same_day_type"] = float(np.median(rho[same]))
            stats[k]["shape_rank_corr_lag1_weekday_weekend_switch"] = float(np.median(rho[~same]))
            lo1, lo0 = np.argsort(x1, 1)[:, :4], np.argsort(x0, 1)[:, :4]
            hi1, hi0 = np.argsort(x1, 1)[:, -4:], np.argsort(x0, 1)[:, -4:]
            overlap[k] = {"cheapest4": float(np.mean([len(set(a) & set(b)) for a, b in zip(lo1, lo0)])),
                          "dearest4": float(np.mean([len(set(a) & set(b)) for a, b in zip(hi1, hi0)])),
                          "same_min_hour": float((x1.argmin(1) == x0.argmin(1)).mean()),
                          "same_max_hour": float((x1.argmax(1) == x0.argmax(1)).mean())}
    stats[k]["by_day_lag"] = {
        "lag_days": day_lags, "pairs": n_pairs[k], "level_corr": lvl_corr[k], "level_median_abs_change_pct": lvl_rel[k],
        "shape_rank_corr_median": shp_rho[k], "shape_rank_corr_p10": shp_p10[k], "mse_share_level": mse_level[k]}
    stats[k]["overlap_with_previous_day_out_of_4"] = overlap[k]
C.write_stats("q7_predictability", stats)

# ================================================================== FIGURE 1: ACF + naive errors
fig = plt.figure(figsize=(13.5, 6.5))
C.titled(fig, "Beyond 24 hours the signals fade: copying yesterday misses by 19–20 %, copying the day before by "
              "24–29 %",
         "Left: correlation of each signal with itself h hours earlier. Right: mean absolute error of copying an "
         "earlier value, as a share of the signal's mean.")
gs = fig.add_gridspec(1, 2, left=0.06, right=0.985, top=0.80, bottom=0.14, wspace=0.38, width_ratios=[1.35, 1])
ax = fig.add_subplot(gs[0])
ax.axhline(0, color=C.AXIS, lw=1.0)
for k in KEYS:
    cfg = C.SIGNALS[k]
    ax.plot(acf_lags, acfs[k].values, color=cfg["color"], lw=2.0, label=cfg["label"])
    ax.scatter(LAGS, [acfs[k][L] for L in LAGS], s=46, color=cfg["color"], edgecolor=C.SURFACE, linewidth=1.5, zorder=3)
for L in LAGS:
    c, p = acfs["co2"][L], acfs["price"][L]
    hi_k, lo_k = ("co2", "price") if c >= p else ("price", "co2")
    ax.annotate(f"{max(c, p):.2f}", xy=(L, max(c, p)), xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=10.5, color=C.INK)
    ax.annotate(f"{min(c, p):.2f}", xy=(L, min(c, p)), xytext=(13, -3), textcoords="offset points", ha="left",
                fontsize=10.5, color=C.INK, bbox=dict(facecolor=C.SURFACE, edgecolor="none", pad=1.0, alpha=0.85))
ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168, 192])
ax.set_xlim(0, 195)
ax.set_ylim(-0.25, 1.08)
ax.set_xlabel("Lag h (hours)")
ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation: CO₂ decays day by day, price repeats weekly")
ax.legend(loc="upper right", fontsize=10.5, ncol=2, bbox_to_anchor=(1.0, 1.0), handlelength=1.6)

ax2 = fig.add_subplot(gs[1])
methods = [("lag_24h", "Same hour 1 day earlier"), ("lag_48h", "Same hour 2 days earlier"),
           ("lag_72h", "Same hour 3 days earlier"), ("lag_168h", "Same hour last week"),
           ("profile", "Mean profile of the month\nand day type")]
y = np.arange(len(methods))[::-1]
hbar = 0.34
for i, k in enumerate(KEYS):
    cfg = C.SIGNALS[k]
    v = [err[k][m]["mae_pct"] for m, _ in methods]
    yy = y + (0.5 - i) * (hbar + 0.04)
    ax2.barh(yy, v, height=hbar, color=cfg["color"], label=cfg["label"])
    for a, b in zip(yy, v):
        ax2.text(b + 0.5, a, f"{b:.0f} %", va="center", fontsize=10.5, color=C.INK)
ax2.set_yticks(y)
ax2.set_yticklabels([n for _, n in methods])
ax2.set_xlim(0, 36)
ax2.set_xlabel("Mean absolute error (% of the signal's mean)")
ax2.grid(axis="y", visible=False)
ax2.set_title("Error of naive forecasts")
ax2.legend(loc="lower right", fontsize=10.5, handlelength=1.0)
C.footnote(fig, f"Sample: train hours whose lagged timestamp is also in train: {err['co2']['lag_24h']['n']:,} pairs at 24 h, "
                f"{err['co2']['lag_48h']['n']:,} at 48 h, {err['co2']['lag_72h']['n']:,} at 72 h, "
                f"{err['co2']['lag_168h']['n']:,} at 168 h. Mean profile is computed leaving the forecast day out.")
C.save(fig, "q7_autocorrelation")

# ================================================================== FIGURE 2: shape vs level
fig = plt.figure(figsize=(13.5, 6.3))
C.titled(fig, "The daily shape is far more stable than the daily level, but the best four hours still move",
         "Day d compared with day d − k. Level = the day's mean. Shape = the ranking of its 24 hours "
         "(Spearman rank correlation, median over day pairs).")
gs = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=0.79, bottom=0.15, wspace=0.27, width_ratios=[1, 1, 0.95])
ax = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
for k in KEYS:
    cfg = C.SIGNALS[k]
    ax.plot(day_lags, lvl_corr[k], color=cfg["color"], lw=2.0, marker="o", ms=7, mec=C.SURFACE, mew=1.5, label=cfg["label"])
    ax2.plot(day_lags, shp_rho[k], color=cfg["color"], lw=2.0, marker="o", ms=7, mec=C.SURFACE, mew=1.5, label=cfg["label"])
for a in (ax, ax2):
    a.set_ylim(0, 1.0)
    a.set_xticks(day_lags)
    a.set_xlabel("k (days apart)")
ax.set_ylabel("Correlation")
ax.set_title("Level: correlation of daily means")
ax2.set_title("Shape: rank correlation of the 24 hours")
ax.legend(loc="upper center", fontsize=10.5)
ax.annotate(f"{lvl_corr['co2'][0]:.2f}", xy=(1, lvl_corr["co2"][0]), xytext=(8, 6), textcoords="offset points", fontsize=10.5)
ax.annotate(f"{lvl_corr['co2'][1]:.2f}", xy=(2, lvl_corr["co2"][1]), xytext=(8, 6), textcoords="offset points", fontsize=10.5)
ax.annotate(f"{lvl_corr['price'][1]:.2f}", xy=(2, lvl_corr["price"][1]), xytext=(0, -18), textcoords="offset points",
            fontsize=10.5, ha="center")
ax.annotate(f"{lvl_corr['price'][6]:.2f}", xy=(7, lvl_corr["price"][6]), xytext=(-4, 9), textcoords="offset points",
            fontsize=10.5, ha="center")
ax.text(3.9, 0.215, "price level follows the\nweekday / weekend cycle", fontsize=10, color=C.INK2, ha="center", va="center")
ax2.annotate(f"{shp_rho['co2'][1]:.2f}", xy=(2, shp_rho["co2"][1]), xytext=(0, 9), textcoords="offset points", fontsize=10.5,
             ha="center")
ax2.annotate(f"{shp_rho['price'][1]:.2f}", xy=(2, shp_rho["price"][1]), xytext=(0, -18), textcoords="offset points",
             fontsize=10.5, ha="center")
ax2.text(4.0, 0.30, f"price, same day type: {stats['price']['shape_rank_corr_lag1_same_day_type']:.2f}\n"
                    f"across a weekday/weekend switch: {stats['price']['shape_rank_corr_lag1_weekday_weekend_switch']:.2f}",
         fontsize=10, color=C.INK2, ha="center", va="center")

ax3 = fig.add_subplot(gs[2])
cats = ["4 cheapest hours", "4 dearest hours"]
x = np.arange(2)
bw = 0.3
for i, k in enumerate(KEYS):
    cfg = C.SIGNALS[k]
    v = [overlap[k]["cheapest4"], overlap[k]["dearest4"]]
    xx = x + (i - 0.5) * (bw + 0.04)
    ax3.bar(xx, v, width=bw, color=cfg["color"], label=cfg["label"])
    for a, b in zip(xx, v):
        ax3.text(a, b + 0.06, f"{b:.1f}", ha="center", fontsize=10.5, color=C.INK)
ax3.set_xticks(x)
ax3.set_xticklabels(cats)
ax3.set_ylim(0, 4)
ax3.set_yticks([0, 1, 2, 3, 4])
ax3.set_ylabel("Hours shared with the previous day (of 4)")
ax3.grid(axis="x", visible=False)
ax3.set_title("Yesterday's best hours today")
ax3.legend(loc="upper left", fontsize=10.5, handlelength=1.0)
C.footnote(fig, f"Sample: pairs of train days k days apart (both in train): {n_pairs['co2'][0]} pairs at k = 1, "
                f"{n_pairs['co2'][1]} at k = 2, {n_pairs['co2'][6]} at k = 7.")
C.save(fig, "q7_shape_vs_level")

for k in KEYS:
    print(k, stats[k]["acf"], {m: round(v["mae_pct"], 1) for m, v in err[k].items()})
    print("   ", stats[k]["by_day_lag"])
    print("   ", overlap[k])
