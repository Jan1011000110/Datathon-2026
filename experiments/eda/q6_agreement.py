"""
Q6 - Do the two objectives agree?  Price vs CO2, and the combined signal w.
  fig q6_agreement.png     : normalised daily profiles + per-day rank correlation
  fig q6_w_decomposition.png : which term drives the within-day variation of w
  stats q6_agreement.json
All 282 train days. w uses the train baseline totals of the organisers' table (5,993 h).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

import common as C

C.set_style()
df = C.signals_frame()
base = C.baseline_totals()
stats: dict = {"baseline": base}

co2_m = C.day_matrix(df["co2_kg_per_kwh"])
pr_m = C.day_matrix(df["precio_eur_per_kwh"])
a_m = C.day_matrix(df["w_co2"])
b_m = C.day_matrix(df["w_price"])
w_m = C.day_matrix(df["w"])
days = co2_m.index
wkend = C.is_weekend(days)
N = len(days)

# ------------------------------------------------------------------ correlations
stats["hourly_pearson"] = float(np.corrcoef(df["co2_kg_per_kwh"], df["precio_eur_per_kwh"])[0, 1])
stats["hourly_spearman"] = float(st.spearmanr(df["co2_kg_per_kwh"], df["precio_eur_per_kwh"])[0])
stats["daily_mean_pearson"] = float(np.corrcoef(co2_m.mean(axis=1), pr_m.mean(axis=1))[0, 1])
rho = pd.Series([st.spearmanr(co2_m.values[i], pr_m.values[i])[0] for i in range(N)], index=days)
stats["within_day_spearman"] = {
    "median": float(rho.median()), "mean": float(rho.mean()), "p10": float(rho.quantile(0.1)),
    "p90": float(rho.quantile(0.9)), "share_negative": float((rho < 0).mean()),
    "share_above_0p5": float((rho > 0.5).mean()), "share_above_0p8": float((rho > 0.8).mean()),
    "median_weekday": float(rho[~wkend].median()), "median_weekend": float(rho[wkend].median()),
    "by_month_median": {C.MONTH_ABBR[m]: float(v) for m, v in rho.groupby(days.month).median().items()},
}
# pooled within-day deviations
a_dev = a_m.values - a_m.values.mean(axis=1, keepdims=True)
b_dev = b_m.values - b_m.values.mean(axis=1, keepdims=True)
w_dev = a_dev + b_dev
stats["within_day_deviation_pearson"] = float((a_dev * b_dev).mean() / np.sqrt((a_dev ** 2).mean() * (b_dev ** 2).mean()))


def var_shares(mask):
    a, b = a_dev[mask], b_dev[mask]
    tot = ((a + b) ** 2).mean()
    return {"co2": float((a ** 2).mean() / tot), "price": float((b ** 2).mean() / tot),
            "covariance": float(2 * (a * b).mean() / tot)}


stats["w_within_day_variance_shares"] = {
    "all": var_shares(np.ones(N, bool)), "weekday": var_shares(~wkend), "weekend": var_shares(wkend),
    "by_month": {C.MONTH_ABBR[m]: var_shares(days.month == m) for m in C.MONTH_ORDER},
}
# share of w's total (hourly) variance that is within-day vs between days
stats["w_share_of_variance_within_day"] = float((w_dev ** 2).mean() / df["w"].var(ddof=0))
r_a = a_m.values.max(axis=1) - a_m.values.min(axis=1)
r_b = b_m.values.max(axis=1) - b_m.values.min(axis=1)
stats["intraday_range_co2_term_median"] = float(np.median(r_a))
stats["intraday_range_price_term_median"] = float(np.median(r_b))
stats["share_days_co2_term_range_larger"] = float((r_a > r_b).mean())

# do they pick the same hours?
def lowk(m, k=4):
    return np.argsort(m.values, axis=1)[:, :k]


def highk(m, k=4):
    return np.argsort(m.values, axis=1)[:, -k:]


def overlap(x, y):
    return float(np.mean([len(set(i) & set(j)) for i, j in zip(x, y)]))


stats["same_hour"] = {
    "min_co2_vs_price": float((co2_m.values.argmin(1) == pr_m.values.argmin(1)).mean()),
    "max_co2_vs_price": float((co2_m.values.argmax(1) == pr_m.values.argmax(1)).mean()),
    "min_w_vs_co2": float((w_m.values.argmin(1) == co2_m.values.argmin(1)).mean()),
    "min_w_vs_price": float((w_m.values.argmin(1) == pr_m.values.argmin(1)).mean()),
    "max_w_vs_co2": float((w_m.values.argmax(1) == co2_m.values.argmax(1)).mean()),
    "max_w_vs_price": float((w_m.values.argmax(1) == pr_m.values.argmax(1)).mean()),
}
stats["overlap_of_4_hours_out_of_4"] = {
    "cheapest_co2_vs_price": overlap(lowk(co2_m), lowk(pr_m)),
    "dearest_co2_vs_price": overlap(highk(co2_m), highk(pr_m)),
    "cheapest_w_vs_co2": overlap(lowk(w_m), lowk(co2_m)), "cheapest_w_vs_price": overlap(lowk(w_m), lowk(pr_m)),
    "dearest_w_vs_co2": overlap(highk(w_m), highk(co2_m)), "dearest_w_vs_price": overlap(highk(w_m), highk(pr_m)),
}
# normalised hourly profiles (1 = import-weighted mean of the year)
prof = df.groupby(["weekend", "hour"])[["w_co2", "w_price", "w"]].mean()
prof["co2_rel"] = prof["w_co2"] / 0.7
prof["price_rel"] = prof["w_price"] / 0.3
stats["weekday_profile_rel"] = {k: {int(h): float(v) for h, v in prof.loc[False, k].items()}
                                for k in ("co2_rel", "price_rel", "w")}
stats["weekend_profile_rel"] = {k: {int(h): float(v) for h, v in prof.loc[True, k].items()}
                                for k in ("co2_rel", "price_rel", "w")}
# stability of the kg-per-euro ratio that sets the weights
off = C.load_hourly_official()
grid = (off["load_kw"] - off["pv_kw"]).clip(lower=0)
kg, eur = grid * off["co2_kg_per_kwh"], grid * off["precio_eur_per_kwh"]
stats["kg_per_eur_by_month"] = {C.MONTH_ABBR[m]: float(v) for m, v in
                                (kg.groupby(off.index.month).sum() / eur.groupby(off.index.month).sum()).items()}
late = off.index.day >= 17
stats["kg_per_eur_days_17_plus"] = float(kg[late].sum() / eur[late].sum())
stats["kg_per_eur_days_1_16"] = float(kg[~late].sum() / eur[~late].sum())
stats["kg_per_eur_leaderboard_public_estimate"] = 71848.0 / 118772.0
# intraday spread and profitable swings of w itself (same definitions as Q4/Q5)
import signals as S  # noqa: E402

ds_w = S.daily_stats(df, "w")
sw_w = S.swings_per_day(df, "w")
stats["w_daily"] = {
    "ratio_median": float(ds_w["ratio"].median()), "ratio_min": float(ds_w["ratio"].min()),
    "ratio_p10": float(ds_w["ratio"].quantile(0.1)), "ratio_p90": float(ds_w["ratio"].quantile(0.9)),
    "share_days_above_break_even": float((ds_w["ratio"] > C.BREAK_EVEN).mean()),
    "ratio4_median": float(ds_w["ratio4"].median()), "ratio4_min": float(ds_w["ratio4"].min()),
    "share_days_ratio4_above_1p5": float((ds_w["ratio4"] > 1.5).mean()),
    "ratio_median_by_month": {C.MONTH_ABBR[m]: float(v) for m, v in ds_w.groupby("month")["ratio"].median().items()},
    "swings_per_day_mean": float(sw_w["swings"].mean()),
    "swings_per_day_distribution": {int(k): int(v) for k, v in sw_w["swings"].value_counts().sort_index().items()},
    "share_days_two_or_more_swings": float((sw_w["swings"] >= 2).mean()),
    "swings_per_day_mean_thr_1.25": float(S.swings_per_day(df, "w", 1.25)["swings"].mean()),
    "swings_per_day_mean_thr_1.5": float(S.swings_per_day(df, "w", 1.5)["swings"].mean()),
}
C.write_stats("q6_agreement", stats)

# ================================================================== FIGURE 1
fig = plt.figure(figsize=(13.5, 6.7))
wd = stats["within_day_spearman"]
C.titled(fig, "Price and CO₂ agree on the midday trough and the evening peak, and disagree at night: "
              "cheap power, dirty grid",
         "Mean daily profile of each signal divided by its yearly mean (1.0 = average hour), and the combined score signal "
         "w = 0.7 × CO₂ + 0.3 × price.")
gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.79, bottom=0.14, wspace=0.20, width_ratios=[1.2, 1.2, 1])
hours = np.arange(24)
for i, (flag, name) in enumerate(((False, "Weekdays"), (True, "Weekends"))):
    ax = fig.add_subplot(gs[i])
    p_ = prof.loc[flag]
    ax.axhline(1.0, color=C.AXIS, lw=1.0)
    ax.plot(hours, p_["co2_rel"], color=C.C_CO2, lw=2.0, label="CO₂ intensity")
    ax.plot(hours, p_["price_rel"], color=C.C_PRICE, lw=2.0, label="PVPC price")
    ax.plot(hours, p_["w"], color=C.C_W, lw=3.0, label="Combined signal w")
    ax.set_ylim(0.2, 2.0)
    C.hour_axis(ax, step=3)
    ax.set_title(name)
    if i == 0:
        ax.set_ylabel("Relative to the yearly mean (= 1.0)")
        ax.legend(loc="upper left", fontsize=10.5, ncol=1, handlelength=1.6)
        ax.annotate("night: price below average,\nCO₂ over 20 % above", xy=(3, 1.22), xytext=(0.2, 0.52), fontsize=10,
                    color=C.INK2, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8, shrinkB=3))
        ax.annotate("peak tariff,\nclean grid", xy=(11.5, 1.10), xytext=(11.2, 1.52), fontsize=10, color=C.INK2,
                    ha="center", va="center", arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8, shrinkB=3))
        ax.annotate("both low", xy=(15, 0.62), xytext=(15.2, 0.36), fontsize=10, color=C.INK2, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8, shrinkB=3))
        ax.annotate("both high", xy=(20.3, 1.8), xytext=(16.6, 1.88), fontsize=10, color=C.INK2, ha="center",
                    va="center", arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8, shrinkB=3))
    else:
        ax.set_yticklabels([])
        ax.text(23.2, p_["co2_rel"].iloc[-1] + 0.07, "CO₂", ha="right", fontsize=10.5, color=C.INK, fontweight="bold")
        ax.text(23.2, p_["price_rel"].iloc[-1] - 0.11, "Price", ha="right", fontsize=10.5, color=C.INK, fontweight="bold")
        ax.text(13, p_["w"].loc[13] + 0.06, "w", ha="center", fontsize=10.5, color=C.INK, fontweight="bold")

ax = fig.add_subplot(gs[2])
bins = np.linspace(-1, 1, 21)
ax.hist([rho[~wkend], rho[wkend]], bins=bins, stacked=True, color=[C.C_WEEKDAY, C.C_WEEKEND],
        label=["Weekdays", "Weekends"], edgecolor=C.SURFACE, linewidth=0.8)
ax.axvline(0, color=C.AXIS, lw=1.0)
ax.set_xlabel("Rank correlation of the 24 hourly values")
ax.set_ylabel("Days")
ax.set_title(f"Per-day agreement: median ρ = {wd['median']:.2f}")
ax.legend(loc="upper left", fontsize=10.5)
ax.text(-0.97, ax.get_ylim()[1] * 0.60,
        f"weekdays {wd['median_weekday']:.2f}\nweekends {wd['median_weekend']:.2f}\n"
        f"negative on {wd['share_negative']:.0%} of days", fontsize=10.5, color=C.INK2, va="top")
C.footnote(fig, f"Sample: all {N} train days ({(~wkend).sum()} weekdays, {wkend.sum()} weekend days). Spearman rank correlation "
                f"between the day's 24 CO₂ values and its 24 prices. Over all {len(df):,} hours: Pearson "
                f"{stats['hourly_pearson']:.2f}.")
C.save(fig, "q6_agreement")

# ================================================================== FIGURE 2
vs = stats["w_within_day_variance_shares"]
fig = plt.figure(figsize=(13.5, 6.4))
C.titled(fig, f"Inside a day, the CO₂ term drives the score signal: {vs['all']['co2']:.0%} of its variance, "
              f"against {vs['all']['price']:.0%} for price alone",
         "Variance of w around its daily mean, split into the CO₂ term (0.7 × CO₂/mean), the price term "
         "(0.3 × price/mean) and their co-movement.")
gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.79, bottom=0.19, wspace=0.22, width_ratios=[1.25, 1])
ax = fig.add_subplot(gs[0])
labels = ["All\ndays", "Mon–\nFri", "Sat–\nSun"] + [C.MONTH_ABBR[m] for m in C.MONTH_ORDER]
rows = [vs["all"], vs["weekday"], vs["weekend"]] + [vs["by_month"][C.MONTH_ABBR[m]] for m in C.MONTH_ORDER]
x = np.r_[0, 1.1, 2.2, np.arange(12) + 3.9]
parts = [("co2", C.C_CO2, "CO₂ term alone"), ("covariance", C.GRAY, "Co-movement of the two"),
         ("price", C.C_PRICE, "Price term alone")]
bottom = np.zeros(len(rows))
for key, col, name in parts:
    v = np.array([r[key] for r in rows]) * 100
    ax.bar(x, v, bottom=bottom, width=0.7, color=col, edgecolor=C.SURFACE, linewidth=1.5, label=name)
    for xi, vi, bi in zip(x[:3], v[:3], bottom[:3]):
        ax.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center", fontsize=10.5,
                color="white" if key != "covariance" else C.INK)
    bottom += v
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25", "50", "75", "100 %"])
ax.set_ylabel("Share of the within-day variance of w")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.5, -0.115), handlelength=1.0, columnspacing=1.6)
ax.set_title("Variance split, overall and by month")

ax2 = fig.add_subplot(gs[1])
ax2.scatter(r_a[~wkend], r_b[~wkend], s=22, color=C.C_WEEKDAY, alpha=0.7, edgecolor=C.SURFACE, linewidth=0.5,
            label="Weekdays")
ax2.scatter(r_a[wkend], r_b[wkend], s=22, color=C.C_WEEKEND, alpha=0.9, edgecolor=C.SURFACE, linewidth=0.5,
            label="Weekends")
lim = max(r_a.max(), r_b.max()) * 1.05
ax2.plot([0, lim], [0, lim], color=C.MUTED, lw=1.0)
ax2.text(lim * 0.70, lim * 0.74, "equal swing", rotation=45, color=C.MUTED, fontsize=10, ha="center", va="center",
         rotation_mode="anchor")
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel("Daily swing of the CO₂ term (max − min)")
ax2.set_ylabel("Daily swing of the price term (max − min)")
ax2.set_title(f"CO₂ swings more on {stats['share_days_co2_term_range_larger']:.0%} of days")
ax2.legend(loc="upper left", fontsize=10.5, handletextpad=0.2)
C.footnote(fig, f"Sample: all {N} train days. w is scaled so that its import-weighted mean is 1; the weights come from the "
                f"train baseline ({base['co2_kg'] / 1000:,.0f} t CO₂, {base['cost_eur'] / 1000:,.0f} k€, "
                f"{base['kg_per_eur']:.3f} kg/€).")
C.save(fig, "q6_w_decomposition")

print({k: v for k, v in stats.items() if not isinstance(v, dict)})
print(stats["within_day_spearman"])
print(stats["w_within_day_variance_shares"]["all"], stats["same_hour"], stats["overlap_of_4_hours_out_of_4"])
