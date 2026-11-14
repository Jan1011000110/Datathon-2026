"""
Q2 - Campus load.
  fig q2_load_profiles.png   : daily profile weekday vs weekend + hour x month heatmap (weekdays)
  fig q2_load_calendar.png   : daily mean load over the year, holidays / closures annotated
  fig q2_net_load_vs_500kw.png : distribution of net load against the battery power limit
  stats q2_load.json
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

C.set_style()
stats: dict = {}

full = C.load_hourly_full()
off = C.load_hourly_official()
load = full["load_kw"].dropna()
weekend = pd.Series(C.is_weekend(load.index), index=load.index)

# hours where the meter reports a comms fault (reads exactly 0): excluded from profiles
valid = load[load > 0]
stats["hours_load_available"] = int(len(load))
stats["hours_load_valid_nonzero"] = int(len(valid))
stats["load_mean_kw"] = float(valid.mean())
stats["load_median_kw"] = float(valid.median())
stats["load_p01_p99_kw"] = [float(valid.quantile(0.01)), float(valid.quantile(0.99))]
stats["load_max_kw"] = float(valid.max())
stats["load_min_nonzero_kw"] = float(valid.min())

# public holidays / closures that fall on train weekdays (Catalonia / Barcelona calendar)
HOLIDAYS = {
    "2025-09-11": "Diada", "2025-12-08": "Immaculada", "2026-01-01": "New Year", "2026-01-06": "Reis",
    "2026-04-03": "Good Friday", "2026-04-06": "Easter Monday", "2026-05-01": "1 May",
}
hol_days = pd.to_datetime(list(HOLIDAYS))

# ------------------------------------------------------------------ daily table
by_day = valid.groupby(valid.index.normalize())
daily = pd.DataFrame({"mean_kw": by_day.mean(), "n": by_day.size()})
daily = daily[daily["n"] >= 20]
daily["weekend"] = C.is_weekend(daily.index)
daily["holiday"] = daily.index.isin(hol_days)
daily["month"] = daily.index.month
stats["days_with_load"] = int(len(daily))

wd = daily[~daily["weekend"]]
we = daily[daily["weekend"]]
stats["weekday_mean_kw"] = float(wd["mean_kw"].mean())
stats["weekend_mean_kw"] = float(we["mean_kw"].mean())
stats["weekend_over_weekday"] = float(we["mean_kw"].mean() / wd["mean_kw"].mean())
stats["monthly_weekday_mean_kw"] = {C.MONTH_ABBR[m]: float(v) for m, v in wd.groupby("month")["mean_kw"].mean().items()}
stats["monthly_weekend_mean_kw"] = {C.MONTH_ABBR[m]: float(v) for m, v in we.groupby("month")["mean_kw"].mean().items()}
stats["holiday_daily_mean_kw"] = {f"{d.date()} ({HOLIDAYS[str(d.date())]})": float(daily.loc[d, "mean_kw"])
                                  for d in hol_days if d in daily.index}
normal_wd = wd[~wd["holiday"]]
aug = normal_wd[(normal_wd.index >= "2026-08-01") & (normal_wd.index <= "2026-08-23")]
stats["august_closure_weekday_mean_kw"] = float(aug["mean_kw"].mean())
stats["aug24_mean_kw"] = float(daily.loc[pd.Timestamp("2026-08-24"), "mean_kw"])
xmas = daily[(daily.index >= "2026-01-02") & (daily.index <= "2026-01-05") & ~daily["weekend"]]
stats["christmas_closure_weekday_mean_kw_jan2_5"] = float(xmas["mean_kw"].mean())
jul = normal_wd[normal_wd["month"] == 7]
stats["july_weekday_mean_kw"] = float(jul["mean_kw"].mean())
spring = normal_wd[normal_wd["month"].isin([3, 4, 5])]
stats["mar_may_weekday_mean_kw"] = float(spring["mean_kw"].mean())

# hourly profile (frozen-night hours flagged)
prof = pd.DataFrame({"load": valid, "hour": valid.index.hour, "weekend": weekend[valid.index].values,
                     "month": valid.index.month})
g = prof.groupby(["weekend", "hour"])["load"]
p_mean, p_lo, p_hi = g.mean().unstack(0), g.quantile(0.10).unstack(0), g.quantile(0.90).unstack(0)
stats["weekday_profile_peak_kw"] = float(p_mean[False].max())
stats["weekday_profile_peak_hour"] = int(p_mean[False].idxmax())
stats["weekday_profile_min_kw"] = float(p_mean[False].min())
stats["weekend_profile_range_kw"] = [float(p_mean[True].min()), float(p_mean[True].max())]

# ================================================================== FIGURE: profiles
fig = plt.figure(figsize=(13.5, 6.6))
C.titled(fig, "Load is an office-hours hump on a ~750 kW base: weekdays peak near 1,500 kW, weekends stay flat",
         "Hourly campus load from the meter (kW). Hours when the meter reports a communication fault (0 kW) are excluded.")
gs = fig.add_gridspec(1, 2, left=0.065, right=0.945, top=0.80, bottom=0.14, wspace=0.20, width_ratios=[1, 1.12])
ax = fig.add_subplot(gs[0])
hours = np.arange(24)
for flag, col, name in ((False, C.C_WEEKDAY, "Weekdays"), (True, C.C_WEEKEND, "Weekends")):
    ax.fill_between(hours, p_lo[flag], p_hi[flag], color=col, alpha=0.14, lw=0)
    ax.plot(hours, p_mean[flag], color=col, lw=2.2, label=f"{name}: mean, with 10–90 % band")
ax.text(13.2, p_mean[False].max() + 60, "Weekdays", color=C.INK, fontsize=11.5, ha="center", fontweight="bold")
ax.text(13.2, p_mean[True].loc[13] - 115, "Weekends", color=C.INK, fontsize=11.5, ha="center", fontweight="bold")
for x0, x1 in ((-0.5, 5.5), (21.5, 23.5)):
    ax.axvspan(x0, x1, color=C.GRID, alpha=0.6, lw=0)
ax.text(2.5, 2290, "22:00–06:00\nmeter not polled:\nvalue held", ha="center", va="top", fontsize=9.5, color=C.INK2)
ax.set_ylim(0, 2350)
ax.set_ylabel("Load (kW)")
C.hour_axis(ax)
ax.set_title("Mean daily profile, all months")
ax.legend(loc="lower right", fontsize=10.5)

ax2 = fig.add_subplot(gs[1])
wd_prof = prof[~prof["weekend"]].groupby(["month", "hour"])["load"].mean().unstack("hour")
mat = wd_prof.reindex(C.MONTH_ORDER)
im = ax2.imshow(mat.values, aspect="auto", cmap=C.SEQ_VIOLET, vmin=600, vmax=2200)
ax2.set_yticks(range(12))
ax2.set_yticklabels([C.MONTH_ABBR[m] for m in C.MONTH_ORDER])
ax2.set_xticks(range(0, 24, 3))
ax2.set_xlabel("Hour of day (local time)")
ax2.grid(False)
ax2.tick_params(length=0)
for sp in ax2.spines.values():
    sp.set_visible(False)
ax2.set_xticks(np.arange(-0.5, 24, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, 12, 1), minor=True)
ax2.grid(which="minor", color=C.SURFACE, linewidth=1.5)
ax2.tick_params(which="minor", length=0)
ax2.text(11.5, 2, "no meter data", ha="center", va="center", fontsize=10.5, color=C.MUTED)
cb = fig.colorbar(im, ax=ax2, pad=0.02, fraction=0.045)
cb.set_label("Mean weekday load (kW)")
cb.outline.set_visible(False)
ax2.set_title("Weekdays by month: summer cooling lifts the peak, August is closed")
C.footnote(fig, f"Sample: {len(valid):,} train hours with meter data ({len(daily)} days; November 2025 missing). "
                "Night values (grey band) are the 22:00 reading repeated, see the coverage section.")
C.save(fig, "q2_load_profiles")

# ================================================================== FIGURE: year calendar of daily means
fig = plt.figure(figsize=(13.5, 6.4))
C.titled(fig, "Holidays and closures drop a weekday to weekend level; summer cooling adds about 40 %",
         "Daily mean campus load (kW), one dot per train day. Grey bands are the hidden test weeks.")
ax = fig.add_axes([0.065, 0.13, 0.915, 0.67])
for d0 in C.test_days()[::7]:
    ax.axvspan(d0, d0 + pd.Timedelta(days=7), color=C.GRID, alpha=0.75, lw=0)
m_wd = ~daily["weekend"] & ~daily["holiday"]
ax.scatter(daily.index[m_wd] + pd.Timedelta(hours=12), daily.loc[m_wd, "mean_kw"], s=30, color=C.C_WEEKDAY,
           edgecolor=C.SURFACE, linewidth=0.7, label="Weekday", zorder=3)
ax.scatter(daily.index[daily["weekend"]] + pd.Timedelta(hours=12), daily.loc[daily["weekend"], "mean_kw"], s=30,
           color=C.C_WEEKEND, edgecolor=C.SURFACE, linewidth=0.7, label="Saturday / Sunday", zorder=3)
ax.scatter(daily.index[daily["holiday"]] + pd.Timedelta(hours=12), daily.loc[daily["holiday"], "mean_kw"], s=46,
           facecolor=C.SURFACE, edgecolor=C.C_WEEKDAY, linewidth=1.8, label="Public holiday on a weekday", zorder=4)
ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-24"), facecolor="none", edgecolor=C.AXIS, hatch="///", lw=0)
ax.text(pd.Timestamp("2025-11-12 12:00"), 1250, "November:\nno meter data", ha="center", va="center", fontsize=10.5,
        color=C.INK2, bbox=dict(facecolor=C.SURFACE, edgecolor="none", pad=2))


def lab(text, x, y, xt, yt, ha="center"):
    ax.annotate(text, xy=(pd.Timestamp(x), y), xytext=(pd.Timestamp(xt), yt), fontsize=10, color=C.INK2, ha=ha,
                va="center", arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8, shrinkA=2, shrinkB=4))


lab("Diada\n11 Sep", "2025-09-11 12:00", 835, "2025-09-13", 520)
lab("8 Dec", "2025-12-08 12:00", 764, "2025-12-08", 560)
lab("Christmas closure\nuntil 6 Jan", "2026-01-03 12:00", 850, "2026-01-06", 520)
lab("12 Feb\n(unexplained)", "2026-02-12 12:00", 964, "2026-02-12", 560)
lab("Easter week\n1–6 Apr", "2026-04-04", 760, "2026-04-04", 500)
lab("1 May", "2026-05-01 12:00", 819, "2026-05-01", 580)
lab("Meter fault\n17–18 Jun", "2026-06-17 20:00", 1100, "2026-06-07", 1150)
lab("August closure:\nweekdays ≈ 870 kW", "2026-08-12", 890, "2026-08-12", 520)
lab("24 Aug:\nback to normal", "2026-08-24 12:00", 1223, "2026-08-14", 1420)
ax.text(pd.Timestamp("2026-07-13"), 1820, "Summer cooling", ha="center", fontsize=10.5, color=C.INK2)
ax.set_ylim(380, 1900)
ax.set_xlim(pd.Timestamp("2025-08-28"), pd.Timestamp("2026-09-03"))
ax.set_ylabel("Daily mean load (kW)")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
for lbl in ax.get_xticklabels():
    lbl.set_ha("left")
ax.legend(loc="upper left", ncol=3, fontsize=10.5, handletextpad=0.2, columnspacing=1.6, bbox_to_anchor=(0.0, 1.075),
          borderaxespad=0)
C.footnote(fig, f"Sample: {len(daily)} train days with at least 20 valid meter hours. Weekday mean "
                f"{stats['weekday_mean_kw']:,.0f} kW, weekend mean {stats['weekend_mean_kw']:,.0f} kW. "
                "Night hours carry the held 22:00 value, which inflates weekday means by roughly 6 %.")
C.save(fig, "q2_load_calendar")

# ================================================================== net load vs 500 kW
net = (off["load_kw"] - off["pv_kw"])
below = net < C.P_MAX
stats["official_hours"] = int(len(net))
stats["net_below_500_hours"] = int(below.sum())
stats["net_below_500_share"] = float(below.mean())
zero = off["load_kw"] == 0
stats["net_below_500_meter_fault_hours"] = int((below & zero).sum())
stats["net_below_500_other_hours"] = int((below & ~zero).sum())
stats["net_below_500_other_list"] = {str(t): float(v) for t, v in net[below & ~zero].items()}
stats["net_min_excluding_faults_kw"] = float(net[~zero].min())
stats["net_p01_kw"] = float(net[~zero].quantile(0.01))
stats["net_p05_kw"] = float(net[~zero].quantile(0.05))
stats["net_median_kw"] = float(net.median())
stats["net_negative_hours"] = int((net < 0).sum())
stats["wasted_if_always_500_discharge_share_of_energy"] = float(
    (C.P_MAX - net.clip(upper=C.P_MAX)).clip(lower=0).sum() / (C.P_MAX * len(net)))
# persistence of the fault state: P(load_t = 0 | load_{t-1} = 0)
l_full = full["load_kw"]
prev_zero = (l_full.shift(1) == 0) & l_full.notna() & (l_full.index.to_series().diff() == pd.Timedelta("1h")).values
stats["p_zero_given_prev_zero"] = float((l_full[prev_zero] == 0).mean())
stats["n_prev_zero"] = int(prev_zero.sum())

fig = plt.figure(figsize=(13.5, 6.2))
C.titled(fig, f"Net load sits above the 500 kW battery limit in {1 - below.mean():.1%} of hours: discharge is almost never capped by demand",
         "Distribution of hourly net load (load − PV) in the organisers' table. A discharge larger than the net load "
         "is exported for free and wasted.")
ax = fig.add_axes([0.065, 0.14, 0.915, 0.66])
bins = np.arange(0, 2601, 50)
wk = ~C.is_weekend(net.index)
ax.hist([net[wk], net[~wk]], bins=bins, stacked=True, color=[C.C_WEEKDAY, C.C_WEEKEND],
        label=["Weekday hours", "Weekend hours"], edgecolor=C.SURFACE, linewidth=0.8)
ax.axvline(C.P_MAX, color=C.INK, lw=1.6)
ymax = ax.get_ylim()[1]
ax.text(C.P_MAX + 18, ymax * 0.97, "Battery power limit\n500 kW", color=C.INK, fontsize=11, va="top", ha="left")
ax.annotate(f"{int(zero.sum())} hours at 0 kW:\nmeter communication faults\n(17–18 Jun, 6 Aug)",
            xy=(25, int(zero.sum()) + 2), xytext=(60, ymax * 0.36), fontsize=10.5, color=C.INK2, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8, shrinkB=2))
ax.text(2570, ymax * 0.76,
        f"Below 500 kW: {int(below.sum())} of {len(net):,} hours ({below.mean():.2%})\n"
        f"of which meter faults: {int((below & zero).sum())}\n"
        f"Lowest real value: {stats['net_min_excluding_faults_kw']:.0f} kW\n"
        f"1st percentile: {stats['net_p01_kw']:.0f} kW",
        fontsize=11, color=C.INK, ha="right", va="top", linespacing=1.5)
ax.set_xlabel("Net load = load − PV (kW)")
ax.set_ylabel("Hours")
ax.set_xlim(0, 2600)
ax.legend(loc="upper right", fontsize=11)
C.footnote(fig, f"Sample: the {len(net):,} train hours of the organisers' table (load and PV both present). "
                "Net load is never negative, so there is no free solar surplus to store.")
C.save(fig, "q2_net_load_vs_500kw")

C.write_stats("q2_load", stats)
print({k: v for k, v in stats.items() if not isinstance(v, (dict, list))})
print(stats["monthly_weekday_mean_kw"])
print(stats["holiday_daily_mean_kw"])
print(stats["net_below_500_other_list"])
