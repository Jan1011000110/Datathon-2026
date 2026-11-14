"""
Q4 (CO2 intensity) and Q5 (PVPC price): level, daily shape, intraday spread, profitable cycles.
  fig q4_co2_profile.png, q4_co2_spread.png
  fig q5_price_profile.png, q5_price_spread.png
  stats q4_co2.json, q5_price.json
All 282 train days are used (both signals are complete).
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import common as C
import signals as S

C.set_style()
df = C.signals_frame()
N_DAYS = df["date"].nunique()


def heat(ax, mat, cmap, vmin, vmax, months=C.MONTH_ORDER, xstep=3):
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(months)))
    ax.set_yticklabels([C.MONTH_ABBR[m] for m in months])
    ax.set_xticks(range(0, 24, xstep))
    ax.set_xlabel("Hour of day (local time)")
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(months), 1), minor=True)
    ax.grid(which="minor", color=C.SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    return im


def common_stats(key: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = C.SIGNALS[key]
    col, sc = cfg["col"], cfg["scale"]
    s = df[col] * sc
    ds = S.daily_stats(df, col)
    sw = S.swings_per_day(df, col)
    st_tab = S.swing_table(df, col)
    stats = {
        "unit": cfg["unit"], "days": int(len(ds)), "hours": int(len(s)),
        "mean": float(s.mean()), "median": float(s.median()), "min": float(s.min()), "max": float(s.max()),
        "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95)),
        "monthly_mean": {C.MONTH_ABBR[m]: float(v) for m, v in s.groupby(df["month"]).mean().items()},
        "hourly_mean": {int(h): float(v) for h, v in s.groupby(df["hour"]).mean().items()},
        "weekday_mean": float(s[~df["weekend"]].mean()), "weekend_mean": float(s[df["weekend"]].mean()),
        "daily_mean_cv": float(ds["mean"].std() / ds["mean"].mean()),
        "daily_mean_p10_p90": [float(ds["mean"].quantile(0.1) * sc), float(ds["mean"].quantile(0.9) * sc)],
        "share_var_between_days": float(ds["mean"].var(ddof=0) / (df[col].var(ddof=0))),
        "ratio_median": float(ds["ratio"].median()), "ratio_min": float(ds["ratio"].min()),
        "ratio_p10": float(ds["ratio"].quantile(0.1)), "ratio_p90": float(ds["ratio"].quantile(0.9)),
        "ratio_max": float(ds["ratio"].max()),
        "share_days_ratio_above_break_even": float((ds["ratio"] > C.BREAK_EVEN).mean()),
        "share_days_ordered_ratio_above_break_even": float((ds["ratio_ordered"] > C.BREAK_EVEN).mean()),
        "ratio4_median": float(ds["ratio4"].median()), "ratio4_min": float(ds["ratio4"].min()),
        "share_days_ratio4_above_break_even": float((ds["ratio4"] > C.BREAK_EVEN).mean()),
        "share_days_ratio4_above_1p5": float((ds["ratio4"] > 1.5).mean()),
        "ratio_median_by_month": {C.MONTH_ABBR[m]: float(v) for m, v in ds.groupby("month")["ratio"].median().items()},
        "ratio4_median_by_month": {C.MONTH_ABBR[m]: float(v) for m, v in ds.groupby("month")["ratio4"].median().items()},
        "spread_median": float(ds["spread"].median() * sc),
        "swings_per_day_mean": float(sw["swings"].mean()),
        "swings_per_day_distribution": {int(k): int(v) for k, v in sw["swings"].value_counts().sort_index().items()},
        "share_days_two_or_more_swings": float((sw["swings"] >= 2).mean()),
        "swings_per_day_mean_by_month": {C.MONTH_ABBR[m]: float(v) for m, v in
                                         sw.groupby(sw.index.month)["swings"].mean().items()},
        "swing_ratio_median": float(st_tab["ratio"].median()),
        "n_swings": int(len(st_tab)),
    }
    # the same count with a stricter threshold: swings that leave a real margin after losses
    for thr in (1.25, 1.5):
        sw2 = S.swings_per_day(df, col, thr)
        stats[f"swings_per_day_mean_thr_{thr}"] = float(sw2["swings"].mean())
        stats[f"share_days_two_or_more_swings_thr_{thr}"] = float((sw2["swings"] >= 2).mean())
    return stats, ds, sw, st_tab


def spread_figure(key: str, stats: dict, ds: pd.DataFrame, sw: pd.DataFrame, title: str, subtitle: str,
                  extra_panel=None):
    cfg = C.SIGNALS[key]
    col, sc, color = cfg["col"], cfg["scale"], cfg["color"]
    fig = plt.figure(figsize=(13.5, 7.6))
    C.titled(fig, title, subtitle, y=0.978)
    n_bottom = 3 if extra_panel else 2
    gs = fig.add_gridspec(2, n_bottom, left=0.065, right=0.965 if extra_panel else 0.98, top=0.835, bottom=0.135,
                          hspace=0.52, wspace=0.24, height_ratios=[1, 1.05])
    # ---- top: daily range over the year
    ax = fig.add_subplot(gs[0, :])
    blocks = C.block_id(ds.index)
    for b in np.unique(blocks):
        d = ds[blocks == b]
        x = d.index + pd.Timedelta(hours=12)
        ax.fill_between(x, d["min"] * sc, d["max"] * sc, color=color, alpha=0.22, lw=0,
                        label="Range between the day's lowest and highest hour" if b == 0 else None)
        ax.plot(x, d["mean"] * sc, color=color, lw=1.8, label="Daily mean" if b == 0 else None)
    ax.set_ylim(0, None)
    ax.set_xlim(pd.Timestamp("2025-08-29"), pd.Timestamp("2026-09-03"))
    ax.set_ylabel(cfg["unit"])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for lbl in ax.get_xticklabels():
        lbl.set_ha("left")
    ax.legend(loc="upper left", ncol=2, fontsize=10.5, bbox_to_anchor=(0.0, 1.17), borderaxespad=0, handlelength=1.6)
    ax.set_title("Level and daily range, day by day (gaps = hidden test weeks)", loc="right", fontweight="normal",
                 fontsize=10.5, color=C.INK2)

    # ---- bottom-left: max/min ratio per day by month
    ax1 = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    for i, m in enumerate(C.MONTH_ORDER):
        r = ds.loc[ds["month"] == m, "ratio"].values
        ax1.scatter(i + rng.uniform(-0.22, 0.22, len(r)), r, s=13, color=color, alpha=0.55, edgecolor="none")
        ax1.plot([i - 0.33, i + 0.33], [np.median(r)] * 2, color=C.INK, lw=2.0, solid_capstyle="butt")
    ax1.axhline(C.BREAK_EVEN, color=C.INK, lw=1.2)
    ax1.set_yscale("log")
    top = max(8, ds["ratio"].max() * 1.15)
    ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 50) if t <= top]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"{t:g}×" for t in ticks])
    ax1.minorticks_off()
    ax1.set_ylim(0.95, top)
    ax1.text(11.5, C.BREAK_EVEN * 1.03, "break-even 1.11×", ha="right", va="bottom", fontsize=10, color=C.INK)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels([C.MONTH_ABBR[m][0] for m in C.MONTH_ORDER])
    ax1.set_xlim(-0.6, 11.6)
    ax1.grid(axis="x", visible=False)
    ax1.set_ylabel("Highest hour / lowest hour")
    ax1.set_title(f"Daily max/min ratio: median {stats['ratio_median']:.1f}×")

    # ---- bottom-middle: profitable swings per day by month
    ax2 = fig.add_subplot(gs[1, 1])
    cats = [("0–1 swing", lambda v: v <= 1), ("2 swings", lambda v: v == 2), ("3 or more", lambda v: v >= 3)]
    shades = [cfg["cmap"](0.28), cfg["cmap"](0.58), cfg["cmap"](0.88)]
    bottom = np.zeros(12)
    for (name, f), shade in zip(cats, shades):
        share = np.array([f(sw.loc[sw.index.month == m, "swings"]).mean() for m in C.MONTH_ORDER]) * 100
        ax2.bar(range(12), share, bottom=bottom, width=0.72, color=shade, edgecolor=C.SURFACE, linewidth=1.5, label=name)
        bottom += share
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0", "25", "50", "75", "100 %"])
    ax2.set_xticks(range(12))
    ax2.set_xticklabels([C.MONTH_ABBR[m][0] for m in C.MONTH_ORDER])
    ax2.set_xlim(-0.6, 11.6)
    ax2.grid(axis="x", visible=False)
    ax2.set_ylabel("Share of days")
    ax2.set_title(f"Profitable swings per day: {stats['swings_per_day_mean']:.1f} on average")
    ax2.legend(loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.12), handlelength=1.0,
               columnspacing=1.2, borderaxespad=0)
    if extra_panel:
        extra_panel(fig.add_subplot(gs[1, 2]))
    return fig


# =====================================================================================  CO2
key = "co2"
stats, ds, sw, st_tab = common_stats(key)
s = df["co2_kg_per_kwh"] * 1000
mat = s.groupby([df["month"], df["hour"]]).mean().unstack().reindex(C.MONTH_ORDER)
stats["hour_month_mean_min"] = float(mat.values.min())
stats["hour_month_mean_max"] = float(mat.values.max())
hm = s.groupby(df["hour"]).mean()
stats["mean_midday_10_16"] = float(hm.loc[10:16].mean())
stats["mean_night_20_08"] = float(pd.concat([hm.loc[20:23], hm.loc[0:7]]).mean())
stats["night_over_midday"] = stats["mean_night_20_08"] / stats["mean_midday_10_16"]
stats["co2_is_0p37_times_emitting_share"] = bool(np.allclose(df["co2_kg_per_kwh"], 0.37 * df["pct_emitting"]))
# what drives the midday trough: emitting generation falls or clean generation rises?
g = df.groupby("hour")[["gen_emitting_mw", "gen_clean_mw"]].mean()
stats["emitting_gw_midday_vs_night"] = [float(g.loc[10:16, "gen_emitting_mw"].mean() / 1000),
                                        float(pd.concat([g.loc[20:23], g.loc[0:7]])["gen_emitting_mw"].mean() / 1000)]
stats["clean_gw_midday_vs_night"] = [float(g.loc[10:16, "gen_clean_mw"].mean() / 1000),
                                     float(pd.concat([g.loc[20:23], g.loc[0:7]])["gen_clean_mw"].mean() / 1000)]
C.write_stats("q4_co2", stats)

# ---- figure: profile
fig = plt.figure(figsize=(13.5, 6.6))
C.titled(fig, f"Grid CO₂ has one deep trough at solar midday: nights are {stats['night_over_midday']:.1f}× dirtier "
              "than 10:00–16:00",
         "Mean CO₂ intensity of the Spanish peninsular mix (g CO₂/kWh). The signal is exactly 0.37 kg/kWh × the share of "
         "emitting generation.")
gs = fig.add_gridspec(2, 4, left=0.06, right=0.985, top=0.80, bottom=0.13, wspace=0.12, hspace=0.42,
                      width_ratios=[1.85, 0.07, 1, 1])
ax = fig.add_subplot(gs[:, 0])
im = heat(ax, mat, C.SEQ_BLUE, 30, 160)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
cb.outline.set_visible(False)
ax.set_title("Mean by month and hour (g CO₂/kWh)")
hours = np.arange(24)
first = None
for i, season in enumerate(C.SEASON_ORDER):
    axs = fig.add_subplot(gs[i // 2, 2 + i % 2], sharey=first)
    first = first or axs
    dsea = df[[C.SEASON_OF_MONTH[m] == season for m in df["month"]]]
    x = (dsea["co2_kg_per_kwh"] * 1000).groupby(dsea["hour"])
    axs.fill_between(hours, x.quantile(0.1), x.quantile(0.9), color=C.C_CO2, alpha=0.18, lw=0)
    axs.plot(hours, x.mean(), color=C.C_CO2, lw=2.0)
    months = sorted({m for m in dsea["month"]}, key=C.MONTH_ORDER.index)
    axs.set_title(f"{season}  ({C.MONTH_ABBR[months[0]]}–{C.MONTH_ABBR[months[-1]]})", fontsize=11.5)
    axs.set_xticks([0, 6, 12, 18])
    axs.set_xlim(0, 23)
    axs.set_ylim(0, 200)
    axs.set_yticks([0, 50, 100, 150, 200])
    if i % 2 == 1:
        plt.setp(axs.get_yticklabels(), visible=False)
    if i // 2 == 1:
        axs.set_xlabel("Hour of day")
fig.text(0.985, 0.842, "Seasonal profiles: mean with 10–90 % band", ha="right", fontsize=10.5, color=C.INK2)
C.footnote(fig, f"Sample: all {N_DAYS} train days ({len(df):,} hours), Sep 2025 – Aug 2026 without the last 7 days of "
                "each month.")
C.save(fig, "q4_co2_profile")

# ---- figure: spread
fig = spread_figure(
    key, stats, ds, sw,
    f"Every one of the {N_DAYS} train days offers a profitable CO₂ cycle: the dirtiest hour is "
    f"{stats['ratio_median']:.1f}× the cleanest (median)",
    "A charge→discharge cycle pays off when the dear hour is more than 1/0.9 = 1.11× the cheap hour. "
    f"Lowest daily ratio in train: {stats['ratio_min']:.2f}×.")
C.footnote(fig, f"Sample: all {N_DAYS} train days. A swing = a rise of at least 11.1 % from a trough to the next peak "
                "(zigzag on the hourly series, attributed to the day of the peak).")
C.save(fig, "q4_co2_spread")

# =====================================================================================  PRICE
key = "price"
stats, ds, sw, st_tab = common_stats(key)
p = df["precio_eur_per_kwh"] * 100
wd, we = df[~df["weekend"]], df[df["weekend"]]
mat_wd = (wd["precio_eur_per_kwh"] * 100).groupby([wd["month"], wd["hour"]]).mean().unstack().reindex(C.MONTH_ORDER)
mat_we = (we["precio_eur_per_kwh"] * 100).groupby([we["month"], we["hour"]]).mean().unstack().reindex(C.MONTH_ORDER)
prof_wd = (wd["precio_eur_per_kwh"] * 100).groupby(wd["hour"])
prof_we = (we["precio_eur_per_kwh"] * 100).groupby(we["hour"])
stats["weekday_hourly_mean"] = {int(h): float(v) for h, v in prof_wd.mean().items()}
stats["weekend_hourly_mean"] = {int(h): float(v) for h, v in prof_we.mean().items()}
stats["weekday_peak_18_22_mean"] = float(prof_wd.mean().loc[18:21].mean())
stats["weekday_flat_14_18_mean"] = float(prof_wd.mean().loc[14:17].mean())
stats["weekday_valley_0_8_mean"] = float(prof_wd.mean().loc[0:7].mean())
stats["weekday_peak_10_14_mean"] = float(prof_wd.mean().loc[10:13].mean())
stats["weekend_midday_10_17_mean"] = float(prof_we.mean().loc[10:16].mean())
stats["weekend_evening_20_22_mean"] = float(prof_we.mean().loc[20:21].mean())
# tariff steps on weekdays: median hour-to-hour change at the period boundaries
dstep = (wd["precio_eur_per_kwh"].diff() * 100)[wd.index.to_series().diff() == pd.Timedelta("1h")]
stats["weekday_median_step_at_hour"] = {int(h): float(v) for h, v in dstep.groupby(dstep.index.hour).median().items()}
for thr in (1, 2, 3, 5):
    m = p < thr
    stats[f"hours_below_{thr}c"] = int(m.sum())
    stats[f"days_with_hour_below_{thr}c"] = int(df.loc[m, "date"].nunique())
low = p < 5
stats["below_5c_weekend_share"] = float(df.loc[low, "weekend"].mean())
stats["weekend_share_of_all_hours"] = float(df["weekend"].mean())
stats["below_5c_share_hours_10_17"] = float(df.loc[low, "hour"].between(9, 17).mean())
stats["below_5c_mean_co2_g"] = float(df.loc[low, "co2_kg_per_kwh"].mean() * 1000)
stats["below_5c_share_apr_sep"] = float(df.loc[low, "month"].isin([4, 5, 6, 7, 8, 9]).mean())
stats["below_5c_by_month"] = {C.MONTH_ABBR[m]: int(v) for m, v in low.groupby(df["month"]).sum().items()}
# fixed-date national holidays are billed as weekends (valley all day)
hol = {}
for day in ("2025-12-08", "2026-01-01", "2026-01-06", "2026-05-01", "2025-09-11", "2026-04-03", "2026-04-06"):
    x = df.loc[day, "precio_eur_per_kwh"] * 100
    hol[day] = {"step_09_to_10": float(x.iloc[10] - x.iloc[9]), "step_17_to_18": float(x.iloc[18] - x.iloc[17])}
stats["holiday_tariff_steps"] = hol
C.write_stats("q5_price", stats)

# ---- figure: profile + tariff structure
fig = plt.figure(figsize=(13.5, 6.6))
C.titled(fig, "Weekday prices follow the tariff clock: cheapest 14:00–18:00, dearest 18:00–22:00; weekends are cheap "
              "all day",
         "Mean PVPC price (c€/kWh). On weekdays the access tariff adds steps at 08, 10, 14, 18 and 22 h on top of the "
         "solar-driven energy price.")
gs = fig.add_gridspec(1, 4, left=0.055, right=0.985, top=0.80, bottom=0.14, wspace=0.12,
                      width_ratios=[1, 1.07, 0.03, 1.3])
vmax = 32
ax = fig.add_subplot(gs[0])
heat(ax, mat_wd, C.SEQ_ORANGE, 2, vmax, xstep=6)
ax.set_title("Weekdays, by month and hour")
ax = fig.add_subplot(gs[1])
im = heat(ax, mat_we, C.SEQ_ORANGE, 2, vmax, xstep=6)
ax.set_yticklabels([])
ax.set_title("Weekends, same colour scale")
cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.05)
cb.outline.set_visible(False)
ax = fig.add_subplot(gs[3])
hours = np.arange(24)
# tariff periods of the weekday access tariff (2.0TD): P1 peak, P2 flat, P3 valley
for a, b, name in ((0, 8, "valley"), (8, 10, "flat"), (10, 14, "peak"), (14, 18, "flat"), (18, 22, "peak"), (22, 24, "flat")):
    shade = {"valley": "#f4f3ef", "flat": "#e9e7e0", "peak": "#d9d7cf"}[name]
    ax.axvspan(a - 0.5, b - 0.5, ymin=0.925, ymax=1.0, color=shade, lw=0)
    ax.text((a + b) / 2 - 0.5, 40.45, name, ha="center", va="center", fontsize=9, color=C.INK2)
    if a:
        ax.axvline(a - 0.5, color=C.GRID, lw=0.8)
for prof, col, name in ((prof_wd, C.C_WEEKDAY, "Weekdays"), (prof_we, C.C_WEEKEND, "Weekends")):
    ax.fill_between(hours, prof.quantile(0.1), prof.quantile(0.9), color=col, alpha=0.13, lw=0, step="mid")
    ax.step(hours, prof.mean(), where="mid", color=col, lw=2.2, label=f"{name}: mean, 10–90 % band")
ax.text(20.5, 34.2, "Weekdays", fontsize=11.5, fontweight="bold", color=C.INK, ha="center")
ax.text(12.5, 0.7, "Weekends", fontsize=11.5, fontweight="bold", color=C.INK, ha="center")
ax.set_ylim(0, 42)
ax.set_ylabel("c€/kWh")
C.hour_axis(ax, step=2)
ax.grid(axis="x", visible=False)
ax.set_title("Mean daily profile and weekday tariff periods")
ax.legend(loc="upper left", fontsize=10.5, bbox_to_anchor=(0.0, 0.925))
C.footnote(fig, f"Sample: all {N_DAYS} train days ({(~ds['weekend']).sum()} weekdays, {ds['weekend'].sum()} weekend days). "
                "Fixed-date national holidays (8 Dec, 1 Jan, 6 Jan, 1 May) are billed like weekends.")
C.save(fig, "q5_price_profile")


# ---- figure: spread, with the near-zero panel
def near_zero_panel(ax):
    cnt = pd.crosstab(df.loc[low, "month"], df.loc[low, "hour"]).reindex(index=C.MONTH_ORDER, columns=range(24)).fillna(0)
    im = heat(ax, cnt, C.SEQ_ORANGE, 0, max(cnt.values.max(), 1), xstep=6)
    ax.set_yticklabels([C.MONTH_ABBR[m][0] for m in C.MONTH_ORDER])
    ax.set_xlabel("Hour of day")
    cb = ax.figure.colorbar(im, ax=ax, pad=0.03, fraction=0.05, ticks=[0, 5, 10, 15])
    cb.outline.set_visible(False)
    cb.set_label("hours in train")
    ax.set_title(f"Hours below 5 c€/kWh: {int(low.sum())} ({low.mean():.1%})")


fig = spread_figure(
    key, stats, ds, sw,
    f"Price spreads are wider still: the dearest hour is {stats['ratio_median']:.1f}× the cheapest (median), "
    f"with {stats['swings_per_day_mean']:.1f} profitable swings a day",
    f"All {N_DAYS} days clear the 1.11× break-even. Near-zero prices appear on spring and summer middays, "
    f"{stats['below_5c_weekend_share']:.0%} of them at weekends.",
    extra_panel=near_zero_panel)
C.footnote(fig, f"Sample: all {N_DAYS} train days. Lowest hourly price {p.min():.2f} c€/kWh, highest {p.max():.1f} c€/kWh. "
                "Swing = rise of at least 11.1 % from a trough to the next peak.")
C.save(fig, "q5_price_spread")

for k in ("q4_co2", "q5_price"):
    import json
    st_ = json.loads((C.STATS_DIR / f"{k}.json").read_text())
    print(k, {a: b for a, b in st_.items() if not isinstance(b, (dict, list))})
