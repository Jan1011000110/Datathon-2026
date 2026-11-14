"""
Q1 - Coverage and quality map.
  fig q1_coverage_calendar.png   : per-day availability of each source, train/test structure visible
  fig q1_meter_night_freeze.png  : the meter is not polled 22:00-06:00 (value held), true night load
                                   recovered from the energy register
  stats q1_coverage.json
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

import common as C

C.set_style()

P_COL = "Contador - POTENCIA ACTIVA EXPORTADA (kW)"
E_COL = "Contador - ENERGIA ACTIVA IMPORTADA (kWh)"
Q_COL = "Contador - NUMERO DE PREGUNTAS"
LR_COL = "Contador - FECHA ULTIMA LECTURA"
INV_COLS = [f"Inversor {i} - POTENCIA ACTIVA (kW)" for i in (1, 2, 3)]
IRR_COL = "Atersa - IRRADIANCIA (W/m2)"

stats: dict = {}

# ------------------------------------------------------------------ raw 5-minute sources
cont = pd.read_csv(C.RAW_DIR / "contador.csv", parse_dates=["Fecha"],
                   usecols=["Fecha", P_COL, E_COL, Q_COL, LR_COL]).set_index("Fecha")
inv = pd.read_csv(C.RAW_DIR / "inversor.csv", parse_dates=["Fecha"], usecols=["Fecha"] + INV_COLS).set_index("Fecha")
met = pd.read_csv(C.RAW_DIR / "meteo.csv", parse_dates=["Fecha"]).set_index("Fecha")
pvpc = pd.read_csv(C.RAW_DIR / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
co2 = pd.read_csv(C.RAW_DIR / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

full = C.load_hourly_full()
official = C.load_hourly_official()
train_days = pd.DatetimeIndex(sorted(full.index.normalize().unique()))
test_days = C.test_days()

load5 = -cont[P_COL]
# A 5-minute row stamped HH:MM belongs to hour HH in resample("1h") (label = left edge).
samples_per_day = {
    "meter": load5.dropna().groupby(load5.dropna().index.normalize()).size(),
    "pv": inv[INV_COLS].dropna(how="all").groupby(inv[INV_COLS].dropna(how="all").index.normalize()).size(),
    "meteo": met[IRR_COL].dropna().groupby(met[IRR_COL].dropna().index.normalize()).size(),
}
hours_per_day = {
    "meter": full["load_kw"].notna().groupby(full.index.normalize()).sum(),
    "pv": full["pv_kw"].notna().groupby(full.index.normalize()).sum(),
    "meteo": full["irradiance_wm2"].notna().groupby(full.index.normalize()).sum(),
    "signals": (full["precio_eur_per_kwh"].notna() & full["co2_kg_per_kwh"].notna()).groupby(full.index.normalize()).sum(),
}
# hours where the meter reads exactly 0 kW (comms outage, see below)
zero_hours = (full["load_kw"] == 0)
zero_per_day = zero_hours.groupby(full.index.normalize()).sum()

# ------------------------------------------------------------------ headline counts
n_train_h = len(full)
stats["train_days"] = len(train_days)
stats["train_hours"] = n_train_h
stats["test_days"] = len(test_days)
stats["test_hours"] = len(test_days) * 24
stats["hours_with_load"] = int(full["load_kw"].notna().sum())
stats["hours_with_pv"] = int(full["pv_kw"].notna().sum())
stats["hours_official_table"] = int(len(official))
stats["share_official_of_train"] = len(official) / n_train_h
stats["hours_missing_load"] = int(full["load_kw"].isna().sum())
stats["hours_missing_pv"] = int(full["pv_kw"].isna().sum())
nov = full.index.month == 11
stats["hours_missing_load_november"] = int(full.loc[nov, "load_kw"].isna().sum())
stats["hours_missing_load_other"] = int(full.loc[~nov, "load_kw"].isna().sum())
miss_other = full.index[full["load_kw"].isna() & ~nov]
stats["missing_load_other_timestamps"] = [str(t) for t in miss_other]
pv_only = full.index[full["pv_kw"].isna() & full["load_kw"].notna()]
stats["hours_pv_missing_but_load_present"] = int(len(pv_only))
stats["pv_last_timestamp"] = str(inv.index.max())
stats["hours_meter_reads_zero"] = int(zero_hours.sum())
stats["zero_hours_list"] = [str(t) for t in full.index[zero_hours]]
stats["days_with_zero_hours"] = {str(d.date()): int(v) for d, v in zero_per_day[zero_per_day > 0].items()}

# 5-minute level zeros (shorter outages that do not zero a whole hour)
z5 = load5[load5 == 0]
runs = (z5.index.to_series().diff() != pd.Timedelta("5min")).cumsum()
zero_runs = z5.groupby(runs.values).agg(start=lambda s: s.index.min(), end=lambda s: s.index.max(), n="size")
stats["zero_runs_5min"] = [{"start": str(r.start), "end": str(r.end), "samples": int(r.n)} for r in zero_runs.itertuples()]

# intra-day gaps in the 5-minute telemetry (other than whole missing days)
d = cont.index.to_series().diff()
gaps = d[(d > pd.Timedelta("5min")) & (d < pd.Timedelta("1D"))]
stats["intraday_gaps"] = [{"resumes": str(t), "gap_minutes": int(g.total_seconds() / 60 - 5)} for t, g in gaps.items()]

# ------------------------------------------------------------------ the night freeze
polled = cont[Q_COL] > 0
stale = (load5.diff() == 0) & (d == pd.Timedelta("5min"))
stats["share_5min_identical_to_previous"] = float(stale.mean())
by_hour_stale = stale.groupby(cont.index.hour).mean()
by_hour_polled = polled.groupby(cont.index.hour).mean()
night = (cont.index.hour >= 22) | (cont.index.hour < 6)
stats["share_stale_22_06"] = float(stale[night].mean())
stats["share_stale_06_22"] = float(stale[~night].mean())
stats["share_polled_23_06"] = float(polled[(cont.index.hour >= 23) | (cont.index.hour < 6)].mean())
stats["share_polled_07_22"] = float(polled[(cont.index.hour >= 7) & (cont.index.hour < 22)].mean())

# daytime check: the energy register (kWh per 5 min) x 12 reproduces the power reading
day = (cont.index.hour >= 7) & (cont.index.hour < 22) & polled & (cont[E_COL] < 300) & (load5 > 0)
ratio = (cont.loc[day, E_COL] * 12) / load5[day]
stats["daytime_energy_x12_over_power_median"] = float(ratio.median())
stats["daytime_energy_x12_vs_power_corr"] = float(np.corrcoef(cont.loc[day, E_COL] * 12, load5[day])[0, 1])

# first poll after each night: the energy register carries the whole gap
lr = pd.to_datetime(cont[LR_COL], unit="s")
gap_h = (lr - lr.shift(1)).dt.total_seconds() / 3600
catch = cont[(cont[E_COL] > 300) & (gap_h > 7.5) & (gap_h < 8.5)].copy()
catch["gap_h"] = gap_h[catch.index]
catch["held_kw"] = load5.shift(1)[catch.index]          # value frozen since 22:00
catch["true_kw"] = catch[E_COL] / catch["gap_h"]         # real mean power over the gap
catch = catch[catch["held_kw"] > 0]
# classify the night by the weekday of the evening it starts on (22:00)
eve = (catch.index - pd.Timedelta(hours=9)).normalize()
catch["eve_dow"] = eve.day_of_week
catch["over_pct"] = (catch["held_kw"] / catch["true_kw"] - 1) * 100
stats["nights_with_energy_catchup"] = int(len(catch))
stats["night_held_kw_median"] = float(catch["held_kw"].median())
stats["night_true_kw_median"] = float(catch["true_kw"].median())
stats["night_true_kw_min"] = float(catch["true_kw"].min())
stats["night_true_kw_p05"] = float(catch["true_kw"].quantile(0.05))
stats["night_overstatement_pct_median"] = float(catch["over_pct"].median())
stats["night_overstatement_pct_mean"] = float(catch["over_pct"].mean())
stats["night_overstatement_pct_p10_p90"] = [float(catch["over_pct"].quantile(0.1)), float(catch["over_pct"].quantile(0.9))]
stats["share_nights_overstated"] = float((catch["over_pct"] > 0).mean())
# effect on the baseline energy of the official table (nights with a catch-up only)
excess_kwh = float(((catch["held_kw"] - catch["true_kw"]) * 8.0).sum())
base = C.baseline_totals()
stats["baseline_energy_overstated_pct"] = excess_kwh / base["energy_kwh"] * 100
# same in CO2 / cost terms: subtract the excess from each frozen hour (22:00..05:00)
corr = official.copy()
adj = pd.Series(0.0, index=corr.index)
for t, r in catch.iterrows():
    start = (t - pd.Timedelta(hours=9)).normalize() + pd.Timedelta(hours=22)
    hrs = pd.date_range(start, periods=8, freq="h")
    hrs = hrs[hrs.isin(corr.index)]
    adj[hrs] = r["held_kw"] - r["true_kw"]
stats["baseline_co2_overstated_pct"] = float((adj * corr["co2_kg_per_kwh"]).sum() / base["co2_kg"] * 100)
stats["baseline_cost_overstated_pct"] = float((adj * corr["precio_eur_per_kwh"]).sum() / base["cost_eur"] * 100)

# other suspicious sensors
stats["wind_sensor_all_zero"] = bool((met["Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)"].fillna(0) == 0).all())
tcol = [c for c in met.columns if "TEMPERATURA AMBIENTE" in c][0]
stats["ambient_temp_max_c"] = float(met[tcol].max())
stats["ambient_temp_hours_above_45c"] = int((met[tcol].resample("1h").mean() > 45).sum())
stats["co2_equals_0p37_times_emitting_share"] = bool(
    np.allclose(co2["co2_kg_per_kwh"], 0.37 * co2["porcentaje_con_emisiones"], atol=1e-9))
stats["inverter_ac_cap_kw_each"] = float(inv[INV_COLS].max().max())
stats["baseline"] = base

C.write_stats("q1_coverage", stats)

# ================================================================== FIGURE 1: calendar
months = [(2025, 9), (2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4),
          (2026, 5), (2026, 6), (2026, 7), (2026, 8)]
ST_OK, ST_PART, ST_MISS, ST_TEST = "#9ec5f4", C.ST_WARN, C.ST_CRIT, "#d9d7cf"
panels = [
    ("meter", "Campus meter (load)  ·  weather station"),
    ("pv", "PV inverters"),
    ("signals", "PVPC price  ·  grid CO₂ (REE)"),
]
test_set = set(test_days)
# 1 Sep 2026 is in the train files for price/CO2 only (the sensors have a single sample): it is
# counted in the statistics but not drawn, to keep the calendar to the 12 competition months.
cal_days = [d for d in train_days if d < pd.Timestamp("2026-09-01")]
train_set = set(cal_days)


def day_status(key: str, day_ts: pd.Timestamp) -> str:
    h = int(hours_per_day[key].get(day_ts, 0))
    z = int(zero_per_day.get(day_ts, 0)) if key == "meter" else 0
    if h <= 1:            # 0 h, or the single stray 00:00 sample that closes a test week
        return "miss"
    return "part" if (h < 24 or z > 0) else "ok"


status = {key: pd.Series({d: day_status(key, d) for d in cal_days}) for key, _ in panels}
stats["calendar_days"] = len(cal_days)
stats["day_status_counts"] = {key: status[key].value_counts().to_dict() for key in status}
stats["partial_days"] = {key: [str(d.date()) for d in status[key].index[status[key] == "part"]] for key in status}
C.write_stats("q1_coverage", stats)

fig = plt.figure(figsize=(13.5, 7.2))
C.titled(fig, "Price and CO₂ are complete; the campus sensors lose all of November and PV stops on 16 August",
         "One cell per day. Rows are months, columns the day of the month. "
         "The hidden test set is the last 7 days of every month.")
axes = fig.subplots(1, 3, gridspec_kw={"left": 0.065, "right": 0.985, "top": 0.775, "bottom": 0.185, "wspace": 0.06})
colmap = {"ok": ST_OK, "part": ST_PART, "miss": ST_MISS}

for ax, (key, name) in zip(axes, panels):
    for r, (y, m) in enumerate(months):
        ndays = (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).day
        for dday in range(1, ndays + 1):
            day_ts = pd.Timestamp(year=y, month=m, day=dday)
            col = ST_TEST if day_ts in test_set else colmap[status[key][day_ts]]
            ax.add_patch(Rectangle((dday - 0.5 + 0.06, r - 0.5 + 0.08), 0.88, 0.84, facecolor=col, edgecolor="none"))
    ax.set_xlim(0.4, 31.6)
    ax.set_ylim(len(months) - 0.5, -0.5)
    ax.set_xticks([1, 8, 15, 22, 29])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(months)))
    ax.set_yticklabels([f"{C.MONTH_ABBR[m]} {str(y)[2:]}" for y, m in months] if ax is axes[0] else [])
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    n_ok = int((status[key] == "ok").sum())
    ax.set_title(name, fontsize=12.5, pad=38)
    ax.text(0.0, 1.058, f"{n_ok} of {len(cal_days)} train days complete", transform=ax.transAxes,
            fontsize=10.5, color=C.INK2, ha="left", va="bottom")

# annotations (text in ink, never in the data colour)
a0, a1 = axes[0], axes[1]
for a in (a0, a1):
    a.text(12, 2, "no telemetry", ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
def note(ax, x_cell, y_cell, text, x_text, ha, y_text=12.3):
    """Leader line running in the gap between two columns, label below the grid."""
    ax.plot([x_cell, x_cell], [y_cell, y_text - 0.3], color=C.MUTED, lw=0.9, clip_on=False, solid_capstyle="butt")
    ax.text(x_text, y_text, text, fontsize=10, color=C.INK2, ha=ha, va="center", clip_on=False)


note(a0, 17.5, 9.46, "17–18 Jun: meter reads 0 kW for 14 h", 12.6, "left")
note(a0, 4.5, 11.46, "4 Aug: 5.5 h gap  ·  6 Aug: meter reads 0 kW for 1 h", 0.6, "left", y_text=13.0)
note(a1, 20.5, 11.46, "inverter log ends 16 Aug 22:00", 20.5, "center")

handles = [Patch(facecolor=ST_OK, label="Train day, all 24 hours"),
           Patch(facecolor=ST_PART, label="Train day, hours missing or meter reading 0 kW"),
           Patch(facecolor=ST_MISS, label="Train day, no data"),
           Patch(facecolor=ST_TEST, label="Hidden test week")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.04), ncol=4, fontsize=11,
           handlelength=1.2, handleheight=1.0, columnspacing=2.2)
C.footnote(fig, f"Train = {len(train_days)} days / {n_train_h:,} hours (incl. 1 Sep 2026, not drawn).  Load present in "
                f"{stats['hours_with_load']:,} h, PV in {stats['hours_with_pv']:,} h, both (the organisers' table) in "
                f"{len(official):,} h ({len(official) / n_train_h:.1%}).", y=0.008)
C.save(fig, "q1_coverage_calendar")

# ================================================================== FIGURE 2: night freeze
fig = plt.figure(figsize=(13.5, 6.6))
C.titled(fig, "The meter is not read between 22:00 and 06:00: the official night load is a frozen 22:00 value",
         "Every night the last reading is repeated for 8 hours. The meter's energy register, read at 06:05, "
         "gives the real night consumption.")
gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.80, bottom=0.14, wspace=0.20, width_ratios=[1.55, 1])
ax = fig.add_subplot(gs[0])

t0, t1 = pd.Timestamp("2025-10-14 12:00"), pd.Timestamp("2025-10-17 12:00")
seg = load5[t0:t1]
seg_polled = polled[t0:t1]
ax.plot(seg.index, seg.where(seg_polled).values, color=C.C_LOAD, lw=1.4, label="Polled reading (5 min)")
held = seg.where(~seg_polled)
ax.plot(held.index, held.values, color=C.ORANGE, lw=2.4, label="Held reading (meter not polled)")
first = True
for t, r in catch[(catch.index >= t0) & (catch.index <= t1)].iterrows():
    s = (t - pd.Timedelta(hours=9)).normalize() + pd.Timedelta(hours=22)
    ax.plot([s, t], [r["true_kw"], r["true_kw"]], color=C.INK, lw=2.0,
            label="Real night mean (energy register)" if first else None)
    ax.axvspan(s, t, color=C.GRID, alpha=0.55, lw=0)
    ax.annotate(f"{r['held_kw']:.0f} kW held\n{r['true_kw']:.0f} kW real", xy=(s + (t - s) / 2, r["true_kw"]),
                xytext=(0, -12), textcoords="offset points", ha="center", va="top", fontsize=10, color=C.INK2)
    first = False
ax.set_ylim(400, 2080)
ax.set_ylabel("Grid import at the meter (kW)")
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b\n%H:%M"))
ax.set_title("Three example nights, 14–17 October 2025")
ax.legend(loc="upper left", ncol=3, fontsize=10.5, handlelength=1.4, columnspacing=1.4, borderaxespad=0.3)
ax.set_xlim(t0, t1)

ax2 = fig.add_subplot(gs[1])
wk = catch["eve_dow"].isin([0, 1, 2, 3, 4])
ax2.scatter(catch.loc[wk, "true_kw"], catch.loc[wk, "held_kw"], s=26, color=C.C_WEEKDAY, alpha=0.75,
            edgecolor=C.SURFACE, linewidth=0.6, label="Night after a weekday")
ax2.scatter(catch.loc[~wk, "true_kw"], catch.loc[~wk, "held_kw"], s=26, color=C.C_WEEKEND, alpha=0.9,
            edgecolor=C.SURFACE, linewidth=0.6, label="Night after Sat / Sun")
lo, hi = 550, 1400
ax2.plot([lo, hi], [lo, hi], color=C.MUTED, lw=1.0)
ax2.text(hi - 10, hi - 40, "no bias", color=C.MUTED, fontsize=10, ha="right", va="top", rotation=38)
ax2.set_xlim(lo, hi)
ax2.set_ylim(lo, hi)
ax2.set_xlabel("Real mean load 22:00–06:00 (kW, energy register)")
ax2.set_ylabel("Load in the official hourly table (kW)")
ax2.set_title(f"{len(catch)} nights: official value is {stats['night_overstatement_pct_median']:.0f} % too high (median)")
ax2.legend(loc="lower right", fontsize=10.5, handletextpad=0.3)
C.footnote(fig, f"Sample: all {len(catch)} train nights with meter data. Daytime check: energy register × 12 equals the power "
                f"reading (median ratio {stats['daytime_energy_x12_over_power_median']:.3f}). "
                f"The freeze inflates the no-battery baseline energy by {stats['baseline_energy_overstated_pct']:.1f} %.")
C.save(fig, "q1_meter_night_freeze")

print({k: v for k, v in stats.items() if not isinstance(v, (list, dict))})
