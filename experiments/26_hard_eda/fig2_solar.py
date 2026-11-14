# Usage: .venv/Scripts/python.exe experiments/26_hard_eda/fig2_solar.py
"""Figure 2: what x12 solar does to the campus net load.
Rows: the organisers' hourly train table (5,993 h on 250 days with load and solar; November missing,
August up to the 16th). Writes reports/15_hard_eda/figs/f2_solar_x12.png and solar_by_month.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, HERE, MONTH_NUM, MONTHS, load_hard
from src.plotstyle import C, INK, SEQ, apply, save

df = load_hard()
load = df["load_kw"]
pv12 = df["pv_kw"]
pv1 = pv12 / 12.0
net1, net12 = load - pv1, load - pv12
sur = np.maximum(-net12, 0.0)
idx = df.index
hour, month = idx.hour, idx.month
weekend = idx.day_of_week >= 5
summer = np.isin(month, [4, 5, 6, 7, 8, 9])

# ---- numbers ---------------------------------------------------------------------------------
day_sur = sur.groupby(idx.date).sum()
print(f"hours {len(df)}, days {len(day_sur)}")
print(f"solar share of consumption: standard {pv1.sum()/load.sum():.3%}, x12 {pv12.sum()/load.sum():.3%}")
print(f"baseline import falls from {np.maximum(net1,0).sum()/1e3:.0f} to {np.maximum(net12,0).sum()/1e3:.0f} MWh "
      f"({1-np.maximum(net12,0).sum()/np.maximum(net1,0).sum():.1%} less)")
print(f"surplus hours: {(sur>0).sum()} ({(sur>0).mean():.1%}); on {(day_sur>0).sum()} of {len(day_sur)} days; "
      f"total {sur.sum()/1e3:.1f} MWh = {sur.sum()/pv12.sum():.1%} of solar; peak hour {sur.max():.0f} kW; "
      f"max day {day_sur.max():.0f} kWh; median surplus day {day_sur[day_sur>0].median():.0f} kWh")
print(f"surplus energy at weekends: {sur[weekend].sum()/sur.sum():.1%}; share of weekend days with surplus: "
      f"{(day_sur[pd.DatetimeIndex(day_sur.index).day_of_week>=5]>0).mean():.1%}; weekdays: "
      f"{(day_sur[pd.DatetimeIndex(day_sur.index).day_of_week<5]>0).mean():.1%}")
print(f"surplus hours between 10:00 and 16:59: {((sur>0)&(hour>=10)&(hour<=16)).sum()/(sur>0).sum():.1%}")
print(f"net load x12: median {net12.median():.0f}, IQR {net12.quantile(.25):.0f}-{net12.quantile(.75):.0f}, "
      f"below 1500 in {(net12<1500).mean():.1%}; midday (11-15) median: std {net1[(hour>=11)&(hour<=15)].median():.0f}, "
      f"x12 {net12[(hour>=11)&(hour<=15)].median():.0f}")
for nm, m in (("weekday", ~weekend), ("weekend", weekend)):
    mid = m & (hour >= 11) & (hour <= 15)
    print(f"  {nm} midday net load median: std {net1[mid].median():.0f} kW, x12 {net12[mid].median():.0f} kW; "
          f"share of midday hours with surplus {(sur[mid]>0).mean():.1%}")

days_m = pd.Series(idx.date, index=idx).groupby(month).nunique()
tab = pd.DataFrame({
    "days": days_m,
    "hours": load.groupby(month).size(),
    "solar_share_pct": 100 * pv12.groupby(month).sum() / load.groupby(month).sum(),
    "surplus_hours": (sur > 0).groupby(month).sum(),
    "surplus_hours_pct": 100 * (sur > 0).groupby(month).mean(),
    "days_with_surplus": (day_sur > 0).groupby(pd.DatetimeIndex(day_sur.index).month).sum(),
    "export_mwh": sur.groupby(month).sum() / 1e3,
    "export_kwh_per_day": sur.groupby(month).sum() / days_m,
    "export_pct_of_solar": 100 * sur.groupby(month).sum() / pv12.groupby(month).sum(),
}).reindex([m for m in MONTH_NUM if m in days_m.index])
tab.index = [MONTHS[MONTH_NUM.index(m)] for m in tab.index]
print(tab.round(1).to_string())
tab.round(2).to_csv(HERE / "solar_by_month.csv")

# ---- figure ----------------------------------------------------------------------------------
apply()
fig = plt.figure(figsize=(11.5, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 0.8], hspace=0.55, wspace=0.12)
panels = [("Oct-Mar, weekdays", ~summer & ~weekend), ("Oct-Mar, weekends", ~summer & weekend),
          ("Apr-Sep, weekdays", summer & ~weekend), ("Apr-Sep, weekends", summer & weekend)]
h = np.arange(24)
ax0 = None
for i, (name, m) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i], sharey=ax0)
    ax0 = ax0 or ax
    for s, col, lab in ((net1, C[0], "Solar as built"), (net12, C[1], "Solar x12")):
        q = s[m].groupby(hour[m]).quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(h, q[0.25], q[0.75], color=col, alpha=0.22, linewidth=0)
        ax.plot(h, q[0.5], color=col, label=lab)
    ax.axhline(0, color=INK["primary"], linewidth=0.9)
    ax.set_title(f"{name}\n{len(set(idx.date[m]))} days", fontsize=10, fontweight="normal", color=INK["secondary"], pad=6)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour of day")
    if i == 0:
        ax.set_ylabel("Net load = load - solar (kW)")
        ax.text(1.0, 1790, "Solar as built", color=C[0], fontsize=9, fontweight="bold")
        ax.text(1.0, 1630, "Solar x12", color=C[1], fontsize=9, fontweight="bold")
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    if i == 3:
        ax.text(0.8, -330, "below zero:\nfree surplus", fontsize=9, color=INK["secondary"], ha="left", va="center")
ax0.set_ylim(-1000, 1950)

xm = np.arange(len(tab))
axb = fig.add_subplot(gs[1, 0:2])
axb.bar(xm, tab["solar_share_pct"], color=SEQ[3], width=0.7)
for xi, v in zip(xm, tab["solar_share_pct"]):
    axb.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
axb.set_xticks(xm, tab.index)
axb.set_ylabel("% of consumption")
axb.set_ylim(0, tab["solar_share_pct"].max() * 1.2)
axb.grid(axis="x", visible=False)
axb.set_title("Solar x12 as a share of campus consumption, by month", fontsize=10, fontweight="bold", color=INK["secondary"])

axc = fig.add_subplot(gs[1, 2:4])
axc.bar(xm, tab["export_kwh_per_day"], color=SEQ[3], width=0.7)
for xi, v in zip(xm, tab["export_kwh_per_day"]):
    axc.text(xi, v + 25, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5)
axc.set_xticks(xm, tab.index)
axc.set_ylabel("kWh per day")
axc.set_ylim(0, tab["export_kwh_per_day"].max() * 1.2)
axc.grid(axis="x", visible=False)
axc.yaxis.set_label_position("left")
axc.set_title("Surplus that would be exported for nothing without a battery", fontsize=10, fontweight="bold", color=INK["secondary"])
pos = axc.get_position()
axc.set_position([pos.x0 + 0.035, pos.y0, pos.width - 0.035, pos.height])
pos = axb.get_position()
axb.set_position([pos.x0, pos.y0, pos.width - 0.035, pos.height])

fig.suptitle("Solar x12 more than halves the midday net load on working days and pushes it below zero on most weekends",
             x=0.075, y=0.995, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.075, 0.025, "Top: median (line) and middle half of hours (band). Train rows: organisers' hourly table, 5,993 hours on 250 days "
         "(no November, August to the 16th). Monthly values are averages over the train days of that month.",
         fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "f2_solar_x12.png")
print("saved")
