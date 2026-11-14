# Usage: .venv/Scripts/python.exe experiments/26_hard_eda/fig6_signal.py
"""Figure 6: daily profile of the score signal w by season, and where its daily minimum falls.
Rows: every train day with price and CO2 (282 days, 6,768 h; price and CO2 have no gaps, so November is
included here). w = co2 + KAPPA * price. Writes reports/15_hard_eda/figs/f6_signal_profile.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, HERE, MONTH_NUM, MONTHS, load_hard, signal
from src.plotstyle import C, GREY, INK, SEQ, apply, save

df = load_hard("filled")
w = signal(df) * 1000.0                       # g CO2-eq per kWh
day = pd.Series(w.index.date, index=w.index)
full_days = day.groupby(day).transform("size") == 24
w = w[full_days.values]
idx = w.index
hour, month = idx.hour, idx.month
print(f"days {len(set(idx.date))}, hours {len(w)}")

tab = w.to_frame("w").assign(date=idx.date, hour=hour).pivot(index="date", columns="hour", values="w")
dmonth = pd.DatetimeIndex(tab.index).month
dwe = pd.DatetimeIndex(tab.index).day_of_week >= 5
hmin = tab.idxmin(axis=1)
hmax = tab.idxmax(axis=1)


def where(h):
    return np.where((h >= 10) & (h <= 17), "midday", np.where((h >= 22) | (h <= 7), "night", "other"))


kmin = pd.Series(where(hmin.to_numpy()), index=tab.index)
print("daily minimum of w falls at:", kmin.value_counts(normalize=True).round(3).to_dict())
print("  weekdays:", kmin[~dwe].value_counts(normalize=True).round(3).to_dict(), " weekends:", kmin[dwe].value_counts(normalize=True).round(3).to_dict())
print("  hour of the minimum (share of days, %):", (hmin.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print("  hour of the maximum (share of days, %):", (hmax.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print(f"  maximum between 18:00 and 22:59: {((hmax >= 18) & (hmax <= 22)).mean():.1%}; between 05:00 and 09:59: {((hmax >= 5) & (hmax <= 9)).mean():.1%}")
# the four cheapest hours of the day (what a 4-hour charge uses) against a fixed 13:00-16:59 window
rank = tab.rank(axis=1, method="first")
cheap4 = rank <= 4
print(f"  of the 4 cheapest hours of a day, in 13:00-16:59: {cheap4[[13, 14, 15, 16]].sum(axis=1).mean():.2f} on average; "
      f"in 11:00-17:59: {cheap4[list(range(11, 18))].sum(axis=1).mean():.2f}; all four inside 10:00-17:59 on "
      f"{(cheap4[list(range(10, 18))].sum(axis=1) == 4).mean():.1%} of days")
print(f"  mean w over 13:00-16:59 vs mean of the day's 4 cheapest hours: ratio median "
      f"{(tab[[13, 14, 15, 16]].mean(axis=1) / tab.where(cheap4).mean(axis=1)).median():.3f}")
print(f"  daily max/min of w: median {(tab.max(axis=1) / tab.min(axis=1)).median():.2f}, min {(tab.max(axis=1) / tab.min(axis=1)).min():.2f}; "
      f"days above the 1.11 break-even: {((tab.max(axis=1) / tab.min(axis=1)) > 1 / 0.9).mean():.1%}")
# night minimum versus midday minimum: how much cheaper is the better of the two
night_min = tab[[22, 23, 0, 1, 2, 3, 4, 5, 6, 7]].min(axis=1)
mid_min = tab[list(range(10, 18))].min(axis=1)
print(f"  midday minimum below night minimum on {(mid_min < night_min).mean():.1%} of days; median ratio midday/night {(mid_min / night_min).median():.2f}")

bym = pd.crosstab(dmonth, kmin.to_numpy(), normalize="index").reindex(columns=["midday", "night", "other"], fill_value=0) * 100
bym = bym.reindex([m for m in MONTH_NUM if m in bym.index])
bym["days"] = pd.Series(dmonth).value_counts().reindex(bym.index).to_numpy()
bym.index = [MONTHS[MONTH_NUM.index(m)] for m in bym.index]
print(bym.round(1).to_string())
bym.round(2).to_csv(HERE / "signal_min_by_month.csv")

seasons = [("Winter (Dec-Feb)", [12, 1, 2]), ("Spring (Mar-May)", [3, 4, 5]), ("Summer (Jun-Aug)", [6, 7, 8]), ("Autumn (Sep-Nov)", [9, 10, 11])]
for name, ms in seasons:
    m = np.isin(month, ms)
    med = w[m].groupby(hour[m]).median()
    print(f"  {name}: median w min {med.min():.0f} at {med.idxmin():02d}:00, max {med.max():.0f} at {med.idxmax():02d}:00, "
          f"night (00-05) {med.loc[0:5].mean():.0f}, midday (12-16) {med.loc[12:16].mean():.0f}")

apply()
fig = plt.figure(figsize=(11.5, 7.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 0.85], hspace=0.55, wspace=0.12)
h = np.arange(24)
ax0 = None
for i, (name, ms) in enumerate(seasons):
    ax = fig.add_subplot(gs[0, i], sharey=ax0)
    ax0 = ax0 or ax
    m = np.isin(month, ms)
    q = w[m].groupby(hour[m]).quantile([0.25, 0.5, 0.75]).unstack()
    ax.fill_between(h, q[0.25], q[0.75], color=SEQ[1], alpha=0.6, linewidth=0)
    ax.plot(h, q[0.5], color=SEQ[4])
    ax.scatter([q[0.5].idxmin()], [q[0.5].min()], s=26, color=SEQ[6], zorder=3)
    ax.text(q[0.5].idxmin(), q[0.5].min() - 9, f"{q[0.5].idxmin():02d}:00", ha="center", va="top", fontsize=8.5, color=SEQ[6])
    ax.set_title(f"{name}\n{len(set(idx.date[m]))} days", fontsize=10, fontweight="normal", color=INK["secondary"], pad=6)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour of day")
    if i == 0:
        ax.set_ylabel("Score signal w (g CO$_2$-eq / kWh)")
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
ax0.set_ylim(0, None)

axb = fig.add_subplot(gs[1, :])
xm = np.arange(len(bym))
axb.bar(xm, bym["midday"], color=C[0], width=0.7, label="Midday, 10:00-17:59")
axb.bar(xm, bym["night"], bottom=bym["midday"], color=C[1], width=0.7, label="Night, 22:00-07:59")
axb.bar(xm, bym["other"], bottom=bym["midday"] + bym["night"], color=GREY, width=0.7, label="Other hours")
for xi, v in zip(xm, bym["midday"]):
    axb.text(xi, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
axb.set_xticks(xm, bym.index)
axb.set_ylim(0, 100)
axb.set_ylabel("% of days")
axb.grid(axis="x", visible=False)
axb.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=9, handlelength=1.2, borderaxespad=0.2)
axb.set_title("Where the cheapest hour of the day falls, by month", fontsize=10, color=INK["secondary"])

pm = (kmin == "midday").mean()
win = np.isin(dmonth, [12, 1, 2])
pn = (kmin[win] == "night").mean()
print(f"Dec-Feb: minimum at night on {pn:.1%} of {win.sum()} days")
fig.suptitle(f"The cheapest hour for the score falls at midday on {pm:.0%} of days; from December to February it is at night on {pn:.0%}",
             x=0.07, y=0.995, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.07, 0.03, "Top: median (line) and middle half of hours (band); the dot marks the lowest median hour. Train rows: all 282 train days "
         "(6,768 hours; price and CO2 have no gaps). w = CO2 + 0.27 x price.", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "f6_signal_profile.png")
print("saved")
