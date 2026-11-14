# Usage: .venv/Scripts/python.exe experiments/26_hard_eda/fig5_night.py
"""Figure 5: the meter's night freeze and the optimal discharge.
Rows: (top) the organisers' hourly train table, pairs of consecutive hours (5,993 h);
(bottom) folds A, B and C, perfect-foresight discharge pooled (5,417 h).
Writes reports/15_hard_eda/figs/f5_night_freeze.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, PMAX, TOL, folds_abc, load_hard, run_oracle
from src.plotstyle import C, GREY, INK, SEQ, apply, save

df = load_hard()
ld = df["load_kw"]
contig = (df.index.to_series().diff() == pd.Timedelta("1h")).to_numpy()
chg = (ld - ld.shift(1)).abs()[contig]
hr = chg.index.hour
stats = pd.DataFrame({"n": chg.groupby(hr).size(), "median": chg.groupby(hr).median(), "p90": chg.groupby(hr).quantile(0.9),
                      "same_1kw": (chg <= 1.0).groupby(hr).mean(), "same_exact": (chg < 1e-9).groupby(hr).mean()})
print(stats.round(3).to_string())
FROZEN = [23, 0, 1, 2, 3, 4, 5]          # hours whose load equals the previous hour's reading
fz = np.isin(hr, FROZEN)
print(f"frozen hours 23:00-05:00: {fz.sum()} pairs, equal within 1 kW in {(chg[fz] <= 1).mean():.2%}, "
      f"within 5 kW {(chg[fz] <= 5).mean():.2%}, max change {chg[fz].max():.1f} kW")
print(f"other hours: median change {chg[~fz].median():.0f} kW, p90 {chg[~fz].quantile(.9):.0f} kW")
# is the night value the 22:00 reading? compare hour 22 with hour 03 of the next day
h22 = ld[ld.index.hour == 22]
h03 = ld[ld.index.hour == 3]
h03.index = h03.index - pd.Timedelta("5h")
j = pd.concat([h22, h03], axis=1, keys=["h22", "h03"]).dropna()
print(f"hour 03 vs hour 22 of the evening before: {len(j)} nights, median abs diff {(j.h22 - j.h03).abs().median():.2f} kW, "
      f"p90 {(j.h22 - j.h03).abs().quantile(.9):.2f}")
# 06:00 = first live reading after the freeze: how far from the held value
h06 = chg[hr == 6]
print(f"06:00 change from the held value: median {h06.median():.0f} kW, p90 {h06.quantile(.9):.0f} kW")

parts = []
for k, f in folds_abc().items():
    tr = run_oracle(f)
    tr["fold"] = k
    parts.append(tr)
tr = pd.concat(parts)
dis = tr[tr.discharge > TOL]
dh = dis.index.hour
e_h = dis.discharge.groupby(dh).sum().reindex(range(24), fill_value=0.0)
share_h = 100 * e_h / e_h.sum()
infz = np.isin(dh, FROZEN)
print(f"discharge hours {len(dis)}; in 23:00-05:00: {infz.sum()} ({infz.mean():.1%}), energy {dis.discharge[infz].sum()/dis.discharge.sum():.1%}")
for k, g in dis.groupby("fold"):
    m = np.isin(g.index.hour, FROZEN)
    print(f"  fold {k}: {m.mean():.1%} of discharge hours, {g.discharge[m].sum()/g.discharge.sum():.1%} of discharge energy")
print(f"  of the frozen-window discharge hours, limited by the load (grid import 0): {(dis.grid_kw[infz] < TOL).mean():.1%}")
m68 = np.isin(dh, [6, 7, 8])
print(f"discharge energy at 06-08: {dis.discharge[m68].sum()/dis.discharge.sum():.1%}; at 06:00 alone {share_h[6]:.1f}%; "
      f"18-22: {share_h[18:23].sum():.1f}%")
ch = tr[tr.charge > TOL]
print(f"charge energy in 23:00-05:00: {ch.charge[np.isin(ch.index.hour, FROZEN)].sum()/ch.charge.sum():.1%}")

order = list(range(12, 24)) + list(range(0, 12))          # noon to noon, night in the middle
pos = {h: i for i, h in enumerate(order)}
x = np.arange(24)
apply()
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True, gridspec_kw={"hspace": 0.32})
lo, hi = pos[23] - 0.5, pos[5] + 0.5
for a in (ax, ax2):
    a.axvspan(lo, hi, color=SEQ[0], alpha=0.55, linewidth=0, zorder=0)
    a.grid(axis="x", visible=False)

med = stats["median"].reindex(order).to_numpy()
p90 = stats["p90"].reindex(order).to_numpy()
ax.bar(x, p90, color=GREY, width=0.72, label="9 hours in 10 change by less than this")
ax.bar(x, med, color=INK["secondary"], width=0.72, label="Median change")
ax.set_ylabel("Change of the load from\nthe previous hour (kW)")
ax.legend(loc="upper left", fontsize=8.5, handlelength=1.2)
ax.text((lo + hi) / 2, p90.max() * 0.62, "Meter not read: the hourly load\nrepeats the last reading, so the\nnext hour is known exactly",
        ha="center", va="center", fontsize=9, color=SEQ[6])
ax.annotate("06:00, first live reading:\nthe largest jump of the day", xy=(pos[6], p90[pos[6]]), xytext=(pos[6] + 1.2, p90.max() * 0.93),
            fontsize=8.5, color=INK["secondary"], ha="left", va="center",
            arrowprops={"arrowstyle": "-", "color": INK["muted"], "linewidth": 0.8})
ax.set_ylim(0, p90.max() * 1.12)
ax.set_title("a. How wrong is 'same as the last hour'?", fontsize=10, color=INK["secondary"])

sh = share_h.reindex(order).to_numpy()
cols = [C[0] if h in FROZEN else GREY for h in order]
ax2.bar(x, sh, color=cols, width=0.72)
for xi, v, h in zip(x, sh, order):
    if v >= 0.95:
        ax2.text(xi, v + 0.25, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK["primary"] if h in FROZEN else INK["muted"])
ax2.set_ylabel("Share of the optimal\ndischarge energy (%)")
ax2.set_ylim(0, sh.max() * 1.18)
ax2.set_xticks(x, [f"{h:02d}" for h in order])
ax2.set_xlabel("Hour of day (noon to noon)")
ax2.set_xlim(-0.6, 23.6)
tot = dis.discharge[infz].sum() / dis.discharge.sum()
ax2.text((lo + hi) / 2, sh.max() * 1.02, f"{tot:.0%} of the discharge energy", ha="center", va="center", fontsize=9.5,
         color=C[0], fontweight="bold")
ax2.set_title("b. When the best schedule discharges", fontsize=10, color=INK["secondary"])

fig.suptitle(f"The meter is frozen from 23:00 to 05:00: {infz.mean():.0%} of the optimal discharge hours fall where the load is known exactly,\n"
             "but the largest discharges, 18:00-22:00 and 06:00-08:00, fall where it moves most",
             x=0.02, y=1.0, ha="left", fontsize=12, fontweight="bold")
fig.text(0.02, 0.01, "Train rows. Top: organisers' hourly table, 5,993 hours, change between consecutive hours. Bottom: folds A, B and C (5,417 hours), "
         f"perfect-foresight schedule, {len(dis):,} discharge hours.", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "f5_night_freeze.png")
print("saved")
