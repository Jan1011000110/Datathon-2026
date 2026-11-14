# Usage: .venv/Scripts/python.exe experiments/26_hard_eda/fig1_week.py
"""Figure 1: one sunny week of hard mode under the perfect-foresight schedule.
Rows: fold B (days 8-14 of every train month, 1,848 h), oracle solved on the whole fold; the week shown
is 8-14 May 2026. Writes reports/15_hard_eda/figs/f1_week.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, PMAX, TOL, folds_abc, run_oracle, signal
from src.plotstyle import C, GREY, INK, SEQ, apply, save

FOLD, START, END = "B", "2026-05-08", "2026-05-14"

fold = folds_abc()[FOLD]
tr = run_oracle(fold)
tr["w"] = signal(fold) * 1000.0                     # g CO2-equivalent per kWh
tr["load"], tr["pv"] = fold["load_kw"], fold["pv_kw"]
tr["surplus"] = np.maximum(-tr["net_kw"], 0.0)
tr["free"] = np.minimum(tr["charge"], tr["surplus"])
tr["paid"] = tr["charge"] - tr["free"]
soc_before = tr["soc_kwh"].shift(1).loc[START:END].iloc[0]
wk = tr.loc[START:END]
assert len(wk) == 168, len(wk)

# numbers quoted in the report
dis = wk[wk.discharge > TOL]
ch = wk[wk.charge > TOL]
print(f"week {START}..{END}: load {wk.load.sum()/1e3:.1f} MWh, solar {wk.pv.sum()/1e3:.1f} MWh, "
      f"surplus {wk.surplus.sum()/1e3:.2f} MWh in {(wk.surplus > 0).sum()} h on "
      f"{(wk.surplus.groupby(wk.index.date).sum() > 0).sum()} days")
print(f"charge {wk.charge.sum()/1e3:.1f} MWh (free {wk.free.sum()/1e3:.2f}, paid {wk.paid.sum()/1e3:.1f}), "
      f"discharge {wk.discharge.sum()/1e3:.1f} MWh, cycles/day {wk.discharge.sum()/0.9**0.5/6000/7:.2f}")
print(f"mean w when charging from grid {np.average(wk.w, weights=wk.paid):.0f} g/kWh, "
      f"when discharging {np.average(wk.w, weights=wk.discharge):.0f} g/kWh, week mean {wk.w.mean():.0f}")
print(f"discharge hours {len(dis)}, of which grid import is zero (discharge = net load): "
      f"{(dis.grid_kw < TOL).sum()}; at 1,500 kW: {(dis.discharge > PMAX - TOL).sum()}")
print(f"score points earned in the week: {wk.pts.sum():.2f} of {tr.pts.sum():.2f} on the fold")
print("charge energy by hour:", (ch.charge.groupby(ch.index.hour).sum() / 1e3).round(1).to_dict())
print("discharge energy by hour:", (dis.discharge.groupby(dis.index.hour).sum() / 1e3).round(1).to_dict())


def stair(ax, s, **kw):
    """Hourly means drawn as steps covering [t, t+1h)."""
    x = list(s.index) + [s.index[-1] + pd.Timedelta("1h")]
    ax.step(x, list(s.values) + [s.values[-1]], where="post", **kw)


apply()
fig, axs = plt.subplots(4, 1, figsize=(10.5, 9.6), sharex=True,
                        gridspec_kw={"height_ratios": [1.25, 0.9, 1.35, 0.9], "hspace": 0.34})
x = wk.index
bw = 1.0 / 24.0

# (a) load and solar
ax = axs[0]
stair(ax, wk.load, color=C[0], label="Campus load")
stair(ax, wk.pv, color=C[1], label="Solar x12")
xs = list(x) + [x[-1] + pd.Timedelta("1h")]
ax.fill_between(xs, list(wk.load) + [wk.load.iloc[-1]], list(wk.pv) + [wk.pv.iloc[-1]],
                where=list(wk.pv > wk.load) + [False], step="post", color=C[1], alpha=0.28, linewidth=0,
                label="Free surplus (solar above load)")
ax.set_ylabel("kW")
ax.set_ylim(0, 1900)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, borderaxespad=0.2)
ax.text(0, 1.04, "a. Campus load and solar", transform=ax.transAxes, fontweight="bold", color=INK["secondary"])

# (b) score signal
ax = axs[1]
stair(ax, wk.w, color=INK["secondary"], linewidth=1.8)
ax.set_ylabel("g CO$_2$-eq / kWh")
ax.set_ylim(0, None)
ax.text(0, 1.05, "b. Score signal w = CO$_2$ + 0.27 x price: what one kWh of grid import costs in score",
        transform=ax.transAxes, fontweight="bold", color=INK["secondary"])

# (c) battery power
ax = axs[2]
ax.bar(x, wk.paid, width=bw, align="edge", color=C[0], label="Charge from the grid", linewidth=0)
ax.bar(x, wk.free, width=bw, align="edge", bottom=wk.paid, color=C[1], label="Charge from free surplus", linewidth=0)
ax.bar(x, -wk.discharge, width=bw, align="edge", color=C[2], label="Discharge", linewidth=0)
stair(ax, -np.maximum(wk.net_kw, 0.0), color=INK["primary"], linewidth=0.9, label="Limit: campus net load")
ax.axhline(0, color=INK["axis"], linewidth=0.8)
for yv in (PMAX, -PMAX):
    ax.axhline(yv, color=INK["muted"], linewidth=0.8, linestyle=(0, (4, 3)))
ax.text(x[-1] + pd.Timedelta("1h"), PMAX, " inverter\n +1,500", va="center", ha="left", fontsize=8, color=INK["muted"])
ax.text(x[-1] + pd.Timedelta("1h"), -PMAX, " inverter\n -1,500", va="center", ha="left", fontsize=8, color=INK["muted"])
ax.set_ylim(-1700, 1700)
ax.set_yticks([-1500, -1000, -500, 0, 500, 1000, 1500])
ax.set_ylabel("kW  (charge +, discharge -)")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, borderaxespad=0.2, columnspacing=1.2, handlelength=1.4)
ax.text(0, 1.04, "c. Best battery schedule", transform=ax.transAxes, fontweight="bold", color=INK["secondary"])

# (d) state of charge (value at the end of each hour)
ax = axs[3]
xe = [x[0]] + [t + pd.Timedelta("1h") for t in x]
soc = [soc_before] + list(wk.soc_kwh)
ax.fill_between(xe, 0, soc, color=SEQ[1], linewidth=0)
ax.plot(xe, soc, color=SEQ[4], linewidth=1.6)
ax.set_ylim(0, 6300)
ax.set_yticks([0, 2000, 4000, 6000])
ax.set_ylabel("kWh")
ax.text(0, 1.05, "d. State of charge (capacity 6,000 kWh)", transform=ax.transAxes, fontweight="bold", color=INK["secondary"])

ax.set_xlim(x[0], x[-1] + pd.Timedelta("1h"))
ax.set_xticks(pd.date_range(START, END, freq="D"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
for a in axs:
    a.grid(axis="x", which="major", color=INK["axis"], linewidth=0.8)
    a.grid(axis="x", which="minor", color=INK["grid"], linewidth=0.4)
for lab in ax.get_xticklabels():
    lab.set_ha("left")

fig.suptitle("One week in May with perfect foresight: the battery fills at the midday trough and on free solar,\n"
             "and empties into the evening and the morning, never faster than the campus draws",
             x=0.065, y=0.975, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.065, 0.045, "Train rows: fold B, week of 8-14 May 2026 (Friday to Thursday). Schedule = perfect-foresight optimum of the whole fold. "
         "Ticks inside each day: 06:00, 12:00, 18:00.", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "f1_week.png")
print("saved")
