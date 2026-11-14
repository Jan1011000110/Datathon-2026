"""Figure: what agent v1 does over one week, next to the perfect-foresight plan.
Run from the repo root:  .venv/Scripts/python.exe experiments/07_overview_figs/week_plot.py"""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.sim import get_folds, simulate, oracle

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, RED, GREY = "#2a78d6", "#e34948", "#b9b8b1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

df = get_folds("filled")["ALL"]
tr = simulate(v1.BatteryAgent(), df, return_trace=True)["trace"]
orc = oracle(df, return_trace=True)["trace"]
w = df["co2_kg_per_kwh"] + v1.KAPPA * df["precio_eur_per_kwh"]

start, end = "2026-03-09", "2026-03-15 23:00"          # Monday -> Sunday, in the train set
s = slice(start, end)
t = df.loc[s].index
p = tr.loc[s, "p_kw"]

fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.25, 1, 1], "hspace": 0.16})
fig.suptitle("The agent charges in the cleanest, cheapest hours and discharges in the dirtiest,\n"
             "matching the perfect-foresight plan almost hour for hour",
             x=0.085, y=0.985, ha="left", fontsize=14.5, fontweight="bold", color=INK)

# 1. signal, with charge / discharge hours shaded
ax = axes[0]
for ts_, pw in p.items():
    if abs(pw) > 1:
        ax.axvspan(ts_, ts_ + pd.Timedelta(hours=1), color=BLUE if pw > 0 else RED, alpha=0.16, lw=0)
ax.step(t, w.loc[s], where="post", color=INK, lw=1.8)
ax.set_ylabel("Score signal w\n(kg CO$_2$-eq per kWh)")
ax.legend(handles=[Line2D([], [], color=INK, lw=1.8, label="w = CO$_2$ intensity + 0.257 × price"),
                   Patch(color=BLUE, alpha=0.3, label="hours the agent charges"),
                   Patch(color=RED, alpha=0.3, label="hours the agent discharges")],
          loc="upper left", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0, 1.13))

# 2. battery power
ax = axes[1]
ax.bar(t[p > 0], p[p > 0], width=1 / 24 * 0.86, align="edge", color=BLUE, label="charge")
ax.bar(t[p < 0], p[p < 0], width=1 / 24 * 0.86, align="edge", color=RED, label="discharge")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylim(-560, 560); ax.set_yticks([-500, -250, 0, 250, 500])
ax.set_ylabel("Battery power (kW)")
ax.text(t[0], 505, "charge", color=INK2, fontsize=9.5, va="bottom")
ax.text(t[0], -505, "discharge", color=INK2, fontsize=9.5, va="top")

# 3. state of charge: agent vs oracle (state after each hour)
ax = axes[2]
tt = t + pd.Timedelta(hours=1)
ax.fill_between(tt, tr.loc[s, "soc_kwh"], step=None, color=GREY, alpha=0.55, lw=0)
ax.plot(tt, tr.loc[s, "soc_kwh"], color=INK2, lw=1.6, label="agent v1 (sees 24 h ahead)")
ax.plot(tt, orc.loc[s, "soc_kwh"], color=INK, lw=1.3, ls=(0, (4, 3)), label="perfect foresight (knows the whole year)")
ax.set_ylim(0, 2150); ax.set_yticks([0, 500, 1000, 1500, 2000])
ax.set_ylabel("Energy stored (kWh)")
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0, 1.16))

for ax in axes:
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.margins(x=0)
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
axes[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
fig.text(0.085, 0.012, "Train data, week of 9–15 March 2026. Battery: 2,000 kWh, 500 kW, 90% round-trip efficiency. "
         "Source: experiments/07_overview_figs/week_plot.py", fontsize=8.5, color=INK2)
axes[-1].set_xlim(t[0], t[-1] + pd.Timedelta(hours=1))
for lab in axes[-1].get_xticklabels()[-1:]:
    lab.set_ha("right")
fig.subplots_adjust(left=0.085, right=0.975, top=0.86, bottom=0.075)
out = ROOT / "reports" / "00_overview" / "figs" / "week_v1.png"
fig.savefig(out, dpi=150)
print("saved", out)
wk = tr.loc[s]
print("cycles this week: %.2f full-equivalent; max |soc diff| vs oracle: %.0f kWh; share of hours identical action: %.1f%%" % (
    wk.p_kw.clip(lower=0).sum() * v1.ETA / 2000, (wk.soc_kwh - orc.loc[s, "soc_kwh"]).abs().max(),
    100 * np.mean(np.abs(wk.p_kw - orc.loc[s, "p_kw"]) < 1.0)))
allp = tr.p_kw
print("whole train set: %.2f full-equivalent cycles per day; identical-action share vs oracle: %.1f%%" % (
    allp.clip(lower=0).sum() * v1.ETA / 2000 / (len(allp) / 24), 100 * np.mean(np.abs(allp - orc.p_kw) < 1.0)))
