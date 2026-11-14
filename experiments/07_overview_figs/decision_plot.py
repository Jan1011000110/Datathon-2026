"""Figure: how agent v1 decides. For three hours of one day it shows the marginal value of
stored energy v(s) computed from the 24 h lookahead, against the cost of adding a kWh (w/eta)
and the value of using one (w*eta).
Run from the repo root:  .venv/Scripts/python.exe experiments/07_overview_figs/decision_plot.py"""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.sim import get_folds, simulate

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

SURFACE, INK, INK2, GRID, BLUE, RED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#2a78d6", "#e34948"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

df = get_folds("filled")["ALL"]
tr = simulate(v1.BatteryAgent(), df, return_trace=True)["trace"]
w_all = (df["co2_kg_per_kwh"] + v1.KAPPA * df["precio_eur_per_kwh"]).to_numpy()
soc_before = np.concatenate([[1000.0], tr.soc_kwh.to_numpy()[:-1]])

cases = [("2026-03-11 03:00", "Clean, cheap hour: charge"),
         ("2026-03-11 05:00", "In-between hour: wait"),
         ("2026-03-11 20:00", "Dirty, expensive hour: discharge")]
from matplotlib.lines import Line2D
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.9), sharey=True, gridspec_kw={"wspace": 0.11})
Y_ARROW = 0.047
for ax, (ts, title) in zip(axes, cases):
    i = df.index.get_loc(pd.Timestamp(ts))
    w = w_all[i:i + 24]
    v = v1.marginal_values(w, np.zeros(v1.NI))
    b, a = w[0] / v1.ETA, w[0] * v1.ETA
    s0, s1, p = soc_before[i], tr.soc_kwh.iloc[i], tr.p_kw.iloc[i]
    ax.stairs(v, v1.GRID, color=INK, lw=2.0, baseline=None)
    ax.axhline(b, color=BLUE, lw=1.6, ls=(0, (5, 3)))
    ax.axhline(a, color=RED, lw=1.6, ls=(0, (5, 3)))
    ax.axvline(s0, color=INK2, lw=0.9, ls=":")
    col = BLUE if p > 1 else RED if p < -1 else INK2
    if abs(s1 - s0) > 1:
        ax.annotate("", xy=(s1, Y_ARROW), xytext=(s0, Y_ARROW),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6, mutation_scale=20, shrinkA=0, shrinkB=0))
        ax.text((s0 + s1) / 2, Y_ARROW + 0.008, ("charge %+.0f kW" if p > 0 else "discharge %+.0f kW") % p,
                color=INK, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    else:
        ax.plot([s0], [Y_ARROW], marker="o", ms=9, color=INK2)
        ax.text(s0 - 45, Y_ARROW, "wait (0 kW)", color=INK, va="center", ha="right", fontsize=10.5, fontweight="bold")
    ax.text(s0, 0.012, (" now: %.0f kWh" if s0 < 1500 else "now: %.0f kWh ") % s0, color=INK2, fontsize=9.5,
            ha="left" if s0 < 1500 else "right", va="bottom")
    ax.set_title("%s\n%s  ·  signal w = %.3f" % (title, pd.Timestamp(ts).strftime("%a %d %b, %H:%M"), w[0]),
                 loc="left", fontsize=11.5, color=INK, fontweight="bold")
    ax.set_xlim(-25, 2025); ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xlabel("Energy in the battery (kWh)")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("kg CO$_2$-eq per kWh")
axes[0].set_ylim(0, 0.29)
fig.suptitle("How the agent decides: is a stored kWh worth more later than it costs or earns now?",
             x=0.06, y=0.985, ha="left", fontsize=14.5, fontweight="bold", color=INK)
fig.legend(handles=[Line2D([], [], color=INK, lw=2.0, label="value later of one more stored kWh, from the 24 known hours (exact)"),
                    Line2D([], [], color=BLUE, lw=1.6, ls=(0, (5, 3)), label="cost of storing one more kWh now  (w / η)"),
                    Line2D([], [], color=RED, lw=1.6, ls=(0, (5, 3)), label="value of using one kWh now  (w × η)")],
           loc="upper left", bbox_to_anchor=(0.052, 0.94), ncol=3, frameon=False, fontsize=10)
fig.text(0.06, 0.845, "Black above blue: charge until they meet.   Black below red: discharge until they meet.   "
         "Black between the two: wait.   The 500 kW limit caps how far one hour can move.", fontsize=10, color=INK2)
fig.text(0.06, 0.012, "Train data, Wednesday 11 March 2026. η = √0.9 (one-way efficiency). "
         "Source: experiments/07_overview_figs/decision_plot.py", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.06, right=0.985, top=0.73, bottom=0.125)
out = ROOT / "reports" / "00_overview" / "figs" / "how_it_decides.png"
fig.savefig(out, dpi=150); print("saved", out)
for ts, _ in cases:
    i = df.index.get_loc(pd.Timestamp(ts)); print(ts, "soc %.0f -> %.0f, p %.0f, w %.3f" % (soc_before[i], tr.soc_kwh.iloc[i], tr.p_kw.iloc[i], w_all[i]))
