"""Figure for experiments/09_public_noise: score difference between two equally good controllers
(perfect foresight vs v1) on the whole test and on several kinds of 'public half'.
Run from the repo root:  .venv/Scripts/python.exe experiments/09_public_noise/plot.py"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SURFACE, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
df = pd.read_csv(HERE / "results.csv")
rows = [("full", "Whole test set\n(what the controller really achieves)"),
        ("first half of each week", "Public half =\nfirst 3.5 days of each week"),
        ("alternate days", "Public half =\nevery other day"),
        ("random days", "Public half =\nrandom days"),
        ("random hours", "Public half =\nrandom hours")]
LEAD = 0.0066
fig, ax = plt.subplots(figsize=(12.5, 6.4))
rng = np.random.default_rng(0)
for i, (key, label) in enumerate(rows):
    x = df[f"{key}: oracle - v1"].to_numpy()
    y = len(rows) - 1 - i + rng.uniform(-0.17, 0.17, len(x))
    ax.scatter(x, y, s=34, color=BLUE, alpha=0.55, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    sd, share = x.std(), np.mean(np.abs(x) >= LEAD)
    ax.text(1.005, len(rows) - 1 - i, f"spread (std) {sd:.4f}\n|difference| ≥ {LEAD}: {share:.0%} of runs",
            transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9.5, color=INK2)
ax.axvline(0, color=INK2, lw=1.0, zorder=1)
ax.axvline(LEAD, color=ORANGE, lw=1.6, ls=(0, (4, 3)), zorder=2)
ax.text(LEAD + 0.002, len(rows) - 0.42, "lead of the 6.44 team\nover the 6.43 pack (+0.0066)", color=INK2, fontsize=9.5, va="top")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows][::-1])
ax.set_ylim(-0.6, len(rows) - 0.25)
lim = max(0.05, float(np.ceil(np.abs(df[[c for c in df.columns if "oracle" in c]].to_numpy()).max() * 50) / 50))
ax.set_xlim(-lim, lim)
ax.set_xlabel("Score of a perfect-foresight controller minus score of agent v1 (points)   ·   each dot = one simulated 12-week test")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
fig.suptitle("Scored on the whole test, perfect foresight adds nothing over our agent.\n"
             "Scored on half of the hours, two equally good controllers differ by hundredths of a point",
             x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.text(0.02, 0.012, f"{len(df)} simulated tests built from train weeks (one random week per month, meter outage excluded). "
         "Source: experiments/09_public_noise/", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.205, right=0.795, top=0.84, bottom=0.115)
out = ROOT / "reports" / "00_overview" / "figs" / "public_split_noise.png"
fig.savefig(out, dpi=150); print("saved", out)
