# Usage: .venv/Scripts/python.exe experiments/20_hard_baseline/fig_losses.py
"""Figure: points lost against perfect foresight, by which information the agent lacks.
Reads decompose.csv, decompose_n.csv, decompose_parts.csv; writes reports/10_hard_mode/figs/losses.png."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plotstyle import C, GREY, INK, apply, save  # noqa: E402

here = Path(__file__).parent
d = pd.concat([pd.read_csv(here / f) for f in ("decompose.csv", "decompose_n.csv", "decompose_parts.csv")])
sc = d.pivot_table(index="config", columns="fold", values="score")
ceil = sc.loc["ORACLE (whole run known)"]
rows = [  # label, config, group
    ("Naive legal agent\n(nothing known beyond obs)", "clim w   + profile n, H=36 (legal)", "legal"),
    ("... with a 10% haircut on discharge", "clim w   + profile n, H=36, shade 0.9 (legal)", "legal"),
    ("Load and solar unknown\n(price/CO2 known 36 h ahead)", "profile everywhere", "what-if"),
    ("of which: only the current hour unknown", "profile hour 0, perfect after", "what-if"),
    ("of which: only later hours unknown", "perfect hour 0, profile after", "what-if"),
    ("of which: only load unknown", "profile load, perfect solar", "what-if"),
    ("of which: only solar unknown", "perfect load, profile solar", "what-if"),
    ("Price/CO2 known 3 h only\n(load and solar known)", "clim w   + perfect n, H=36", "what-if"),
]
loss = pd.DataFrame({lab: ceil - sc.loc[cfg] for lab, cfg, _ in rows}).T
loss["mean"] = loss.mean(axis=1)
print(loss.round(3).to_string())
loss.round(4).to_csv(here / "losses.csv")

apply()
fig, ax = plt.subplots(figsize=(8.6, 4.6))
y = range(len(rows))[::-1]
colors = [C[0] if g == "legal" else GREY for _, _, g in rows]
ax.barh(list(y), loss["mean"], height=0.62, color=colors)
for yi, (lab, _, _) in zip(y, rows):
    v = loss.loc[lab, "mean"]
    ax.text(max(v, loss.loc[lab, ["A", "B", "C"]].max()) + 0.05, yi, f"{v:.2f}", va="center", ha="left", color=INK["primary"], fontsize=10)
    ax.scatter(loss.loc[lab, ["A", "B", "C"]], [yi] * 3, s=14, color=INK["secondary"], zorder=3, linewidths=0)
ax.set_yticks(list(y))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 3.1)
ax.set_xlabel("Score points lost against perfect foresight (mean of folds A, B, C; dots = each fold)")
ax.grid(axis="y", visible=False)
ax.set_title("Hard mode: the score is lost mostly in the hour being decided, and mostly on load")
fig.text(0.01, -0.03, "Ceiling with perfect foresight: 21.26 points (22.44 / 21.42 / 19.92). Grey bars are diagnostic runs that give the "
         "agent information it will not have;\nblue bars are legal agents with naive average-day forecasts.",
         fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, ROOT / "reports/10_hard_mode/figs/losses.png")
print("saved")
