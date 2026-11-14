# Usage: .venv/Scripts/python.exe experiments/23_signal/figs2.py
"""Round 2 figures (reads need.csv, errstruct_hour.csv, errstruct_horizon.csv)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.plotstyle import C, GREY, INK, apply, save  # noqa: E402

FIGS = ROOT / "reports" / "13_signal_forecast" / "figs"


def fig_need():
    d = pd.read_csv(HERE / "need.csv")
    m = d.groupby("variant")[["pn", "comb"]].mean()
    base = m.loc["forecast (round 1)"]
    rows = [
        ("truth in +3..+3", "True value of hour +3"),
        ("truth in +3..+5", "True values of hours +3 to +5"),
        ("truth in +3..+8", "True values of hours +3 to +8"),
        ("truth in +12..+35", "True values of hours +12 to +35"),
        None,
        ("true level, forecast shape, +3..+8", "True mean of +3 to +8, forecast shape"),
        ("true min and max, forecast in between, +3..+8", "True minimum and maximum of +3 to +8"),
        ("true values in the forecast's order, +3..+8", "True values of +3 to +8, in the forecast's order"),
        ("forecast values in the true order, +3..+8", "Forecast values of +3 to +8, in the true order"),
        ("forecast level, true shape, +3..+8", "True shape of +3 to +8, forecast mean"),
    ]
    labels, a, b = [], [], []
    for r in rows:
        if r is None:
            labels.append(""); a.append(np.nan); b.append(np.nan)
        else:
            labels.append(r[1]); a.append(m.loc[r[0], "pn"] - base["pn"]); b.append(m.loc[r[0], "comb"] - base["comb"])
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(y + 0.2, a, 0.38, color=C[0], label="Net load known")
    ax.barh(y - 0.2, b, 0.38, color=C[1], label="Combined legal agent")
    for yy, v in list(zip(y + 0.2, a)) + list(zip(y - 0.2, b)):
        if np.isfinite(v):
            ax.text(v + (0.008 if v >= 0 else -0.008), yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=INK["secondary"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color=INK["axis"], lw=0.8)
    ax.set_xlabel("Score gained over the current forecast when it is given this piece of the truth (points, mean of folds A, B, C)")
    ax.set_xlim(min(np.nanmin(a), np.nanmin(b)) - 0.08, max(np.nanmax(a), np.nanmax(b)) + 0.08)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("The planner needs the level and the extremes of the next six hours, not their order")
    save(fig, FIGS / "what_the_planner_needs.png")


def fig_level_by_hour():
    h = pd.read_csv(HERE / "errstruct_hour.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    hard = h.mae_level_g >= 13.0
    ax.bar(h.hour, h.mae_level_g, 0.8, color=np.where(hard, C[0], GREY))
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day at which the forecast is made")
    ax.set_ylabel("Mean absolute error of the mean of w over hours +3 to +8\n(g CO2-equivalent per kWh)")
    ax.set_title("The level of the next six hours is hardest to call before the morning ramp and before the evening peak")
    save(fig, FIGS / "level_error_by_hour.png")


def fig_need_hour():
    d = pd.read_csv(HERE / "need_hour.csv")
    base = d[d.band == "none"].set_index("fold")
    d = d[d.band != "none"].copy()
    d["gain"] = d.comb - d.fold.map(base.comb)
    g = d.groupby("band").agg(lvl=("lvl_mae_g", "mean"), gain=("gain", "mean"), hours=("hours", "mean"))
    g["gain100"] = g.gain / g.hours * 100
    x = np.arange(len(g))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    top = g.gain100 >= 0.04
    axes[0].bar(x, g.lvl, 0.7, color=GREY)
    axes[0].set_ylabel("Mean absolute error of the mean of w\nover hours +3 to +8 (g CO2-eq. per kWh)")
    axes[0].set_title("Size of the forecast error", fontsize=10, fontweight="normal")
    axes[1].bar(x, g.gain100 * 100, 0.7, color=np.where(top, C[0], GREY))
    axes[1].set_ylabel("Score gained per 100 forecasts if hours +3 to +8\nwere known (hundredths of a point)")
    axes[1].set_title("What the error costs (combined legal agent)", fontsize=10, fontweight="normal")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace(" h", "") for b in g.index], fontsize=9)
        ax.set_xlabel("Hours of day at which the forecast is made")
    fig.suptitle("The forecast is worst in the early afternoon, but it costs most at night (22 h to 3 h)", x=0.02, y=1.03,
                 ha="left", fontweight="bold", fontsize=12)
    save(fig, FIGS / "where_the_forecast_matters.png")


if __name__ == "__main__":
    apply()
    for f in (fig_need, fig_need_hour):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
