# Usage: .venv/Scripts/python.exe experiments/23_signal/figs.py
"""Figures for reports/13_signal_forecast (reads offline.csv, closed_loop.csv, diag_days.csv)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAPPA, ROOT, get_folds, replay, signal_fit_table, truth  # noqa: E402
from signal_fc import SignalFC  # noqa: E402
from src.hard_mpc import ClimW  # noqa: E402
from src.plotstyle import C, GREY, INK, apply, save  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = ROOT / "reports" / "13_signal_forecast" / "figs"
BEST = sys.argv[1] if len(sys.argv) > 1 else "FINAL"
BEST_CL = sys.argv[2] if len(sys.argv) > 2 else "FINAL|perfectN|H36"


def fig_closed_loop():
    cl = pd.read_csv(HERE / "closed_loop.csv")
    piv = cl.pivot(index="config", columns="fold", values="score")
    names = ["clim|perfectN|H36", BEST_CL, "perfect|perfectN|H36"]
    labels = ["Climatology beyond +2 h", "This forecaster", "Perfect signal"]
    cols = [GREY, C[0], INK["muted"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(4)
    for j, (n, lab, col) in enumerate(zip(names, labels, cols)):
        v = [piv.loc[n, f] for f in "ABC"] + [piv.loc[n, ["A", "B", "C"]].mean()]
        b = ax.bar(x + (j - 1) * 0.27, v, 0.25, color=col, label=lab)
        for xi, vi in zip(x + (j - 1) * 0.27, v):
            ax.text(xi, vi + 0.05, f"{vi:.2f}", ha="center", va="bottom", fontsize=8, color=INK["secondary"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Fold A (cold start)", "Fold B", "Fold C", "Mean"])
    lo = min(piv.loc[names[0]].min(), piv.loc[names[1]].min()) - 1.0
    ax.set_ylim(lo, piv.loc[names[2]].max() + 1.1)
    ax.set_ylabel("Score (points), net load known")
    m = [piv.loc[n, ["A", "B", "C"]].mean() for n in names]
    ax.set_title(f"The forecast recovers {100 * (m[1] - m[0]) / (m[2] - m[0]):.0f} % of the score lost to the 3-hour window")
    ax.legend(loc="upper center", ncol=3)
    save(fig, FIGS / "closed_loop.png")


def fig_mae():
    off = pd.read_csv(HERE / "offline.csv")
    off = off[off.fold.isin(["B", "C"])]
    hs = [3, 6, 12, 18, 24, 36, 47]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (sig, unit, scale, ttl) in zip(axes, [("co2", "g CO2/kWh", 1e3, "CO2 intensity"), ("price", "c EUR/kWh", 1e2, "Price")]):
        for name, col, lab in (("ClimW", GREY, "Climatology"), (BEST, C[0], "This forecaster")):
            r = off[off.model == name][[f"{sig}_{h}" for h in hs]].mean() * scale
            ax.plot(hs, r.values, marker="o", color=col, label=lab)
        ax.set_xlabel("Hours ahead")
        ax.set_ylabel(f"Mean absolute error ({unit})")
        ax.set_xticks(hs)
        ax.set_xlim(2, 48)
        ax.set_ylim(0, None)
        ax.set_title(ttl, fontsize=10, fontweight="normal")
    axes[0].legend(loc="lower right")
    fig.suptitle("Forecast error by horizon, folds B and C: lower at every horizon, most for the price", x=0.02, y=1.03,
                 ha="left", fontweight="bold", fontsize=12)
    save(fig, FIGS / "mae_by_horizon.png")


def fig_example(fold="B", start="2026-03-12 12:00", H=36):
    df = get_folds()[fold]
    fit = signal_fit_table(fold)
    fc = SignalFC(fit, KAPPA, variant="A48", use_lw=False, cross=True, ridge=0.01, interp=True)
    _, _, W = replay(fc, df, H=H)
    _, _, Wc = replay(ClimW(fit), df, H=H, parts=False)
    T = truth(df, H=H)["w"]
    origins = [pd.Timestamp(start) + pd.Timedelta(hours=24 * j) for j in range(3)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, t0 in zip(axes, origins):
        i = df.index.get_loc(t0)
        x = np.arange(H)
        ax.plot(x, T[i] * 1e3, color=INK["primary"], lw=1.6, label="Actual")
        ax.plot(x[2:], Wc[i, 2:] * 1e3, color=GREY, label="Climatology")
        ax.plot(x[2:], W[i, 2:] * 1e3, color=C[0], label="This forecaster")
        ax.axvspan(-0.5, 2.5, color=INK["grid"], alpha=0.6, lw=0)
        ax.set_title(f"{t0:%a %d %b, %H:%M}", fontsize=10, fontweight="normal")
        ax.set_xlabel("Hours ahead")
    axes[0].set_ylabel("Signal w (g CO2-equivalent per kWh)")
    axes[1].legend(loc="upper right")
    fig.suptitle("Forecasts made at noon on three consecutive days: the timing of troughs and peaks is right, the height "
                 "of the evening peak is the hard part (shaded: the 3 known hours)", x=0.02, y=1.04, ha="left",
                 fontweight="bold", fontsize=12)
    save(fig, FIGS / "example_forecasts.png")


def fig_diag():
    t = pd.read_csv(HERE / "diag_days.csv")
    order = ["Tue-Fri", "Monday", "Saturday", "Sunday", "holiday"]
    g = t.groupby("kind")[["loss_fc", "loss_clim"]].mean().reindex(order).dropna()
    n = t.groupby("kind").size().reindex(g.index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (grp, ttl, idx) in zip(axes, [(g, "By kind of day", g.index),
                                           (t.groupby("co2_shape")[["loss_fc", "loss_clim"]].mean(), "By shape of the day's CO2 curve", None)]):
        idx = grp.index if idx is None else idx
        cnt = t.groupby("co2_shape").size().reindex(idx) if ttl.startswith("By shape") else n
        x = np.arange(len(idx))
        ax.bar(x - 0.2, grp["loss_clim"].values * 100, 0.38, color=GREY, label="Climatology")
        ax.bar(x + 0.2, grp["loss_fc"].values * 100, 0.38, color=C[0], label="This forecaster")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{str(a).split(' (')[0]}\n({int(c)} days)" for a, c in zip(idx, cnt)], fontsize=9)
        ax.set_title(ttl, fontsize=10, fontweight="normal")
    axes[0].set_ylabel("Score lost per day against a perfect signal\n(hundredths of a point)")
    axes[0].legend()
    fig.suptitle("The forecast loses less than climatology on all kinds of day but Mondays (both near zero); holidays, Sundays and flat "
                 "CO2 curves stay the most expensive (folds A, B, C)", x=0.02, y=1.04, ha="left", fontweight="bold", fontsize=12)
    save(fig, FIGS / "loss_by_day_kind.png")


if __name__ == "__main__":
    apply()
    only = sys.argv[3:]
    for f in (fig_closed_loop, fig_mae, fig_example, fig_diag):
        if only and f.__name__ not in only:
            continue
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:      # keep the other figures if one input is missing
            print("FAILED", f.__name__, repr(e))
