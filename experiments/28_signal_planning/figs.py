# Usage: .venv/Scripts/python.exe experiments/28_signal_planning/figs.py
"""Figures for reports/19_signal_planning: (1) what the signal-forecast errors look like and the
scenario paths built from them, (2) the scan of the blend weight and the number of paths.
Reads errors_B.npz, the tapes of experiments/27_combined and results_*.csv."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.plotstyle import C, GREY, INK, apply, save  # noqa: E402
import sigplan as sp  # noqa: E402

FIGS = ROOT / "reports/19_signal_planning/figs"
REF = "certainty-equivalent, exact planner"
apply()

# --------------------------------------------------------------------------------------
# Figure 1: the errors and the paths
# --------------------------------------------------------------------------------------
z = np.load(HERE / "errors_B.npz")
E, F = z["E"], z["F"]
ok = np.isfinite(E[:, 3:24]).all(1)
level = float(np.nanmean(F[:, 0]))
ks = np.arange(3, 24)
rmse = np.sqrt(np.mean(E[ok][:, 3:24] ** 2, axis=0)) / level * 100
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.24})
ax1.plot(ks, rmse, color=C[0], marker="o", ms=4)
ax1.set_xlabel("Hours ahead (0-2 are given by the organisers)")
ax1.set_ylabel("Forecast error (RMSE), % of the mean signal")
ax1.set_ylim(0, None)
ax1.set_xticks([3, 6, 9, 12, 15, 18, 21])
ax1.set_title(f"The error doubles between +3 h and +5 h\nand is {rmse[8]:.0f}% of the signal at +11 h", fontsize=10.5)

z2 = np.load(ROOT / "experiments/27_combined/cache/tape_B.npz")
W = z2["w_sig"]
t = 24 * 9 + 6                                             # one origin of fold B, 06:00
pool = sp.Paths("B", K=10, m=18)
w = W[t]
paths = pool.paths(w, 6)
hrs = np.arange(paths.shape[1])
for i in range(len(paths)):
    ax2.plot(hrs[2:], paths[i, 2:], color=GREY, lw=1.0, label="10 scenario paths (forecast + real past error vectors)" if i == 0 else None)
ax2.plot(np.arange(18), w[:18], color=C[0], label="Forecast the plan used to trust")
ax2.plot([0, 1, 2], w[:3], color=INK["primary"], marker="o", ms=5, lw=2.0, label="Known (this hour, +1, +2)")
ax2.set_xlabel("Hours ahead")
ax2.set_ylabel("Signal w = CO2 + kappa x price (kg/kWh)")
ax2.set_xticks([0, 3, 6, 9, 12, 15])
ax2.legend(loc="best")
ax2.set_title("One 06:00 decision: the paths fan out from the\nlast known hour and keep the shape of real errors", fontsize=10.5)
fig.suptitle(f"Beyond the 3 known hours the signal is a forecast with a {rmse[0]:.0f}-{rmse[8]:.0f}% error, so the plan is given paths, not one line",
             x=0.065, ha="left", fontsize=12.5, fontweight="bold", y=1.04)
fig.text(0.065, -0.05, "Errors: SignalForecaster fitted without each 7-day block of the fit rows and replayed on it (cross-fitted); here the fit rows of fold B. "
         "Paths: whole error vectors from origins at the same hour of day.", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "errors_and_paths.png")

# --------------------------------------------------------------------------------------
# Figure 2: scan of the blend weight and the number of paths
# --------------------------------------------------------------------------------------
files = [f for f in ("main", "grid") if (HERE / f"results_{f}.csv").exists()]
if files:
    d = pd.concat([pd.read_csv(HERE / f"results_{f}.csv") for f in files]).drop_duplicates(["config", "setting", "fold"], keep="last")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, s, ttl in ((axes[0], "iso", "Signal alone (perfect net load)"), (axes[1], "comb", "Combined agent (all forecasts)")):
        piv = d[d.setting == s].pivot(index="config", columns="fold", values="score")
        ref = piv.loc[REF]
        for i, K in enumerate((5, 10, 20, 50)):
            xs, ys = [0.0], [0.0]
            for lam in (0.25, 0.5, 0.75, 1.0):
                key = f"paths K={K}, lam={lam:.2f}"
                if key in piv.index:
                    xs.append(lam)
                    ys.append(float((piv.loc[key] - ref).mean()))
            ax.plot(xs, ys, color=C[i], marker="o", ms=4, label=f"K = {K} paths")
        key = "paths K=20, lam=0.50"
        if key in piv.index:
            g = piv.loc[key] - ref
            ax.scatter([0.5] * 3, g.values, s=16, color=INK["secondary"], zorder=3, linewidths=0)
            for f_, v in g.items():
                ax.text(0.515, v, f_, fontsize=8, color=INK["secondary"], va="center")
        ax.axhline(0, color=INK["axis"], lw=1.0)
        ax.set_xlabel("Blend weight lam (0 = trust the forecast, 1 = average over paths)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(ttl, fontsize=10.5)
    axes[0].set_ylabel("Score gain over the plan that trusts the forecast\n(mean of folds A, B, C)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Half forecast, half paths is the plateau (about +0.03); paths alone give nothing", x=0.065, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.065, -0.04, "Grey dots: the three folds for K = 20, lam = 0.5. Paths and forecasters fitted outside each fold; paths cover the next 12 hours. Differences of 0.01-0.02 between K are noise (see random draws).",
             fontsize=8.5, color=INK["secondary"], ha="left", va="top")
    save(fig, FIGS / "scan.png")
print("saved")

# --------------------------------------------------------------------------------------
# Figure 3: what each variant gave
# --------------------------------------------------------------------------------------
files = [f for f in ("main", "grid", "fix") if (HERE / f"results_{f}.csv").exists()]
if files:
    d = pd.concat([pd.read_csv(HERE / f"results_{f}.csv") for f in files]).drop_duplicates(["config", "setting", "fold"], keep="last")
    rows = [("Average of the paths' first actions", "paths K=20, average of first actions"),
            ("Median of the paths' first actions", "paths K=20, median of first actions"),
            ("Path-averaged value, lam = 1\n(perfect-information relaxation)", "paths K=20, lam=1.00"),
            ("Blend, lam = 0.5", "paths K=20, lam=0.50"),
            ("Mean correction only\n(last revealed error)", "certainty-equivalent + AR correction x1.0"),
            ("Blend lam = 0.5 + mean correction\n(recommended)", "paths K=20, lam=0.50 + AR correction x1.0")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, gridspec_kw={"wspace": 0.06})
    y = np.arange(len(rows))[::-1]
    for ax, s, ttl in ((axes[0], "iso", "Signal alone (perfect net load): reference 20.69, ceiling 21.26"),
                       (axes[1], "comb", "Combined agent: reference 20.05, perfect signal about +0.7")):
        piv = d[d.setting == s].pivot(index="config", columns="fold", values="score")
        ref = piv.loc[REF]
        for yi, (lab, key) in zip(y, rows):
            g = piv.loc[key] - ref
            col = C[0] if "recommended" in lab else GREY
            ax.barh(yi, g.mean(), height=0.6, color=col)
            ax.scatter(g.values, [yi] * 3, s=14, color=INK["secondary"], zorder=3, linewidths=0)
            x = max(g.max(), g.mean(), 0.0) + 0.012
            ax.text(x, yi, f"{g.mean():+.3f}", va="center", ha="left", fontsize=9.5, color=INK["primary"])
        ax.axvline(0, color=INK["axis"], lw=1.0)
        ax.set_xlim(-0.46, 0.2)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Score gain (points; dots = folds A, B, C)")
        ax.set_title(ttl, fontsize=10.5)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([r[0] for r in rows])
    fig.suptitle("Planning against signal paths recovers a few hundredths of a point, not the 0.6-0.7 lost to the signal forecast",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold", y=1.03)
    fig.text(0.02, -0.04, "Gain over the plan that trusts the forecast. K = 20 paths over the next 12 hours, drawn from cross-fitted forecast errors at the same hour of day. "
             "Everything fitted outside each fold.", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
    save(fig, FIGS / "variants.png")
    print("saved variants")
