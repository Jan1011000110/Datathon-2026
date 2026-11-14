# Usage: .venv/Scripts/python.exe experiments/24_planner/figs.py
"""Figures for reports/14_planner: the first-step rule under uncertainty, and the closed-loop
value of planning against the forecast distribution. Reads closed_main.csv / closed_ctrl.csv."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, train_complement  # noqa: E402
from src.plotstyle import C, GREY, INK, apply, save  # noqa: E402
import planner as P  # noqa: E402

here = Path(__file__).parent
FIGS = ROOT / "reports/14_planner/figs"
apply()

# --------------------------------------------------------------------------------------
# Figure 1: the first-step rule
# --------------------------------------------------------------------------------------
df = get_folds()["B"]
fit = train_complement(df)
qn = P.QuantileN(fit, H=2)
net = (fit["load_kw"] - fit["pv_kw"])
wk = fit.index.day_of_week < 5
lev = (np.arange(qn.FINE) + 0.5) / qn.FINE

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"wspace": 0.27})

hod = 13
f0 = float(net[wk & (fit.index.hour == hod)].median())          # a typical weekday 13:00 forecast
q = f0 + qn.table[hod, 0]                                        # forecast distribution of this hour's net load
d = np.linspace(0, 1500, 601)
surv = 1.0 - np.searchsorted(q, d, side="right") / len(q)        # P(net load > d)
ax1.plot(d, surv, color=C[0])
ax1.axvline(f0, color=GREY, lw=1.2)
ax1.text(f0 + 12, 1.02, f"point forecast\n{f0:.0f} kW", color=INK["secondary"], fontsize=8.5, va="bottom", ha="left")
for r, lab in ((0.9, "a stored kWh is worth 90% as much later"), (0.6, "worth 60% later"), (0.25, "worth 25% later")):
    dstar = float(np.quantile(q, 1 - r))
    ax1.plot([0, dstar], [r, r], color=INK["muted"], lw=1.0, ls=(0, (4, 3)))
    ax1.plot([dstar, dstar], [0, r], color=INK["muted"], lw=1.0, ls=(0, (4, 3)))
    ax1.scatter([dstar], [r], s=42, color=C[1], zorder=4, edgecolors=INK["surface"], linewidths=1.5)
    ax1.text(dstar + 28, r + 0.012, f"{lab}: discharge {dstar:.0f} kW", fontsize=8.5, color=INK["primary"],
             va="bottom", ha="left")
ax1.set_xlim(0, 1500)
ax1.set_ylim(0, 1.18)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xlabel("Discharge this hour, d (kW)")
ax1.set_ylabel("Chance that the d-th kW still displaces import:  P(net load > d)")
ax1.set_title("Weekday 13:00: discharge until the chance\nof being useful drops to what the kWh is worth later", fontsize=10.5)

r = np.linspace(0.02, 0.98, 97)
for i, (h, lab) in enumerate(((4, "04:00 (no sun, steady load)"), (13, "13:00 (solar uncertain)"), (19, "19:00"))):
    fh = float(net[wk & (fit.index.hour == h)].median())
    qh = fh + qn.table[h, 0]
    dd = np.clip(np.quantile(qh, 1 - r), 0, 1500) / fh
    ax2.plot(r, dd, color=C[i], label=f"{lab}, forecast {fh:.0f} kW")
ax2.axhline(1.0, color=GREY, lw=1.2)
ax2.axhline(0.9, color=GREY, lw=1.2, ls=(0, (4, 3)))
ax2.text(0.985, 1.012, "plan that trusts the forecast", fontsize=8.5, color=INK["secondary"], ha="right", va="bottom")
ax2.text(0.015, 0.888, "fixed 10% haircut", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
ax2.set_xlim(0, 1)
ax2.set_xlabel("Value of keeping one kWh for later / value of using it now  (v / (w·η))")
ax2.set_ylabel("Optimal discharge / point forecast of net load")
ax2.legend(loc="lower left")
ax2.set_title("The right haircut is not a constant: it depends on\nthe hour's uncertainty and on what the energy is worth later", fontsize=10.5)
fig.suptitle("Under uncertainty the best discharge is a quantile of the net-load forecast, not the forecast itself",
             x=0.065, ha="left", fontsize=12.5, fontweight="bold", y=1.04)
fig.text(0.065, -0.05, "Rule: discharge the d-th kW only if  w·η·P(net load > d) > v,  where w = this hour's CO2+price signal and v = marginal value of stored energy "
         "after this hour (from the plan).\nForecast distribution: ProfileN point forecast + its empirical errors by hour of day, "
         "train rows outside fold B. Example forecasts are the weekday median net load of that hour.",
         fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "first_step_rule.png")

# --------------------------------------------------------------------------------------
# Figure 2: closed loop, stochastic plan vs every fixed haircut
# --------------------------------------------------------------------------------------
res = pd.read_csv(here / "closed_main.csv")
if (here / "closed_ctrl.csv").exists():
    res = pd.concat([res, pd.read_csv(here / "closed_ctrl.csv")])
sc = res.groupby("config")["score"].mean()
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=False, gridspec_kw={"wspace": 0.22})
for ax, wl, ttl in ((axes[0], "clim w", "Legal agent (naive price/CO2 forecast)"),
                    (axes[1], "perfect w", "Diagnostic: price/CO2 known 36 h ahead")):
    xs, ys = [], []
    for cfg, v in sc.items():
        if cfg.startswith(f"{wl} + profile n | det, d_shade="):
            xs.append(float(cfg.split("=")[1]))
            ys.append(v)
    xs.append(1.0)
    ys.append(sc[f"{wl} + profile n | det"])
    o = np.argsort(xs)
    xs, ys = np.array(xs)[o], np.array(ys)[o]
    ax.plot(xs, ys, color=GREY, marker="o", ms=5, label="Deterministic plan with a fixed haircut")
    s_all = sc[f"{wl} + profile n | stoch all, Q=9"]
    s_h0 = sc[f"{wl} + profile n | stoch h0, Q=9"]
    ax.axhline(s_all, color=C[0])
    ax.axhline(s_h0, color=C[1])
    ax.text(0.702, s_all + 0.03, f"Stochastic plan, every hour: {s_all:.2f}", color=INK["primary"], fontsize=9, va="bottom")
    ax.text(0.702, s_h0 - 0.03, f"Stochastic plan, this hour only: {s_h0:.2f}", color=INK["primary"], fontsize=9, va="top")
    ib = int(np.argmax(ys))
    ax.plot([xs[ib], xs[ib]], [ys[ib] - 0.06, ys[ib] - 0.5], color=INK["muted"], lw=0.8)
    ax.text(xs[ib], ys[ib] - 0.53, f"best fixed haircut: {ys[ib]:.2f}", fontsize=9, color=INK["primary"],
            ha="center", va="top")
    ax.set_xlim(0.69, 1.01)
    ax.set_ylim(min(ys) - 0.1, s_all + 0.3)
    ax.set_xlabel("Fixed haircut: discharge at most this fraction of the forecast net load")
    ax.set_title(ttl, fontsize=10.5)
axes[0].set_ylabel("Score (mean of folds A, B, C)")
axes[0].legend(loc="lower right")
gain = sc["clim w + profile n | stoch all, Q=9"] - max(v for c, v in sc.items() if c.startswith("clim w + profile n | det"))
fig.suptitle(f"Planning against the forecast distribution beats every fixed haircut by {gain:.2f} points",
             x=0.065, ha="left", fontsize=12.5, fontweight="bold", y=1.02)
fig.text(0.065, -0.04, "Same forecasts in every run (ProfileN net load; errors by hour of day and horizon from train rows outside the fold); "
         "only the planner changes. H = 36 h, 9 scenarios per hour.",
         fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "stoch_vs_haircut.png")
print("saved")

# --------------------------------------------------------------------------------------
# Figure 3: how sensitive is the gain to the distribution we feed the planner?
# --------------------------------------------------------------------------------------
if (here / "closed_ctrl.csv").exists():
    wl = "clim w"
    best_shade = max(v for c, v in sc.items() if c.startswith(f"{wl} + profile n | det"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True, gridspec_kw={"wspace": 0.08})
    sp = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    ys = [sc[f"{wl} + profile n | stoch all, Q=9"] if s == 1.0 else sc[f"{wl} + profile n | stoch all, Q=9, spread x{s:.2f}"] for s in sp]
    ax1.plot(sp, ys, color=C[0], marker="o")
    for x, y in zip(sp, ys):
        ax1.text(x, y + 0.035, f"{y:.2f}", ha="center", va="bottom", fontsize=9, color=INK["primary"])
    ax1.set_xlabel("Width of the error distribution given to the planner\n(1 = as measured on train)")
    ax1.set_ylabel("Score (mean of folds A, B, C)")
    ax1.set_title("A distribution 25% too narrow or too wide costs 0.02-0.04", fontsize=10.5)
    ks = [0, 1, 2, 3, 6, 12, 24, 36]
    lab = {0: "det", 1: "stoch h0, Q=9", 2: "stoch h0-1, Q=9", 3: "stoch h0-2, Q=9", 6: "stoch h0-5, Q=9",
           12: "stoch h0-11, Q=9", 24: "stoch h0-23, Q=9", 36: "stoch all, Q=9"}
    yk = [sc[f"{wl} + profile n | {lab[k]}"] for k in ks]
    ax2.plot(range(len(ks)), yk, color=C[0], marker="o")
    for x, y in zip(range(len(ks)), yk):
        if x == 0:
            ax2.text(x + 0.15, y, f"{y:.2f}", ha="left", va="center", fontsize=9, color=INK["primary"])
        else:
            ax2.text(x, y + 0.035, f"{y:.2f}", ha="center", va="bottom", fontsize=9, color=INK["primary"])
    ax2.set_xticks(range(len(ks)))
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("Leading hours planned against scenarios\n(0 = deterministic plan; later hours use the point forecast)")
    ax2.set_title("The hour being decided brings most of the gain; 6-12 hours bring the rest", fontsize=10.5)
    for ax in (ax1, ax2):
        ax.axhline(best_shade, color=GREY, lw=1.2, ls=(0, (4, 3)))
        ax.set_ylim(18.7, 19.7)
    ax1.text(2.0, best_shade - 0.02, f"best fixed haircut: {best_shade:.2f}", fontsize=8.5, color=INK["secondary"], ha="right", va="top")
    fig.suptitle("The stochastic plan is robust: even a distribution twice too wide beats the best fixed haircut",
                 x=0.065, ha="left", fontsize=12.5, fontweight="bold", y=1.03)
    fig.text(0.065, -0.1, "Legal agent (ClimW signal forecast + ProfileN net-load forecast), H = 36 h, 9 scenarios per hour, folds A, B, C.",
             fontsize=8.5, color=INK["secondary"], ha="left", va="top")
    save(fig, FIGS / "sensitivity.png")
    print("saved sensitivity")
