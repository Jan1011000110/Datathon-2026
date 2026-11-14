"""
experiments/29_stress_hard/figs.py  --  figures of reports/21_stress_hard (reads out/*.csv).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plotstyle import C, GREY, INK, apply, save  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
FIGS = ROOT / "reports" / "21_stress_hard" / "figs"
apply()


def suite():
    fs = [OUT / f"suite_{a}_{f}_layout-lowload-faults.csv" for a in ("h1", "h2") for f in ("B", "C")]
    return pd.concat([pd.read_csv(f) for f in fs])


def fig_scenarios():
    d = suite()
    pick = [("12 weeks on the test calendar + both clock changes", "Test calendar, 12 weeks, clock changes\n(weekday labels shifted: not a fair score)"),
            ("12 weeks, test calendar, lists shortened at every week end", "Test calendar + price lists cut\nto 1-2 values at every week end"),
            ("shifted by whole weeks (weekdays kept)", "Weeks moved to days 22-28, weekdays kept"),
            ("gaps of 2-10 h, one per week", "One gap of 2-10 h per week"),
            ("12 duplicated hours", "12 duplicated hours"),
            ("Dec week load x0.6", "December week, load x0.6"),
            ("Dec week: weekdays take the Sunday profile", "December week closed (Sunday profile)"),
            ("Jul week load x0.6", "July week, load x0.6"),
            ("Jul week: weekdays take the Sunday profile", "July week closed (Sunday profile)"),
            ("solar 0 for the last week, irradiance normal", "Solar reads 0 in the last week"),
            ("meter 0 for a whole week (Oct), solar normal", "Meter reads 0 for a week"),
            ("meter 0 for 14 h from 12:00, every week", "Meter reads 0 for 14 h, every week (x11)")]
    t = d.pivot_table(index="scenario", columns="agent", values="d_regret", aggfunc="mean")
    names = [p[1] for p in pick][::-1]
    v2 = [t.loc[p[0], "h2"] for p in pick][::-1]
    v1 = [t.loc[p[0], "h1"] for p in pick][::-1]
    y = np.arange(len(pick))
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.barh(y + 0.19, v2, height=0.34, color=C[0], label="h2 (specialist forecasters)")
    ax.barh(y - 0.19, v1, height=0.34, color=C[1], label="h1 (average-day forecasts)")
    for yy, a, b in zip(y, v2, v1):
        ax.text(max(a, 0) + 0.012, yy + 0.19, f"{a:+.2f}", va="center", fontsize=8, color=INK["secondary"])
        ax.text(max(b, 0) + 0.012, yy - 0.19, f"{b:+.2f}", va="center", fontsize=8, color=INK["secondary"])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color=INK["axis"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("extra regret against the clean run, points (mean of folds B and C; regret = perfect-foresight score - agent score)")
    ax.set_title("No scenario breaks either agent: a low-load week costs 0.1-0.2 points,\nprice lists cut at week ends 0.2-0.4 on top of the calendar run", fontsize=11)
    ax.legend(loc="center right")
    save(fig, FIGS / "scenario_losses.png")


def fig_hidden():
    d = pd.concat([pd.read_csv(OUT / "hidden_parts_B.csv"), pd.read_csv(OUT / "hidden_parts_C.csv")])
    order = ["load", "solar", "load+solar", "signal"]
    lab = ["load rows", "solar rows", "load and solar rows\n(the real November case)", "price / CO2 rows\n(not the case: train has\nNovember prices)"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, p in enumerate(order):
        x = d[d.hidden_from == p]
        ax.scatter(np.full(len(x), k) + np.linspace(-0.12, 0.12, len(x)), x["d_run"], s=36, color=GREY, zorder=2,
                   label="one fold and month (B, C x October, December)" if k == 0 else None)
        m = x["d_run"].mean()
        ax.plot([k - 0.25, k + 0.25], [m, m], color=C[0], linewidth=2.5, zorder=3, label="mean" if k == 0 else None)
        ax.text(k + 0.28, m, f"{m:+.3f}", va="center", fontsize=9, color=INK["secondary"])
    ax.axhline(0, color=INK["axis"], linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_xlim(-0.5, len(order) - 0.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("change in the run's score, points")
    ax.set_title("h2 with a whole month removed from one fit table:\nmissing load and solar history costs about 0.01, missing prices would cost 0.08")
    ax.legend(loc="lower left")
    save(fig, FIGS / "hidden_month.png")


def fig_lowload_days():
    d = pd.read_csv(OUT / "lowload_days.csv")
    d["part"] = np.where(d.day == 1, "day 1", np.where(d.day == 2, "day 2", "days 3-7"))
    n = d.groupby(["fold", "week", "scenario"]).ngroups
    t = d.groupby(["agent", "part"])["extra_regret"].sum().div(n).unstack("agent")
    parts = ["day 1", "day 2", "days 3-7"]
    x = np.arange(len(parts))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - 0.18, t.loc[parts, "h2"], width=0.32, color=C[0], label="h2")
    ax.bar(x + 0.18, t.loc[parts, "h1"], width=0.32, color=C[1], label="h1")
    for xx, a, b in zip(x, t.loc[parts, "h2"], t.loc[parts, "h1"]):
        ax.text(xx - 0.18, max(a, 0) + 0.003, f"{a:+.3f}", ha="center", fontsize=8.5, color=INK["secondary"])
        ax.text(xx + 0.18, max(b, 0) + 0.003, f"{b:+.3f}", ha="center", fontsize=8.5, color=INK["secondary"])
    ax.set_ylim(-0.03, 0.1)
    ax.axhline(0, color=INK["axis"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["first day", "second day", "days 3 to 7 (sum)"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("extra regret in the low-load week, points")
    ax.set_title("A low-load week is paid for in its first two days\n(mean of 8 cases: December and July weeks, folds B and C, load x0.6 or Sunday profile)")
    ax.legend(loc="upper right")
    save(fig, FIGS / "lowload_days.png")


if __name__ == "__main__":
    fig_scenarios()
    fig_hidden()
    fig_lowload_days()
    print("figures written to", FIGS)
