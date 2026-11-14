"""Figures for reports/02_robustness: what the agent does through an outage, with and without
the guard (real outage of 17-18 June 2026) and through a partial logger glitch (v1 vs v1.1).

    .venv/Scripts/python.exe experiments/02_stress/figs.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *          # noqa: F401,F403

FIGS = ROOT / "reports" / "02_robustness" / "figs"
SURFACE, INK, INK2, GRID_C = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"no guard": "#eb6834", "v1": "#2a78d6", "v1.1": "#1baf7a"}      # colour follows the agent
BAND = "#efeeea"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID_C, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID_C, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def panel_figure(df, traces, t0, t1, band, title, subtitle, notes, fname, kappa=0.2571):
    """4 stacked panels on one time axis: signal, recorded net load, battery power, stored energy."""
    s = slice(t0, t1)
    idx = df.loc[s].index
    w = (df["co2_kg_per_kwh"] + kappa * df["precio_eur_per_kwh"]).loc[s]
    net = (df["load_kw"] - df["pv_kw"]).loc[s]
    fig, ax = plt.subplots(4, 1, figsize=(9.2, 8.6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.5, 1.2], "hspace": 0.32})
    for a in ax:
        a.axvspan(pd.Timestamp(band[0]), pd.Timestamp(band[1]), color=BAND, lw=0, zorder=0)
    ax[0].step(idx, w, where="post", color=INK2, lw=1.6)
    ax[0].set_title("What an hour is worth to the score  (CO$_2$ + 0.257 × price, per kWh moved)")
    ax[0].set_ylim(0, None)
    ax[1].fill_between(idx, net, step="post", color="#b9b8b2", lw=0)
    ax[1].set_title("Net load recorded by the meter (kW): what a discharge can displace")
    ax[1].set_ylim(0, None)
    ax[1].text(pd.Timestamp(band[0]) + (pd.Timestamp(band[1]) - pd.Timestamp(band[0])) / 2, ax[1].get_ylim()[1] * 0.45,
               notes["band"], ha="center", va="center", color=INK2, fontsize=8.5)
    for name, tr in traces.items():
        p = tr["p_kw"].loc[s]
        ax[2].step(idx, p, where="post", color=COL[name], lw=2, label=name)
        ax[3].step(idx, tr["soc_kwh"].loc[s], where="post", color=COL[name], lw=2, label=name)
    ax[2].axhline(0, color=INK2, lw=0.8)
    ax[2].set_title("Battery command (kW):  above 0 = charging,  below 0 = discharging")
    ax[2].set_ylim(-560, 560)
    ax[2].legend(loc="upper left", ncol=len(traces), frameon=False, fontsize=8.5, handlelength=1.6)
    ax[3].set_title("Stored energy (kWh)")
    ax[3].set_ylim(-60, 2150)
    for (panel, x, y, text, xy) in notes.get("arrows", []):
        ax[panel].annotate(text, xy=(pd.Timestamp(xy[0]), xy[1]), xytext=(pd.Timestamp(x), y), fontsize=8.5, color=INK,
                           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), ha="left", va="center")
    # direct labels at the right end of the stored-energy lines
    for name, tr in traces.items():
        v = tr["soc_kwh"].loc[s]
        ax[3].annotate(name, xy=(idx[-1], v.iloc[-1]), xytext=(5, notes.get("label_dy", {}).get(name, 0)), textcoords="offset points",
                       color=INK, fontsize=8.5, va="center")
    ax[3].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.suptitle(title, x=0.065, y=0.985, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.065, 0.948, subtitle, ha="left", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.065, right=0.93, top=0.905, bottom=0.07)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / fname, dpi=160)
    plt.close(fig)
    print("saved", FIGS / fname)


def main():
    v1 = load_agent_module(ROOT / "agents" / "v1" / "agent.py", "fig_v1")
    cand = load_agent_module(HERE / "candidate" / "agent.py", "fig_cand")
    NG = make_noguard(v1)

    # ---- figure 1: the real 14-hour outage, fold C (train calendar = real dates) -------------
    df = folds("train")["C"]
    tr = {"no guard": run(NG(), df, trace=True), "v1": run(v1.BatteryAgent(), df, trace=True)}
    sc = {k: r["raw"] for k, r in tr.items()}
    exp = {k: float(np.maximum(-(df["load_kw"] - df["pv_kw"] + r["trace"]["p_kw"]), 0).loc["2026-06-17 12:00":"2026-06-18 12:00"].sum()) for k, r in tr.items()}
    panel_figure(
        df, {k: r["trace"] for k, r in tr.items()}, "2026-06-17 06:00", "2026-06-18 14:00",
        ("2026-06-17 17:00", "2026-06-18 07:00"),
        "Through a real meter outage: the guard holds the energy instead of exporting it for nothing",
        "17-18 June 2026 (fold C). Without the guard the optimiser sends %.0f kWh to the grid, worth nothing, and re-buys it: "
        "fold score %.3f vs %.3f with v1's guard." % (exp["no guard"], sc["no guard"], sc["v1"]),
        {"band": "meter reads exactly 0 for 14 hours",
         "arrows": [(2, "2026-06-17 23:30", -330, "no guard: evening peak\ndischarged into zero load", ("2026-06-17 21:30", -480)),
                    (2, "2026-06-18 08:40", 330, "v1: blocked one hour too long\n(last reading still 0)", ("2026-06-18 07:30", 0))],
         "label_dy": {"no guard": -6, "v1": 6}},
        "outage_guard.png")

    # ---- figure 2: a partial glitch hour, v1 vs v1.1 ------------------------------------------
    base = folds("test")["A"]
    df2 = glitches_frozen(base, 110 / 60, 19.4, weeks=[6], day=2)
    day = df2.index[168 * 6 + 48].normalize()
    tr2 = {"v1": run(v1.BatteryAgent(), df2, trace=True), "v1.1": run(cand.BatteryAgent(), df2, trace=True)}
    sc2 = {k: r["raw"] for k, r in tr2.items()}
    a, b = day + pd.Timedelta(hours=12), day + pd.Timedelta(hours=34)
    exp2 = {k: float(np.maximum(-(df2["load_kw"] - df2["pv_kw"] + r["trace"]["p_kw"]), 0).loc[a:b].sum()) for k, r in tr2.items()}
    panel_figure(
        df2, {k: r["trace"] for k, r in tr2.items()}, a, b,
        (day + pd.Timedelta(hours=19), day + pd.Timedelta(hours=22)),
        "A logger glitch that starts mid-hour: v1 trusts the half-empty reading, v1.1 reads it as a warning",
        "Injected glitch 19:24-21:14 (same shape as the real one of 6 Aug 2026). The 19:00 reading is 40%% of normal; v1 still discharges 95%% of it "
        "into a dead hour.\nEnergy exported for nothing: v1 %.0f kWh, v1.1 %.0f kWh. Run score %.4f vs %.4f." % (exp2["v1"], exp2["v1.1"], sc2["v1"], sc2["v1.1"]),
        {"band": "glitch",
         "arrows": [(2, (day + pd.Timedelta(hours=22, minutes=30)).isoformat(), -200, "v1: 95% of a\npartial reading", ((day + pd.Timedelta(hours=20, minutes=30)).isoformat(), -300))],
         "label_dy": {"v1": -6, "v1.1": 6}},
        "glitch_guard.png")


if __name__ == "__main__":
    main()
