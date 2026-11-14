"""
Q7 (operational part) + extras - how much value is in the signals and how much lookahead it takes.
Runs LP benchmarks on all 6,768 train hours (load/PV gaps imputed with load_hourly_filled; the
load cap binds in ~0.5 % of hours so the imputation barely matters):

  * perfect-foresight optimum for the score, for CO2 only and for price only
  * receding-horizon LP with a lookahead of 4 .. 72 hours (24 h is what `obs` provides)
  * 24 h known + a naive "same hour yesterday" forecast for hours 24-47
  * sensitivity of the score to the CO2 weight assumed by the controller

Simulation results are cached in reports/01_eda/stats/q9_runs.json (delete to recompute).
  fig q9_lookahead_value.png, q9_oracle_schedule.png, q9_weight_sensitivity.png
  stats q9_foresight.json
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import battery_lp as B
import common as C

C.set_style()
CACHE = C.STATS_DIR / "q9_runs.json"
HORIZONS = [2, 4, 6, 8, 12, 16, 20, 24, 30, 36, 48, 72]

df = C.load_hourly_filled()
net = (df["load_kw"] - df["pv_kw"]).values
w = B.weights(df) * 1e6          # scaled for the solver; evaluate() does the real accounting
T = len(df)
days = T / 24

# ------------------------------------------------------------------ simulations (cached)
if CACHE.exists():
    runs = json.loads(CACHE.read_text(encoding="utf-8"))
else:
    runs = {"mpc": {}, "weight": {}, "meta": {"hours": T}}
    t0 = time.time()
    c, d, s = B.solve_lp(w, net)
    runs["oracle"] = B.evaluate(c - d, df)
    runs["oracle_power"] = [round(float(v), 3) for v in (c - d)]
    for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ww = B.weights(df, a, 1 - a) * 1e6
        c2, d2, _ = B.solve_lp(ww, net)
        runs["weight"][f"{a:.1f}"] = B.evaluate(c2 - d2, df)
    print(f"oracles done in {time.time() - t0:.1f}s", flush=True)
    for h in HORIZONS:
        t0 = time.time()
        p = B.run_mpc(w, net, h)
        runs["mpc"][str(h)] = B.evaluate(p, df)
        if h == 24:
            runs["mpc24_power"] = [round(float(v), 3) for v in p]
        print(f"mpc H={h}: score {runs['mpc'][str(h)]['score']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    # naive extension: hours 24..47 forecast = the value 24 h earlier (inside the known window)
    fc = np.full((T, 24), np.nan)
    for k in range(24):
        fc[:, k] = np.r_[w[k:], np.full(k, w[-1])][:T]        # w[t+k]  ~ forecast for t+24+k
    p = B.run_mpc(w, net, 48, known=24, w_forecast=fc)
    runs["mpc48_persistence"] = B.evaluate(p, df)
    print(f"mpc 24 known + 24 persistence: {runs['mpc48_persistence']['score']:.3f}", flush=True)
    CACHE.write_text(json.dumps(runs), encoding="utf-8")

oracle = runs["oracle"]
stats = {
    "hours": T, "days": days,
    "oracle": oracle,
    "oracle_co2_only": runs["weight"]["1.0"],
    "oracle_price_only": runs["weight"]["0.0"],
    "mpc": {h: v for h, v in runs["mpc"].items()},
    "mpc_score_share_of_oracle": {h: v["score"] / oracle["score"] for h, v in runs["mpc"].items()},
    "mpc48_persistence": runs["mpc48_persistence"],
    "weight_sensitivity_score": {a: v["score"] for a, v in runs["weight"].items()},
}

# ------------------------------------------------------------------ oracle schedule by month/hour
p_or = pd.Series(runs["oracle_power"], index=df.index)
p24 = pd.Series(runs["mpc24_power"], index=df.index)
stats["mpc24_vs_oracle_mean_abs_diff_kw"] = float((p_or - p24).abs().mean())
stats["mpc24_hours_identical_action_share"] = float(((p_or - p24).abs() < 1.0).mean())
dis = (-p_or).clip(lower=0)
chg = p_or.clip(lower=0)
daily_dis = dis.groupby(dis.index.normalize()).sum() / (B.CAP * B.ETA)      # equivalent full cycles
stats["oracle_cycles_per_day_mean"] = float(daily_dis.mean())
stats["oracle_cycles_per_day_quantiles"] = {q: float(daily_dis.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
stats["oracle_cycles_per_day_by_month"] = {C.MONTH_ABBR[m]: float(v) for m, v in
                                           daily_dis.groupby(daily_dis.index.month).mean().items()}
stats["oracle_share_days_ge_1p5_cycles"] = float((daily_dis >= 1.5).mean())
stats["oracle_share_days_le_1p1_cycles"] = float((daily_dis <= 1.1).mean())
stats["oracle_share_hours_idle"] = float((p_or.abs() < 1).mean())
stats["oracle_share_hours_full_power"] = float((p_or.abs() > 499).mean())

# value by month: score points earned per day (additive: sum w_t * (d - c) / totals)
wt = pd.Series(B.weights(df), index=df.index)
gain = 100 * wt * (-p_or)
stats["oracle_score_points_per_100_days_by_month"] = {
    C.MONTH_ABBR[m]: float(v) for m, v in (gain.groupby(gain.index.month).sum()
                                           / (gain.groupby(gain.index.month).size() / 24) * 100).items()}
co2_saved = (df["co2_kg_per_kwh"] * (-p_or))
eur_saved = (df["precio_eur_per_kwh"] * (-p_or))
stats["oracle_kg_saved_per_day_by_month"] = {C.MONTH_ABBR[m]: float(v) for m, v in
                                             (co2_saved.groupby(co2_saved.index.month).sum()
                                              / (co2_saved.groupby(co2_saved.index.month).size() / 24)).items()}
stats["oracle_eur_saved_per_day_by_month"] = {C.MONTH_ABBR[m]: float(v) for m, v in
                                              (eur_saved.groupby(eur_saved.index.month).sum()
                                               / (eur_saved.groupby(eur_saved.index.month).size() / 24)).items()}
C.write_stats("q9_foresight", stats)

# ================================================================== FIGURE: value of lookahead
hs = np.array(HORIZONS)
sc = np.array([runs["mpc"][str(h)]["score"] for h in HORIZONS])
fig = plt.figure(figsize=(13.5, 6.4))
s24 = runs["mpc"]["24"]["score"]
C.titled(fig, f"The 24-hour lookahead already captures {s24 / oracle['score']:.2%} of the perfect-foresight score",
         "Score on the train year of a receding-horizon optimiser that knows the next H hours of price and CO₂ "
         "exactly, against the full-year optimum.")
ax = fig.add_axes([0.065, 0.14, 0.60, 0.66])
ax.axhline(oracle["score"], color=C.INK, lw=1.2)
ax.text(72, oracle["score"] + 0.07, f"Perfect foresight of the whole year: {oracle['score']:.2f} points", ha="right",
        va="bottom", fontsize=10.5, color=C.INK)
ax.plot(hs, sc, color=C.C_W, lw=2.2, marker="o", ms=7, mec=C.SURFACE, mew=1.5)
ax.axvline(24, color=C.MUTED, lw=1.0)
ax.text(24.8, 0.25, "what the agent is given:\nthe next 24 hours", fontsize=10.5, color=C.INK2, va="bottom")
ax.annotate(f"{s24:.2f}", xy=(24, s24), xytext=(-12, -22), textcoords="offset points", fontsize=11, color=C.INK,
            ha="center", fontweight="bold")
for h in (2, 4, 6, 8, 12):
    ax.annotate(f"{runs['mpc'][str(h)]['score']:.2f}", xy=(h, runs["mpc"][str(h)]["score"]), xytext=(10, -14),
                textcoords="offset points", fontsize=10.5, color=C.INK2)
ax.set_xlim(0, 74)
ax.set_ylim(0, oracle["score"] * 1.12)
ax.set_xticks([0, 6, 12, 24, 36, 48, 72])
ax.set_xlabel("Lookahead H known to the optimiser (hours)")
ax.set_ylabel("Score (points)")
ax.set_title("Score against lookahead length")

# right: what extra information adds beyond 24 h
# The gains are a thousand times smaller than the score, so they are shown as numbers, not bars.
ax2 = fig.add_axes([0.715, 0.14, 0.265, 0.66])
ax2.axis("off")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.text(0.0, 1.0, "What more information would add", fontsize=12.5, fontweight="bold", color=C.INK, va="bottom")
ax2.text(0.0, 0.925, "Score change against the 24 h lookahead, in points", fontsize=10.5, color=C.INK2, va="center")
rows = [("Naive forecast of hours 24–47\n(same hour one day earlier)", runs["mpc48_persistence"]["score"] - s24),
        ("Perfect knowledge of hours 24–47", runs["mpc"]["48"]["score"] - s24),
        ("Perfect knowledge of hours 24–71", runs["mpc"]["72"]["score"] - s24),
        ("Perfect knowledge of the whole year", oracle["score"] - s24)]
stats["gain_vs_24h_points"] = {name.replace("\n", " "): float(v) for name, v in rows}
C.write_stats("q9_foresight", stats)
for i, (name, v) in enumerate(rows):
    y = 0.78 - i * 0.19
    ax2.plot([0, 1], [y + 0.095, y + 0.095], color=C.GRID, lw=1.0)
    ax2.text(0.0, y, name, fontsize=11, color=C.INK2, va="center", linespacing=1.35)
    ax2.text(1.0, y, f"{v:+.4f}", fontsize=17, fontweight="bold", color=C.INK, va="center", ha="right")
ax2.plot([0, 1], [0.78 - 3 * 0.19 - 0.095] * 2, color=C.GRID, lw=1.0)
ax2.text(0.0, 0.035, f"For scale: the score itself is {s24:.2f} points and\none physical infraction costs 0.20.",
         fontsize=10.5, color=C.INK2, va="center", linespacing=1.4)
C.footnote(fig, f"Sample: all {T:,} train hours simulated as one sequence ({days:.0f} days, SOC carried over, load/PV gaps "
                "imputed). Optimiser = linear programme re-solved every hour; no forecast error inside the horizon.")
C.save(fig, "q9_lookahead_value")

# ================================================================== FIGURE: oracle schedule
fig = plt.figure(figsize=(13.5, 6.4))
C.titled(fig, "The optimal battery charges at the solar midday trough and again before dawn, and discharges at both peaks",
         "Mean power of the perfect-foresight schedule by month and hour (kW). Blue = charging, red = discharging.")
ax = fig.add_axes([0.065, 0.14, 0.56, 0.66])
mat = p_or.groupby([p_or.index.month, p_or.index.hour]).mean().unstack().reindex(C.MONTH_ORDER)
im = ax.imshow(-mat.values, aspect="auto", cmap=C.DIV_BLUE_RED, vmin=-500, vmax=500)
ax.set_yticks(range(12))
ax.set_yticklabels([C.MONTH_ABBR[m] for m in C.MONTH_ORDER])
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day (local time)")
ax.grid(False)
ax.tick_params(length=0)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
ax.grid(which="minor", color=C.SURFACE, linewidth=1.5)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045, ticks=[-500, -250, 0, 250, 500])
cb.ax.set_yticklabels(["500 charge", "250", "0", "250", "500 discharge"])
cb.outline.set_visible(False)
ax.set_title("Mean battery power (kW)")

ax2 = fig.add_axes([0.745, 0.14, 0.235, 0.66])
cyc = pd.Series(stats["oracle_cycles_per_day_by_month"]).reindex([C.MONTH_ABBR[m] for m in C.MONTH_ORDER])
ypos = np.arange(12)[::-1]
ax2.barh(ypos, cyc.values, height=0.5, color=C.C_W)
for y, v in zip(ypos, cyc.values):
    ax2.text(v + 0.03, y, f"{v:.2f}", va="center", fontsize=10.5, color=C.INK)
ax2.set_yticks(ypos)
ax2.set_yticklabels(cyc.index)
ax2.set_xlim(0, 2.1)
ax2.set_xlabel("Equivalent full cycles per day")
ax2.set_title(f"Cycles per day: {stats['oracle_cycles_per_day_mean']:.2f} on average")
ax2.grid(axis="y", visible=False)
C.footnote(fig, f"Sample: perfect-foresight linear programme on all {T:,} train hours, objective = official score. "
                "One full cycle = 2,000 kWh charged and discharged.")
C.save(fig, "q9_oracle_schedule")

# ================================================================== FIGURE: weight sensitivity
a_vals = np.array(sorted(float(a) for a in runs["weight"]))
sc_a = np.array([runs["weight"][f"{a:.1f}"]["score"] for a in a_vals])
co2_a = np.array([runs["weight"][f"{a:.1f}"]["co2_red_pct"] for a in a_vals])
cost_a = np.array([runs["weight"][f"{a:.1f}"]["cost_red_pct"] for a in a_vals])
fig = plt.figure(figsize=(13.5, 6.2))
C.titled(fig, f"Chasing CO₂ alone keeps {runs['weight']['1.0']['score'] / oracle['score']:.0%} of the best score, "
              f"chasing price alone only {runs['weight']['0.0']['score'] / oracle['score']:.0%}",
         "Perfect-foresight optimum when the controller weights CO₂ by a and price by 1 − a. The official score "
         "uses a = 0.7.")
ax = fig.add_axes([0.065, 0.14, 0.915, 0.66])
ax.plot(a_vals, co2_a, color=C.C_CO2, lw=2.0, marker="o", ms=6, mec=C.SURFACE, mew=1.2, label="CO₂ reduction (%)")
ax.plot(a_vals, cost_a, color=C.C_PRICE, lw=2.0, marker="o", ms=6, mec=C.SURFACE, mew=1.2, label="Cost reduction (%)")
ax.plot(a_vals, sc_a, color=C.C_W, lw=2.6, marker="o", ms=7, mec=C.SURFACE, mew=1.2,
        label="Score = 0.7 × CO₂ + 0.3 × cost")
ax.axvline(0.7, color=C.MUTED, lw=1.0)
ax.text(0.705, 0.35, "official weights", fontsize=10.5, color=C.INK2, rotation=90, va="bottom")
ax.text(1.012, co2_a[-1], "CO₂", va="center", fontsize=11, color=C.INK, fontweight="bold")
ax.text(1.012, cost_a[-1], "Cost", va="center", fontsize=11, color=C.INK, fontweight="bold")
ax.text(1.012, sc_a[-1], "Score", va="center", fontsize=11, color=C.INK, fontweight="bold")
for a_, v in zip(a_vals, sc_a):
    if round(a_, 1) in (0.0, 0.5, 0.7, 1.0):
        below = round(a_, 1) == 1.0
        ax.annotate(f"{v:.2f}", xy=(a_, v), xytext=(0, -19 if below else 9), textcoords="offset points", ha="center",
                    fontsize=10.5, color=C.INK)
ax.set_xlim(-0.03, 1.08)
ax.set_ylim(0, max(cost_a.max(), co2_a.max()) * 1.12)
ax.set_xticks(a_vals)
ax.set_xlabel("CO₂ weight a used by the controller  (0 = price only, 1 = CO₂ only)")
ax.set_ylabel("Reduction vs no battery (%)  /  score (points)")
ax.legend(loc="upper center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.5, 1.09), borderaxespad=0)
C.footnote(fig, f"Sample: all {T:,} train hours, perfect foresight. The flat top between a = 0.5 and 0.9 means the exact "
                "CO₂-to-cost normalisation of the hidden test set matters little.")
C.save(fig, "q9_weight_sensitivity")

# ================================================================== FIGURE: two example weeks
sig = C.add_w(df)                      # w normalised with the organisers' train baseline (mean = 1)
soc = pd.Series(B.SOC0 + np.cumsum(np.where(p_or.values > 0, p_or.values * B.ETA, p_or.values / B.ETA)), index=df.index)
examples = [("Winter week: 12–16 January 2026", "2026-01-12", "2026-01-17"),
            ("Summer week: 6–10 July 2026", "2026-07-06", "2026-07-11")]
CHG, DIS = "#9ec5f4", "#f3a9a3"
fig = plt.figure(figsize=(13.5, 7.0))
C.titled(fig, "The optimal schedule fills the battery in each trough of the score signal and empties it on the next peak",
         "Top: combined signal w (1.0 = average hour) with the hours in which the perfect-foresight schedule charges or "
         "discharges. Bottom: the resulting state of charge.", y=0.978)
gs = fig.add_gridspec(2, 2, left=0.065, right=0.985, top=0.80, bottom=0.11, wspace=0.10, hspace=0.20, height_ratios=[1.5, 1])
import matplotlib.dates as mdates
from matplotlib.patches import Patch

for j, (name, a, b) in enumerate(examples):
    sl = slice(pd.Timestamp(a), pd.Timestamp(b) - pd.Timedelta(hours=1))
    wj, pj, sj = sig["w"][sl], p_or[sl], soc[sl]
    ax = fig.add_subplot(gs[0, j])
    for t, v in pj.items():
        if abs(v) > 1:
            ax.axvspan(t, t + pd.Timedelta(hours=1), color=CHG if v > 0 else DIS, alpha=min(1.0, 0.20 + 0.60 * abs(v) / 500),
                       lw=0)
    ax.step(wj.index, wj.values, where="post", color=C.INK, lw=1.8)
    ax.set_ylim(0, 2.6)
    ax.set_title(name)
    ax.set_xlim(pd.Timestamp(a), pd.Timestamp(b))
    ticks = pd.date_range(a, periods=5, freq="D")
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    ax2 = fig.add_subplot(gs[1, j])
    ax2.fill_between(sj.index + pd.Timedelta(hours=1), 0, sj.values, color=C.GRAY, alpha=0.35, lw=0)
    ax2.plot(sj.index + pd.Timedelta(hours=1), sj.values, color=C.INK2, lw=1.6)
    ax2.set_ylim(0, 2100)
    ax2.set_yticks([0, 1000, 2000])
    ax2.set_xlim(pd.Timestamp(a), pd.Timestamp(b))
    ax2.set_xticks(ticks)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    for lbl in ax2.get_xticklabels():
        lbl.set_ha("left")
    if j == 0:
        ax.set_ylabel("Score signal w")
        ax2.set_ylabel("State of charge (kWh)")
        ax.legend(handles=[Patch(facecolor=CHG, label="Charging"), Patch(facecolor=DIS, label="Discharging"),
                           plt.Line2D([], [], color=C.INK, lw=1.8, label="w")],
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=10.5, handlelength=1.3,
                  columnspacing=1.4, borderaxespad=0.2)
    else:
        ax.set_yticklabels([])
        ax2.set_yticklabels([])
    cyc = (-pj.clip(upper=0)).sum() / (B.CAP * B.ETA) / 5
    ax2.set_title(f"{cyc:.1f} full cycles per day", loc="right", fontsize=10.5, fontweight="normal", color=C.INK2, pad=4)
C.footnote(fig, "Sample: two five-day windows (Monday to Friday) cut from the perfect-foresight schedule of the whole train "
                "year. Shading strength = share of the 500 kW limit used.")
C.save(fig, "q9_example_weeks")

print(json.dumps({k: v for k, v in stats.items() if k not in ("mpc",)}, indent=1)[:3500])
