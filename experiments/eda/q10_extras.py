"""
Extras that bear on how the battery should be run.
  1. A test-like sample: the last 7 train days of each month, concatenated like the hidden test
     (12 x 168 h, SOC carried across the jumps). Perfect-foresight and 24 h-lookahead scores,
     value of the free initial 1,000 kWh, baseline per hour vs the public-leaderboard baseline.
  2. Fixed clock schedules against the optimum (what does day-ahead optimisation buy?).
  3. Where the value is: CO2 and euros saved per day, by month.
  fig q10_value_by_month.png, q10_clock_vs_optimiser.png
  stats q10_extras.json
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import battery_lp as B
import common as C

C.set_style()
stats: dict = {}
df = C.load_hourly_filled()
runs = json.loads((C.STATS_DIR / "q9_runs.json").read_text(encoding="utf-8"))
oracle_full = runs["oracle"]


def simulate(p_req: np.ndarray, soc0: float = B.SOC0) -> np.ndarray:
    """Apply a requested power series with the simulator's clamps (no infractions counted)."""
    soc = soc0
    out = np.zeros(len(p_req))
    for t, a in enumerate(p_req):
        if a > 0:
            a = min(a, B.PMAX, (B.CAP - soc) / B.ETA)
            soc += a * B.ETA
        elif a < 0:
            a = -min(-a, B.PMAX, soc * B.ETA)
            soc += a / B.ETA
        out[t] = a
    return out


# ------------------------------------------------------------------ 1. test-like sample
days = pd.DatetimeIndex(sorted(df.index.normalize().unique()))
days = days[days < "2026-09-01"]
last7 = []
for (y, m), g in pd.Series(days, index=days).groupby([days.year, days.month]):
    last7.extend(g.index[-7:])
last7 = pd.DatetimeIndex(last7)


def run_sample(sel_days: pd.DatetimeIndex, label: str) -> dict:
    sub = df[df.index.normalize().isin(sel_days)]
    net = (sub["load_kw"] - sub["pv_kw"]).values
    w = B.weights(sub) * 1e6
    c, d, _ = B.solve_lp(w, net)
    res = {"hours": int(len(sub)), "oracle": B.evaluate(c - d, sub)}
    res["mpc24"] = B.evaluate(B.run_mpc(w, net, 24), sub)
    c0, d0, _ = B.solve_lp(w, net, soc0=0.0)
    res["oracle_soc0_empty"] = B.evaluate(c0 - d0, sub)
    res["value_of_initial_1000kwh_points"] = res["oracle"]["score"] - res["oracle_soc0_empty"]["score"]
    # same controller but with the weights of the whole train year (what the agent can know)
    w_train = B.weights(df)
    w_tr = pd.Series(w_train, index=df.index)[sub.index].values * 1e6
    res["mpc24_train_weights"] = B.evaluate(B.run_mpc(w_tr, net, 24), sub)
    # CO2-only and price-only controllers, to show the stakes of the weighting
    base = np.maximum(net, 0)
    res["baseline_kg_per_hour"] = float((base * sub["co2_kg_per_kwh"].values).sum() / len(sub))
    res["baseline_eur_per_hour"] = float((base * sub["precio_eur_per_kwh"].values).sum() / len(sub))
    res["kg_per_eur"] = res["baseline_kg_per_hour"] / res["baseline_eur_per_hour"]
    res["imputed_hours"] = int(sub["imputed"].sum())
    print(label, {k: (round(v["score"], 3) if isinstance(v, dict) else v) for k, v in res.items()})
    return res


stats["test_like_12_weeks"] = run_sample(last7, "12 weeks")
no_nov_aug = last7[~last7.month.isin([11, 8])]
stats["test_like_10_weeks_without_nov_aug"] = run_sample(no_nov_aug, "10 weeks")
stats["test_like_days"] = [str(d.date()) for d in last7]
# leaderboard snapshot (WEBSITE_NOTES.md): 4,289.4 kg = 5.97 %, 8,907.87 EUR = 7.5 % on the public half
pub_kg, pub_eur = 4289.4 / 0.0597, 8907.87 / 0.075
off = C.load_hourly_official()
grid = (off["load_kw"] - off["pv_kw"]).clip(lower=0)
stats["public_leaderboard_baseline"] = {
    "kg_total": pub_kg, "eur_total": pub_eur, "kg_per_eur": pub_kg / pub_eur,
    "kg_per_hour_if_1008h": pub_kg / 1008, "eur_per_hour_if_1008h": pub_eur / 1008,
    "kg_per_hour_if_840h": pub_kg / 840, "eur_per_hour_if_840h": pub_eur / 840,
    "train_kg_per_hour": float((grid * off["co2_kg_per_kwh"]).sum() / len(off)),
    "train_eur_per_hour": float((grid * off["precio_eur_per_kwh"]).sum() / len(off)),
}
pl = stats["public_leaderboard_baseline"]
pl["implied_hours_at_train_kg_rate"] = pub_kg / pl["train_kg_per_hour"]
pl["implied_hours_at_train_eur_rate"] = pub_eur / pl["train_eur_per_hour"]

# ------------------------------------------------------------------ 2. fixed clock schedules
hour = df.index.hour.values


def clock(charge_hours, discharge_hours) -> np.ndarray:
    p = np.zeros(len(df))
    p[np.isin(hour, charge_hours)] = B.PMAX
    p[np.isin(hour, discharge_hours)] = -B.PMAX
    return simulate(p)


schedules = {
    "One cycle: charge 13–17 h,\ndischarge 18–22 h": clock([13, 14, 15, 16, 17], [18, 19, 20, 21]),
    "Two cycles: also charge 01–05 h,\ndischarge 06–09 h": clock([1, 2, 3, 4, 13, 14, 15, 16, 17], [6, 7, 8, 18, 19, 20, 21]),
}
clock_res = {name: B.evaluate(p, df) for name, p in schedules.items()}
clock_res["24 h lookahead, re-optimised\nevery hour"] = runs["mpc"]["24"]
stats["clock_schedules"] = {k.replace("\n", " "): v for k, v in clock_res.items()}
stats["clock_share_of_oracle"] = {k.replace("\n", " "): v["score"] / oracle_full["score"] for k, v in clock_res.items()}

# ------------------------------------------------------------------ 3. value by month
p_or = pd.Series(runs["oracle_power"], index=df.index)
kg = (df["co2_kg_per_kwh"] * (-p_or))
eur = (df["precio_eur_per_kwh"] * (-p_or))
n_days = p_or.groupby(p_or.index.month).size() / 24
kg_day = (kg.groupby(kg.index.month).sum() / n_days).reindex(C.MONTH_ORDER)
eur_day = (eur.groupby(eur.index.month).sum() / n_days).reindex(C.MONTH_ORDER)
stats["oracle_kg_saved_per_day_by_month"] = {C.MONTH_ABBR[m]: float(v) for m, v in kg_day.items()}
stats["oracle_eur_saved_per_day_by_month"] = {C.MONTH_ABBR[m]: float(v) for m, v in eur_day.items()}
stats["oracle_kg_saved_per_day_mean"] = float(kg.sum() / (len(df) / 24))
stats["oracle_eur_saved_per_day_mean"] = float(eur.sum() / (len(df) / 24))
stats["oracle_kg_saved_total"] = float(kg.sum())
stats["oracle_eur_saved_total"] = float(eur.sum())
summer = [6, 7, 8]
winter = [12, 1, 2]
stats["summer_over_winter_kg_per_day"] = float(kg_day.loc[summer].mean() / kg_day.loc[winter].mean())
stats["summer_over_winter_eur_per_day"] = float(eur_day.loc[summer].mean() / eur_day.loc[winter].mean())

# ------------------------------------------------------------------ 4. check in the official simulator
# Replay the perfect-foresight schedule in starter_kit/environment.py, once as it comes out of the
# solver and once through the website's safety clamp (eps = 1e-4). The optimum sits exactly on the
# SOC limits most days, so float rounding alone triggers infractions (0.20 points each).
import sys

sys.path.insert(0, str(C.ROOT / "starter_kit"))
from environment import CampusBatteryEnv  # noqa: E402

core = df[["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]]


def replay(p: np.ndarray, clamp: bool) -> dict:
    env = CampusBatteryEnv(core)
    env.reset()
    eta, eps = 0.9 ** 0.5, 1e-4
    for a in p:
        if clamp:
            max_c = max(0.0, min(500.0, (2000.0 - env.soc) / eta - eps))
            max_d = max(0.0, min(500.0, env.soc * eta - eps))
            a = float(np.clip(a, -max_d, max_c))
        env.step(a)
    r = env.evaluate_results()
    return {"score_reported": r["score"], "co2_red_pct": r["co2_estalviat_pct"], "cost_red_pct": r["cost_estalviat_pct"],
            "infractions": r["infraccions"],
            "score_before_penalties": 0.7 * r["co2_estalviat_pct"] + 0.3 * r["cost_estalviat_pct"]}


stats["official_simulator_replay_raw"] = replay(p_or.values, clamp=False)
stats["official_simulator_replay_with_eps_clamp"] = replay(p_or.values, clamp=True)
soc = B.SOC0 + np.cumsum(np.where(p_or.values > 0, p_or.values * B.ETA, p_or.values / B.ETA))
stats["oracle_hours_ending_on_a_soc_limit"] = int(((soc < 1.0) | (soc > B.CAP - 1.0)).sum())
stats["oracle_share_hours_ending_on_a_soc_limit"] = float(((soc < 1.0) | (soc > B.CAP - 1.0)).mean())
C.write_stats("q10_extras", stats)

# ================================================================== FIGURE: value by month
fig = plt.figure(figsize=(13.5, 6.0))
C.titled(fig, f"A summer day is worth {stats['summer_over_winter_kg_per_day']:.1f}× a winter day in CO₂ avoided: "
              "summer and autumn weeks weigh most",
         "What the perfect-foresight schedule saves per day in each month. The official score adds these savings over "
         "the whole test period, so high-value days weigh more.")
gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.79, bottom=0.14, wspace=0.20)
x = np.arange(12)
for i, (vals, col, name, unit, mean) in enumerate((
        (kg_day, C.C_CO2, "CO₂ avoided per day", "kg CO₂ per day", stats["oracle_kg_saved_per_day_mean"]),
        (eur_day, C.C_PRICE, "Cost saved per day", "€ per day", stats["oracle_eur_saved_per_day_mean"]))):
    ax = fig.add_subplot(gs[i])
    ax.bar(x, vals.values, width=0.6, color=col)
    for xi, v in zip(x, vals.values):
        ax.text(xi, v + vals.max() * 0.015, f"{v:.0f}", ha="center", va="bottom", fontsize=10, color=C.INK)
    ax.axhline(mean, color=C.INK, lw=1.0)
    ax.text(5.5 if i == 0 else 3.5, mean + vals.max() * 0.02, f"year mean {mean:.0f}", ha="center", va="bottom",
            fontsize=10, color=C.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([C.MONTH_ABBR[m] for m in C.MONTH_ORDER])
    ax.set_ylabel(unit)
    ax.set_ylim(0, vals.max() * 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_title(name)
C.footnote(fig, f"Sample: perfect-foresight schedule on all {len(df):,} train hours (21–24 days per month). Totals over the "
                f"282 train days: {kg.sum() / 1000:.1f} t CO₂ and {eur.sum() / 1000:.1f} k€.")
C.save(fig, "q10_value_by_month")

# ================================================================== FIGURE: clock vs optimiser
fig = plt.figure(figsize=(13.5, 5.6))
names = list(clock_res)
sc = np.array([clock_res[n]["score"] for n in names])
C.titled(fig, f"A fixed daily timetable gets {sc[0] / oracle_full['score']:.0%} of the optimum; reading the day-ahead signals "
              "gets the rest",
         "Score on the train year of two fixed clock schedules and of the hourly re-optimisation on the 24 h lookahead.")
ax = fig.add_axes([0.30, 0.16, 0.66, 0.60])
y = np.arange(len(names))[::-1]
ax.barh(y, sc, height=0.5, color=[C.GRAY, C.GRAY, C.C_W])
ax.axvline(oracle_full["score"], color=C.INK, lw=1.2)
ax.text(oracle_full["score"] + 0.04, y[0] + 0.42, f"perfect foresight\n{oracle_full['score']:.2f}", fontsize=10.5,
        color=C.INK, va="top")
for yi, n in zip(y, names):
    r = clock_res[n]
    ax.text(r["score"] - 0.06, yi, f"{r['score']:.2f} points   (CO₂ −{r['co2_red_pct']:.1f} %, cost −{r['cost_red_pct']:.1f} %)",
            va="center", ha="right", fontsize=11, color="white" if n == names[-1] else C.INK)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=11)
ax.set_xlim(0, oracle_full["score"] * 1.14)
ax.set_xlabel("Score (points)")
ax.grid(axis="y", visible=False)
C.footnote(fig, f"Sample: all {len(df):,} train hours simulated as one sequence. Clock schedules run at 500 kW until the "
                "battery is full or empty.")
C.save(fig, "q10_clock_vs_optimiser")

print(json.dumps({k: v for k, v in stats.items() if k not in ("test_like_days",)}, indent=1)[:6000])
