"""
experiments/04_whatif/a2_pareto.py
Analysis 2 -- CO2 versus cost: the Pareto frontier of the battery.

Sweep the weight on CO2 (alpha) from 0 (minimise the bill only) to 1 (minimise CO2 only):
    w_t = alpha * co2_t / CO2_base + (1 - alpha) * price_t / Cost_base
and record the euros and kg saved by the optimal schedule. The weights imply an exchange rate
between kg and EUR -- an implicit carbon price:
    EUR per tonne = 1000 * alpha / (1 - alpha) * Cost_base / CO2_base.

Outputs: figs/a2_pareto.png, stats/a2_pareto.json
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

LEX = 1e-4            # tilt at the end points so that they are Pareto-efficient (lexicographic)
ALPHAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
          0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
ACT_KW = 25.0         # |power| above this counts as charging / discharging


def action(p: np.ndarray) -> np.ndarray:
    return np.where(p > ACT_KW, 1, np.where(p < -ACT_KW, -1, 0))


def main() -> None:
    C.set_style()
    prob = C.Problem()
    idx = prob.idx

    rows, sols = [], {}
    for a in ALPHAS:
        a_eff = min(max(a, LEX), 1 - LEX)
        sol = C.solve(prob, prob.weights(a_eff))
        acc = C.account(prob, sol)
        ann = C.annual_savings(prob, sol)
        rows.append({"alpha": a, "carbon_price_eur_t": prob.carbon_price_eur_per_t(a_eff) if 0 < a < 1 else
                     (0.0 if a == 0 else np.inf),
                     "co2_saved_kg": acc["co2_saved_kg"], "cost_saved_eur": acc["cost_saved_eur"],
                     "co2_pct": acc["co2_pct"], "cost_pct": acc["cost_pct"], "score_70_30": acc["score"],
                     "co2_t_year": ann["co2_t_per_year"], "eur_year": ann["eur_per_year"],
                     "cycles_per_day": acc["cycles_per_day"]})
        sols[a] = sol
    fr = pd.DataFrame(rows).set_index("alpha")

    # what would a market carbon price do?  (alpha that corresponds to a given EUR/t)
    k = prob.cost_base / prob.co2_base                       # EUR of bill per kg of emissions (baseline)
    market = {}
    for cp in (80.0, 200.0, 1000.0):
        a = (cp / 1000.0) / (cp / 1000.0 + k)
        sol = C.solve(prob, prob.weights(a))
        ann = C.annual_savings(prob, sol)
        market[f"{cp:.0f}"] = {"alpha": a, **ann, "co2_pct_of_max": None}

    ref, cost_only, co2_only = fr.loc[0.7], fr.loc[0.0], fr.loc[1.0]
    for m in market.values():
        m["co2_pct_of_max"] = 100 * m["co2_t_per_year"] / co2_only["co2_t_year"]
    give_up_eur = cost_only["eur_year"] - ref["eur_year"]
    gain_t = ref["co2_t_year"] - cost_only["co2_t_year"]
    # local slope of the frontier around 70/30 (EUR given up per extra tonne)
    lo, hi = fr.loc[0.65], fr.loc[0.75]
    local_slope = (lo["eur_year"] - hi["eur_year"]) / (hi["co2_t_year"] - lo["co2_t_year"])

    # ---------------------------------------------------------------- do the signals disagree?
    a_cost, a_co2, a_ref = action(sols[0.0]["p"]), action(sols[1.0]["p"]), action(sols[0.7]["p"])
    opposite = (a_cost * a_co2) == -1
    same_active = (a_cost == a_co2) & (a_cost != 0)
    both_idle = (a_cost == 0) & (a_co2 == 0)
    one_idle = ~(opposite | same_active | both_idle)
    any_active = ~both_idle
    price, co2 = pd.Series(prob.price, index=idx), pd.Series(prob.co2, index=idx)
    day = idx.normalize()
    r_all = float(np.corrcoef(price, co2)[0, 1])
    r_within = float(np.corrcoef(price - price.groupby(day).transform("mean"),
                                 co2 - co2.groupby(day).transform("mean"))[0, 1])
    # per day: rank agreement and overlap of the 4 cheapest / 4 cleanest hours (battery = 4 h)
    dm_p = pd.DataFrame({"v": prob.price, "d": day, "h": idx.hour}).pivot(index="d", columns="h", values="v")
    dm_c = pd.DataFrame({"v": prob.co2, "d": day, "h": idx.hour}).pivot(index="d", columns="h", values="v")
    rp, rc = dm_p.rank(axis=1).to_numpy(), dm_c.rank(axis=1).to_numpy()
    spearman = np.array([np.corrcoef(rp[i], rc[i])[0, 1] for i in range(len(rp))])
    low_overlap = np.mean([len(set(np.argsort(dm_p.to_numpy()[i])[:4]) & set(np.argsort(dm_c.to_numpy()[i])[:4]))
                           for i in range(len(rp))]) / 4
    high_overlap = np.mean([len(set(np.argsort(dm_p.to_numpy()[i])[-4:]) & set(np.argsort(dm_c.to_numpy()[i])[-4:]))
                            for i in range(len(rp))]) / 4
    opp_by_hour = pd.Series(opposite, index=idx).groupby(idx.hour).mean()
    prof = {name: pd.Series(sols[a]["p"], index=idx).groupby(idx.hour).mean() for name, a in
            (("cost", 0.0), ("co2", 1.0), ("ref", 0.7))}
    # 70/30 against each pure objective
    opp_ref_cost = float(np.mean((a_ref * a_cost) == -1))
    opp_ref_co2 = float(np.mean((a_ref * a_co2) == -1))
    same_ref_cost = float(np.mean(a_ref == a_cost))
    same_ref_co2 = float(np.mean(a_ref == a_co2))

    # ---------------------------------------------------------------- figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 7.0), gridspec_kw={"width_ratios": [1.0, 1.08]})
    fig.subplots_adjust(left=0.07, right=0.975, top=0.81, bottom=0.14, wspace=0.24)
    C.titled(fig, f"At 70/30 UPC gives up {give_up_eur / 1000:.1f} k€ a year to avoid {gain_t:.1f} t more CO₂ "
                  f"than a cost-only battery",
             f"The 70/30 weights value CO₂ at ≈{prob.carbon_price_eur_per_t():,.0f} €/t; the trade actually made "
             f"costs ≈{give_up_eur / gain_t:,.0f} € per extra tonne on average")

    x, y = fr["eur_year"] / 1000, fr["co2_t_year"]
    ax.plot(x, y, color=C.GRAY, lw=2, zorder=2)
    ax.scatter(x, y, s=18, color=C.GRAY, zorder=3, edgecolor=C.SURFACE, linewidth=0.8)
    marks = ((0.0, C.C_COST, "Cost only (0/100)", (-12, -4), "right", "center"),
             (1.0, C.C_CO2, "CO₂ only (100/0)", (-6, -12), "left", "top"),
             (0.7, C.INK, "Challenge 70/30", (12, 6), "left", "bottom"))
    for a, col, lab, off, ha, va in marks:
        r = fr.loc[a]
        ax.scatter([r["eur_year"] / 1000], [r["co2_t_year"]], s=130, color=col, zorder=5, edgecolor=C.SURFACE,
                   linewidth=2)
        ax.annotate(f"{lab}\n{r['co2_t_year']:.1f} t · {r['eur_year'] / 1000:.1f} k€", (r["eur_year"] / 1000,
                    r["co2_t_year"]), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=11,
                    color=C.INK, fontweight="bold" if a == 0.7 else "normal")
    for a in (0.3, 0.5, 0.9, 0.1):
        r = fr.loc[a]
        ax.annotate(f"{int(a * 100)}/{int(100 - a * 100)}", (r["eur_year"] / 1000, r["co2_t_year"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=10, color=C.MUTED, ha="left", va="bottom")
    ax.set_xlabel("Bill saving (k€ per year)")
    ax.set_ylabel("CO₂ avoided (t per year)")
    ax.set_title("Pareto frontier: every weighting from cost-only to CO₂-only")
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    ax.set_xlim(x.min() - 0.45 * xr, x.max() + 0.12 * xr)
    ax.set_ylim(y.min() - 0.12 * yr, y.max() + 0.14 * yr)

    hrs = np.arange(24)
    ax2.axhline(0, color=C.AXIS, lw=1.0)
    for name, col, lab, lw in (("cost", C.C_COST, "Cost only", 2.2), ("co2", C.C_CO2, "CO₂ only", 2.2),
                               ("ref", C.INK, "70/30", 2.6)):
        ax2.plot(hrs, prof[name].to_numpy(), color=col, lw=lw, label=lab, zorder=4 if name == "ref" else 3)
    ax2.set_xticks(range(0, 24, 3))
    ax2.set_xlim(-0.3, 23.3)
    ax2.set_ylim(-520, 520)
    ax2.set_yticks([-500, -250, 0, 250, 500])
    ax2.set_xlabel("Hour of day (local time)")
    ax2.set_ylabel("Mean battery power (kW)   ← discharge · charge →")
    ax2.set_title(f"Cost-only and CO₂-only do opposite things in {100 * opposite.mean():.0f} % of hours, "
                  f"{100 * opp_by_hour.max():.0f} % at {int(opp_by_hour.idxmax())}:00")
    ax2.legend(loc="lower left", ncol=3, handlelength=1.6, columnspacing=1.4, bbox_to_anchor=(0.0, -0.005))
    C.footnote(fig, C.FOOT)
    C.save(fig, "a2_pareto")

    stats = {
        "baseline_eur_per_kg": k,
        "implied_carbon_price_70_30_eur_per_t": prob.carbon_price_eur_per_t(0.7),
        "alpha_equivalent_to_80_eur_t": (0.08) / (0.08 + k),
        "frontier": fr.reset_index().to_dict(orient="records"),
        "market_carbon_prices": market,
        "vs_cost_only": {
            "eur_given_up_per_year": give_up_eur, "co2_gained_t_per_year": gain_t,
            "avg_abatement_cost_eur_per_t": give_up_eur / gain_t,
            "eur_given_up_pct": 100 * give_up_eur / cost_only["eur_year"],
            "co2_gain_pct": 100 * gain_t / cost_only["co2_t_year"],
        },
        "vs_co2_only": {
            "co2_forgone_t_per_year": co2_only["co2_t_year"] - ref["co2_t_year"],
            "eur_gained_per_year": ref["eur_year"] - co2_only["eur_year"],
            "co2_captured_pct_of_max": 100 * ref["co2_t_year"] / co2_only["co2_t_year"],
            "eur_captured_pct_of_max": 100 * ref["eur_year"] / cost_only["eur_year"],
        },
        "frontier_local_slope_at_70_30_eur_per_t": local_slope,
        "extremes": {"cost_only": cost_only.to_dict(), "co2_only": co2_only.to_dict(), "ref": ref.to_dict()},
        "disagreement": {
            "threshold_kw": ACT_KW,
            "hours_opposite_pct": 100 * float(opposite.mean()),
            "hours_same_action_pct": 100 * float(same_active.mean()),
            "hours_one_idle_pct": 100 * float(one_idle.mean()),
            "hours_both_idle_pct": 100 * float(both_idle.mean()),
            "opposite_pct_of_hours_where_either_acts": 100 * float(opposite.sum() / any_active.sum()),
            "identical_action_pct_all_hours": 100 * float(np.mean(a_cost == a_co2)),
            "corr_price_co2_hourly": r_all, "corr_price_co2_within_day": r_within,
            "daily_rank_corr_median": float(np.nanmedian(spearman)),
            "daily_rank_corr_negative_days_pct": 100 * float(np.nanmean(spearman < 0)),
            "cheapest4_also_cleanest4_overlap_pct": 100 * float(low_overlap),
            "dearest4_also_dirtiest4_overlap_pct": 100 * float(high_overlap),
            "opposite_by_hour_pct": (100 * opp_by_hour).round(1).tolist(),
            "ref_vs_cost_only_opposite_pct": 100 * opp_ref_cost, "ref_vs_co2_only_opposite_pct": 100 * opp_ref_co2,
            "ref_vs_cost_only_identical_pct": 100 * same_ref_cost, "ref_vs_co2_only_identical_pct": 100 * same_ref_co2,
        },
        "mean_power_by_hour": {n: v.round(1).tolist() for n, v in prof.items()},
    }
    C.write_stats("a2_pareto", stats)
    pd.set_option("display.width", 220)
    print(fr.round(3).to_string())
    for kk in ("vs_cost_only", "vs_co2_only", "market_carbon_prices"):
        print(kk, stats[kk])
    print("implied carbon price", stats["implied_carbon_price_70_30_eur_per_t"], "local slope", local_slope,
          "alpha@80", stats["alpha_equivalent_to_80_eur_t"])
    print({a: b for a, b in stats["disagreement"].items() if not isinstance(b, list)})
    print("opp by hour", stats["disagreement"]["opposite_by_hour_pct"])


if __name__ == "__main__":
    main()
