"""
experiments/04_whatif/a7_peak_import.py
Analysis 7 (extra) -- the side effect the score does not see: peak grid import.

The 70/30 schedule charges at 500 kW in the cleanest/cheapest hours, some of which are hours of
high campus demand, so the campus's maximum hourly import goes UP. A real supply contract bills
contracted power and penalises exceeding it; the challenge score does not. This script measures
the effect and prices the remedy:

* monthly peak import with and without the battery (optimal 70/30 schedule)
* the same optimisation with grid import capped at f x the month's no-battery peak,
  f = 100 % (never worse than today) down to 80 % (peak shaving)

The cap needs the campus load to be known in advance, so this is an upper bound in a stronger
sense than the rest: our controller does not forecast load.

Outputs: figs/a7_peak_import.png, stats/a7_peak_import.json
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

FRACS = [1.00, 0.975, 0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80]


def main() -> None:
    C.set_style()
    prob = C.Problem()
    idx = prob.idx
    w = prob.weights()
    ym = idx.to_period("M")

    sol, acc = C.run(prob)
    ann = C.annual_savings(prob, sol)
    base = pd.Series(prob.base, index=idx)
    grid = pd.Series(sol["g"], index=idx)
    peak_base_m = base.groupby(ym).max()
    peak_batt_m = grid.groupby(ym).max()
    cap_hourly = base.groupby(ym).transform("max").to_numpy()        # the month's no-battery peak, per hour
    over = grid.to_numpy() > cap_hourly + 1e-6
    fac = C.annual_factors(idx).to_numpy()

    monthly = pd.DataFrame({"month": [C.MONTH_ABBR[p.month] for p in peak_base_m.index],
                            "peak_no_battery_kw": peak_base_m.to_numpy(), "peak_with_battery_kw": peak_batt_m.to_numpy()})
    monthly["increase_kw"] = monthly["peak_with_battery_kw"] - monthly["peak_no_battery_kw"]
    monthly["increase_pct"] = 100 * monthly["increase_kw"] / monthly["peak_no_battery_kw"]
    monthly["hours_above_no_battery_peak"] = pd.Series(over, index=idx).groupby(ym).sum().to_numpy()

    rows = []
    for f in FRACS:
        try:
            s = C.solve(prob, w, grid_cap=f * cap_hourly)
        except RuntimeError as e:                                    # cap not reachable with this battery
            rows.append({"cap_pct_of_monthly_peak": 100 * f, "feasible": False, "note": str(e)[:60]})
            continue
        a = C.account(prob, s)
        an = C.annual_savings(prob, s)
        rows.append({"cap_pct_of_monthly_peak": 100 * f, "feasible": True, "score": a["score"],
                     "score_retained_pct": 100 * a["score"] / acc["score"],
                     "co2_t_year": an["co2_t_per_year"], "eur_year": an["eur_per_year"],
                     "co2_retained_pct": 100 * an["co2_t_per_year"] / ann["co2_t_per_year"],
                     "eur_retained_pct": 100 * an["eur_per_year"] / ann["eur_per_year"],
                     "cycles_per_day": a["cycles_per_day"],
                     "mean_monthly_peak_cut_kw": float(np.mean((1 - f) * peak_base_m.to_numpy()))})
    caps = pd.DataFrame(rows)
    ok = caps[caps["feasible"]]

    # Robustness to the held night reading (the official 22:00-06:00 load is ~31 % too high, EDA):
    # redo the peak comparison on daytime hours only, and relax the cap at night.
    night = np.isin(idx.hour, C.NIGHT_HOURS)
    day_base, day_grid = base[~night], grid[~night]
    ym_day = ym[~night]
    peak_base_day = day_base.groupby(ym_day).max()
    peak_batt_day = day_grid.groupby(ym_day).max()
    inc_day = (peak_batt_day - peak_base_day).to_numpy()
    cap_day_only = np.where(night, np.inf, cap_hourly)
    s_day = C.solve(prob, w, grid_cap=cap_day_only)
    a_day, an_day = C.account(prob, s_day), C.annual_savings(prob, s_day)
    daytime = {
        "monthly_peak_no_battery_is_daytime_in_all_months": bool(np.allclose(peak_base_day.to_numpy(), peak_base_m.to_numpy())),
        "monthly_peak_increase_kw_daytime_min": float(inc_day.min()), "monthly_peak_increase_kw_daytime_max": float(inc_day.max()),
        "monthly_peak_increase_kw_daytime_mean": float(inc_day.mean()),
        "monthly_peak_increase_kw_daytime": inc_day.round(1).tolist(),
        "hours_above_monthly_peak_daytime": int(over[~night].sum()), "hours_above_monthly_peak_night": int(over[night].sum()),
        "hours_above_monthly_peak_daytime_excl_nov": int(over[(~night) & (idx.month != 11)].sum()),
        "cap_daytime_only_score_retained_pct": 100 * a_day["score"] / acc["score"],
        "cap_daytime_only_co2_retained_pct": 100 * an_day["co2_t_per_year"] / ann["co2_t_per_year"],
        "cap_daytime_only_eur_retained_pct": 100 * an_day["eur_per_year"] / ann["eur_per_year"],
    }

    # when does the battery push the campus above its no-battery peak?
    hours_over = pd.Series(over, index=idx)
    charge_at_high_load = float(np.mean(sol["c"][prob.base > np.quantile(prob.base, 0.9)] > 1))

    # ---------------------------------------------------------------- figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.7), gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(left=0.07, right=0.975, top=0.79, bottom=0.15, wspace=0.22)
    r100 = ok[ok["cap_pct_of_monthly_peak"] == 100].iloc[0]
    r90 = ok[ok["cap_pct_of_monthly_peak"] == 90].iloc[0] if (ok["cap_pct_of_monthly_peak"] == 90).any() else None
    C.titled(fig, f"Side effect: the 70/30 schedule raises the campus's peak grid import by "
                  f"{monthly['increase_kw'].min():.0f}–{monthly['increase_kw'].max():.0f} kW every month",
             f"Capping import at today's monthly peak keeps {r100['score_retained_pct']:.0f} % of the saving · the "
             "challenge does not charge for peak power, a real supply contract does")
    x = np.arange(len(monthly))
    bw = 0.36
    ax.bar(x - bw / 2 - 0.02, monthly["peak_no_battery_kw"], width=bw, color=C.GRAY, label="No battery", zorder=3)
    ax.bar(x + bw / 2 + 0.02, monthly["peak_with_battery_kw"], width=bw, color=C.VIOLET, label="With battery, 70/30 optimum",
           zorder=3)
    i_top = int(monthly["peak_with_battery_kw"].idxmax())
    ax.text(x[i_top] + bw / 2 + 0.02, monthly.loc[i_top, "peak_with_battery_kw"] + 35,
            f"{monthly.loc[i_top, 'peak_with_battery_kw']:,.0f} kW", ha="center", va="bottom", fontsize=10.5, color=C.INK,
            fontweight="bold")

    ax.set_xticks(x, monthly["month"])
    ax.set_ylabel("Highest hourly grid import in the month (kW)")
    ax.set_ylim(0, monthly["peak_with_battery_kw"].max() * 1.22)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, handlelength=1.2)
    ax.set_title("Monthly peak import, with and without the battery")

    ax2.plot(ok["cap_pct_of_monthly_peak"], ok["co2_retained_pct"], color=C.C_CO2, lw=2.4, zorder=3)
    ax2.plot(ok["cap_pct_of_monthly_peak"], ok["eur_retained_pct"], color=C.C_COST, lw=2.4, zorder=3)
    last = ok.iloc[-1]
    hi = "co2" if last["co2_retained_pct"] >= last["eur_retained_pct"] else "eur"
    for key, lab in (("co2", "CO₂"), ("eur", "€")):
        ax2.annotate(f"{lab} {last[key + '_retained_pct']:.0f} %", (last["cap_pct_of_monthly_peak"], last[key + "_retained_pct"]),
                     xytext=(4, 8 if key == hi else -8), textcoords="offset points", ha="left",
                     va="bottom" if key == hi else "top", fontsize=11, color=C.INK)
    ax2.scatter([100], [r100["score_retained_pct"]], s=110, color=C.INK, edgecolor=C.SURFACE, linewidth=2, zorder=5)
    ax2.annotate(f"Never above today's peak:\n{r100['score_retained_pct']:.1f} % of the score kept", (100, r100["score_retained_pct"]),
                 xytext=(14, -62), textcoords="offset points", ha="left", va="top", fontsize=11, fontweight="bold")
    ax2.invert_xaxis()
    ax2.set_xlim(101, ok["cap_pct_of_monthly_peak"].min() - 2.2)
    ax2.set_ylim(0, 108)
    ax2.set_xlabel("Cap on grid import (% of each month's no-battery peak)")
    ax2.set_ylabel("Saving kept (% of the uncapped optimum)")
    ax2.set_title("Shaving the peak as well: what it costs")
    C.footnote(fig, C.FOOT + "\nPeak caps assume the campus load is known in advance (our controller does not forecast "
                             "load). Hourly means: sub-hourly peaks are higher. Monthly peaks all fall in daytime hours.")
    C.save(fig, "a7_peak_import")

    stats = {
        "peak_no_battery_kw": float(base.max()), "peak_with_battery_kw": float(grid.max()),
        "peak_increase_kw": float(grid.max() - base.max()),
        "p999_no_battery_kw": float(base.quantile(0.999)), "p999_with_battery_kw": float(grid.quantile(0.999)),
        "hours_above_monthly_no_battery_peak": int(over.sum()),
        "hours_above_monthly_no_battery_peak_per_year": float(np.sum(over * fac)),
        "months_with_higher_peak": int((monthly["increase_kw"] > 1).sum()),
        "mean_monthly_peak_increase_kw": float(monthly["increase_kw"].mean()),
        "max_monthly_peak_increase_kw": float(monthly["increase_kw"].max()),
        "hours_over_by_hour_of_day": hours_over.groupby(idx.hour).sum().tolist(),
        "share_of_top_decile_load_hours_in_which_battery_charges_pct": 100 * charge_at_high_load,
        "monthly": monthly.to_dict(orient="records"),
        "caps": caps.to_dict(orient="records"),
        "uncapped": {"score": acc["score"], **ann},
        "daytime_only_check": daytime,
    }
    C.write_stats("a7_peak_import", stats)
    pd.set_option("display.width", 220)
    print(monthly.round(1).to_string(index=False))
    print(caps.round(2).to_string(index=False))
    print({k: v for k, v in stats.items() if not isinstance(v, (list, dict))})
    print("hours over by hour of day", stats["hours_over_by_hour_of_day"])
    print("daytime check", daytime)


if __name__ == "__main__":
    main()
