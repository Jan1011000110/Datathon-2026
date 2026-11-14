"""
experiments/04_whatif/a5_pv_export.py
Analysis 5 (side note) -- what if PV were larger, or exports were paid?

* PV scaled by 1x .. 20x (same hourly shape): when does the campus start to have surplus
  solar, how much of it the battery soaks up, and what the battery is then worth.
* Export tariff of 0.04 EUR/kWh (mentioned once on the website; the simulator uses 0).

Every scenario is compared with ITS OWN no-battery baseline (same PV, same export tariff), and
all use today's exchange rate between kg and EUR (the 70/30 weights of the real baseline), so
that the value judgement stays fixed while the hardware changes.
NB the official environment.py would credit the agent's exports but not the baseline's; that
quirk is not reproduced here.

Outputs: figs/a5_pv_export.png, stats/a5_pv_export.json
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

SCALES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
EXPORT_PRICE = 0.04


def main() -> None:
    C.set_style()
    base = C.Problem()
    fac = C.annual_factors(base.idx).to_numpy()
    credit = (1 - C.ALPHA) * EXPORT_PRICE / base.cost_base            # export tariff in score-weight units

    rows = []
    sched = {}
    for k in SCALES:
        for xp in (0.0, EXPORT_PRICE):
            prob = C.Problem(pv_scale=k, export_price=xp)
            w = prob.weights(ref=base)
            sol = C.solve(prob, w, export_credit=credit if xp > 0 else 0.0)
            ann = C.annual_savings(prob, sol)
            surplus = prob.base_export                                # kWh per hour, no battery
            from_surplus = np.minimum(sol["c"], surplus)              # charging covered by surplus solar
            rows.append({
                "pv_scale": k, "export_price": xp, "pv_peak_kw": float(prob.pv.max()),
                "pv_share_of_load_pct": 100 * float(np.sum(prob.pv) / np.sum(prob.load)),
                "surplus_hours_pct": 100 * float(np.mean(surplus > 0)),
                "surplus_mwh_year": float(np.sum(surplus * fac) / 1000),
                "surplus_absorbed_mwh_year": float(np.sum(from_surplus * fac) / 1000),
                "charge_from_surplus_pct": 100 * float(np.sum(from_surplus) / np.sum(sol["c"])),
                "exported_with_battery_mwh_year": float(np.sum(sol["e"] * fac) / 1000),
                "co2_t_year": ann["co2_t_per_year"], "eur_year": ann["eur_per_year"],
                "baseline_co2_t_year": float(np.sum(prob.base * prob.co2 * fac) / 1000),
                "baseline_eur_year": float(np.sum((prob.base * prob.price - surplus * xp) * fac)),
                "cycles_per_day": C.account(prob, sol)["cycles_per_day"],
            })
            sched[(k, xp)] = sol["p"]
    res = pd.DataFrame(rows)
    r0 = res[res["export_price"] == 0].set_index("pv_scale")
    rx = res[res["export_price"] > 0].set_index("pv_scale")

    # first surplus: smallest PV multiple at which any real hour exports
    ratio = pd.Series(base.pv / np.where(base.load > 50, base.load, np.nan), index=base.idx)
    first_k = 1.0 / float(ratio.max())
    net_min = float(np.min(base.net[base.load > 50]))
    # multiples at which surplus hours reach 1 % and 5 %
    fine = np.arange(4.0, 20.01, 0.25)
    share = np.array([np.mean((base.load - k * base.pv)[base.load > 50] < 0) for k in fine])
    k_1pct = float(fine[np.argmax(share >= 0.01)])
    k_5pct = float(fine[np.argmax(share >= 0.05)])
    # effect of the export tariff today
    same_schedule_today = float(np.max(np.abs(sched[(1, 0.0)] - sched[(1, EXPORT_PRICE)])))

    # ---------------------------------------------------------------- figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.6))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.79, bottom=0.16, wspace=0.24)
    C.titled(fig, f"Surplus solar only appears above {first_k:.1f}× today's PV; a 0.04 €/kWh export tariff changes nothing today",
             f"Today PV peaks at {base.pv.max():.0f} kW and the campus never draws less than {net_min:.0f} kW net: "
             "the battery works off the grid signal, not off the campus's own sun")

    x = np.array(SCALES, float)
    absorbed = r0["surplus_absorbed_mwh_year"].to_numpy()
    spilled = (r0["surplus_mwh_year"] - r0["surplus_absorbed_mwh_year"]).to_numpy()
    ax.bar(x, absorbed, width=0.7, color=C.AQUA, label="Stored by the battery", zorder=3)
    ax.bar(x, spilled, width=0.7, bottom=absorbed, color=C.GRAY, label="Still spilled to the grid", zorder=3)
    for i, k in enumerate(SCALES):
        if k in (10, 15, 20):
            ax.text(x[i], absorbed[i] + spilled[i] + r0["surplus_mwh_year"].max() * 0.015,
                    f"{r0.loc[k, 'surplus_hours_pct']:.0f} % of hours", ha="center", va="bottom", fontsize=10.5, color=C.INK2)
    ax.axvline(first_k, color=C.INK2, lw=1.0, zorder=2)
    ax.text(first_k + 0.3, r0["surplus_mwh_year"].max() * 0.52, f"first hour of surplus\nat {first_k:.1f}× today's PV",
            ha="left", va="center", fontsize=11, color=C.INK)
    ax.set_xticks([1, 5, 10, 15, 20], ["1×", "5×", "10×", "15×", "20×"])
    ax.set_xlim(0, 21)
    ax.set_xlabel(f"PV size relative to today ({base.pv.max():.0f} kW peak)")
    ax.set_ylabel("Surplus solar (MWh per year)")
    ax.set_title("Surplus solar by PV size (labels: share of hours with surplus)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", handlelength=1.2, bbox_to_anchor=(0.21, 1.0))

    for col, color, lab in (("co2_t_year", C.C_CO2, "CO₂ avoided"), ("eur_year", C.C_COST, "Bill saving, exports unpaid")):
        ax2.plot(x, 100 * r0[col] / r0.loc[1, col], color=color, lw=2.4, label=lab, zorder=3)
    ax2.plot(x, 100 * rx["eur_year"] / r0.loc[1, "eur_year"], color=C.C_COST, lw=2.0, ls=(0, (4, 2)),
             label="Bill saving, exports paid 0.04 €/kWh", zorder=3)
    ax2.axhline(100, color=C.AXIS, lw=1.0, zorder=1)
    ax2.set_xticks([1, 5, 10, 15, 20], ["1×", "5×", "10×", "15×", "20×"])
    ax2.set_xlim(0, 21)
    ax2.set_xlabel("PV size relative to today")
    ax2.set_ylabel("Battery saving (today = 100)")
    ax2.set_ylim(0, 170)
    ax2.set_title("What the battery adds on top of each PV size")
    ax2.legend(loc="lower left", handlelength=2.2)
    C.footnote(fig, C.FOOT + "\nPV scaled with today's hourly shape. Each case is compared with the same PV and tariff "
                             "without a battery; kg-to-€ exchange rate fixed at today's 70/30.")
    C.save(fig, "a5_pv_export")

    stats = {
        "pv_today": {"peak_kw": float(base.pv.max()), "share_of_load_pct": 100 * float(np.sum(base.pv) / np.sum(base.load)),
                     "max_pv_over_load": float(ratio.max()), "when": str(ratio.idxmax()),
                     "min_net_load_kw_excl_meter_faults": net_min,
                     "hours_net_below_500kw": int(np.sum((base.net < 500) & (base.load > 50)))},
        "pv_multiple_for_first_surplus_hour": first_k,
        "pv_multiple_for_surplus_in_1pct_of_hours": k_1pct,
        "pv_multiple_for_surplus_in_5pct_of_hours": k_5pct,
        "export_tariff_today": {
            "max_abs_schedule_change_kw": same_schedule_today,
            "eur_year_no_tariff": float(r0.loc[1, "eur_year"]), "eur_year_with_tariff": float(rx.loc[1, "eur_year"]),
            "exported_mwh_year_with_tariff": float(rx.loc[1, "exported_with_battery_mwh_year"]),
            "tariff_as_pct_of_mean_import_price": 100 * EXPORT_PRICE / (base.cost_base / base.energy_base),
            "hours_pvpc_below_tariff_pct": 100 * float(np.mean(base.price < EXPORT_PRICE)),
        },
        "scenarios_export_unpaid": r0.reset_index().to_dict(orient="records"),
        "scenarios_export_paid": rx.reset_index().to_dict(orient="records"),
    }
    C.write_stats("a5_pv_export", stats)
    pd.set_option("display.width", 250)
    cols = ["pv_peak_kw", "pv_share_of_load_pct", "surplus_hours_pct", "surplus_mwh_year", "surplus_absorbed_mwh_year",
            "charge_from_surplus_pct", "co2_t_year", "eur_year", "baseline_co2_t_year", "baseline_eur_year", "cycles_per_day"]
    print("exports unpaid\n", r0[cols].round(2).to_string())
    print("exports paid 0.04\n", rx[cols + ["exported_with_battery_mwh_year"]].round(2).to_string())
    for kk in ("pv_today", "pv_multiple_for_first_surplus_hour", "pv_multiple_for_surplus_in_1pct_of_hours",
               "pv_multiple_for_surplus_in_5pct_of_hours", "export_tariff_today"):
        print(kk, stats[kk])


if __name__ == "__main__":
    main()
