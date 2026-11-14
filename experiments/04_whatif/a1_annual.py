"""
experiments/04_whatif/a1_annual.py
Analysis 1 -- what the battery delivers per year.

* reference optimum (2,000 kWh / 500 kW / 90 %, 70/30 weights) on the 281 train days
* cross-check: src.sim.oracle convention and our real-time controller (agents/v1)
* scaled to a full year month by month (see common.annual_factors for the assumption)
* equivalent full cycles per day, hour-of-day x month schedule

Outputs: figs/a1_annual_savings.png, figs/a1_schedule_heatmap.png, stats/a1_annual.json
"""
from __future__ import annotations

import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
from src.sim import simulate


def load_agent():
    spec = importlib.util.spec_from_file_location("agent_v1", C.ROOT / "agents" / "v1" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BatteryAgent()


def main() -> None:
    C.set_style()
    prob = C.Problem()
    idx = prob.idx
    fac = C.annual_factors(idx).to_numpy()

    sol, acc = C.run(prob)                               # energy-neutral reference
    _, acc_free = C.run(prob, terminal="free", tie=0.0)  # = src.sim.oracle
    ctrl = simulate(load_agent(), prob.df[["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]])

    # ---------------------------------------------------------------- hourly accounting
    saved_kwh = prob.base - sol["g"]
    h = pd.DataFrame({
        "co2_saved": saved_kwh * prob.co2, "eur_saved": saved_kwh * prob.price,
        "co2_base": prob.base * prob.co2, "eur_base": prob.base * prob.price,
        "energy_base": prob.base, "c": sol["c"], "d": sol["d"], "p": sol["p"], "fac": fac,
    }, index=idx)
    ym = idx.to_period("M")
    rows = []
    for per, g in h.groupby(ym):
        n_days = len(g) / 24
        k = per.days_in_month / n_days
        rows.append({
            "month": C.MONTH_ABBR[per.month] + f" {per.year}", "m": per.month,
            "train_days": int(n_days), "days_in_month": per.days_in_month,
            "co2_saved_t": g["co2_saved"].sum() * k / 1000, "eur_saved_k": g["eur_saved"].sum() * k / 1000,
            "co2_base_t": g["co2_base"].sum() * k / 1000, "eur_base_k": g["eur_base"].sum() * k / 1000,
            "co2_saved_kg_per_day": g["co2_saved"].sum() / n_days, "eur_saved_per_day": g["eur_saved"].sum() / n_days,
            "cycles_per_day": g["d"].sum() / (0.9 ** 0.5) / C.CAP / n_days,
        })
    mon = pd.DataFrame(rows)
    mon["co2_pct"] = 100 * mon["co2_saved_t"] / mon["co2_base_t"]
    mon["eur_pct"] = 100 * mon["eur_saved_k"] / mon["eur_base_k"]

    year = {
        "co2_saved_t": float(np.sum(h["co2_saved"] * fac) / 1000),
        "eur_saved": float(np.sum(h["eur_saved"] * fac)),
        "co2_base_t": float(np.sum(h["co2_base"] * fac) / 1000),
        "eur_base": float(np.sum(h["eur_base"] * fac)),
        "energy_base_mwh": float(np.sum(h["energy_base"] * fac) / 1000),
        "charged_mwh": float(np.sum(h["c"] * fac) / 1000),
        "discharged_mwh": float(np.sum(h["d"] * fac) / 1000),
    }
    year["co2_pct"] = 100 * year["co2_saved_t"] / year["co2_base_t"]
    year["eur_pct"] = 100 * year["eur_saved"] / year["eur_base"]
    year["losses_mwh"] = year["charged_mwh"] - year["discharged_mwh"]
    year["losses_pct_of_campus_import"] = 100 * year["losses_mwh"] / year["energy_base_mwh"]
    year["throughput_pct_of_campus_import"] = 100 * year["discharged_mwh"] / year["energy_base_mwh"]
    naive = 365 / prob.days
    # the official night load is held at the 22:00 reading and is too high (EDA): correct the baseline
    over = C.BASELINE_OVERSTATED
    year["co2_base_t_night_corrected"] = year["co2_base_t"] / (1 + over["co2"])
    year["eur_base_night_corrected"] = year["eur_base"] / (1 + over["cost"])
    year["energy_base_mwh_night_corrected"] = year["energy_base_mwh"] / (1 + over["energy"])
    year["co2_pct_night_corrected"] = 100 * year["co2_saved_t"] / year["co2_base_t_night_corrected"]
    year["eur_pct_night_corrected"] = 100 * year["eur_saved"] / year["eur_base_night_corrected"]
    year["score_equivalent_night_corrected"] = (C.ALPHA * year["co2_pct_night_corrected"]
                                                + (1 - C.ALPHA) * year["eur_pct_night_corrected"])
    year["score_equivalent_official_baseline"] = C.ALPHA * year["co2_pct"] + (1 - C.ALPHA) * year["eur_pct"]
    year["co2_saved_t_naive_365_over_281"] = acc["co2_saved_kg"] * naive / 1000
    year["eur_saved_naive_365_over_281"] = acc["cost_saved_eur"] * naive

    # robustness: the 15 meter-fault hours (load = PV = 0) give the battery nothing to displace.
    # Replace them by the month / day-type / hour mean and re-solve.
    df2 = prob.df.copy()
    fault = df2["load_kw"] < 50
    key = [df2.index.month, df2.index.day_of_week >= 5, df2.index.hour]
    for col in ("load_kw", "pv_kw"):
        df2.loc[fault, col] = df2[col].where(~fault).groupby(key).transform("mean")[fault]
    prob2 = C.Problem(df2)
    sol2 = C.solve(prob2, prob.weights())                 # same weights: only the fault hours change
    ann2 = C.annual_savings(prob2, sol2)
    fault_check = {"fault_hours": int(fault.sum()), "co2_t_year_if_meter_had_worked": ann2["co2_t_per_year"],
                   "eur_year_if_meter_had_worked": ann2["eur_per_year"],
                   "co2_lost_to_faults_pct": 100 * (ann2["co2_t_per_year"] / year["co2_saved_t"] - 1),
                   "eur_lost_to_faults_pct": 100 * (ann2["eur_per_year"] / year["eur_saved"] - 1)}

    # signal levels seen by the battery (energy-weighted)
    sig = {
        "co2_at_charge_g": 1000 * float(np.sum(sol["c"] * prob.co2) / np.sum(sol["c"])),
        "co2_at_discharge_g": 1000 * float(np.sum(sol["d"] * prob.co2) / np.sum(sol["d"])),
        "price_at_charge_c": 100 * float(np.sum(sol["c"] * prob.price) / np.sum(sol["c"])),
        "price_at_discharge_c": 100 * float(np.sum(sol["d"] * prob.price) / np.sum(sol["d"])),
        "campus_avg_co2_g": 1000 * prob.co2_base / prob.energy_base,
        "campus_avg_price_c": 100 * prob.cost_base / prob.energy_base,
    }

    # daily activity
    day = h.groupby(idx.normalize()).agg(d=("d", "sum"), c=("c", "sum"))
    efc = day["d"] / (0.9 ** 0.5) / C.CAP
    active = {
        "cycles_per_day_mean": float(efc.mean()), "cycles_per_day_p10": float(efc.quantile(0.1)),
        "cycles_per_day_median": float(efc.median()), "cycles_per_day_p90": float(efc.quantile(0.9)),
        "days_ge_1p5_cycles_pct": float(100 * (efc >= 1.5).mean()),
        "days_lt_0p5_cycles_pct": float(100 * (efc < 0.5).mean()),
        "hours_charging_pct": float(100 * np.mean(sol["c"] > 1)), "hours_discharging_pct": float(100 * np.mean(sol["d"] > 1)),
        "hours_idle_pct": float(100 * np.mean((sol["c"] <= 1) & (sol["d"] <= 1))),
        "hours_full_power_pct_of_active": float(100 * np.mean(np.abs(sol["p"][np.abs(sol["p"]) > 1]) > 499)),
        "cycles_per_year": float(efc.mean() * 365),
    }

    # hour x month mean power
    hm = h["p"].groupby([idx.month, idx.hour]).mean().unstack().reindex(C.MONTH_ORDER)

    # ---------------------------------------------------------------- figure 1: monthly savings
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.94, top=0.84, bottom=0.115, hspace=0.24)
    C.titled(fig, f"The battery avoids ≈{year['co2_saved_t']:.0f} t CO₂ and ≈{year['eur_saved'] / 1000:.0f} k€ a year: "
                  f"{year['co2_pct']:.1f} % of emissions, {year['eur_pct']:.1f} % of the bill",
             f"Official baseline ≈{year['co2_base_t']:.0f} t CO₂ and ≈{year['eur_base'] / 1e6:.2f} M€ a year "
             f"({year['co2_pct_night_corrected']:.1f} % and {year['eur_pct_night_corrected']:.1f} % once the overstated "
             f"night load is corrected) · {acc['cycles_per_day']:.2f} full cycles a day")
    x = np.arange(12)
    labels = [C.MONTH_ABBR[m] for m in mon["m"]]
    for ax, col, color, unit, fmt in ((axes[0], "co2_saved_t", C.C_CO2, "t CO₂ avoided per month", "{:.1f} t"),
                                      (axes[1], "eur_saved_k", C.C_COST, "k€ saved per month", "{:.1f} k€")):
        v = mon[col].to_numpy()
        ax.bar(x, v, width=0.62, color=color, zorder=3)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, v.max() * 1.22)
        for i in (int(np.argmax(v)), int(np.argmin(v))):
            ax.text(x[i], v[i] + v.max() * 0.03, fmt.format(v[i]), ha="center", va="bottom", fontsize=11,
                    color=C.INK, fontweight="bold")
        ax.axhline(v.mean(), color=C.INK2, lw=1.0, zorder=4)
        ax.text(11.68, v.mean(), f"mean\n{fmt.format(v.mean())}", ha="left", va="center", fontsize=10.5, color=C.INK2)
    axes[1].set_xticks(x, labels)
    axes[1].set_xlim(-0.6, 11.6)
    C.footnote(fig, C.FOOT + "\nEach month's ~3 train weeks scaled by days-in-month / train-days. "
                             "Nov: campus load imputed (no meter data); price and CO₂ signals are real.")
    C.save(fig, "a1_annual_savings")

    # ---------------------------------------------------------------- figure 2: schedule heatmap
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.83, bottom=0.14)
    C.titled(fig, "A full cycle from afternoon solar to the evening peak, plus a part cycle from night to the morning peak",
             "Mean battery power by hour of day and month, optimal schedule (kW; 500 kW = inverter limit)")
    im = ax.imshow(hm.to_numpy(), aspect="auto", cmap=C.DIV_BLUE_RED.reversed(), vmin=-500, vmax=500,
                   extent=(-0.5, 23.5, 11.5, -0.5))
    ax.set_yticks(range(12), [C.MONTH_ABBR[m] for m in C.MONTH_ORDER])
    ax.set_xticks(range(0, 24, 1))
    ax.set_xlabel("Hour of day (local time, hour beginning)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax.grid(which="minor", color=C.SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-500, -250, 0, 250, 500])
    cb.ax.set_yticklabels(["500\ndischarge", "250", "0", "250", "500\ncharge"])
    cb.set_label("Mean power (kW)")
    cb.outline.set_visible(False)
    C.footnote(fig, C.FOOT + "\nRed = discharging to the campus, blue = charging from the grid. "
                             f"Battery is active {100 - active['hours_idle_pct']:.0f} % of hours.")
    C.save(fig, "a1_schedule_heatmap")

    # ---------------------------------------------------------------- stats
    stats = {
        "sample": {"hours": prob.n, "days": prob.days, "imputed_hours": int(prob.df["imputed"].sum())},
        "train_reference_energy_neutral": acc,
        "train_oracle_convention_free_end": acc_free,
        "free_initial_charge_worth_points": acc_free["score"] - acc["score"],
        "controller_v1_on_train": {k: ctrl[k] for k in ("score", "co2_pct", "cost_pct", "co2_saved_kg",
                                                         "cost_saved_eur", "infractions")},
        "controller_gap_points": acc_free["score"] - ctrl["score"],
        "controller_gap_pct_of_oracle": 100 * (1 - ctrl["score"] / acc_free["score"]),
        "train_baseline": {"co2_kg": prob.co2_base, "cost_eur": prob.cost_base, "energy_kwh": prob.energy_base,
                           "kg_per_eur": prob.co2_base / prob.cost_base},
        "year": year, "signals_seen": sig, "activity": active, "meter_fault_check": fault_check,
        "public_leaderboard_cross_check": {
            "note": "6.43-point entry in docs/WEBSITE_NOTES.md: 4,289.4 kg and 8,907.87 EUR on the public half; "
                    "the number of public hours with meter data is unknown (840 to 1,008)",
            "co2_t_year_if_1008h": 4289.4 / 1008 * 8760 / 1000, "co2_t_year_if_840h": 4289.4 / 840 * 8760 / 1000,
            "eur_year_if_1008h": 8907.87 / 1008 * 8760, "eur_year_if_840h": 8907.87 / 840 * 8760},
        "monthly": mon.drop(columns="m").to_dict(orient="records"),
        "hour_month_mean_power_kw": {C.MONTH_ABBR[m]: hm.loc[m].round(1).tolist() for m in C.MONTH_ORDER},
        "hour_mean_power_kw": h["p"].groupby(idx.hour).mean().round(1).tolist(),
    }
    C.write_stats("a1_annual", stats)

    pd.set_option("display.width", 200)
    print(mon.drop(columns="m").round(2).to_string(index=False))
    for k in ("year", "signals_seen", "activity", "meter_fault_check", "public_leaderboard_cross_check"):
        print(k, {a: (round(b, 3) if not isinstance(b, str) else b) for a, b in stats[k].items()})
    print("reference", {a: round(b, 4) for a, b in acc.items()})
    print("oracle   ", {a: round(b, 4) for a, b in acc_free.items()})
    print("controller", stats["controller_v1_on_train"], "gap pts", round(stats["controller_gap_points"], 4),
          "gap %", round(stats["controller_gap_pct_of_oracle"], 3))


if __name__ == "__main__":
    main()
