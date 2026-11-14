"""
experiments/04_whatif/a4_efficiency.py
Analysis 4 -- round-trip efficiency.

* optimal savings for a round-trip efficiency of 70 % .. 100 % (one-way = sqrt, as in the brief)
* at the real 90 %: how much of the spread the battery trades on is spent on conversion losses

Accounting used for the split (x = CO2 intensity, price or the score weight):
    A = sum_t d_t x_t      what discharging avoids
    B = sum_t c_t x_t      what charging adds                       net saving = A - B
    of B, the share (1 - eta_rt) buys energy that never comes back (losses);
    eta_rt * B buys the energy that is delivered. Gross spread on delivered energy = A - eta_rt*B.

Outputs: figs/a4_efficiency.png, stats/a4_efficiency.json
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

RTES = [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00]


def main() -> None:
    C.set_style()
    prob = C.Problem()
    w = prob.weights()
    fac = C.annual_factors(prob.idx).to_numpy()

    rows, sols = [], {}
    for r in RTES:
        sol = C.solve(prob, w, rte=r)
        acc = C.account(prob, sol)
        ann = C.annual_savings(prob, sol)
        rows.append({"rte": r, "score": acc["score"], "co2_pct": acc["co2_pct"], "cost_pct": acc["cost_pct"],
                     "co2_t_year": ann["co2_t_per_year"], "eur_year": ann["eur_per_year"],
                     "cycles_per_day": acc["cycles_per_day"],
                     "charged_mwh_year": float(np.sum(sol["c"] * fac) / 1000),
                     "delivered_mwh_year": float(np.sum(sol["d"] * fac) / 1000)})
        sols[r] = sol
    sw = pd.DataFrame(rows).set_index("rte")
    ref, ideal = sw.loc[0.90], sw.loc[1.00]

    # ---------------------------------------------------------------- split at 90 %
    sol = sols[0.90]
    c, d = sol["c"] * fac, sol["d"] * fac                       # annualised kWh per train hour
    split = {}
    for name, x, unit_scale in (("co2_t", prob.co2, 1 / 1000), ("eur", prob.price, 1.0), ("score_pts", w * 100, None)):
        if unit_scale is None:                                   # score: no annual scaling
            A, B = float(np.sum(sol["d"] * x)), float(np.sum(sol["c"] * x))
        else:
            A, B = float(np.sum(d * x) * unit_scale), float(np.sum(c * x) * unit_scale)
        gross = A - 0.9 * B
        split[name] = {
            "avoided_by_discharging": A, "added_by_charging": B, "net_saving": A - B,
            "charging_for_delivered_energy": 0.9 * B, "charging_for_losses": 0.1 * B,
            "gross_spread_on_delivered_energy": gross,
            "losses_pct_of_gross_spread": 100 * 0.1 * B / gross,
        }
    key = {"co2_t": "co2_t_year", "eur": "eur_year", "score_pts": "score"}
    vs_ideal = {}
    for name, col in key.items():
        gap = ideal[col] - ref[col]
        conv = split[name]["charging_for_losses"]
        vs_ideal[name] = {
            "lossless_battery": ideal[col], "at_90pct": ref[col], "gap": gap, "gap_pct_of_lossless": 100 * gap / ideal[col],
            "of_which_conversion_losses_on_trades_made": conv,
            "of_which_trades_forgone_or_smaller": gap - conv,
            "conversion_share_of_gap_pct": 100 * conv / gap,
        }
    # sensitivity around the design point
    sens = {col: (sw.loc[0.925, col] - sw.loc[0.875, col]) / 5.0 for col in ("score", "co2_t_year", "eur_year")}
    sens_pct = {col: 100 * sens[col] / ref[col] for col in sens}
    # the spread the battery actually trades on
    spread = {
        "co2_at_charge_g": 1000 * float(np.sum(sol["c"] * prob.co2) / np.sum(sol["c"])),
        "co2_at_discharge_g": 1000 * float(np.sum(sol["d"] * prob.co2) / np.sum(sol["d"])),
        "price_at_charge_c": 100 * float(np.sum(sol["c"] * prob.price) / np.sum(sol["c"])),
        "price_at_discharge_c": 100 * float(np.sum(sol["d"] * prob.price) / np.sum(sol["d"])),
        "w_ratio_discharge_over_charge": float((np.sum(sol["d"] * w) / np.sum(sol["d"])) /
                                               (np.sum(sol["c"] * w) / np.sum(sol["c"]))),
        "break_even_ratio": 1 / 0.9,
    }
    spread["co2_ratio"] = spread["co2_at_discharge_g"] / spread["co2_at_charge_g"]
    spread["price_ratio"] = spread["price_at_discharge_c"] / spread["price_at_charge_c"]

    # ---------------------------------------------------------------- figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.8), gridspec_kw={"width_ratios": [1.0, 1.0]})
    fig.subplots_adjust(left=0.07, right=0.975, top=0.79, bottom=0.15, wspace=0.26)
    s_co2 = split["co2_t"]
    C.titled(fig, f"At 90 % efficiency the battery keeps {100 * ref['score'] / ideal['score']:.0f} % of what a lossless one would earn",
             f"Each efficiency point is worth ≈{sens_pct['score']:.0f} % of the saving · conversion losses eat "
             f"{s_co2['losses_pct_of_gross_spread']:.0f} % of the CO₂ spread and "
             f"{split['eur']['losses_pct_of_gross_spread']:.0f} % of the price spread the battery trades on")

    xs = 100 * sw.index.to_numpy()
    ax.plot(xs, sw["score"], color=C.C_SCORE, lw=2.6, zorder=3)
    ax.axvline(90, color=C.AXIS, lw=1.0, zorder=1)
    for r in (0.70, 0.80, 1.00):
        ax.scatter([100 * r], [sw.loc[r, "score"]], s=46, color=C.C_SCORE, edgecolor=C.SURFACE, linewidth=1.5, zorder=4)
        ax.annotate(f"{sw.loc[r, 'score']:.2f}", (100 * r, sw.loc[r, "score"]), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, color=C.INK)
    ax.scatter([90], [ref["score"]], s=120, color=C.INK, edgecolor=C.SURFACE, linewidth=2, zorder=5)
    ax.annotate(f"Design: 90 %\n{ref['score']:.2f} pts", (90, ref["score"]), xytext=(-10, 10), textcoords="offset points",
                ha="right", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("Round-trip efficiency (%)")
    ax.set_ylabel("Score (points)")
    ax.set_title("Score against efficiency (CO₂ and € move in step with it)")
    ax.set_xlim(68.5, 101.5)
    ax.set_ylim(0, sw["score"].max() * 1.18)

    # waterfall, CO2 at 90 %
    A, Bd, Bl, net = (s_co2["avoided_by_discharging"], s_co2["charging_for_delivered_energy"],
                      s_co2["charging_for_losses"], s_co2["net_saving"])
    labels = ["Avoided by\ndischarging", "Charging the\nenergy delivered", "Charging to\ncover losses", "Net CO₂\navoided"]
    bottoms = [0, A - Bd, A - Bd - Bl, 0]
    heights = [A, Bd, Bl, net]
    colors = [C.C_CO2, C.GRAY, C.RED, C.C_CO2]
    ax2.bar(range(4), heights, bottom=bottoms, color=colors, width=0.6, zorder=3)
    for i, (h, b, sign) in enumerate(zip(heights, bottoms, ("+", "−", "−", "="))):
        ax2.text(i, b + h + A * 0.015, f"{sign} {h:.1f} t" if sign != "=" else f"{h:.1f} t", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold" if i in (0, 3) else "normal", color=C.INK)
    for i in range(3):                                           # connectors
        yc = [A, A - Bd, A - Bd - Bl][i]
        ax2.plot([i + 0.3, i + 0.7], [yc, yc], color=C.MUTED, lw=1.0, zorder=2)
    ax2.set_xticks(range(4), labels)
    ax2.set_ylabel("t CO₂ per year")
    ax2.set_ylim(0, A * 1.13)
    ax2.set_xlim(-0.55, 3.55)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("CO₂ balance of the battery at 90 % efficiency")
    ax2.text(3.5, A * 1.10, f"charges at {spread['co2_at_charge_g']:.0f} g/kWh on average,\n"
                            f"discharges at {spread['co2_at_discharge_g']:.0f} g/kWh", ha="right", va="top",
             fontsize=11, color=C.INK2)
    C.footnote(fig, C.FOOT + "\nOne-way efficiency = √(round-trip), as in the brief. Losses = 10 % of the energy charged, "
                             "priced at the hours in which the battery charges.")
    C.save(fig, "a4_efficiency")

    stats = {"sweep": sw.reset_index().to_dict(orient="records"), "split_at_90pct": split, "vs_lossless": vs_ideal,
             "sensitivity_per_efficiency_point": sens, "sensitivity_pct_of_saving_per_point": sens_pct,
             "spread_traded": spread,
             "pct_of_lossless_kept": {col: 100 * ref[col] / ideal[col] for col in ("score", "co2_t_year", "eur_year")},
             "at_80pct_vs_90pct": {col: 100 * sw.loc[0.80, col] / ref[col] for col in ("score", "co2_t_year", "eur_year")},
             "at_70pct_vs_90pct": {col: 100 * sw.loc[0.70, col] / ref[col] for col in ("score", "co2_t_year", "eur_year")}}
    C.write_stats("a4_efficiency", stats)
    pd.set_option("display.width", 220)
    print(sw.round(3).to_string())
    for k in ("split_at_90pct", "vs_lossless", "sensitivity_per_efficiency_point", "sensitivity_pct_of_saving_per_point",
              "spread_traded", "pct_of_lossless_kept", "at_80pct_vs_90pct", "at_70pct_vs_90pct"):
        print(k, stats[k])


if __name__ == "__main__":
    main()
