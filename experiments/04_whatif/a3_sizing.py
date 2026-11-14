"""
experiments/04_whatif/a3_sizing.py
Analysis 3 -- sizing: is the battery energy-limited or power-limited?

* 9 x 9 grid of capacity (500 .. 8,000 kWh) x inverter power (125 .. 2,000 kW), 70/30 weights
* two finer 1-D cuts through the design point (2,000 kWh / 500 kW)
* marginal value of one more kWh and one more kW at the design point: LP duals, cross-checked
  with central finite differences (which also give the split into kg and EUR)

Outputs: figs/a3_sizing.png, stats/a3_sizing.json, stats/a3_sizing_grid.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

CAPS = [500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
POWS = [125, 187.5, 250, 375, 500, 750, 1000, 1500, 2000]
CAP_CUT = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000]
POW_CUT = [125, 187.5, 250, 312.5, 375, 437.5, 500, 625, 750, 875, 1000, 1250, 1500, 1750, 2000]


def point(prob, w, cap, pmax, **kw) -> dict:
    sol = C.solve(prob, w, cap=cap, pmax=pmax, **kw)
    acc = C.account(prob, sol)
    ann = C.annual_savings(prob, sol)
    return {"cap_kwh": cap, "pmax_kw": pmax, "duration_h": cap / pmax, "score": acc["score"],
            "co2_t_year": ann["co2_t_per_year"], "eur_year": ann["eur_per_year"],
            "cycles_per_day": acc["cycles_per_day"], "peak_import_kw": acc["peak_import_kw"], "_sol": sol}


def main() -> None:
    C.set_style()
    prob = C.Problem()
    w = prob.weights()

    grid = pd.DataFrame([{k: v for k, v in point(prob, w, c, p).items() if k != "_sol"}
                         for c in CAPS for p in POWS])
    cut_c = pd.DataFrame([{k: v for k, v in point(prob, w, c, C.PMAX).items() if k != "_sol"} for c in CAP_CUT])
    cut_p = pd.DataFrame([{k: v for k, v in point(prob, w, C.CAP, p).items() if k != "_sol"} for p in POW_CUT])
    S = grid.pivot(index="pmax_kw", columns="cap_kwh", values="score")

    # ------------------------------------------------------------- marginal values at the design point
    ref = point(prob, w, C.CAP, C.PMAX, duals=True)
    sol = ref["_sol"]
    dual = {
        "score_pts_per_kwh_capacity": -100 * sol["d_obj_d_cap"],
        "score_pts_per_kw_power": -100 * (sol["d_obj_d_pmax_c"] + sol["d_obj_d_pmax_d"]),
        "score_pts_per_kw_charge_side": -100 * sol["d_obj_d_pmax_c"],
        "score_pts_per_kw_discharge_side": -100 * sol["d_obj_d_pmax_d"],
        "hours_full_pct": 100 * sol["hours_at_cap"] / prob.n, "hours_empty_pct": 100 * sol["hours_at_empty"] / prob.n,
        "hours_charge_at_pmax_pct": 100 * sol["hours_pmax_c"] / prob.n,
        "hours_discharge_at_pmax_pct": 100 * sol["hours_pmax_d"] / prob.n,
    }
    dc, dp = 100.0, 25.0
    c_hi, c_lo = point(prob, w, C.CAP + dc, C.PMAX), point(prob, w, C.CAP - dc, C.PMAX)
    p_hi, p_lo = point(prob, w, C.CAP, C.PMAX + dp), point(prob, w, C.CAP, C.PMAX - dp)
    fd = {
        "per_kwh_capacity": {k: (c_hi[k] - c_lo[k]) / (2 * dc) for k in ("score", "co2_t_year", "eur_year")},
        "per_kw_power": {k: (p_hi[k] - p_lo[k]) / (2 * dp) for k in ("score", "co2_t_year", "eur_year")},
    }
    mv_e, mv_p = fd["per_kwh_capacity"], fd["per_kw_power"]
    elast = {
        "capacity": {k: mv_e[k] * C.CAP / ref[k] for k in mv_e},
        "power": {k: mv_p[k] * C.PMAX / ref[k] for k in mv_p},
    }
    mrs = mv_p["score"] / mv_e["score"]           # kWh of capacity worth the same as 1 kW of power

    # doubling experiments (same total "size" added in two ways)
    dbl = {
        "double_capacity_4000kwh_500kw": {k: grid.query("cap_kwh==4000 and pmax_kw==500")[k].item() for k in
                                          ("score", "co2_t_year", "eur_year")},
        "double_power_2000kwh_1000kw": {k: grid.query("cap_kwh==2000 and pmax_kw==1000")[k].item() for k in
                                        ("score", "co2_t_year", "eur_year")},
        "double_both_4000kwh_1000kw": {k: grid.query("cap_kwh==4000 and pmax_kw==1000")[k].item() for k in
                                       ("score", "co2_t_year", "eur_year")},
        "half_capacity_1000kwh_500kw": {k: grid.query("cap_kwh==1000 and pmax_kw==500")[k].item() for k in
                                        ("score", "co2_t_year", "eur_year")},
        "half_power_2000kwh_250kw": {k: grid.query("cap_kwh==2000 and pmax_kw==250")[k].item() for k in
                                     ("score", "co2_t_year", "eur_year")},
    }
    for d in dbl.values():
        d["score_vs_design_pct"] = 100 * d["score"] / ref["score"]

    # value per kWh of capacity along constant-duration rays (should be ~flat: constant returns)
    grid["score_per_mwh"] = grid["score"] / grid["cap_kwh"] * 1000
    grid["eur_per_kwh_year"] = grid["eur_year"] / grid["cap_kwh"]
    grid["kgco2_per_kwh_year"] = 1000 * grid["co2_t_year"] / grid["cap_kwh"]
    by_dur = grid.groupby("duration_h").agg(n=("score", "size"), score_per_mwh=("score_per_mwh", "mean"),
                                            score_per_mwh_min=("score_per_mwh", "min"),
                                            score_per_mwh_max=("score_per_mwh", "max"),
                                            eur_per_kwh_year=("eur_per_kwh_year", "mean"),
                                            kgco2_per_kwh_year=("kgco2_per_kwh_year", "mean"))
    by_dur["score_per_mw"] = by_dur["score_per_mwh"] * by_dur.index           # per MW of inverter

    # elasticities along the 2,000 kWh row and the 500 kW column (log-log slopes of the cuts)
    def loglog_slope(x, y):
        lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
        return np.gradient(ly, lx)

    cut_c["elasticity"] = loglog_slope(cut_c["cap_kwh"], cut_c["score"])
    cut_p["elasticity"] = loglog_slope(cut_p["pmax_kw"], cut_p["score"])

    # ------------------------------------------------------------- duration sweep (constant returns)
    # V(E, P) = P * f(E / P): with a fixed 500 kW inverter, sweep the duration h = E / P.
    DUR = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24]
    dur = pd.DataFrame([{k: v for k, v in point(prob, w, C.PMAX * h, C.PMAX).items() if k != "_sol"} for h in DUR])
    for col in ("score", "co2_t_year", "eur_year"):
        dur[f"marg_{col}_per_kwh"] = np.gradient(dur[col], dur["cap_kwh"])          # marginal value of capacity
        dur[f"elast_{col}"] = np.gradient(np.log(dur[col]), np.log(dur["cap_kwh"]))  # capacity share of value
    hh = np.linspace(0.5, 24, 941)

    def balanced(col):                       # duration where capacity and power each hold half the marginal value
        e = np.interp(hh, dur["duration_h"], dur[f"elast_{col}"])
        below = np.flatnonzero(e < 0.5)
        return float(hh[below[0]]) if len(below) else float("nan")

    def best_duration(col, r):               # max value per unit of capex, a kW costing as much as r kWh
        f = np.interp(hh, dur["duration_h"], dur[col])
        return float(hh[np.argmax(f / (hh + r))])

    balanced_h = {col: balanced(col) for col in ("score", "co2_t_year", "eur_year")}
    best_h = {f"kw_costs_{r}_kwh": {col: best_duration(col, r) for col in ("score", "co2_t_year", "eur_year")}
              for r in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)}

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(14.2, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], left=0.065, right=0.975, top=0.80, bottom=0.155,
                          wspace=0.20, hspace=0.50)
    ax = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])
    C.titled(fig, f"Energy-limited: doubling capacity adds {dbl['double_capacity_4000kwh_500kw']['score_vs_design_pct'] - 100:.0f} % "
                  f"to the score, doubling power only {dbl['double_power_2000kwh_1000kw']['score_vs_design_pct'] - 100:.0f} %",
             f"Score of the optimal schedule for each battery size · at the design point one more kW of inverter is worth "
             f"as much as {mrs:.1f} kWh of capacity")

    X, Y = np.meshgrid(S.columns.to_numpy(float), S.index.to_numpy(float))
    levels = np.arange(0, np.ceil(S.to_numpy().max() / 2) * 2 + 2, 2.0)
    cf = ax.contourf(X, Y, S.to_numpy(), levels=levels, cmap=C.SEQ_AQUA, extend="neither")
    cs = ax.contour(X, Y, S.to_numpy(), levels=levels[1:-1], colors=C.SURFACE, linewidths=1.2)
    ax.clabel(cs, fmt="%.0f", fontsize=10, colors=C.INK, inline=True, inline_spacing=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for d_h in (1, 2, 4, 8, 16):
        xs = np.array([max(500, 125 * d_h), min(8000, 2000 * d_h)], float)
        ax.plot(xs, xs / d_h, color=C.INK2, lw=1.0, ls=(0, (1, 3)), zorder=3)
        x0 = xs[0] * 1.09
        ax.text(x0, x0 / d_h * 1.035, f"{d_h} h", fontsize=10.5, color=C.INK2, ha="left", va="bottom",
                rotation=39, rotation_mode="anchor")
    ax.scatter([C.CAP], [C.PMAX], s=170, color=C.INK, edgecolor=C.SURFACE, linewidth=2, zorder=6)
    ax.annotate(f"Design: 2,000 kWh / 500 kW\n{ref['score']:.2f} pts", (C.CAP, C.PMAX), xytext=(-12, 14),
                textcoords="offset points", ha="right", va="bottom", fontsize=11, fontweight="bold", color=C.INK)
    ax.set_xticks([500, 1000, 2000, 4000, 8000], ["500", "1,000", "2,000", "4,000", "8,000"])
    ax.set_yticks([125, 250, 500, 1000, 2000], ["125", "250", "500", "1,000", "2,000"])
    ax.minorticks_off()
    ax.set_xlim(500, 8000)
    ax.set_ylim(125, 2000)
    ax.set_xlabel("Capacity (kWh)")
    ax.set_ylabel("Inverter power (kW)")
    ax.set_title("Score (points) by capacity and power · dotted = equal duration (kWh/kW)")
    ax.grid(False)
    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Score (points)")
    cb.outline.set_visible(False)

    for a, cut, xcol, xd, xlabel, title in (
            (ax_c, cut_c, "cap_kwh", C.CAP, "Capacity (kWh), power fixed at 500 kW",
             "More capacity: CO₂ keeps growing, euros level off"),
            (ax_p, cut_p, "pmax_kw", C.PMAX, "Inverter power (kW), capacity fixed at 2,000 kWh",
             "More power: some extra euros, little extra CO₂")):
        a.axhline(100, color=C.AXIS, lw=1.0, zorder=1)
        a.axvline(xd, color=C.AXIS, lw=1.0, zorder=1)
        ends = {col: 100 * cut[col].iloc[-1] / ref[col] for col in ("co2_t_year", "eur_year")}
        top = max(ends, key=ends.get)
        for col, color, lab in (("co2_t_year", C.C_CO2, "CO₂"), ("eur_year", C.C_COST, "€")):
            yv = 100 * cut[col] / ref[col]
            a.plot(cut[xcol], yv, color=color, lw=2.4, zorder=3)
            a.annotate(f"{lab} {yv.iloc[-1]:.0f}", (cut[xcol].iloc[-1], yv.iloc[-1]),
                       xytext=(0, 7 if col == top else -7), textcoords="offset points", ha="right",
                       va="bottom" if col == top else "top", fontsize=11, color=C.INK)
        a.scatter([xd], [100], s=110, color=C.INK, edgecolor=C.SURFACE, linewidth=2, zorder=5)
        a.set_xlabel(xlabel)
        a.set_ylabel("Saving (design = 100)")
        a.set_title(title)
        a.set_ylim(0, 230)
        a.set_yticks([0, 50, 100, 150, 200])
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_c.set_xlim(0, 8400)
    ax_p.set_xlim(0, 2100)
    C.footnote(fig, C.FOOT + "\nState of charge starts at 50 % of each capacity. Net campus load (≈1,100 kW average) "
                             "caps the useful discharge of the largest inverters.")
    C.save(fig, "a3_sizing")

    # ------------------------------------------------------------- stats
    grid.to_csv(C.STATS_DIR / "a3_sizing_grid.csv", index=False)
    stats = {
        "design": {k: v for k, v in ref.items() if k != "_sol"},
        "duals_design_point": dual, "finite_differences_design_point": fd,
        "annual_value_of_one_more_kwh": {"kg_co2": 1000 * mv_e["co2_t_year"], "eur": mv_e["eur_year"]},
        "annual_value_of_one_more_kw": {"kg_co2": 1000 * mv_p["co2_t_year"], "eur": mv_p["eur_year"]},
        "kwh_of_capacity_worth_one_kw": {"score": mrs, "co2": mv_p["co2_t_year"] / mv_e["co2_t_year"],
                                         "eur": mv_p["eur_year"] / mv_e["eur_year"]},
        "elasticity_design_point": elast,
        "euler_check_score": {"E*dV/dE + P*dV/dP": mv_e["score"] * C.CAP + mv_p["score"] * C.PMAX, "V": ref["score"]},
        "doubling": dbl,
        "by_duration": by_dur.reset_index().to_dict(orient="records"),
        "duration_sweep_500kw": dur.to_dict(orient="records"),
        "balanced_duration_h": balanced_h, "best_duration_h_by_cost_ratio": best_h,
        "cut_capacity_at_500kw": cut_c.to_dict(orient="records"),
        "cut_power_at_2000kwh": cut_p.to_dict(orient="records"),
        "score_grid_rows_power_cols_capacity": {"capacity_kwh": CAPS, "power_kw": POWS,
                                                "score": S.round(3).to_numpy().tolist()},
    }
    C.write_stats("a3_sizing", stats)
    pd.set_option("display.width", 250)
    print("SCORE grid (rows = power kW, cols = capacity kWh)\n", S.round(2).to_string())
    print("\nCO2 t/yr\n", grid.pivot(index="pmax_kw", columns="cap_kwh", values="co2_t_year").round(1).to_string())
    print("\nk EUR/yr\n", (grid.pivot(index="pmax_kw", columns="cap_kwh", values="eur_year") / 1000).round(1).to_string())
    print("\ncycles/day\n", grid.pivot(index="pmax_kw", columns="cap_kwh", values="cycles_per_day").round(2).to_string())
    print("\nby duration\n", by_dur.round(3).to_string())
    print("\ncut capacity\n", cut_c.round(3).to_string(index=False))
    print("\ncut power\n", cut_p.round(3).to_string(index=False))
    print("\nduration sweep\n", dur.round(3).to_string(index=False))
    for k in ("balanced_duration_h", "best_duration_h_by_cost_ratio", "duals_design_point",
              "finite_differences_design_point", "annual_value_of_one_more_kwh", "annual_value_of_one_more_kw", "kwh_of_capacity_worth_one_kw", "elasticity_design_point",
              "euler_check_score", "doubling"):
        print(k, stats[k])


if __name__ == "__main__":
    main()
