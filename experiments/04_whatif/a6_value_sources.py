"""
experiments/04_whatif/a6_value_sources.py
Analysis 6 -- where the value comes from: season, kind of day, grid conditions.

Day-level attribution needs days that stand on their own, so each calendar day is solved as a
separate problem with a periodic state of charge (end of day = start of day, level free). The
sum of these 281 one-day optima is compared with the full-horizon optimum to show how little
value comes from moving energy between days.

Outputs: figs/a6_value_sources.png, stats/a6_value_sources.json, stats/a6_daily.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
from src.data import load_hourly_full


def r2(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def main() -> None:
    C.set_style()
    prob = C.Problem()
    idx = prob.idx
    w = prob.weights()
    day_id = idx.normalize()

    full_sol, full_acc = C.run(prob)
    sol = C.solve(prob, w, cyclic=day_id.asi8)
    acc = C.account(prob, sol)

    saved = prob.base - sol["g"]
    h = pd.DataFrame({"co2_saved": saved * prob.co2, "eur_saved": saved * prob.price, "pts": 100 * saved * w,
                      "d": sol["d"], "co2": prob.co2, "price": prob.price, "w": w, "load": prob.load,
                      "p": sol["p"]}, index=idx)
    g = h.groupby(day_id)
    top4 = lambda s: s.nlargest(4).mean()        # noqa: E731
    bot4 = lambda s: s.nsmallest(4).mean()       # noqa: E731
    day = pd.DataFrame({
        "co2_saved_kg": g["co2_saved"].sum(), "eur_saved": g["eur_saved"].sum(), "score_pts": g["pts"].sum(),
        "cycles": g["d"].sum() / 0.9 ** 0.5 / C.CAP,
        "co2_mean_g": 1000 * g["co2"].mean(), "co2_min_g": 1000 * g["co2"].min(), "co2_max_g": 1000 * g["co2"].max(),
        "co2_spread_g": 1000 * (g["co2"].max() - g["co2"].min()),
        "co2_spread_4h_g": 1000 * (g["co2"].agg(top4) - g["co2"].agg(bot4)),
        "price_spread_c": 100 * (g["price"].max() - g["price"].min()),
        "price_spread_4h_c": 100 * (g["price"].agg(top4) - g["price"].agg(bot4)),
        "price_mean_c": 100 * g["price"].mean(),
        "w_ratio_4h": g["w"].agg(top4) / g["w"].agg(bot4),
        "hour_of_co2_min": g["co2"].agg(lambda s: s.idxmin().hour),
        "load_mean_kw": g["load"].mean(),
    })
    day["weekend"] = day.index.day_of_week >= 5
    day["month"] = day.index.month
    day["season"] = day["month"].map(C.SEASON_OF_MONTH)
    # grid mix on the day (peninsular system, from co2_red.csv)
    mix = load_hourly_full().reindex(idx)
    mg = mix.groupby(day_id)
    day["emitting_share_mean_pct"] = 100 * mg["pct_emitting"].mean()          # CO2 intensity = 0.37 kg/kWh x this share
    day["emitting_share_min_pct"] = 100 * mg["pct_emitting"].min()
    day["emitting_share_max_pct"] = 100 * mg["pct_emitting"].max()
    day["emitting_gen_swing_gw"] = (mg["gen_emitting_mw"].max() - mg["gen_emitting_mw"].min()) / 1000
    day["clean_gen_swing_gw"] = (mg["gen_clean_mw"].max() - mg["gen_clean_mw"].min()) / 1000

    tot = day[["co2_saved_kg", "eur_saved", "score_pts"]].sum()
    n_days = len(day)

    def group(by, order=None) -> list:
        out = []
        gg = day.groupby(by, observed=True)
        keys = order if order is not None else list(gg.groups)
        for k in keys:
            d = gg.get_group(k)
            out.append({"group": str(k), "days": len(d), "days_pct": 100 * len(d) / n_days,
                        "co2_kg_per_day": d["co2_saved_kg"].mean(), "eur_per_day": d["eur_saved"].mean(),
                        "score_pts_per_day": d["score_pts"].mean(), "cycles_per_day": d["cycles"].mean(),
                        "co2_share_pct": 100 * d["co2_saved_kg"].sum() / tot["co2_saved_kg"],
                        "eur_share_pct": 100 * d["eur_saved"].sum() / tot["eur_saved"],
                        "score_share_pct": 100 * d["score_pts"].sum() / tot["score_pts"],
                        "co2_spread_g_mean": d["co2_spread_g"].mean(), "price_spread_c_mean": d["price_spread_c"].mean()})
        return out

    day["spread_tercile"] = pd.qcut(day["co2_spread_g"], 3, labels=["low spread", "mid spread", "high spread"])
    day["daytype"] = np.where(day["weekend"], "weekend", "weekday")
    by_season = group("season", C.SEASON_ORDER)
    by_daytype = group("daytype", ["weekday", "weekend"])
    by_tercile = group("spread_tercile", ["low spread", "mid spread", "high spread"])
    by_month = group("month", C.MONTH_ORDER)
    for r in by_month:
        r["group"] = C.MONTH_ABBR[int(r["group"])]

    # concentration: share of the saving delivered by the best 20 % of days
    def top_share(col, q=0.2):
        s = day[col].sort_values(ascending=False)
        return 100 * float(s.iloc[: int(round(q * len(s)))].sum() / s.sum())

    fits = {
        "co2_saved_vs_co2_spread_maxmin_r2": r2(day["co2_spread_g"], day["co2_saved_kg"]),
        "co2_saved_vs_co2_spread_4h_r2": r2(day["co2_spread_4h_g"], day["co2_saved_kg"]),
        "co2_saved_vs_co2_mean_r2": r2(day["co2_mean_g"], day["co2_saved_kg"]),
        "eur_saved_vs_price_spread_maxmin_r2": r2(day["price_spread_c"], day["eur_saved"]),
        "eur_saved_vs_price_spread_4h_r2": r2(day["price_spread_4h_c"], day["eur_saved"]),
        "score_vs_w_ratio_4h_r2": r2(day["w_ratio_4h"], day["score_pts"]),
        "co2_saved_vs_clean_gen_swing_r2": r2(day["clean_gen_swing_gw"], day["co2_saved_kg"]),
        "co2_saved_vs_emitting_gen_swing_r2": r2(day["emitting_gen_swing_gw"], day["co2_saved_kg"]),
        "co2_saved_vs_load_mean_r2": r2(day["load_mean_kw"], day["co2_saved_kg"]),
        "co2_spread_vs_co2_mean_r2": r2(day["co2_mean_g"], day["co2_spread_g"]),
        "corr_daily_co2_saved_vs_eur_saved": float(np.corrcoef(day["co2_saved_kg"], day["eur_saved"])[0, 1]),
    }
    slope, icpt = np.polyfit(day["co2_spread_g"], day["co2_saved_kg"], 1)
    fits["co2_saved_kg_per_g_of_spread"] = float(slope)
    fits["co2_saved_intercept_kg"] = float(icpt)

    # profile of the grid on the best and worst thirds of days (by CO2 saved)
    day["value_tercile"] = pd.qcut(day["co2_saved_kg"], 3, labels=["worst", "middle", "best"])
    lab = pd.Series(day["value_tercile"].astype(str).reindex(day_id).to_numpy(), index=idx)
    prof = (1000 * h["co2"]).groupby([lab, idx.hour]).mean().unstack(0)
    prof_p = h["p"].groupby([lab, idx.hour]).mean().unstack(0)
    best_days, worst_days = day[day["value_tercile"] == "best"], day[day["value_tercile"] == "worst"]

    # ---------------------------------------------------------------- figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.9))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.79, bottom=0.15, wspace=0.22)
    C.titled(fig, f"The battery earns its keep on days with a wide CO₂ swing: the top third of days deliver "
                  f"{by_tercile[2]['co2_share_pct']:.0f} % of the CO₂ saving",
             f"Daily CO₂ spread alone explains {100 * fits['co2_saved_vs_co2_spread_maxmin_r2']:.0f} % of the "
             f"day-to-day variation in CO₂ avoided · each dot is one of {n_days} train days")
    for we, color, name in ((False, C.C_WEEKDAY, "Weekday"), (True, C.C_WEEKEND, "Weekend")):
        d = day[day["weekend"] == we]
        ax.scatter(d["co2_spread_g"], d["co2_saved_kg"], s=30, color=color, alpha=0.85, edgecolor=C.SURFACE,
                   linewidth=0.7, label=f"{name} ({len(d)} days)", zorder=3)
    xs = np.array([day["co2_spread_g"].min(), day["co2_spread_g"].max()])
    ax.plot(xs, slope * xs + icpt, color=C.INK, lw=1.6, zorder=4)
    ax.text(0.97, 0.10, f"+{slope:.1f} kg avoided per g/kWh of spread\n≈ one full cycle ({C.CAP * 0.9 ** 0.5 / 1000:.1f} MWh delivered)\n"
                        "moved across the day's swing", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=C.INK)
    ax.axhline(0, color=C.AXIS, lw=1.0, zorder=1)
    ax.set_xlabel("Daily spread of grid CO₂ intensity, max − min (g CO₂/kWh)")
    ax.set_ylabel("CO₂ avoided that day (kg)")
    ax.set_title("CO₂ avoided per day against that day's CO₂ spread")
    ax.legend(loc="upper left", handlelength=1.0, markerscale=1.4)

    hrs = np.arange(24)
    for name, color, text in (("best", C.BLUE, f"Best third of days ({best_days['co2_saved_kg'].mean():.0f} kg CO₂ avoided per day)"),
                              ("worst", C.GRAY, f"Worst third of days ({worst_days['co2_saved_kg'].mean():.0f} kg per day)")):
        ax2.plot(hrs, prof[name].to_numpy(), color=color, lw=2.6, zorder=3, label=text)
    ax2.legend(loc="upper center", handlelength=1.6)
    ax2.set_xticks(range(0, 24, 3))
    ax2.set_xlim(-0.3, 23.3)
    ax2.set_ylim(0, max(prof["best"].max(), prof["worst"].max()) * 1.28)
    ax2.set_xlabel("Hour of day (local time)")
    ax2.set_ylabel("Mean grid CO₂ intensity (g CO₂/kWh)")
    ax2.set_title("What a good day looks like: a solar valley and a dirty evening")
    C.footnote(fig, C.FOOT.replace("battery ends each run as full as it started", "each day solved on its own")
               + f"\nOne-day optima add up to {100 * acc['score'] / full_acc['score']:.1f} % of the full-horizon optimum: "
                 "day-by-day attribution is fair, little value comes from carrying energy between days.")
    C.save(fig, "a6_value_sources")

    # ---------------------------------------------------------------- stats
    stats = {
        "days": n_days,
        "daily_cyclic_vs_full_horizon": {
            "score_daily": acc["score"], "score_full": full_acc["score"], "pct": 100 * acc["score"] / full_acc["score"],
            "co2_kg_daily": acc["co2_saved_kg"], "co2_kg_full": full_acc["co2_saved_kg"],
            "eur_daily": acc["cost_saved_eur"], "eur_full": full_acc["cost_saved_eur"]},
        "by_season": by_season, "by_daytype": by_daytype, "by_co2_spread_tercile": by_tercile, "by_month": by_month,
        "fits": fits,
        "concentration": {"best_20pct_days_share_of_co2_pct": top_share("co2_saved_kg"),
                          "best_20pct_days_share_of_eur_pct": top_share("eur_saved"),
                          "days_with_negative_co2_saving_pct": 100 * float((day["co2_saved_kg"] < 0).mean()),
                          "co2_saved_kg_p10_p50_p90": day["co2_saved_kg"].quantile([0.1, 0.5, 0.9]).tolist(),
                          "eur_saved_p10_p50_p90": day["eur_saved"].quantile([0.1, 0.5, 0.9]).tolist(),
                          "cv_co2": float(day["co2_saved_kg"].std() / day["co2_saved_kg"].mean()),
                          "cv_eur": float(day["eur_saved"].std() / day["eur_saved"].mean())},
        "best_vs_worst_third_by_co2_saved": {
            name: {"days": len(d), "co2_saved_kg_per_day": d["co2_saved_kg"].mean(), "eur_per_day": d["eur_saved"].mean(),
                   "co2_mean_g": d["co2_mean_g"].mean(), "co2_min_g": d["co2_min_g"].mean(), "co2_max_g": d["co2_max_g"].mean(),
                   "co2_spread_g": d["co2_spread_g"].mean(), "weekend_pct": 100 * d["weekend"].mean(),
                   "co2_min_between_10h_17h_pct": 100 * d["hour_of_co2_min"].between(10, 17).mean(),
                   "emitting_share_mean_pct": d["emitting_share_mean_pct"].mean(),
                   "emitting_share_min_pct": d["emitting_share_min_pct"].mean(),
                   "emitting_share_max_pct": d["emitting_share_max_pct"].mean(),
                   "emitting_gen_swing_gw": d["emitting_gen_swing_gw"].mean(),
                   "clean_gen_swing_gw": d["clean_gen_swing_gw"].mean(),
                   "months": d["month"].map(C.MONTH_ABBR).value_counts().to_dict()}
            for name, d in (("best", best_days), ("worst", worst_days))},
        "co2_intensity_over_emitting_share_kg_per_kwh": float(np.median(mix["co2_kg_per_kwh"] / mix["pct_emitting"])),
        "co2_profile_g_by_hour": {k: prof[k].round(1).tolist() for k in ("best", "middle", "worst")},
        "power_profile_kw_by_hour": {k: prof_p[k].round(0).tolist() for k in ("best", "middle", "worst")},
        "hour_of_daily_co2_min_between_10h_17h_pct": 100 * float(day["hour_of_co2_min"].between(10, 17).mean()),
    }
    C.write_stats("a6_value_sources", stats)
    day.round(4).to_csv(C.STATS_DIR / "a6_daily.csv")
    pd.set_option("display.width", 250)
    for name in ("by_season", "by_daytype", "by_co2_spread_tercile", "by_month"):
        print(name, "\n", pd.DataFrame(stats[name]).round(2).to_string(index=False))
    for k in ("daily_cyclic_vs_full_horizon", "fits", "concentration", "best_vs_worst_third_by_co2_saved",
              "hour_of_daily_co2_min_between_10h_17h_pct"):
        print(k, stats[k])


if __name__ == "__main__":
    main()
