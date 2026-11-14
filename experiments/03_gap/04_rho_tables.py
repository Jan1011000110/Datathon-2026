"""RHO = CO2_base / Cost_base: how much it varies, and how well it can be predicted without leakage.

1. RHO by month and by fold (train).
2. Prediction of a fold's RHO from calendar-keyed tables fitted on train_complement(fold) only:
   per-month mean hourly baseline CO2 and cost, each month weighted equally (a test-like fold has
   168 h per month).
3. The same table fitted on the whole train set -> prediction for the hidden test weeks, compared
   with the public-leaderboard baseline reconstructed by the project lead
   (1,008 h: CO2 71,797-71,832 kg, cost 118,733-118,750 EUR, RHO 0.6048).
"""
from lib import *
from src.data import load_hourly_full, TEST_MONTHS

pd.set_option("display.width", 220)


def month_table(tr):
    """tr: hourly frame (may contain NaN load/pv). Returns per-(year, month) mean hourly baseline
    CO2 (kg/h) and cost (EUR/h). Hours without meter data use the frame's own mean net load by
    (weekend, hour) -- only matters for Nov 2025, which has no meter data at all."""
    net = (tr["load_kw"] - tr["pv_kw"].fillna(0.0)).clip(lower=0)
    we = (tr.index.day_of_week >= 5).astype(int)
    prof = net.groupby([we, tr.index.hour]).transform("mean")
    net = net.fillna(prof)
    d = pd.DataFrame({"co2": net * tr["co2_kg_per_kwh"], "cost": net * tr["precio_eur_per_kwh"],
                      "has_load": tr["load_kw"].notna().astype(float)})
    g = d.groupby([tr.index.year, tr.index.month]).mean()
    g["rho"] = g["co2"] / g["cost"]
    return g


if __name__ == "__main__":
    folds = get_folds("filled")
    full = load_hourly_full()
    tab = month_table(full)
    print("per-month baseline (train, all days):")
    print(tab.round(3).to_string())
    print("\ntrain-wide RHO (hours with meter data only): %.4f" % true_rho(full.dropna(subset=["load_kw"]).fillna({"pv_kw": 0.0})))

    rows = []
    for f in ("A", "B", "C"):
        df = folds[f]
        comp = train_complement(df, "full")
        t = month_table(comp)
        t = t.loc[[k for k in t.index if k in TEST_MONTHS]]
        pred = t["co2"].sum() / t["cost"].sum()
        pred_wo_nov = t.drop((2025, 11))["co2"].sum() / t.drop((2025, 11))["cost"].sum()
        # same thing with the fold's own months actually present / fold truth by month
        truth_m = month_table(df)
        rows.append({"fold": f, "true": true_rho(df), "month-table pred (complement)": pred,
                     "pred w/o Nov": pred_wo_nov, "complement pooled": true_rho(comp.dropna(subset=["load_kw"]).fillna({"pv_kw": 0.0})),
                     "fixed 0.60 err": 0.60 - true_rho(df), "table err": pred - true_rho(df),
                     "per-month |err| mean": float(np.mean(np.abs(t["rho"] - truth_m["rho"])))})
    print("\nfold RHO vs calendar-table prediction (fitted on the complement only):")
    print(pd.DataFrame(rows).set_index("fold").round(4).to_string())

    # prediction for the hidden test: all train days of each month, equal weight per month
    t = tab.loc[[k for k in tab.index if k in TEST_MONTHS]]
    for label, tt in (("12 months", t), ("11 months (no Nov 2025)", t.drop((2025, 11)))):
        n_m = len(tt)
        co2_1008 = tt["co2"].sum() * 84 * 12 / n_m
        cost_1008 = tt["cost"].sum() * 84 * 12 / n_m
        print("\ntest prediction, %s: RHO %.4f ; per 1,008 h: CO2 %.0f kg, cost %.0f EUR  (public LB: 71,797-71,832 kg, 118,733-118,750 EUR, RHO 0.6048)" % (
            label, tt["co2"].sum() / tt["cost"].sum(), co2_1008, cost_1008))
    # what if the Nov week were in the test with zero load? then it adds nothing to either total
    tt = t.drop((2025, 11))
    print("if Nov is in the test with load=0 (adds 0 to both totals): per 1,008 h CO2 %.0f kg, cost %.0f EUR" % (
        tt["co2"].sum() * 84, tt["cost"].sum() * 84))
    print("\nfold baselines per 1,008 h (for scale):")
    for f in FOLDS:
        df = folds[f]
        base = np.maximum(df.load_kw - df.pv_kw, 0)
        print("  %-3s CO2 %.0f kg  cost %.0f EUR  rho %.4f" % (f, (base * df.co2_kg_per_kwh).sum() * 1008 / len(df),
                                                      (base * df.precio_eur_per_kwh).sum() * 1008 / len(df), true_rho(df)))
