# Usage: .venv/Scripts/python.exe experiments/26_hard_eda/fig4_value.py
"""Figure 4: where the perfect-foresight points come from.
Rows: folds A, B and C (5,417 train hours). Four perfect-foresight runs per fold:
  1. standard mode (2,000 kWh / 500 kW, solar as built)
  2. battery x3, solar as built
  3. battery x3, solar x12, free surplus removed (net load clipped at 0): pure arbitrage
  4. hard mode as it is (surplus can be stored)
Free-surplus value = run 4 - run 3, hour by hour. Each run is scored against its own no-battery baseline.
Writes reports/15_hard_eda/figs/f4_value_sources.png and value_sources.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, HERE, MONTH_NUM, MONTHS, folds_abc, run_oracle, signal
from src.plotstyle import C, GREY, INK, apply, save

rows, monthly = [], []
for k, f in folds_abc().items():
    load, pv = f["load_kw"].to_numpy(float), f["pv_kw"].to_numpy(float)
    std = run_oracle(f, net=load - pv / 12, cap=2000.0, pmax=500.0, soc0=1000.0)
    big = run_oracle(f, net=load - pv / 12)
    arb = run_oracle(f, net=np.maximum(load - pv, 0.0))
    full = run_oracle(f)
    w = signal(f).to_numpy(float) * 1000
    sur = np.maximum(pv - load, 0.0)
    free = np.minimum(full.charge, sur)
    paid = full.charge - free
    rows.append({"fold": k, "standard": std.pts.sum(), "battery_x3": big.pts.sum(), "arbitrage_only": arb.pts.sum(),
                 "hard_full": full.pts.sum(), "surplus_value": full.pts.sum() - arb.pts.sum(),
                 "base_std_mwh": std.base_kw.sum() / 1e3, "base_hard_mwh": full.base_kw.sum() / 1e3,
                 "charge_mwh": full.charge.sum() / 1e3, "free_mwh": free.sum() / 1e3, "surplus_mwh": sur.sum() / 1e3,
                 "discharge_mwh": full.discharge.sum() / 1e3,
                 "w_paid_charge": np.average(w, weights=paid), "w_discharge": np.average(w, weights=full.discharge),
                 "w_mean": w.mean(), "cycles_per_day": full.discharge.sum() / 0.9 ** 0.5 / 6000 / (len(f) / 24),
                 "arb_charge_mwh": arb.charge.sum() / 1e3,
                 "base_std_co2_t": float((std.base_kw * f.co2_kg_per_kwh).sum() / 1e3),
                 "base_hard_co2_t": float((full.base_kw * f.co2_kg_per_kwh).sum() / 1e3),
                 "base_std_keur": float((std.base_kw * f.precio_eur_per_kwh).sum() / 1e3),
                 "base_hard_keur": float((full.base_kw * f.precio_eur_per_kwh).sum() / 1e3),
                 "std_discharge_mwh": std.discharge.sum() / 1e3, "big_discharge_mwh": big.discharge.sum() / 1e3,
                 "arb_discharge_mwh": arb.discharge.sum() / 1e3,
                 "full_co2_saved_t": float(((full.base_kw - full.grid_kw) * f.co2_kg_per_kwh).sum() / 1e3),
                 "arb_co2_saved_t": float(((arb.base_kw - arb.grid_kw) * f.co2_kg_per_kwh).sum() / 1e3),
                 "big_co2_saved_t": float(((big.base_kw - big.grid_kw) * f.co2_kg_per_kwh).sum() / 1e3),
                 "full_eur_saved_k": float(((full.base_kw - full.grid_kw) * f.precio_eur_per_kwh).sum() / 1e3),
                 "arb_eur_saved_k": float(((arb.base_kw - arb.grid_kw) * f.precio_eur_per_kwh).sum() / 1e3),
                 "big_eur_saved_k": float(((big.base_kw - big.grid_kw) * f.precio_eur_per_kwh).sum() / 1e3)})
    m = pd.DataFrame({"month": f.index.month, "arb": arb.pts.values, "full": full.pts.values, "hours": 1.0,
                      "free_kwh": free.values}).groupby("month").sum()
    m["fold"] = k
    monthly.append(m.reset_index())
res = pd.DataFrame(rows).set_index("fold")
res.loc["mean"] = res.mean()
print(res.round(3).T.to_string())
res.round(4).to_csv(HERE / "value_sources.csv")
mm = pd.concat(monthly).groupby("month")[["arb", "full", "hours", "free_kwh"]].sum()
mm["arb_wk"] = mm["arb"] / mm["hours"] * 168
mm["sur_wk"] = (mm["full"] - mm["arb"]) / mm["hours"] * 168
mm["free_kwh_wk"] = mm["free_kwh"] / mm["hours"] * 168
mm = mm.reindex([m for m in MONTH_NUM if m in mm.index])
mm.index = [MONTHS[MONTH_NUM.index(m)] for m in mm.index]
mm["sur_share"] = mm["sur_wk"] / (mm["arb_wk"] + mm["sur_wk"])
print(mm.round(3).to_string())
mm.round(4).to_csv(HERE / "value_sources_by_month.csv")

mean = res.loc["mean"]
apply()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.5})
labels = ["Standard mode\n2,000 kWh / 500 kW, solar as built", "Battery x3\nsolar as built",
          "Battery x3, solar x12\nsurplus thrown away", "Hard mode\nbattery x3, solar x12, surplus stored"]
keys = ["standard", "battery_x3", "arbitrage_only", "hard_full"]
y = np.arange(4)[::-1]
ax.barh(y[0], mean["standard"], color=GREY, height=0.6)
ax.barh(y[1], mean["battery_x3"], color=GREY, height=0.6)
ax.barh(y[2], mean["arbitrage_only"], color=C[0], height=0.6)
ax.barh(y[3], mean["arbitrage_only"], color=C[0], height=0.6)
ax.barh(y[3], mean["surplus_value"], left=mean["arbitrage_only"], color=C[1], height=0.6)
for yi, kx in zip(y, keys):
    vals = res.loc[["A", "B", "C"], kx]
    ax.scatter(vals, [yi] * 3, s=14, color=INK["primary"], zorder=3, linewidths=0)
    ax.text(max(vals.max(), mean[kx]) + 0.4, yi, f"{mean[kx]:.2f}", va="center", ha="left", fontsize=10)
ax.text(mean["arbitrage_only"] / 2, y[3], "arbitrage", color="white", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(mean["arbitrage_only"], y[3] + 0.5, f"free surplus +{mean['surplus_value']:.2f}", color=C[1], fontsize=9.5,
        fontweight="bold", ha="left", va="center")
ax.set_yticks(y, labels)
ax.set_xlim(0, 26.5)
ax.set_ylim(-0.75, 3.5)
ax.set_xlabel("Best possible score, points (mean of folds A, B, C; dots = each fold)")
ax.grid(axis="y", visible=False)
ax.set_title("From standard to hard mode, step by step", fontsize=10, color=INK["secondary"])

xm = np.arange(len(mm))
ax2.bar(xm, mm["arb_wk"], color=C[0], width=0.7, label="Arbitrage: grid energy moved from low w to high w")
ax2.bar(xm, mm["sur_wk"], bottom=mm["arb_wk"], color=C[1], width=0.7, label="Free surplus solar stored")
for xi, (a, s) in enumerate(zip(mm["arb_wk"], mm["sur_wk"])):
    ax2.text(xi, a + s + 0.03, f"{a + s:.2f}", ha="center", va="bottom", fontsize=8.5)
ax2.set_xticks(xm, mm.index)
ax2.set_ylabel("Score points per week of that month")
ax2.set_ylim(0, (mm["arb_wk"] + mm["sur_wk"]).max() * 1.32)
ax2.grid(axis="x", visible=False)
ax2.legend(loc="upper left", fontsize=8.5, handlelength=1.2)
ax2.set_title("Hard-mode points by month and by source", fontsize=10, color=INK["secondary"])

share = mean["surplus_value"] / mean["hard_full"]
fig.suptitle(f"Of the {mean['hard_full']:.1f} points available, {mean['arbitrage_only']:.1f} are plain arbitrage with a battery three times larger; "
             f"free solar adds {mean['surplus_value']:.1f} ({share:.0%})",
             x=-0.06, y=1.03, ha="left", fontsize=12.5, fontweight="bold")
fig.text(-0.06, -0.04,"Train rows: folds A, B and C (5,417 hours), perfect foresight. 'Surplus thrown away' = same run with the net load clipped at zero, "
         "so the battery must buy every kWh it stores.\nEach bar is scored against its own no-battery baseline. Monthly bars: points earned in that month's "
         "hours, pooled over the three folds, per 168 hours.", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "f4_value_sources.png")
print("saved")
