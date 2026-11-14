"""
Q3 - Photovoltaic generation.
  fig q3_pv.png : size against load, hour x month heatmap, relation to irradiance
  stats q3_pv.json
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

import common as C

C.set_style()
stats: dict = {}

full = C.load_hourly_full()
off = C.load_hourly_official()
off = off[off["load_kw"] > 0]                     # drop meter-fault hours
pv = full["pv_kw"].dropna()
pv = pv[pv.index.month != 11]                    # November: one stray 00:00 sample only

stats["hours_pv"] = int(len(pv))
stats["pv_mean_kw"] = float(pv.mean())
stats["pv_max_kw"] = float(pv.max())
stats["pv_p99_kw"] = float(pv.quantile(0.99))
stats["pv_share_of_load_energy"] = float(off["pv_kw"].sum() / off["load_kw"].sum())
ratio = off["pv_kw"] / off["load_kw"]
stats["pv_over_load_max"] = float(ratio.max())
stats["pv_over_load_max_when"] = str(ratio.idxmax())
stats["pv_over_load_p99"] = float(ratio.quantile(0.99))
stats["hours_pv_exceeds_load"] = int((off["pv_kw"] > off["load_kw"]).sum())
stats["pv_daily_energy_kwh_mean"] = float(pv.groupby(pv.index.normalize()).sum().mean())
stats["pv_daily_energy_kwh_max"] = float(pv.groupby(pv.index.normalize()).sum().max())
stats["battery_capacity_over_best_pv_day"] = float(C.E_CAP / pv.groupby(pv.index.normalize()).sum().max())
by_m = pv.groupby(pv.index.month)
stats["pv_monthly_mean_kw"] = {C.MONTH_ABBR[m]: float(v) for m, v in by_m.mean().items()}
stats["pv_monthly_peak_hour_mean_kw"] = {
    C.MONTH_ABBR[m]: float(g.groupby(g.index.hour).mean().max()) for m, g in by_m}

# relation to irradiance (hours with both, daylight only)
both = full[["pv_kw", "irradiance_wm2", "temp_panel_c"]].dropna()
dl = both[both["irradiance_wm2"] > 20]
lin = st.linregress(dl["irradiance_wm2"], dl["pv_kw"])
stats["irr_hours"] = int(len(dl))
stats["pv_vs_irr_slope_kw_per_wm2"] = float(lin.slope)
stats["pv_vs_irr_intercept_kw"] = float(lin.intercept)
stats["pv_vs_irr_r2"] = float(lin.rvalue ** 2)
k0 = float((dl["pv_kw"] * dl["irradiance_wm2"]).sum() / (dl["irradiance_wm2"] ** 2).sum())   # through the origin
stats["pv_vs_irr_slope_origin"] = k0
stats["pv_at_1000wm2_kw"] = k0 * 1000
resid = dl["pv_kw"] - k0 * dl["irradiance_wm2"]
stats["pv_vs_irr_mae_kw"] = float(resid.abs().mean())
stats["hours_above_140kw"] = int((pv > 140).sum())

# ================================================================== FIGURE
fig = plt.figure(figsize=(13.5, 6.6))
C.titled(fig, f"PV is marginal: {stats['pv_share_of_load_energy']:.1%} of campus energy, never more than "
              f"{stats['pv_over_load_max']:.0%} of the load in any hour",
         "Three 50 kW inverters. Generation never exceeds demand, so the battery cannot be charged from free solar surplus.")
gs = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=0.80, bottom=0.14, wspace=0.27, width_ratios=[1, 1.12, 1])

# (a) load vs PV on one axis
ax = fig.add_subplot(gs[0])
h = np.arange(24)
lp = off.groupby(off.index.hour)[["load_kw", "pv_kw"]].mean()
ax.plot(h, lp["load_kw"], color=C.C_LOAD, lw=2.2, label="Load")
ax.plot(h, lp["pv_kw"], color=C.C_PV, lw=2.2, label="PV")
ax.fill_between(h, 0, lp["pv_kw"], color=C.C_PV, alpha=0.18, lw=0)
ax.text(12.5, lp["load_kw"].max() + 55, "Load", ha="center", color=C.INK, fontsize=11.5, fontweight="bold")
ax.text(12.5, lp["pv_kw"].max() + 45, "PV", ha="center", color=C.INK, fontsize=11.5, fontweight="bold")
ax.set_ylim(0, 1650)
ax.set_ylabel("Mean power (kW)")
C.hour_axis(ax, step=6)
ax.set_title("Mean day: load vs PV, same axis")
ax.legend(loc="upper left", fontsize=10.5, ncol=2, handlelength=1.4, columnspacing=1.2)

# (b) hour x month heatmap
ax2 = fig.add_subplot(gs[1])
mat = pv.groupby([pv.index.month, pv.index.hour]).mean().unstack().reindex(C.MONTH_ORDER)
im = ax2.imshow(mat.values, aspect="auto", cmap=C.SEQ_YELLOW, vmin=0, vmax=120)
ax2.set_yticks(range(12))
ax2.set_yticklabels([C.MONTH_ABBR[m] for m in C.MONTH_ORDER])
ax2.set_xticks(range(0, 24, 6))
ax2.set_xlabel("Hour of day (local time)")
ax2.grid(False)
ax2.tick_params(length=0)
for sp in ax2.spines.values():
    sp.set_visible(False)
ax2.set_xticks(np.arange(-0.5, 24, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, 12, 1), minor=True)
ax2.grid(which="minor", color=C.SURFACE, linewidth=1.2)
ax2.tick_params(which="minor", length=0)
ax2.text(11.5, 2, "no inverter data", ha="center", va="center", fontsize=10, color=C.MUTED)
cb = fig.colorbar(im, ax=ax2, pad=0.02, fraction=0.05)
cb.outline.set_visible(False)
ax2.set_title("Mean PV by month and hour (kW)")

# (c) PV vs irradiance
ax3 = fig.add_subplot(gs[2])
ax3.scatter(dl["irradiance_wm2"], dl["pv_kw"], s=7, color=C.C_PV, alpha=0.35, edgecolor="none", rasterized=True)
xx = np.array([0, 1080])
ax3.plot(xx, k0 * xx, color=C.INK, lw=1.6)
ax3.axhline(150, color=C.MUTED, lw=1.0)
ax3.text(15, 153, "inverter limit 3 × 50 kW", color=C.INK2, fontsize=10, va="bottom")
ax3.text(1075, 48, f"PV ≈ {k0 * 1000:.0f} kW per\n1,000 W/m²\nR² = {stats['pv_vs_irr_r2']:.2f}", color=C.INK,
         fontsize=10.5, ha="right", va="top")
ax3.set_xlim(0, 1100)
ax3.set_ylim(0, 165)
ax3.set_xlabel("Irradiance on the panel plane (W/m²)")
ax3.set_ylabel("PV output (kW)")
ax3.set_title("PV follows irradiance almost linearly")
C.footnote(fig, f"Sample: {len(pv):,} train hours with inverter data (none in November, none after 16 Aug 2026); "
                f"scatter uses the {len(dl):,} daylight hours with irradiance above 20 W/m².")
C.save(fig, "q3_pv")

C.write_stats("q3_pv", stats)
print({k: v for k, v in stats.items() if not isinstance(v, dict)})
print(stats["pv_monthly_mean_kw"])
