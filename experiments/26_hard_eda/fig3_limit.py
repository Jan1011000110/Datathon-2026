# Usage: .venv/Scripts/python.exe experiments/26_hard_eda/fig3_limit.py
"""Figure 3: what limits the discharge in the perfect-foresight schedule.
Rows: folds A, B and C (days 1-7, 8-14, 15-21 of every train month; 5,417 h in total), oracle solved on
each fold separately, discharge hours pooled. Writes reports/15_hard_eda/figs/f3_discharge_limit.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, PMAX, TOL, folds_abc, load_hard, run_oracle
from src.plotstyle import C, GREY, INK, apply, save

parts = []
for k, f in folds_abc().items():
    tr = run_oracle(f)
    tr["fold"] = k
    parts.append(tr)
tr = pd.concat(parts)
dis = tr[tr.discharge > TOL].copy()
dis["kind"] = "partial"
dis.loc[(dis.grid_kw < TOL) & (dis.discharge < PMAX - TOL), "kind"] = "load"
dis.loc[dis.discharge >= PMAX - TOL, "kind"] = "inverter"

n = len(dis)
e = dis.discharge.sum()
print(f"hours in the three folds {len(tr)}, discharge hours {n} ({n/len(tr):.1%}), energy {e/1e3:.0f} MWh")
for kind in ("load", "inverter", "partial"):
    s = dis[dis.kind == kind]
    print(f"  {kind:9s}: {len(s):4d} h ({len(s)/n:.1%}), {s.discharge.sum()/e:.1%} of energy, median power {s.discharge.median():.0f} kW")
print("  by fold, share of discharge hours limited by the load:",
      {k: round(float((g.kind == 'load').mean()), 3) for k, g in dis.groupby('fold')})
print(f"net load in discharge hours: median {dis.net_kw.median():.0f}, IQR {dis.net_kw.quantile(.25):.0f}-{dis.net_kw.quantile(.75):.0f}, "
      f"p5 {dis.net_kw.quantile(.05):.0f}, p95 {dis.net_kw.quantile(.95):.0f}, share below 1500: {(dis.net_kw < PMAX).mean():.1%}")
print(f"discharge power: median {dis.discharge.median():.0f} kW, mean {dis.discharge.mean():.0f}; "
      f"hours above 1,000 kW: {(dis.discharge > 1000).mean():.1%}; above 1,400: {(dis.discharge > 1400).mean():.1%}")
ch = tr[tr.charge > TOL]
print(f"charge hours {len(ch)}; at 1,500 kW: {(ch.charge >= PMAX - TOL).mean():.1%} of hours, "
      f"{ch.charge[ch.charge >= PMAX - TOL].sum()/ch.charge.sum():.1%} of energy; median {ch.charge.median():.0f} kW")
# what a blind full-power discharge would waste in those same hours
spill = np.maximum(PMAX - np.maximum(dis.net_kw, 0), 0)
print(f"discharging 1,500 kW in every discharge hour would export {spill.sum()/(PMAX*n):.1%} of the output")
# standard mode reference on the same rows
full = load_hard()
net_std = full.load_kw - full.pv_kw / 12
print(f"whole table: net load below 1,500 kW with x12 in {((full.load_kw-full.pv_kw) < 1500).mean():.1%} of hours; "
      f"standard mode: net load below 500 kW in {(net_std < 500).mean():.2%}")

apply()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.22})
bins = np.arange(0, 2001, 50)
order = [("load", C[0], "Load-limited: discharge = net load"),
         ("partial", GREY, "Partial: the battery runs out"),
         ("inverter", C[1], "Inverter-limited: 1,500 kW")]
ax.hist([dis.net_kw[dis.kind == k].clip(0, 1999) for k, _, _ in order], bins=bins, stacked=True,
        color=[c for _, c, _ in order], label=[l for _, _, l in order], linewidth=0)
ax.axvline(PMAX, color=INK["primary"], linewidth=1.1, linestyle=(0, (4, 3)))
ax.text(PMAX - 20, ax.get_ylim()[1] * 0.80, "inverter limit\n1,500 kW", ha="right", va="top", fontsize=9, color=INK["primary"])
ax.set_xlabel("Campus net load in the hour (kW)")
ax.set_ylabel("Discharge hours")
ax.set_xlim(0, 2000)
ax.legend(loc="upper left", fontsize=8.5, handlelength=1.2)
ax.set_title("Net load in the hours the best schedule discharges", fontsize=10, color=INK["secondary"])

for k, c, _ in order:
    s = dis[dis.kind == k]
    ax2.scatter(s.net_kw, s.discharge, s=9, color=c, linewidths=0, alpha=0.75, zorder=3 if k != "load" else 2)
ax2.plot([0, 2000], [0, 2000], color=INK["muted"], linewidth=0.8, zorder=1)
ax2.axhline(PMAX, color=INK["primary"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
ax2.text(60, PMAX + 25, "inverter limit", fontsize=9, color=INK["primary"], va="bottom")
ax2.text(300, 380, "discharge = net load", fontsize=9, color=INK["secondary"], rotation=45, rotation_mode="anchor",
         transform_rotates_text=True)
ax2.set_xlim(0, 2000)
ax2.set_ylim(0, 1700)
ax2.set_xlabel("Campus net load in the hour (kW)")
ax2.set_ylabel("Battery discharge (kW)")
ax2.set_title("Discharge power against net load, same hours", fontsize=10, color=INK["secondary"])

pl = (dis.kind == "load").mean()
pi = (dis.kind == "inverter").mean()
fig.suptitle(f"The load, not the inverter, sets the discharge: in {pl:.0%} of discharge hours the battery covers the whole campus "
             f"and stops;\nit reaches 1,500 kW in {pi:.1%}", x=0.07, y=1.04, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.07, -0.03, f"Train rows: folds A, B and C (5,417 hours), perfect-foresight schedule of each fold; {n:,} discharge hours pooled.",
         fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, FIGS / "f3_discharge_limit.png")
print("saved")
