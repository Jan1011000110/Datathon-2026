# Usage: .venv/Scripts/python.exe experiments/22_solar/figs.py
"""Figures for reports/12_solar_forecast. Needs offline_final.csv, offline_noweather.csv, closed.csv, traces/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.plotstyle import C, GREY, INK, apply, save  # noqa: E402
from src.hard import load_hard  # noqa: E402
import solar_fc as S  # noqa: E402

HERE = Path(__file__).parent
OUT = ROOT / "reports/12_solar_forecast/figs"
apply()

# ---- 1. MAE by horizon -------------------------------------------------------------------------
o = pd.read_csv(HERE / "offline_final.csv", parse_dates=["ts"])
nw = pd.read_csv(HERE / "offline_noweather.csv", parse_dates=["ts"])
day = lambda d: d[(d.hour_t >= 8) & (d.hour_t <= 19)]   # noqa: E731
od, nd = day(o), day(nw)
mae = pd.DataFrame({"base": (od.base - od.y).abs().groupby(od.h).mean(),
                    "nw": (nd.fc - nd.y).abs().groupby(nd.h).mean(),
                    "fc": (od.fc - od.y).abs().groupby(od.h).mean()})
print(mae.round(1).to_string())
fig, ax = plt.subplots(figsize=(8, 4.4))
x = np.arange(len(mae))
ax.plot(x, mae.base, color=GREY, marker="o")
ax.plot(x, mae.nw, color=C[1], marker="o")
ax.plot(x, mae.fc, color=C[0], marker="o")
ax.text(x[-1] + 0.12, mae.base.iloc[-1], "Month-hour profile\n(baseline)", color=INK["secondary"], va="center", fontsize=9)
ax.text(x[-1] + 0.12, mae.fc.iloc[-1] - 9, "Envelope + clearness\n+ panel temperature", color=C[0], va="center", fontsize=9)
ax.text(x[3] - 0.1, mae.nw.iloc[3] + 5, "Envelope + clearness", color=C[1], va="bottom", ha="right", fontsize=9)
for i in (0, 3, 4):
    ax.text(x[i], mae.fc.iloc[i] - 7, f"{100 * (mae.fc.iloc[i] / mae.base.iloc[i] - 1):+.0f}%", color=C[0], ha="center", va="top", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"+{h} h" if h else "hour being\ndecided" for h in mae.index])
ax.set_xlim(-0.4, len(mae) + 0.7)
ax.set_ylim(60, 205)
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("Mean absolute error, kW (targets 08-19 h, folds B+C)")
ax.set_title("Solar forecast: 7% less error for the hour being decided, 12-14% less at 3-6 hours")
save(fig, OUT / "mae_by_horizon.png")

# ---- 2. night-time panel temperature vs next-day clearness ----------------------------------------
df = load_hard()
fc = S.SolarFC(df, fit_quantiles=False)
keys = df.index.as_unit("ns").asi8 // S.HNS
env = np.array([fc._env(k) for k in keys.tolist()])
d = pd.DataFrame({"pv": df.pv_kw.to_numpy(), "E": env, "dT": (df.temp_panel_c - df.temp_ambient_c).to_numpy(),
                  "date": df.index.normalize(), "hour": df.index.hour})
sun = d[d.E >= S.E_MIN].groupby("date").agg(pv=("pv", "sum"), E=("E", "sum"), n=("pv", "size"))
night = d[(d.hour >= 2) & (d.hour <= 5)].groupby("date").dT.mean()
j = pd.DataFrame({"k": (sun.pv / sun.E)[sun.n >= 6], "dT": night}).dropna()
r = np.corrcoef(j.dT, j.k)[0, 1]
print("days", len(j), "corr night dT vs day clearness", round(r, 3))
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.scatter(j.dT, j.k, s=16, color=C[0], alpha=0.7, linewidths=0)
bins = pd.cut(j.dT, [-3, -1.6, -1.2, -0.8, -0.4, 0.5])
med = j.groupby(bins, observed=True).k.median()
ax.plot([iv.mid for iv in med.index], med.to_numpy(), color=INK["primary"], marker="o", linewidth=1.5)
ax.text(med.index[-1].mid + 0.03, med.iloc[-1] - 0.06, "median", color=INK["primary"], fontsize=9)
ax.set_xlabel("Panel minus ambient temperature, mean of 02-05 h (°C)")
ax.set_ylabel("Clearness of the following day (output / clear-sky envelope)")
ax.set_title(f"A panel colder than the air before dawn announces a clear day (r = {r:.2f}, {len(j)} days)")
save(fig, OUT / "night_panel_temperature.png")

# ---- 3. example week: hour-ahead forecast with band ---------------------------------------------
h0 = o[o.h == 0].set_index("ts")
h0["date"] = h0.index.normalize()
h0["wk"] = h0.fold + h0.index.strftime("-%Y-%m")
dk = h0[h0.y > 50].groupby(["wk", "date"]).apply(lambda g: g.y.sum() / max(g.q90.sum(), 1), include_groups=False)
wk = dk.groupby("wk").std().idxmax()
g = h0[h0.wk == wk]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
cov = float(((h0.y >= h0.q10) & (h0.y <= h0.q90))[h0.q90 - h0.q10 > 1.0].mean())
ax.fill_between(g.index, g.q10, g.q90, color=C[0], alpha=0.22, linewidth=0,
                label=f"10-90% band ({100 * cov:.0f}% of sunlit hours inside, folds B+C)")
ax.plot(g.index, g.fc, color=C[0], linewidth=1.6, label="Forecast made one hour before")
ax.plot(g.index, g.y, color=INK["primary"], linewidth=1.2, label="Actual")
ax.plot(g.index, g.base, color=GREY, linewidth=1.2, label="Profile baseline", zorder=1)
ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))
ax.set_ylim(0, max(g.y.max(), g.q90.max()) * 1.18)
ax.set_ylabel("Solar output, kW (x12 plant)")
ax.set_xlabel(f"Fold {wk[0]}, week of {g.index[0]:%d %b %Y}")
ax.set_title("Hour-ahead solar forecast: the band is narrow on clear days and wide under broken cloud")
save(fig, OUT / "example_week.png")

# ---- 4. where the remaining solar error costs score ----------------------------------------------
cl = pd.read_csv(HERE / "closed.csv")
names = {0: "perfect", 1: "fc", 3: "base"}
loss = {}
for f in "ABC":
    tr = {v: pd.read_csv(HERE / "traces" / f"{i}_{f}.csv", index_col=0, parse_dates=True) for i, v in names.items()}
    for v in ("fc", "base"):
        pts = 100 * tr[v].w * (tr[v].grid_kw - tr["perfect"].grid_kw)
        loss[(v, f)] = pts.groupby(pts.index.hour).sum()
L = pd.DataFrame(loss).T.groupby(level=0).mean().T
print(L.round(3).T.to_string()); print("total loss vs perfect solar:", L.sum().round(3).to_dict())
fig, ax = plt.subplots(figsize=(8.6, 4.2))
wd = 0.4
ax.bar(L.index - wd / 2, L.base, width=wd, color=GREY, label=f"Profile baseline (total {L.base.sum():.2f} points)")
ax.bar(L.index + wd / 2, L.fc, width=wd, color=C[0], label=f"SolarFC (total {L.fc.sum():.2f} points)")
ax.axhline(0, color=INK["axis"], linewidth=0.8)
ax.legend(loc="upper left")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Score points lost against perfect solar knowledge\n(perfect load and price/CO2; mean of folds A, B, C)")
ax.set_title(f"Solar error costs score between 07 and 13 h; the new forecast recovers {L.base.sum() - L.fc.sum():.2f} of {L.base.sum():.2f} points")
save(fig, OUT / "loss_by_hour.png")
print("saved")
