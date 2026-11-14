# Usage: .venv/Scripts/python.exe experiments/21_load/figs.py [tag] [variant]
"""Figures for reports/11_load_forecast/figs/ from offline_<tag>_errors.csv, closed_loop.csv and diag_*_by_hour.csv."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plotstyle import C, GREY, INK, apply, save  # noqa: E402
from src.hard import get_folds  # noqa: E402

here = Path(__file__).parent
tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
var = sys.argv[2] if len(sys.argv) > 2 else "v2"
QQ = sys.argv[3] if len(sys.argv) > 3 else "0.25"
OUT = ROOT / "reports/11_load_forecast/figs"
apply()
e = pd.read_csv(here / f"offline_{tag}_errors.csv")
e["ae"] = e["err"].abs()

# 1. horizon-0 MAE by hour of day ---------------------------------------------------------
x = e[e.fold.isin(["B", "C"]) & (e.k == 0)].pivot_table(index="hour", columns="model", values="ae")
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.plot(x.index, x["persistence"], color=GREY, label="Last reading (persistence)")
ax.plot(x.index, x["ProfileN"], color=C[1], label="Baseline profile")
ax.plot(x.index, x[var], color=C[0], label="Load model")
ax.legend(loc="upper right")
ax.set_xticks(range(0, 24, 2)); ax.set_xlim(-0.3, 23.3); ax.set_ylim(0, None)
ax.set_xlabel("Hour being decided"); ax.set_ylabel("Mean absolute error of the load forecast (kW)")
m = e[e.fold.isin(["B", "C"]) & (e.k == 0)].groupby("model")["ae"].mean()
ax.set_title(f"Load error for the hour being decided: {m['ProfileN']:.0f} kW with the baseline, {m[var]:.0f} kW with the model")
fig.text(0.01, -0.02, "Folds B and C (days 8-21 of each month), model fitted on the other train days. The meter is not read from 22:00 to 06:00, "
         "so hours 23-05 repeat the 22:00 value:\nthe model copies it (error 0); 06:00, when the true reading returns, stays the hardest hour.",
         fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, OUT / "h0_mae_by_hour.png")

# 2. MAE by horizon -------------------------------------------------------------------------
x = e[e.fold.isin(["B", "C"]) & (e.model != "persistence")].pivot_table(index="k", columns="model", values="ae")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
pos = np.arange(len(x))
ax.plot(pos, x["ProfileN"], color=C[1], marker="o"); ax.plot(pos, x[var], color=C[0], marker="o")
ax.annotate("Baseline profile", (pos[-1], x["ProfileN"].iloc[-1]), xytext=(-4, 9), textcoords="offset points", color=C[1], ha="right", fontsize=9)
ax.annotate("Load model", (pos[-1], x[var].iloc[-1]), xytext=(-4, -14), textcoords="offset points", color=C[0], ha="right", fontsize=9)
ax.set_xticks(pos); ax.set_xticklabels([f"+{k} h" if k else "now" for k in x.index]); ax.set_ylim(0, None)
ax.set_xlabel("Forecast horizon"); ax.set_ylabel("Mean absolute error (kW)")
ax.set_title("The model roughly halves the load error at every horizon")
save(fig, OUT / "mae_by_horizon.png")

# 3. a holiday week: 1-7 January (fold A) -------------------------------------------------
df = get_folds()["A"]
a = e[(e.fold == "A")].copy(); a["ts"] = df.index[a["t"].to_numpy()]
a["target"] = a["ts"] + pd.to_timedelta(a["k"], unit="h"); a["pred"] = a["y"] + a["err"]
w = a[(a.target >= "2026-01-01") & (a.target < "2026-01-08")]
fig, ax = plt.subplots(figsize=(9.2, 4.2))
act = w[(w.model == var) & (w.k == 0)].set_index("target")["y"]
ax.plot(act.index, act.values, color=INK["primary"], lw=1.6, label="Measured load")
ax.plot(*zip(*w[(w.model == "ProfileN") & (w.k == 0)][["target", "pred"]].values), color=C[1], lw=1.4, label="Baseline profile, hour being decided")
ax.plot(*zip(*w[(w.model == var) & (w.k == 0)][["target", "pred"]].values), color=C[0], lw=1.4, label="Load model, hour being decided")
ax.plot(*zip(*w[(w.model == var) & (w.k == 12)][["target", "pred"]].values), color=C[0], lw=1.2, ls=(0, (1, 1.5)), label="Load model, 12 h ahead")
ax.legend(loc="upper left", ncol=2); ax.set_ylabel("Campus load (kW)"); ax.set_ylim(0, None)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b"))
h = w[(w.k == 0)].groupby("model")["err"].apply(lambda s: s.abs().mean())
ax.set_title(f"Christmas closure, 1-7 January: the model follows the low week ({h[var]:.0f} kW error against {h['ProfileN']:.0f} kW)")
fig.text(0.01, -0.02, "Fold A (cold start: the days before 1 January are hidden test days). Weekdays 1, 2, 5 and 6 January run at weekend level; "
         "the campus is back on the 7th.", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, OUT / "holiday_week_jan.png")

# 4. where discharge is still thrown away, by hour ----------------------------------------
pb, pv2 = here / "diag_ProfileN_by_hour.csv", here / f"diag_{var}_by_hour.csv"
if pb.exists() and pv2.exists():
    b = pd.read_csv(pb, index_col=0); v = pd.read_csv(pv2, index_col=0)
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(b.index - 0.2, b["dspill_pts"], width=0.4, color=GREY, label="Baseline profile")
    ax.bar(v.index + 0.2, v["dspill_pts"], width=0.4, color=C[0], label="Load model")
    ax.legend(loc="upper right"); ax.set_xticks(range(0, 24, 2)); ax.set_xlim(-0.7, 23.7)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Score points of discharge sent to the grid for nothing")
    ax.set_title(f"Discharge thrown away: {b['dspill_pts'].sum():.2f} points with the baseline, {v['dspill_pts'].sum():.2f} with the model")
    fig.text(0.01, -0.02, "Mean of folds A, B, C; perfect price/CO2 signal and perfect solar, so load is the only unknown. "
             "Energy discharged above the true net load, valued at the score weight of its hour.", fontsize=8.5, color=INK["secondary"], ha="left", va="top")
    save(fig, OUT / "spill_by_hour.png")

# 5. closed-loop ladder ---------------------------------------------------------------------
cl = pd.read_csv(here / "closed_loop.csv").drop_duplicates(["variant", "config", "fold"], keep="last")
ref = pd.read_csv(ROOT / "experiments/20_hard_baseline/decompose.csv")
rm = ref.groupby("config")["score"].mean()
m = cl.groupby(["config", "variant"])["score"].mean()
rows = [  # label, baseline, model, model with cautious hour 0, perfect load
    ("Legal agent (own price/CO2\nforecast, profile solar)", max(rm["clim w   + profile n, H=36 (legal)"], rm["clim w   + profile n, H=36, shade 0.9 (legal)"]),
     max(m.get(("legal_s0.95", var), np.nan), m.get(("legal_s1.0", var), np.nan)), m.get(("legal_s1.0", var + "_q" + QQ), np.nan), np.nan),
    ("Perfect price/CO2,\nprofile solar", rm["perfect w + profile n, H=36"], m.get(("perfW_profPV", var), np.nan), m.get(("perfW_profPV", var + "_q" + QQ), np.nan), np.nan),
    ("Perfect price/CO2,\nperfect solar", m.get(("perfW_perfPV", "ProfileN"), np.nan), m.get(("perfW_perfPV", var), np.nan), m.get(("perfW_perfPV", var + "_q" + QQ), np.nan), m.get(("perfW_perfPV", "perfect"), np.nan)),
]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
for yi, (lab, b, v, q, pf) in zip(range(len(rows))[::-1], rows):
    ax.plot([b, np.nanmax([v, q, pf])], [yi, yi], color=INK["axis"], lw=1.2, zorder=1)
    ax.scatter([b], [yi], s=60, color=GREY, zorder=3, label="Baseline load profile" if yi == 0 else None)
    ax.scatter([v], [yi], s=60, color=C[0], zorder=3, label="Load model" if yi == 0 else None)
    ax.scatter([q], [yi], s=60, color=C[2], zorder=3, label=f"Load model, cautious hour 0 ({float(QQ) * 100:.0f} % quantile)" if yi == 0 else None)
    if pf == pf:
        ax.scatter([pf], [yi], s=60, color=INK["primary"], zorder=3, label="Perfect load")
        ax.annotate(f"{pf:.2f}", (pf, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{b:.2f}", (b, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK["secondary"])
    ax.annotate(f"{q:.2f}", (q, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK["primary"])
ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("Score (mean of folds A, B, C)"); ax.grid(axis="y", visible=False)
ax.legend(loc="upper right", fontsize=8.5)
ax.set_title(f"The load model adds about {rows[0][3] - rows[0][1]:.1f} points in every setting")
fig.text(0.01, -0.03, "Legal-agent baseline: the better of discharge shade 1.0 and 0.9 with the baseline profile. Same planner (MPC, 36 h) in all runs.",
         fontsize=8.5, color=INK["secondary"], ha="left", va="top")
save(fig, OUT / "closed_loop_ladder.png")
print(pd.DataFrame(rows, columns=["setting", "baseline", "model", "model_q35", "perfect"]).round(3).to_string())
print("saved figures to", OUT)
