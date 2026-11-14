# Usage: .venv/Scripts/python.exe experiments/27_combined/make_figs.py
"""Figures and the hour-of-day split of the remaining gap, from results_*.csv, calib_h0.csv and the traces
saved by `sweep.py gap --trace`. Writes reports/18_combined/figs/*.png and gap_by_hour.csv, gap_by_block.csv."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined as cb  # noqa: E402
from src.hard import get_folds, score_weights  # noqa: E402
from src.plotstyle import apply, C, GREY, INK, save  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = cb.ROOT / "reports" / "18_combined" / "figs"
apply()


def mean_scores(name):
    d = pd.read_csv(HERE / f"results_{name}.csv")
    assert d["infractions"].sum() == 0
    return d.groupby("config", sort=False)["score"].mean()


# ------------------------------------------------------------------ 1. ladder
m = mean_scores("ladder2")
combos = [("naive", "naive", "naive load\nnaive solar"), ("fc", "naive", "LoadFC\nnaive solar"),
          ("naive", "fc", "naive load\nSolarFC"), ("fc", "fc", "LoadFC\nSolarFC")]
sigs = [("clim", "naive signal (ClimW)", GREY), ("fc", "SignalForecaster", C[0]), ("perfect", "perfect signal (not legal)", C[1])]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
x = np.arange(len(combos))
for j, (sk, sl, col) in enumerate(sigs):
    y = [m[f"{sk} | load {a} + solar {b} | conv"] for a, b, _ in combos]
    ax.plot(x, y, marker="o", ms=9, ls="none", color=col, label=sl)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK["secondary"])
    ax.annotate(sl, (x[-1], y[-1]), textcoords="offset points", xytext=(12, -3), fontsize=9, color=col if col != GREY else INK["secondary"])
ax.axhline(21.26, color=INK["muted"], lw=1, ls=(0, (4, 3)))
ax.annotate("ceiling 21.26 (everything known)", (0, 21.26), textcoords="offset points", xytext=(0, 5), fontsize=9, color=INK["secondary"])
ax.set_xticks(x, [c[2] for c in combos])
ax.set_xlim(-0.3, len(combos) - 1 + 1.25)
ax.set_ylim(19.3, 21.5)
ax.set_ylabel("score, mean of folds A, B, C")
ax.set_title("Load forecaster +0.19 to +0.20, solar forecaster +0.05 to +0.06, signal forecaster +0.21 to +0.24:\nthe three gains add up")
save(fig, FIGS / "ladder.png")

# ------------------------------------------------------------------ 2. calibration
cal = pd.read_csv(HERE / "calib_h0.csv")
cal = cal[(cal.load == "fc") & (cal.solar == "fc") & ~cal.block.str.startswith("23-05")]
qs = [10, 25, 50, 75, 90]
blocks = list(dict.fromkeys(cal.block))
fig, axes = plt.subplots(1, len(blocks), figsize=(11, 3.4), sharey=True)
for ax, b in zip(axes, blocks):
    ax.plot([0, 100], [0, 100], color=INK["muted"], lw=1, ls=(0, (4, 3)))
    for meth, col, lab in (("emp", GREY, "errors of the difference"), ("conv", C[0], "LoadFC x SolarFC quantiles")):
        d = cal[(cal.block == b) & (cal.method == meth)]
        y = [100 * d[f"below_{q}"].mean() for q in qs]
        ax.plot(qs, y, marker="o", color=col, label=lab)
    ax.set_title(f"{b} h", fontsize=10, fontweight="normal")
    ax.set_xticks(qs)
    ax.set_yticks(qs)
    ax.set_xlabel("scenario quantile (%)")
axes[0].set_ylabel("actual net load below it (% of hours)")
axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle("Both scenario sets are calibrated for the hour being decided (dashed = perfect), folds A, B, C",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, FIGS / "calibration.png")

# ------------------------------------------------------------------ 3. remaining gap by hour of day
res = pd.read_csv(HERE / "results_gap.csv")
idx = res.drop_duplicates("i").set_index("config")["i"]
I_CEIL = idx["ceiling: perfect signal + perfect net load, deterministic"]
I_LEGAL = idx["fc | best net-load side, horizon cut at the true end of the run"]
I_PW = idx["perfect | best net-load side (LoadFC + SolarFC, conv)"]
I_H0 = idx["fc | + true net load for hours 0..0, horizon cut at the true end of the run"]
folds = get_folds()
rows = []
for f in "ABC":
    df = folds[f]
    w = score_weights(df) * 100.0
    tr = {k: pd.read_pickle(cb.CACHE / f"trace_gap_{i}_{f}.pkl") for k, i in
          (("ceil", I_CEIL), ("legal", I_LEGAL), ("pw", I_PW), ("h0", I_H0))}
    d = pd.DataFrame({"fold": f, "hour": df.index.hour, "month": df.index.strftime("%Y-%m"),
                      "daytype": np.where(df.index.day_of_week >= 5, "weekend", "weekday"),
                      "ceiling_gain": w * (np.maximum(df["load_kw"] - df["pv_kw"], 0.0).to_numpy() - tr["ceil"]["grid_kw"].to_numpy()),
                      "loss_legal": w * (tr["legal"]["grid_kw"] - tr["ceil"]["grid_kw"]).to_numpy(),
                      "loss_perfect_signal": w * (tr["pw"]["grid_kw"] - tr["ceil"]["grid_kw"]).to_numpy(),
                      "loss_legal_h0_known": w * (tr["h0"]["grid_kw"] - tr["ceil"]["grid_kw"]).to_numpy()})
    for k in ("legal", "pw", "ceil"):
        t = tr[k]
        p, net = t["p_kw"].to_numpy(), t["net_kw"].to_numpy()
        d[f"overdischarge_kwh_{k}"] = np.maximum(-p - np.maximum(net, 0.0), 0.0) * (p < 0)     # discharged into nothing
        d[f"discharge_kwh_{k}"] = np.maximum(-p, 0.0)
        d[f"charge_grid_kwh_{k}"] = np.maximum(np.minimum(p, net + p), 0.0)                   # charged from the grid
        d[f"spill_kwh_{k}"] = np.maximum(-(net + p), 0.0)
    rows.append(d)
D = pd.concat(rows)
byh = D.drop(columns=["month", "daytype"]).groupby(["fold", "hour"]).sum(numeric_only=True).groupby("hour").mean()
for key in ("month", "daytype"):
    t = D.groupby(["fold", key])[["ceiling_gain", "loss_legal", "loss_perfect_signal", "loss_legal_h0_known"]].sum().groupby(key).mean()
    t.to_csv(HERE / f"gap_by_{key}.csv")
    print(t.round(3).to_string())
byh.to_csv(HERE / "gap_by_hour.csv")
blk = {"23-05": (23, 0, 1, 2, 3, 4, 5), "06-08": (6, 7, 8), "09-12": (9, 10, 11, 12), "13-16": (13, 14, 15, 16),
       "17-22": (17, 18, 19, 20, 21, 22)}
byb = pd.DataFrame({b: byh.loc[list(h)].sum() for b, h in blk.items()}).T
byb.loc["all"] = byh.sum()
byb.to_csv(HERE / "gap_by_block.csv")
pd.set_option("display.width", 250)
print(byb.round(3).T.to_string())

fig, ax = plt.subplots(figsize=(9, 4.2))
h = np.arange(24)
ax.bar(h - 0.2, byh["loss_legal"], width=0.4, color=C[0],
       label=f"best legal agent, end of run known (total {byh['loss_legal'].sum():.2f} points)")
ax.bar(h + 0.2, byh["loss_perfect_signal"], width=0.4, color=C[1],
       label=f"same with a perfect signal (total {byh['loss_perfect_signal'].sum():.2f})")
ax.axhline(0, color=INK["axis"], lw=0.8)
ax.set_xticks(h)
ax.set_xlabel("hour of day")
ax.set_ylabel("extra grid cost against the ceiling (score points)")
ax.legend(loc="upper right")
ax.set_title("Against the ceiling the agent buys more from 05:00 to 13:00 (it holds discharge back while the\nnet load is uncertain) and gets part of it back at night; a perfect signal does not change that")
save(fig, FIGS / "gap_by_hour.png")

# ------------------------------------------------------------------ 4. what each piece of knowledge would add
g = mean_scores("gap")
items = [("best legal agent", g["fc | best net-load side (LoadFC + SolarFC, conv)"], C[0]),
         ("+ net load of this hour known", g["fc | + true net load for hours 0..0"], GREY),
         ("+ net load of hours 0-2 known", g["fc | + true net load for hours 0..2"], GREY),
         ("+ net load of hours 0-5 known", g["fc | + true net load for hours 0..5"], GREY),
         ("+ net load of all hours known", g["fc | + true net load for hours 0..35"], GREY),
         ("+ only the load known (all hours)", g["fc | true load (all hours) + SolarFC"], GREY),
         ("+ only the solar known (all hours)", g["fc | LoadFC + true solar (all hours)"], GREY),
         ("+ signal of hours 3-5 known", g["fc | best net-load side + true signal for hours 0..5"], GREY),
         ("+ signal of hours 3-11 known", g["fc | best net-load side + true signal for hours 0..11"], GREY),
         ("+ end of the run known", g["fc | best net-load side, horizon cut at the true end of the run"], GREY),
         ("+ signal of all hours and end of run known", g["perfect | best net-load side (LoadFC + SolarFC, conv)"], GREY),
         ("everything known (ceiling)", g["ceiling: perfect signal + perfect net load, deterministic"], C[1])]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
y = np.arange(len(items))[::-1]
base = items[0][1]
ax.barh(y, [v for _, v, _ in items], color=[c for _, _, c in items], height=0.62)
for yi, (lab, v, _) in zip(y, items):
    ax.annotate(f"{v:.2f}" + ("" if v == base else f"  (+{v - base:.2f})"), (v, yi), textcoords="offset points", xytext=(5, -3), fontsize=9,
                color=INK["secondary"])
ax.set_yticks(y, [i[0] for i in items])
ax.set_xlim(19.5, 21.75)
ax.set_xlabel("score, mean of folds A, B, C (each bar: the legal agent plus one piece of perfect knowledge)")
ax.grid(axis="y", visible=False)
ax.set_title("What is still missing: the signal 3 to 11 hours ahead (+0.55), the net load\nof the hour being decided (+0.30) and the end of the run (+0.18)")
save(fig, FIGS / "knowledge.png")

# ------------------------------------------------------------------ 5. sensitivity
s = mean_scores("sens")
b0 = s["BEST: fc signal, LoadFC + SolarFC, conv, Q=9, K=12, H=36"]
panels = [("scenarios per hour (Q)", [("5", s["Q=5"]), ("9", b0), ("15", s["Q=15"]), ("25", s["Q=25"])]),
          ("hours with scenarios (K)", [("1", s["scenario hours K=1"]), ("3", s["scenario hours K=3"]), ("6", s["scenario hours K=6"]),
                                        ("12", b0), ("24", s["scenario hours K=24"]), ("36", s["scenario hours K=all (36)"])]),
          ("width multiplier", [(f"{v:g}", b0 if v == 1 else s[f"width x{v:.2f}"]) for v in (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)]),
          ("horizon H (a) and terminal value, share of mean signal (b: H=36, c: H=24)",
           [("a 24", s["H=24"]), ("a 36", b0), ("b 0.5", s["v_end = 0.50 x mean(w) x eta, H=36"]),
            ("b 1", s["v_end = 1.00 x mean(w) x eta, H=36"]), ("c 0.5", s["v_end = 0.50 x mean(w) x eta, H=24"]),
            ("c 1", s["v_end = 1.00 x mean(w) x eta, H=24"])])]
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True, gridspec_kw={"width_ratios": [4, 6, 7, 7]})
for ax, (title, pts) in zip(axes, panels):
    xs = np.arange(len(pts))
    vals = [v for _, v in pts]
    ax.plot(xs, vals, marker="o", color=C[0])
    for xi, (lab, v) in zip(xs, pts):
        if v == b0:
            ax.plot([xi], [v], marker="o", color=C[1], ms=7)
    ax.set_xticks(xs, [l for l, _ in pts], fontsize=8)
    ax.set_xlim(-0.5, len(pts) - 0.5)
    ax.set_title(title, fontsize=10, fontweight="normal")
axes[0].set_ylabel("score, mean of folds A, B, C")
axes[0].set_ylim(19.85, 20.05)
fig.suptitle("Every setting of the planner is on a plateau: only one scenario hour or a width off by 50 % costs more than 0.03"
             "  (orange = setting used)", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, FIGS / "sensitivity.png")
print("figures written to", FIGS)
