"""Waterfall table + figures for reports/03_gap.

Reads out/decomp_scores.csv (01_decompose.py) and out/rho_sweep.csv (06_rho.py); also re-runs
the chain on 'patched' folds where the two meter anomalies are replaced by the load of the
same hours one day earlier (the clean-data gap).
Writes reports/03_gap/figs/gap_waterfall.png, rho_loss.png and out/waterfall.csv, out/clean.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import *

OUT = Path(__file__).parent / "out"
REP = ROOT / "reports" / "03_gap" / "figs"
REP.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)

EVENTS = [("2026-06-17 17:00", "2026-06-18 06:00"),    # 14 h meter outage (fold C)
          ("2026-08-06 07:00", "2026-08-06 09:00")]    # 2 h meter dropout across 3 clock hours (fold A)

# ---- 1. waterfall on the real folds -----------------------------------------------------------
sc = pd.read_csv(OUT / "decomp_scores.csv", index_col=0)
chain = [("RHO 0.60 instead of the fold's true value", "v1", "rho=true"),
         ("Nothing known beyond the 24 h window", "rho=true", "rho=true,h=full"),
         ("Guard's 5% margin", "rho=true,h=full", "rho=true,h=full,margin=0.000"),
         ("This hour's load unknown (last hour's reading used)", "rho=true,h=full,margin=0.000", "rho=true,h=full,guard=true-load"),
         ("Future load unknown (cannot plan around a dropout)", "rho=true,h=full,guard=true-load", "oracle")]
wf = pd.DataFrame({name: (sc.loc[b] - sc.loc[a]) * 1e4 for name, a, b in chain}).T[list(FOLDS)]
wf.loc["TOTAL gap (oracle - v1)"] = (sc.loc["oracle"] - sc.loc["v1"]) * 1e4
wf.to_csv(OUT / "waterfall.csv")
print("waterfall, 1e-4 points (each row = one more piece of knowledge given to v1):")
print(wf.round(2).to_string())
single = pd.DataFrame({"true RHO only": sc.loc["rho=true"] - sc.loc["v1"], "whole future of w only": sc.loc["h=full"] - sc.loc["v1"],
                       "no guard margin only": sc.loc["margin=0.000"] - sc.loc["v1"], "this hour's true load only": sc.loc["guard=true-load"] - sc.loc["v1"],
                       "no guard at all": sc.loc["guard=off"] - sc.loc["v1"], "no outage heuristic": sc.loc["outage=off"] - sc.loc["v1"]}).T[list(FOLDS)] * 1e4
print("\none change at a time on top of v1 (gain, 1e-4 points):")
print(single.round(2).to_string())
print("\nhorizon cheats on top of v1 (gain, 1e-4 points):")
print(((sc.loc[["h=+%d" % h for h in (1, 2, 3, 6, 12, 24, 48, 96, 168)] + ["h=full"]] - sc.loc["v1"]) * 1e4)[list(FOLDS)].round(2).to_string())
print("\nguard margin on top of v1 (gain, 1e-4 points):")
print(((sc.loc[[i for i in sc.index if i.startswith("margin=")]] - sc.loc["v1"]) * 1e4)[list(FOLDS)].round(2).to_string())


# ---- 2. clean-data gap: same folds with the two meter anomalies patched ----------------------
def patch(df):
    d = df.copy()
    for lo, hi in EVENTS:
        idx = d.index[(d.index >= lo) & (d.index <= hi)]
        for ts in idx:
            d.loc[ts, "load_kw"] = d.loc[ts - pd.Timedelta(hours=24), "load_kw"]
    return d


folds = get_folds("filled")
rows = []
for f in FOLDS:
    d = patch(folds[f])
    rt = true_rho(d)
    s_v1 = simulate(v1.BatteryAgent(), d)["score"]
    s_r = simulate(GapAgent(rho=rt), d)["score"]
    s_h = simulate(GapAgent(rho=rt, cheat_h="full", df=d), d)["score"]
    s_g = simulate(GapAgent(rho=rt, cheat_h="full", df=d, cheat_guard=True), d)["score"]
    s_o = oracle(d)["score"]
    rows.append({"fold": f, "hours patched": int((d["load_kw"] != folds[f]["load_kw"]).sum()), "true rho": rt, "v1": s_v1, "oracle": s_o,
                 "gap": (s_o - s_v1) * 1e4, "RHO": (s_r - s_v1) * 1e4, "horizon": (s_h - s_r) * 1e4,
                 "this hour's load": (s_g - s_h) * 1e4, "future load": (s_o - s_g) * 1e4})
clean = pd.DataFrame(rows).set_index("fold")
clean.to_csv(OUT / "clean.csv")
print("\nclean folds (meter anomalies patched), gap and its parts in 1e-4 points:")
print(clean.round(4).to_string())

# ---- 3. figures --------------------------------------------------------------------------------
SURF, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRIDC, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})

short = ["RHO", "Beyond the 24 h window", "Guard's 5% margin", "This hour's load unknown", "Future load unknown"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
labels = ["A  (days 1-7)", "B  (days 8-14)", "C  (days 15-21)", "ALL train days"]
y = np.arange(len(FOLDS))[::-1]
left = np.zeros(len(FOLDS))
for k in range(5):
    vals = wf.iloc[k][list(FOLDS)].to_numpy()
    ax.barh(y, vals, left=left, height=0.42, color=SER[k], edgecolor=SURF, linewidth=1.5, label=short[k])
    left += vals
for yi, tot in zip(y, wf.loc["TOTAL gap (oracle - v1)"][list(FOLDS)]):
    ax.text(tot + 3, yi, "%.1f" % tot, va="center", ha="left", color=INK, fontsize=10)
ax.set_yticks(y); ax.set_yticklabels(labels, color=INK)
ax.set_xlim(0, 215); ax.set_xlabel("gap to the perfect-foresight oracle, in 0.0001 points")
ax.xaxis.grid(True, color=GRIDC, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("Real folds: the gap is two meter anomalies", loc="left", fontsize=11, color=INK)
h, l = ax.get_legend_handles_labels()
l[2] = "Guard's 5% margin (0 everywhere)"
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.6)

ax = axes[1]
left = np.zeros(len(FOLDS))
parts = [("RHO", SER[0]), ("horizon", SER[1]), ("this hour's load", SER[3]), ("future load", SER[4])]
for col, color in parts:
    vals = clean[col].reindex(FOLDS).to_numpy().clip(min=0)
    ax.barh(y, vals, left=left, height=0.42, color=color, edgecolor=SURF, linewidth=1.5)
    left += vals
for yi, tot in zip(y, clean["gap"].reindex(FOLDS)):
    ax.text(tot + 0.2, yi, "%.1f" % tot, va="center", ha="left", color=INK, fontsize=10)
ax.axvline(10, color=INK2, linewidth=1)
ax.text(10.15, y[0] + 0.33, "0.001 points", color=INK2, fontsize=9, va="center")
ax.set_yticks(y); ax.set_yticklabels(["A", "B", "C", "ALL"], color=INK)
ax.set_xlim(0, 12.5); ax.set_xlabel("gap to the oracle, in 0.0001 points")
ax.xaxis.grid(True, color=GRIDC, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("Same folds with the two anomalies patched", loc="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(REP / "gap_waterfall.png", dpi=170)
plt.close(fig)

sw = pd.read_csv(OUT / "rho_sweep.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
for k, f in enumerate(FOLDS):
    ax.plot(sw.index, sw[f], color=SER[k], linewidth=2, solid_capstyle="round", label="fold %s (true RHO %.3f)" % (f, true_rho(folds[f])))
    rt = true_rho(folds[f])
    ax.plot([rt], [0], marker="o", markersize=8, color=SER[k], markeredgecolor=SURF, markeredgewidth=2, clip_on=False, zorder=5)
for x, lab in ((0.60, "v1: 0.60"),):
    ax.axvline(x, color=INK2, linewidth=1)
    ax.text(x + 0.002, 55, lab, color=INK2, fontsize=9, ha="left")
ax.axhline(10, color=INK2, linewidth=1)
ax.text(0.5005, 11.2, "0.001 points", color=INK2, fontsize=9)
ax.set_xlim(0.50, 0.70); ax.set_ylim(0, 60)
ax.set_xlabel("RHO used by the agent (kg CO2 per EUR of baseline import)")
ax.set_ylabel("score lost vs the fold's true RHO, 0.0001 points")
ax.yaxis.grid(True, color=GRIDC, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("A wrong RHO costs about 0.4 x (error)^2 points; dots = each fold's true RHO", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.3, 1.0))
fig.tight_layout()
fig.savefig(REP / "rho_loss.png", dpi=170)
plt.close(fig)
print("\nfigures written to", REP)
