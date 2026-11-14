# Usage: .venv/Scripts/python.exe experiments/30_boundaries/figs.py
"""Figures of the week-boundary study -> reports/23_boundaries/figs/. Also writes q4_all.csv (every h2 candidate:
change of the arrival charge, of the whole run and of both halves under the 84-hour cut)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib as L
from src.plotstyle import C, GREY, INK, apply, save

OUT = L.ROOT / "reports" / "23_boundaries" / "figs"
FCOL = {"A": C[0], "B": C[1], "C": C[2]}


def fig_offsets():
    b = pd.read_csv(L.HERE / "q1_boundaries.csv")
    b = b[(b.agent == "h2") & (~b.is_last)]
    x = (b.soc_end - b.soc_end_opt) / 1000.0
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axhline(0, color=INK["axis"], lw=0.8)
    ax.scatter(x, b.reg_day7, s=42, color=C[0], edgecolor=INK["surface"], lw=1.2, label="last day of the week before", zorder=3)
    ax.scatter(x, b.reg_next_day1, s=42, color=C[1], edgecolor=INK["surface"], lw=1.2, label="first day of the week after", zorder=3)
    ax.scatter(x, b.reg_day7 + b.reg_next_day1, s=30, marker="D", color=INK["primary"], edgecolor=INK["surface"], lw=1.0,
               label="the two days together", zorder=4)
    ax.set_xlabel("h2's charge at the end of the week minus the optimum's (MWh), 30 boundaries of folds A, B, C")
    ax.set_ylabel("regret against the optimum (points)")
    ax.set_title("Where h2 arrives with less charge, it has already gained on the last day\nmost of what it then loses on the first day")
    ax.legend(loc="lower right")
    save(fig, OUT / "carry_offsets.png")


def fig_days():
    r = pd.read_csv(L.HERE / "results_days_h2.csv")
    piv = r.pivot(index="config", columns="fold", values="score")
    gain = piv - piv.loc["base"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    days = np.arange(1, 8)
    for f in L.FOLDS:
        y = [gain.loc[f"true signal + net load for the hours of day {d}", f] for d in days]
        ax.plot(days, y, color=FCOL[f], marker="o", ms=6, label=f"fold {f}")
        ax.annotate(f"fold {f}", (7, y[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK["secondary"])
    ax.set_xlim(0.7, 7.9)
    ax.set_ylim(0, None)
    ax.set_xticks(days)
    ax.set_xlabel("day of the week whose hours h2 is told exactly (signal and net load)")
    ax.set_ylabel("gain on the whole run (points)")
    ax.set_title("The first day of a week is not harder than the others on B and C;\nonly the cold-start fold A stands out")
    ax.legend(loc="lower left", ncol=3)
    save(fig, OUT / "day_truth.png")


def fig_rules():
    res = pd.concat([pd.read_csv(L.HERE / f"results_{s}_h2.csv") for s in ("diag", "net", "sig", "value")]).drop_duplicates(["config", "fold"])
    piv = res.pivot(index="config", columns="fold", values="score")
    gain = piv - piv.loc["base"]
    rows = [("true signal + net beyond the jump", "CEILING: true signal and net load beyond the jump"),
            ("day-profile net beyond the jump", "net load beyond the jump for the right calendar day"),
            ("signal beyond the jump x1.06", "signal beyond the jump x1.06"),
            ("signal beyond the jump x1.15", "signal beyond the jump x1.15"),
            ("signal beyond the jump: 0.5 on the stored day before the block", "signal: half weight on the stored day before the block"),
            ("value at the week end: 0.5 on the train value of the month", "value at the week end: half train value of the month"),
            ("value at the week end: 1.0 on the train value of the month", "value at the week end: train value of the month"),
            ("floor: the first 3000 kWh worth at least 1.0 x train value", "floor: first 3,000 kWh worth at least the train value"),
            ("floor: the first 4500 kWh worth at least 1.2 x train value", "floor: first 4,500 kWh worth 1.2 x the train value"),
            ("profile net (3, 7), scenario width beyond the jump x2.0", "wider net-load scenarios beyond the jump (x2)")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(rows))[::-1]
    m = [gain.loc[k].mean() for k, _ in rows]
    ax.barh(y, m, height=0.55, color=[INK["muted"]] + [GREY] * (len(rows) - 1), zorder=2)
    for f, mk in zip(L.FOLDS, ("o", "s", "^")):
        ax.scatter([gain.loc[k, f] for k, _ in rows], y, s=34, marker=mk, color=FCOL[f], edgecolor=INK["surface"], lw=1.0, zorder=3,
                   label=f"fold {f}")
    ax.axvline(0, color=INK["axis"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows])
    ax.set_xlabel("change of the whole-run score against h2 (points); bar = mean of the three folds")
    ax.set_title("No boundary rule gains on the whole run; even the truth beyond the jump\nis worth 0.05 on average and 0.01 on folds B and C")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, OUT / "rules_whole_run.png")


def fig_transfer():
    from q3 import halves
    folds = L.get_folds()
    rows = []
    grids = {}
    for s in ("diag", "net", "sig", "value"):
        z = np.load(L.CACHE / f"grid_{s}_h2.npz")
        for k in z.files:
            grids.setdefault(k, z[k])
    configs = sorted({k.rsplit("|", 1)[0] for k in grids if not k.startswith("soc|")} - {"base"})
    base_h = {f: halves(folds[f], f, grids[f"base|{f}"]) for f in L.FOLDS}
    for c in configs:
        acc = []
        for f in L.FOLDS:
            df = folds[f]
            _, hs, he = L.positions(df, f)
            ends = [e for e in np.flatnonzero(he == 0) if e + 1 < len(df)]
            h = halves(df, f, grids[f"{c}|{f}"])
            w = L.score_weights(df) * 100.0
            acc.append(dict(arrival=float(np.mean(grids[f"soc|{c}|{f}"][ends] - grids[f"soc|base|{f}"][ends])) / 1000.0,
                            whole=float((w * (grids[f"base|{f}"] - grids[f"{c}|{f}"])).sum()),
                            public=h["first 84 h"][0] - base_h[f]["first 84 h"][0], private=h["first 84 h"][1] - base_h[f]["first 84 h"][1]))
        a = pd.DataFrame(acc).mean()
        rows.append(dict(config=c, kind="diagnostic (truth)" if "true " in c else "rule", **a.to_dict()))
    cr = pd.read_csv(L.HERE / "q3_carry.csv")
    cr = cr[cr.hyp == "first 84 h"].groupby("extra").agg(arrival=("carried_mwh", "mean"), whole=("d_whole", "mean"), public=("d_public", "mean"),
                                                         private=("d_private", "mean")).reset_index()
    for _, r in cr.iterrows():
        rows.append(dict(config=f"forced extra charge of {int(r.extra)} kWh in the last hour of every week", kind="forced carry",
                         arrival=r.arrival, whole=r.whole, public=r.public, private=r["private"]))
    d = pd.DataFrame(rows)
    d.to_csv(L.HERE / "q4_all.csv", index=False)
    x = d.arrival.to_numpy()
    sl = {k: float(x @ d[k].to_numpy() / (x @ x)) for k in ("public", "private", "whole")}
    print("slopes per MWh of arrival charge (through the origin):", {k: round(v, 3) for k, v in sl.items()}, "n =", len(d))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhline(0, color=INK["axis"], lw=0.8)
    xx = np.linspace(x.min(), x.max(), 2)
    for k, col, lab in (("public", C[0], "first 84 h of each week"), ("private", C[1], "last 84 h of each week"),
                        ("whole", INK["primary"], "whole run")):
        ax.plot(xx, sl[k] * xx, color=col, lw=1.2, alpha=0.6, zorder=2)
        ax.scatter(x, d[k], s=30, color=col, edgecolor=INK["surface"], lw=0.9, label=lab, zorder=3)
        ax.annotate(f"{sl[k]:+.2f} per MWh", (xx[-1], sl[k] * xx[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK["secondary"])
    ax.set_xlim(x.min() - 0.08, x.max() + 0.42)
    ax.set_xlabel("change of the mean charge held at the end of a week (MWh); one point = one variant of h2, mean of folds")
    ax.set_ylabel("change of the score on those hours (points)")
    ax.set_title("With an 84-hour cut, energy carried into a week moves points between the halves:\nthe whole run does not move")
    ax.legend(loc="upper left", title="score computed on")
    save(fig, OUT / "transfer.png")


def fig_halves():
    h = pd.read_csv(L.HERE / "q3_halves.csv")
    h = h[h.agent.isin(["h2", "optimum"])]
    hyps = ["first 84 h", "last 84 h", "alt days (1,3,5,7)", "alt days (2,4,6)", "alt weeks (Sep, Nov, ...)", "alt weeks (Oct, Dec, ...)", "random hours"]
    labs = ["first 84 h", "last 84 h", "days 1, 3, 5, 7", "days 2, 4, 6", "weeks of Sep, Nov, ...", "weeks of Oct, Dec, ...", "random half of the hours"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(hyps))[::-1]
    for f, mk, off in zip(L.FOLDS, ("o", "s", "^"), (0.18, 0.0, -0.18)):
        d = h[(h.agent == "h2") & (h.fold == f)].set_index("hyp").loc[hyps]
        ax.scatter(d.public - d.whole, y + off, s=42, marker=mk, color=FCOL[f], edgecolor=INK["surface"], lw=1.0, zorder=3, label=f"fold {f}")
    ax.axvline(0, color=INK["axis"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("h2: score on the public half minus score on the whole run (points)")
    ax.set_title("The cut alone moves h2's public score by up to 1.8 points on a fold;\non B and C an 84-hour cut moves it by 0.2 at most")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, OUT / "halves.png")


if __name__ == "__main__":
    apply()
    fig_offsets()
    fig_days()
    fig_rules()
    fig_transfer()
    fig_halves()
