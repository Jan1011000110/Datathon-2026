# Usage: .venv/Scripts/python.exe experiments/28_signal_planning/analyse.py [results csv names ...]
"""Guard against choosing knobs on the folds we score on. For each setting and each family of
configurations: (1) in-sample: the configuration with the best mean over A, B, C; (2) honest: for
each fold, pick the configuration that is best on the OTHER two folds and score it on the held-out
fold. Gains are against the certainty-equivalent plan (exact planner). Writes nested.csv."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REF = "certainty-equivalent, exact planner"


def load(names):
    d = pd.concat([pd.read_csv(HERE / f"results_{n}.csv") for n in names])
    return d.drop_duplicates(["config", "setting", "fold"], keep="last")


def nested(piv, members):
    """piv: config x fold scores. Returns (in-sample best config, its gain, honest gain, picks)."""
    ref = piv.loc[REF]
    cand = piv.loc[[m for m in members if m in piv.index]]
    best = cand.mean(axis=1).idxmax()
    gain_in = cand.loc[best].mean() - ref.mean()
    picks, g = {}, []
    for f in "ABC":
        others = [x for x in "ABC" if x != f]
        pick = cand[others].mean(axis=1).idxmax()
        if cand.loc[pick, others].mean() <= ref[others].mean():      # nothing beats the reference: keep it
            pick = REF
        picks[f] = pick
        g.append(piv.loc[pick, f] - ref[f])
    return best, gain_in, sum(g) / 3, picks


if __name__ == "__main__":
    names = sys.argv[1:] or ["main"]
    d = load(names)
    rows = []
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    for s in sorted(d.setting.unique()):
        piv = d[d.setting == s].pivot(index="config", columns="fold", values="score")
        grid = [f"paths K={K}, lam={lam:.2f}" for K in (5, 10, 20, 50) for lam in (0.25, 0.5, 0.75, 1.0)]
        paths_only = [c for c in piv.index if c.startswith("paths") and "AR" not in c and "bias" not in c
                      and "first actions" not in c]
        ar_only = [c for c in piv.index if c.startswith("certainty-equivalent + AR")]
        fam = {"paths: K x lam grid (16)": grid,
               "paths: every variant scanned": paths_only,
               "AR correction only (multiplier)": ar_only,
               "paths + AR correction": [c for c in piv.index if c.startswith("paths") and "AR" in c],
               "everything scanned": [c for c in piv.index if c != REF]}
        for name, members in fam.items():
            if not members:
                continue
            best, gi, gh, picks = nested(piv, members)
            rows.append({"setting": s, "family": name, "n configs": len(members), "best in-sample": best,
                         "gain in-sample": gi, "gain honest (pick on 2, score on 3rd)": gh,
                         "pick for A": picks["A"], "pick for B": picks["B"], "pick for C": picks["C"]})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "nested.csv", index=False)
    # fixed configurations: gain per fold
    fixed = ["paths K=20, lam=1.00", "paths K=20, lam=0.50", "paths K=10, lam=0.50", "paths K=20, lam=0.75",
             "certainty-equivalent + AR correction x1.0", "paths K=20, lam=0.50 + AR correction x1.0",
             "paths K=20, average of first actions", "paths K=20, median of first actions"]
    for s_ in sorted(d.setting.unique()):
        piv = d[d.setting == s_].pivot(index="config", columns="fold", values="score")
        g = piv.loc[[c for c in fixed if c in piv.index]] - piv.loc[REF]
        g["mean"] = g.mean(axis=1)
        print("gain per fold,", s_)
        print(g.round(3).to_string())
    print(out.round(3).T.to_string())
