"""Gap decomposition: run v1 variants with one or more 'cheats' on folds A, B, C, ALL.

Every run stores its per-hour score contributions (points), so later scripts can build
paired per-week differences. Output: out/decomp_<fold>.npz + out/decomp_scores.csv
"""
import time
from joblib import Parallel, delayed
from lib import *

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

RHOS = [0.45, 0.50, 0.55, 0.573, 0.587, 0.60, 0.615, 0.65, 0.70, 0.75]
HORIZ = [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72, 96, 168]
MARGINS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20]


def configs(df):
    rt = true_rho(df)
    c = {"v1": dict()}
    for r in RHOS:
        c["rho=%.3f" % r] = dict(rho=r)
    c["rho=true"] = dict(rho=rt)
    for h in HORIZ:
        c["rho=true,h=+%d" % h] = dict(rho=rt, cheat_h=h, df=df)
        c["h=+%d" % h] = dict(cheat_h=h, df=df)
    c["rho=true,h=full"] = dict(rho=rt, cheat_h="full", df=df)
    c["h=full"] = dict(cheat_h="full", df=df)
    for m in MARGINS:
        c["margin=%.3f" % m] = dict(guard_margin=m)
        c["rho=true,h=full,margin=%.3f" % m] = dict(rho=rt, cheat_h="full", df=df, guard_margin=m)
    c["guard=off"] = dict(guard=False)
    c["outage=off"] = dict(outage=False)
    c["guard=true-load"] = dict(cheat_guard=True, df=df)
    c["rho=true,guard=true-load"] = dict(rho=rt, cheat_guard=True, df=df)
    c["rho=true,h=full,guard=true-load"] = dict(rho=rt, cheat_h="full", cheat_guard=True, df=df)
    c["rho=true,h=full,guard=off"] = dict(rho=rt, cheat_h="full", df=df, guard=False)
    c["rho=true,h=full,outage=off"] = dict(rho=rt, cheat_h="full", df=df, outage=False)
    return c


def one(fold, name):
    df = get_folds("filled")[fold]
    kw = configs(df)[name]
    r = run(GapAgent(**kw), df)
    return fold, name, r["score"], r["infractions"], r["contrib"]


if __name__ == "__main__":
    folds = get_folds("filled")
    t0 = time.time()
    tasks = [(f, n) for f in FOLDS for n in configs(folds[f])]
    res = Parallel(n_jobs=4, batch_size=4)(delayed(one)(f, n) for f, n in tasks)
    rows = []
    for f in FOLDS:
        df = folds[f]
        o = oracle(df, return_trace=True)
        d = {"oracle": contributions(df, o["trace"])}
        rows.append((f, "oracle", o["score"], 0))
        for ff, n, s, k, c in res:
            if ff == f:
                d[n] = c
                rows.append((f, n, s, k))
        np.savez_compressed(OUT / ("decomp_%s.npz" % f), **d)
    tab = pd.DataFrame(rows, columns=["fold", "config", "score", "infr"]).pivot(index="config", columns="fold", values="score")[list(FOLDS)]
    tab.to_csv(OUT / "decomp_scores.csv")
    gap = (tab.loc["oracle"] - tab) * 1e4
    pd.set_option("display.width", 200)
    print("gap to oracle, in 1e-4 points"); print(gap.round(2).to_string())
    print("max infractions:", max(r[3] for r in rows), "  %.0fs" % (time.time() - t0))
