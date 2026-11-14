"""Sensitivity of the one terminal value that was never negative in 10_terminal_level.py:
     v_end(s) = mult * mean(w seen during the last `days` days of the run, plus the known window)
and the level-preserving version of the scenario-averaged water value (the last K days of the
run as they were, NOT rescaled to the window level).
Paired against v1 per week. Output: out/terminal_recent.csv
"""
from joblib import Parallel, delayed
from lib import *

OUT = Path(__file__).parent / "out"
_F = {}


def fold(f):
    if not _F:
        _F.update(get_folds("filled"))
    return _F[f]


def vend_recent(days, mult):
    def g(agent, obs, w):
        if not hasattr(agent, "seen"):
            agent.seen = []
        agent.seen.append(float(w[0]))
        hist = agent.seen[-24 * days:]
        return np.full(NI, mult * (sum(hist) + float(np.sum(w[1:24]))) / (len(hist) + 23))
    return g


def vend_scen_unscaled(K):
    def g(agent, obs, w):
        if not hasattr(agent, "seen"):
            agent.seen = []
        agent.seen.append(float(w[0]))
        t = len(agent.seen) - 1                       # steps seen so far (only non-end-game steps call this hook)
        vs = [v1.marginal_values(np.concatenate([[1.0], w[:24]]), ZERO)]
        for d in range(2, K + 1):
            lo = t + 24 - 24 * d
            if lo < 0:
                break
            vs.append(v1.marginal_values(np.concatenate([[1.0], agent.seen[lo:lo + 24]]), ZERO))
        return np.mean(vs, axis=0)
    return g


def variants(f):
    v = {"v1": dict(), "CHEAT true future, whole run": dict(cheat_h="full", df=fold(f))}
    for days in (1, 2, 4, 7, 14, 28):
        for mult, lab in ((0.80, "0.80"), (ETA, "eta"), (1.0, "1.00"), (1.0 / ETA, "1/eta"), (1.25, "1.25")):
            v["recent mean %2dd x %s" % (days, lab)] = dict(vend=vend_recent(days, mult))
    for K in (3, 7, 14):
        v["own-history water value, unscaled, K=%d" % K] = dict(vend=vend_scen_unscaled(K))
    return v


def one(f, name):
    df = fold(f)
    r = run(GapAgent(**variants(f)[name]), df)
    return f, name, r["score"], r["infractions"], {"contrib": r["contrib"], "trace": r["trace"][["soc_kwh"]]}, r["trace"]["p_kw"].to_numpy()


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    tasks = [(f, n) for f in FOLDS for n in variants(f)]
    res = Parallel(n_jobs=4, batch_size=2)(delayed(one)(f, n) for f, n in tasks)
    store = {(f, n): (s, k, c, p) for f, n, s, k, c, p in res}
    rows = []
    for f, n in tasks:
        s, k, c, p = store[(f, n)]
        ref = store[(f, "v1")]
        st = summarize(paired(fold(f), c, ref[2]))
        rows.append({"fold": f, "variant": n, "gain_1e-4": (s - ref[0]) * 1e4, "hours_changed": int(np.sum(np.abs(p - ref[3]) > 1e-6)),
                     "weeks+": st["pos"], "weeks-": st["neg"], "worst_week_1e-4": st["min"] * 1e4, "infr": k})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "terminal_recent.csv", index=False)
    g = tab.pivot(index="variant", columns="fold", values="gain_1e-4")[list(FOLDS)]
    g["weeks-"] = tab.groupby("variant")["weeks-"].sum()
    g["hours_changed"] = tab.groupby("variant")["hours_changed"].sum()
    g["min"] = g[list(FOLDS)].min(axis=1)
    print("gain vs v1 in 1e-4 points; weeks- = number of weeks (all folds) with a loss:")
    print(g.loc[list(variants("A"))].round(2).to_string())
