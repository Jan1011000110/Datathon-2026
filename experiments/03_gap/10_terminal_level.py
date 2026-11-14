"""Follow-up to 07: the only windows where the horizon matters are very flat, very clean days
(w ~ 0.07, no cycling inside the window), where 'energy left at the end is worth 0' makes v1
stock up too little. Window-relative terminal values cannot see that (they scale with the
window); an ABSOLUTE level can. Test:
   v_end(s) = c                                  for a grid of constants c (in units of w)
   v_end(s) = eta * quantile_q(w of train)       fitted on the train complement only
   v_end(s) = eta * mean(w seen in the last 7 days of the run)
   v_end(s) = m * mean(window)                   (relative, as a control)
Paired against v1, per week. Output: out/terminal_level.csv
"""
from joblib import Parallel, delayed
from lib import *

OUT = Path(__file__).parent / "out"
KAPPA = v1.KAPPA
_F, _Q = {}, {}


def fold(f):
    if not _F:
        _F.update(get_folds("filled"))
    return _F[f]


def train_w(f):
    if f not in _Q:
        from src.data import load_hourly_full
        comp = train_complement(fold(f), "full") if f != "ALL" else load_hourly_full()
        _Q[f] = (comp["co2_kg_per_kwh"] + KAPPA * comp["precio_eur_per_kwh"]).to_numpy(float)
    return _Q[f]


def vend_abs(c):
    arr = np.full(NI, float(c))
    return lambda agent, obs, w: arr


def vend_rel(m):
    return lambda agent, obs, w: np.full(NI, m * float(np.mean(w[:24])))


def vend_recent(days=7, mult=ETA):
    def g(agent, obs, w):
        if not hasattr(agent, "seen"):
            agent.seen = []
        agent.seen.append(float(w[0]))
        hist = agent.seen[-24 * days:]
        return np.full(NI, mult * float(np.mean(hist + list(w[1:24]))))
    return g


def variants(f):
    v = {"v1": dict(), "CHEAT true future, whole run": dict(cheat_h="full", df=fold(f))}
    for c in (0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.50, 10.0):
        v["abs c=%.2f" % c] = dict(vend=vend_abs(c))
    tw = train_w(f)
    for q in (0.05, 0.10, 0.25, 0.50):
        v["eta*q%02d(train w)=%.3f" % (int(q * 100), ETA * np.quantile(tw, q))] = dict(vend=vend_abs(ETA * np.quantile(tw, q)))
    v["eta*mean(last 7 days seen)"] = dict(vend=vend_recent(7))
    for m in (1.0, 1.25, 1.5, 2.0, 3.0):
        v["rel %.2f x mean(window)" % m] = dict(vend=vend_rel(m))
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
    orc = {f: oracle(fold(f))["score"] for f in FOLDS}
    rows = []
    for f, n in tasks:
        s, k, c, p = store[(f, n)]
        ref = store[(f, "v1")]
        full = store[(f, "CHEAT true future, whole run")]
        st = summarize(paired(fold(f), c, ref[2]))
        rows.append({"fold": f, "variant": n, "gain_1e-4": (s - ref[0]) * 1e4, "gain_%gap": 100 * (s - ref[0]) / (orc[f] - ref[0]),
                     "hours_changed": int(np.sum(np.abs(p - ref[3]) > 1e-6)), "hours_unlike_cheat": int(np.sum(np.abs(p - full[3]) > 1e-6)),
                     "weeks+": st["pos"], "weeks-": st["neg"], "worst_week_1e-4": st["min"] * 1e4, "best_week_1e-4": st["max"] * 1e4, "infr": k})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "terminal_level.csv", index=False)
    print("train-complement w quantiles (fold A): ", np.round(np.quantile(train_w("A"), [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]), 3))
    for f in FOLDS:
        print("\nfold %s  (gap %.2fe-4)%s" % (f, (orc[f] - store[(f, "v1")][0]) * 1e4, "  [train quantiles IN-SAMPLE]" if f == "ALL" else ""))
        print(tab[tab.fold == f].drop(columns=["fold"]).round(2).to_string(index=False))
