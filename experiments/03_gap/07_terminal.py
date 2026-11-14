"""Idea 2: recover the 'beyond the window' part without leakage.

Every variant keeps v1 unchanged (RHO 0.60, guard, clamp) and only changes what the solver
assumes after the 24 known hours:
  * constants          v_end(s) = eta * quantile_q(window w)
  * own-history water  average of the exact marginal-value functions of K continuation scenarios
                       = the same clock hours of the last K days seen in this run, rescaled to the
                       level of the current window (K=1 is the periodic continuation)
  * train-analog water table by (month, hour of day) of the average normalised marginal-value
                       function of the following 24 h, fitted on train_complement(fold) only
  * point forecasts    hours 24-47 = yesterday x level trend / month-hour climatology rescaled
and the cheats (true future for +6, +12, +24, +48 h, whole run) as the ceiling.

Paired against v1 per week. Output: out/terminal.csv
"""
from joblib import Parallel, delayed
from lib import *

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
KAPPA = v1.KAPPA
_F, _T = {}, {}


def fold(f):
    if not _F:
        _F.update(get_folds("filled"))
    return _F[f]


def track(agent, w):
    """Keep the run's own history of w (one value per step) and the start of the current block."""
    if not hasattr(agent, "w_hist"):
        agent.w_hist, agent.block_start = [], 0
    if len(agent.w_hist) <= agent.t:
        if agent.jump:
            agent.block_start = agent.t
        agent.w_hist.extend([np.nan] * (agent.t - len(agent.w_hist)))
        agent.w_hist.append(float(w[0]))


def water(scen):
    """Marginal-value function at the start of a continuation scenario (zero value after it)."""
    return v1.marginal_values(np.concatenate([[1.0], scen]), ZERO)


# ---- terminal-value hooks ---------------------------------------------------------------------
def vend_const(q, mult=ETA):
    def f(agent, obs, w):
        return np.full(NI, mult * float(np.quantile(w[:24], q)))
    return f


def vend_own(K):
    def f(agent, obs, w):
        track(agent, w)
        t, lvl = agent.t, float(np.mean(w[:24]))
        vs = [water(w[:24])]                                  # d = 1: the known window itself
        for d in range(2, K + 1):
            lo = t + 24 - 24 * d
            if lo < agent.block_start:
                break
            s = np.asarray(agent.w_hist[lo:lo + 24])
            if len(s) < 24 or np.isnan(s).any():
                break
            vs.append(water(s * (lvl / float(np.mean(s)))))
        return np.mean(vs, axis=0)
    return f


def analog_table(f, horizon=24):
    """(month, hour) -> mean normalised water-value function, from the train complement only
    (for ALL there is no complement: fitted in-sample, flagged in the output)."""
    key = (f, horizon)
    if key in _T:
        return _T[key]
    from src.data import load_hourly_full
    comp = train_complement(fold(f), "full") if f != "ALL" else load_hourly_full()
    w = (comp["co2_kg_per_kwh"] + KAPPA * comp["precio_eur_per_kwh"]).to_numpy(float)
    blk = block_ids(comp.index)
    acc = {}
    for i in range(24, len(w) - horizon + 1):
        if blk[i - 24] != blk[i + horizon - 1]:
            continue
        k = (comp.index[i].month, comp.index[i].hour)
        acc.setdefault(k, []).append(water(w[i:i + horizon]) / np.mean(w[i - 24:i]))
    _T[key] = {k: np.mean(v, axis=0) for k, v in acc.items()}
    return _T[key]


def vend_analog(f, horizon=24):
    tab = analog_table(f, horizon)

    def g(agent, obs, w):
        ts = obs["timestamp"]
        v = tab.get((ts.month, ts.hour))
        return ZERO if v is None else v * float(np.mean(w[:24]))
    return g


def climatology(f):
    key = (f, "clim")
    if key in _T:
        return _T[key]
    from src.data import load_hourly_full
    comp = train_complement(fold(f), "full") if f != "ALL" else load_hourly_full()
    w = comp["co2_kg_per_kwh"] + KAPPA * comp["precio_eur_per_kwh"]
    g = w.groupby([comp.index.month, comp.index.hour]).mean()
    _T[key] = {m: np.array([g.get((m, h), np.nan) for h in range(24)]) for m in range(1, 13)}
    return _T[key]


# ---- point-forecast hooks (extension of the window by 24 h) ------------------------------------
def ext_periodic(agent, obs, w):
    return w[:24]


def ext_trend(agent, obs, w):
    track(agent, w)
    t = agent.t
    trend = 1.0
    if t - 24 >= agent.block_start:
        prev = np.asarray(agent.w_hist[t - 24:t])
        if len(prev) == 24 and not np.isnan(prev).any():
            trend = float(np.clip(np.mean(w[:24]) / np.mean(prev), 0.5, 2.0))
    return w[:24] * trend


def ext_clim(f):
    cl = climatology(f)

    def g(agent, obs, w):
        ts = obs["timestamp"]
        prof = np.roll(cl[ts.month], -ts.hour)                # profile starting at the current hour
        if np.isnan(prof).any():
            return w[:24]
        return prof * (float(np.mean(w[:24])) / float(np.mean(prof)))
    return g


def variants(f):
    df = fold(f)
    v = {"v1": dict()}
    for h in (6, 12, 24, 48):
        v["CHEAT true future +%dh" % h] = dict(cheat_h=h, df=df)
    v["CHEAT true future, whole run"] = dict(cheat_h="full", df=df)
    for q in (0.25, 0.5, 0.75):
        v["const eta*q%02d(window)" % int(q * 100)] = dict(vend=vend_const(q))
    v["const min(window)/eta"] = dict(vend=vend_const(0.0, 1.0 / ETA))
    for K in (1, 3, 7):
        v["own-history water value K=%d" % K] = dict(vend=vend_own(K))
    v["train-analog water value (24h)"] = dict(vend=vend_analog(f, 24))
    v["train-analog water value (48h)"] = dict(vend=vend_analog(f, 48))
    v["forecast: periodic"] = dict(ext=ext_periodic)
    v["forecast: yesterday x trend"] = dict(ext=ext_trend)
    v["forecast: month-hour climatology"] = dict(ext=ext_clim(f))
    return v


def one(f, name):
    df = fold(f)
    r = run(GapAgent(**variants(f)[name]), df)
    return f, name, r["score"], r["infractions"], {"contrib": r["contrib"], "trace": r["trace"][["soc_kwh"]]}, r["trace"]["p_kw"].to_numpy()


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    tasks = [(f, n) for f in FOLDS for n in variants(f)]
    res = Parallel(n_jobs=4, batch_size=1)(delayed(one)(f, n) for f, n in tasks)
    store = {(f, n): (s, k, c, p) for f, n, s, k, c, p in res}
    orc = {f: oracle(fold(f))["score"] for f in FOLDS}
    rows = []
    for f, n in tasks:
        s, k, c, p = store[(f, n)]
        ref = store[(f, "v1")]
        g = paired(fold(f), c, ref[2])
        st = summarize(g)
        rows.append({"fold": f, "variant": n, "score": s, "gain_1e-4": (s - ref[0]) * 1e4,
                     "gain_%gap": 100 * (s - ref[0]) / (orc[f] - ref[0]), "hours_changed": int(np.sum(np.abs(p - ref[3]) > 1e-6)),
                     "weeks+": st["pos"], "weeks-": st["neg"], "worst_week_1e-4": st["min"] * 1e4, "best_week_1e-4": st["max"] * 1e4, "infr": k})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "terminal.csv", index=False)
    for f in FOLDS:
        print("\nfold %s   v1 %.4f   oracle %.4f   gap %.2fe-4%s" % (f, store[(f, "v1")][0], orc[f], (orc[f] - store[(f, "v1")][0]) * 1e4,
                                                                    "   (tables fitted IN-SAMPLE for ALL)" if f == "ALL" else ""))
        print(tab[tab.fold == f].drop(columns=["fold", "score"]).round(2).to_string(index=False))
