"""Is there a horizon-related gap above 0.001 ANYWHERE? 45 test-like runs.

Runs (each = 7 consecutive train days from every month, concatenated like the hidden test):
   15 'sliding' runs : days s..s+6 of every month, s = 1..15   (s=1, 8, 15 are folds A, B, C)
   30 'random' runs  : an independent random start day per month (seed 0)
For each run the same waterfall as 01_decompose.py (every step uses the run's true RHO except v1):
   v1 -> true RHO -> + whole future of w -> + this hour's true load -> oracle
plus every hour where the full-foresight agent first departs from the 24 h agent.
Output: out/many_runs.csv, out/many_runs_episodes.csv
"""
from joblib import Parallel, delayed
from lib import *

OUT = Path(__file__).parent / "out"
_ALL = {}


def all_df():
    if not _ALL:
        _ALL["df"] = get_folds("filled")["ALL"]
    return _ALL["df"]


def build(starts):
    """starts: dict (year, month) -> first day of the 7-day block."""
    df = all_df()
    keep = np.zeros(len(df), bool)
    ym = list(zip(df.index.year, df.index.month))
    day = df.index.day
    for i, k in enumerate(ym):
        s = starts[k]
        keep[i] = s <= day[i] <= s + 6
    return df[keep]


def plans():
    df = all_df()
    last = pd.Series(df.index.day, index=df.index).groupby([df.index.year, df.index.month]).max()
    keys = list(last.index)
    out = [("slide s=%02d" % s, {k: s for k in keys}) for s in range(1, 16)]
    rng = np.random.default_rng(0)
    for j in range(30):
        out.append(("random %02d" % j, {k: int(rng.integers(1, last[k] - 5)) for k in keys}))
    return out


def one(name, starts):
    df = build(starts)
    rt = true_rho(df)
    net = (df.load_kw - df.pv_kw).to_numpy()
    a = run(v1.BatteryAgent(), df)
    r = run(GapAgent(rho=rt), df)
    h = run(GapAgent(rho=rt, cheat_h="full", df=df), df)
    g = simulate(GapAgent(rho=rt, cheat_h="full", df=df, cheat_guard=True), df)["score"]
    h60 = simulate(GapAgent(cheat_h="full", df=df), df)["score"]
    o = oracle(df)["score"]
    row = {"run": name, "hours": len(df), "true_rho": rt, "fault_hours(net<50)": int((net < 50).sum()), "v1": a["score"], "oracle": o,
           "gap": (o - a["score"]) * 1e4, "RHO": (r["score"] - a["score"]) * 1e4, "horizon": (h["score"] - r["score"]) * 1e4,
           "horizon_on_v1": (h60 - a["score"]) * 1e4, "this_hour_load": (g - h["score"]) * 1e4, "future_load": (o - g) * 1e4,
           "infr": a["infractions"]}
    # horizon episodes: first hour where the two (same RHO) agents act differently from the same SOC
    pa, pb = r["trace"].p_kw.to_numpy(), h["trace"].p_kw.to_numpy()
    sa, sb = np.r_[1000.0, r["trace"].soc_kwh.to_numpy()[:-1]], np.r_[1000.0, h["trace"].soc_kwh.to_numpy()[:-1]]
    first = (np.abs(pa - pb) > 1e-6) & (np.abs(sa - sb) < 1e-6)
    wk = week_ids(df)
    gw = paired(df, h, r)
    w = (df.co2_kg_per_kwh + (0.3 / 0.7) * rt * df.precio_eur_per_kwh).to_numpy()
    eps = []
    for i in np.where(first)[0]:
        day_w = w[max(0, i - 12):i + 12]
        eps.append({"run": name, "ts": df.index[i], "p_24h": round(pa[i]), "p_full": round(pb[i]), "soc": round(sa[i]),
                    "week_gain_1e-4": gw[wk[i]] * 1e4, "w_now": w[i], "spread_24h(max/min)": float(day_w.max() / day_w.min())})
    return row, eps


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    res = Parallel(n_jobs=4, batch_size=1)(delayed(one)(n, s) for n, s in plans())
    tab = pd.DataFrame([r for r, _ in res]).set_index("run")
    eps = pd.DataFrame([e for _, ee in res for e in ee])
    tab.to_csv(OUT / "many_runs.csv")
    eps.to_csv(OUT / "many_runs_episodes.csv", index=False)
    print(tab.round(3).to_string())
    clean = tab[tab["fault_hours(net<50)"] == 0]
    print("\nALL %d runs  : horizon part  mean %.3f  max %.3f (1e-4 pts) ; on top of v1 (RHO 0.60): mean %.3f max %.3f min %.3f" % (
        len(tab), tab["horizon"].mean(), tab["horizon"].max(), tab["horizon_on_v1"].mean(), tab["horizon_on_v1"].max(), tab["horizon_on_v1"].min()))
    print("             RHO part      mean %.3f  max %.3f" % (tab["RHO"].mean(), tab["RHO"].max()))
    print("CLEAN %d runs: total gap    mean %.3f  median %.3f  max %.3f ; RHO mean %.3f max %.3f ; horizon mean %.3f max %.3f ; load parts max %.3f" % (
        len(clean), clean["gap"].mean(), clean["gap"].median(), clean["gap"].max(), clean["RHO"].mean(), clean["RHO"].max(),
        clean["horizon"].mean(), clean["horizon"].max(), (clean["this_hour_load"] + clean["future_load"]).max()))
    dirty = tab[tab["fault_hours(net<50)"] > 0]
    print("FAULT %d runs: total gap    mean %.3f  min %.3f  max %.3f ; load parts mean %.3f ; RHO+horizon max %.3f" % (
        len(dirty), dirty["gap"].mean(), dirty["gap"].min(), dirty["gap"].max(), (dirty["this_hour_load"] + dirty["future_load"]).mean(),
        (dirty["RHO"] + dirty["horizon"]).max()))
    print("\nhorizon episodes (distinct timestamps):")
    if len(eps):
        e = eps.groupby("ts").agg(runs=("run", "count"), p_24h=("p_24h", "first"), p_full=("p_full", "first"), soc=("soc", "first"),
                                  max_week_gain=("week_gain_1e-4", "max"), w_now=("w_now", "mean"), spread=("spread_24h(max/min)", "mean"))
        print(e.round(3).to_string())
