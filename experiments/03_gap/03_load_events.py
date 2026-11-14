"""Load-side gap: list every low-load hour, what v1 did there and what it cost.

For each fold (true rho, so the RHO effect is out of the way):
  * hours with net load < 526 kW (500/0.95: where the guard can bind)
  * energy v1 exported (wasted) and discharge it gave up because of the guard
  * per-day gap to the oracle, largest first (the days that make up the whole gap)
"""
from lib import *

pd.set_option("display.width", 250)
folds = get_folds("filled")
for f in FOLDS:
    df = folds[f]
    rt = true_rho(df)
    a = run(GapAgent(rho=rt), df)
    o = oracle(df, return_trace=True)
    oc = contributions(df, o["trace"])
    net = (df["load_kw"] - df["pv_kw"]).to_numpy()
    last = np.r_[net[0], net[:-1]]
    p = a["trace"]["p_kw"].to_numpy()
    po = o["trace"]["p_kw"].to_numpy()
    w = score_weights(df) * 1e6
    export = np.maximum(-p - net, 0.0)
    cap = np.minimum(PMAX, np.maximum(0.0, last * 0.95))
    day = df.index.normalize()
    daily = pd.Series(oc - a["contrib"], index=df.index).groupby(day).sum() * 1e4
    top = daily.sort_values(ascending=False)
    print("\n=== fold %s  gap(true rho) = %.2f e-4 ; exported (wasted) energy by v1: %.0f kWh in %d h" % (
        f, (o["score"] - a["score"]) * 1e4, export.sum(), int((export > 1e-6).sum())))
    print("days with |gap| > 0.05e-4 (1e-4 pts):")
    print(top[top.abs() > 0.05].round(2).to_string())
    low = net < 526.4
    t = pd.DataFrame({"net": net.round(0), "last_net": last.round(0), "w": w.round(3), "p_v1": p.round(0), "p_orc": po.round(0),
                      "export": export.round(0), "soc_after": a["trace"]["soc_kwh"].round(0).to_numpy()}, index=df.index)
    # w rank within the day (1 = highest)
    t["w_rank_day"] = pd.Series(w, index=df.index).groupby(day).rank(ascending=False).astype(int)
    print("low-load hours (net < 526 kW) and neighbours:")
    idx = np.where(low)[0]
    show = sorted(set(np.clip(np.concatenate([idx - 1, idx, idx + 1]), 0, len(df) - 1)))
    print(t.iloc[show].to_string() if f != "ALL" else "(skipped for ALL: %d h)" % low.sum())
