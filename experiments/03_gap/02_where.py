"""Where does the horizon gap come from? Compare v1 (true rho) with the same agent given the
whole future of w, hour by hour, and describe the hours where the first action differs."""
from lib import *

pd.set_option("display.width", 250)
folds = get_folds("filled")
for f in FOLDS:
    df = folds[f]
    rt = true_rho(df)
    a = run(GapAgent(rho=rt), df)
    b = run(GapAgent(rho=rt, cheat_h="full", df=df), df)
    pa, pb = a["trace"]["p_kw"].to_numpy(), b["trace"]["p_kw"].to_numpy()
    sa, sb = a["trace"]["soc_kwh"].to_numpy(), b["trace"]["soc_kwh"].to_numpy()
    diff = np.abs(pa - pb) > 1e-6
    wk = week_ids(df)
    g = np.bincount(wk, weights=b["contrib"] - a["contrib"])
    print("\n=== fold %s: full-horizon gain %.5f pts; hours with a different action: %d of %d; same SOC before: %d" % (
        f, b["score"] - a["score"], diff.sum(), len(df), int(np.sum(diff & (np.abs(np.r_[1000.0, sa[:-1]] - np.r_[1000.0, sb[:-1]]) < 1e-6)))))
    print("weekly gains (1e-4 pts):", np.round(g * 1e4, 2))
    # first divergence of each episode: hour where actions differ while the SOC before was equal
    soc_a0 = np.r_[1000.0, sa[:-1]]; soc_b0 = np.r_[1000.0, sb[:-1]]
    first = diff & (np.abs(soc_a0 - soc_b0) < 1e-6)
    hod = df.index.hour.to_numpy()
    blk = block_ids(df.index)
    # hours until the end of the block
    to_end = np.zeros(len(df), int)
    for bid in np.unique(blk):
        m = np.where(blk == bid)[0]
        to_end[m] = m[-1] - m
    t = pd.DataFrame({"ts": df.index[first], "hod": hod[first], "to_block_end": to_end[first], "soc": soc_a0[first].round(0),
                      "p_v1": pa[first].round(0), "p_full": pb[first].round(0)})
    print("divergence episodes: %d ; by hour of day:" % first.sum())
    print(pd.Series(hod[first]).value_counts().sort_index().to_string())
    print("episodes starting within 24 h of a block end: %d" % int((to_end[first] < 24).sum()))
    kind = np.where(pb[first] > pa[first] + 1e-6, "full charges more / discharges less", "full discharges more / charges less")
    print(pd.Series(kind).value_counts().to_string())
    if f != "ALL":
        print(t.head(60).to_string())
