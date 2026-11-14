"""
experiments/29_stress_hard/boundary.py
The three-week jumps between test weeks: where in the week does each packaged agent lose, and does answering
the first hours after a jump with h1 (h2's own embedded fallback) help h2?

1. Regret by day position in the week (1 = first day after the jump), agents h1 and h2, folds A, B, C.
2. Wrapper "h2, but h1 answers the first K hours after a jump in the timestamps" for K = 6, 12, 24, 48.
   Both agents are stepped every hour (h2's forecasters need every reading); only the answer is switched.
    boundary.py      -> out/boundary_daypos.csv, out/boundary_switch.csv
"""
import numpy as np
import pandas as pd

import lib as L
from src.hard import oracle


class Switch:
    """h1 answers the first K hours of the run and of every stretch that starts after a jump of more than `gap` h."""
    def __init__(self, h2, h1, K, gap=24):
        self.h2, self.h1, self.K, self.gap = h2, h1, K, gap
        self.prev = None
        self.since = 0
        self.fallbacks = 0

    def act(self, obs):
        ts = pd.Timestamp(str(obs["timestamp"]))
        if self.prev is None or abs((ts - self.prev) / pd.Timedelta(hours=1)) > self.gap:
            self.since = 0
        else:
            self.since += 1
        self.prev = ts
        p2 = self.h2.act(obs)
        self.fallbacks = self.h2.fallbacks
        if self.since < self.K:
            return self.h1.act(obs)
        return p2


if __name__ == "__main__":
    folds = L.clean_folds()
    pos_rows, sw_rows = [], []
    for fold, df in folds.items():
        otr = oracle(df, return_trace=True)["trace"]
        wk = L.weeks(df)
        first = pd.Series(df.index.normalize(), index=df.index).groupby(wk).transform("min")
        daypos = ((df.index.normalize() - pd.DatetimeIndex(first)) / pd.Timedelta(days=1)).astype(int) + 1
        base = {}
        for kind in ("h1", "h2"):
            r = L.run(L.load_agent(kind, L.state_path(kind, fold)), df, trace=True)
            base[kind] = r["score"]
            reg = 100 * r["trace"]["w"].to_numpy() * (r["trace"]["grid_kw"].to_numpy() - otr["grid_kw"].to_numpy())
            g = pd.Series(reg).groupby(np.asarray(daypos)).sum()
            for p, v in g.items():
                pos_rows.append(dict(agent=kind, fold=fold, day=int(p), regret=float(v)))
            # hours since the jump, first two days
            hsj = pd.Series(np.arange(len(df))).groupby(wk).transform(lambda s: s - s.iloc[0]).to_numpy()
            for a, b in ((0, 6), (6, 12), (12, 24), (24, 48)):
                m = (hsj >= a) & (hsj < b)
                pos_rows.append(dict(agent=kind, fold=fold, day=f"h{a}-{b}", regret=float(reg[m].sum())))
            print(kind, fold, "score %.3f" % r["score"], flush=True)
        for K in (6, 12, 24, 48):
            a = Switch(L.load_agent("h2", L.state_path("h2", fold)), L.load_agent("h1", L.state_path("h1", fold)), K)
            r = L.run(a, df)
            sw_rows.append(dict(fold=fold, K=K, score=r["score"], d_vs_h2=r["score"] - base["h2"], infractions=r["infractions"],
                                fallbacks=r["fallbacks"]))
            print(sw_rows[-1], flush=True)
    p = pd.DataFrame(pos_rows)
    p.to_csv(L.HERE / "out" / "boundary_daypos.csv", index=False)
    s = pd.DataFrame(sw_rows)
    s.to_csv(L.HERE / "out" / "boundary_switch.csv", index=False)
    pd.set_option("display.width", 200)
    print(p.pivot_table(index=["agent", "fold"], columns="day", values="regret", aggfunc="sum").round(3))
    print(s.pivot_table(index="K", columns="fold", values="d_vs_h2").round(4))
