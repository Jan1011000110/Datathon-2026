"""
experiments/29_stress_hard/fix_endrule.py
Candidate fix: make the "fewer than 3 prices = the run is ending" rule safe against a server that serves short
(or no) price / CO2 lists in the middle of the run.

Today (h1 `end_of_run`, h2 `truncate_end`): any obs with fewer than 3 values makes the plan cover only those
hours, i.e. the agent empties the battery and stops charging. If the lists are absent altogether the agent
believes the run ends every hour.

Gated rule tested here (a wrapper that flips the agents' own switch each hour; the targets are not edited):
the short list counts as the end of the run only when
    * the lists are really there (at least one finite value in each), and
    * the month is July or August (the only months in which the test calendar can end).
On the expected test (one run ending 2026-08-31 23:00) and on folds A, B, C (which end in August) nothing changes.
    fix_endrule.py     -> out/fix_endrule.csv
"""
import numpy as np
import pandas as pd

import lib as L
import suite as S


class Gated:
    def __init__(self, agent, months=(7, 8)):
        self.a, self.months = agent, months
        self.fallbacks = 0

    @staticmethod
    def _n_finite(x):
        try:
            return int(np.isfinite(np.asarray(x, float)).sum())
        except Exception:
            return 0

    def act(self, obs):
        ok = False
        try:
            month = int(str(obs.get("timestamp"))[5:7])
            ok = (month in self.months and self._n_finite(obs.get("preus_properes_24h")) >= 1
                  and self._n_finite(obs.get("co2_properes_24h")) >= 1)
        except Exception:
            ok = False
        a = self.a
        if hasattr(a, "end_of_run"):                 # h1
            a.end_of_run = ok
        if getattr(a, "main", None) is not None:     # h2: the combined agent and the embedded h1
            a.main.truncate_end = ok
        if getattr(a, "h1", None) is not None:
            a.h1.end_of_run = ok
        p = a.act(obs)
        self.fallbacks = getattr(a, "fallbacks", np.nan)
        return p


if __name__ == "__main__":
    folds = L.clean_folds()
    rows = []
    for kind in ("h1", "h2"):
        for fold in ("A", "B", "C"):
            df = folds[fold]
            cases = [("clean", df, None)]
            if fold != "A":
                d12 = L.add_week(df, S.wk_of(df, 10), 28) if S.wk_of(df, 11) is None else df
                t12 = L.relabel(d12, "last7")
                cases += [("lists shortened at every week end", df, L.o_lists("short_weekend")),
                          ("lists shortened at random hours (1 in 50)", df, L.o_lists("short_random")),
                          ("price / CO2 lists absent", df, L.o_lists("missing")),
                          ("empty lists at random hours (1 in 50)", df, L.o_lists("empty_random")),
                          ("12 weeks on the test calendar", t12, None),
                          ("12 weeks, test calendar, lists shortened at every week end", t12, L.o_lists("short_weekend")),
                          ("run ends after 5 weeks", df[L.weeks(df) < 5], None)]
            for name, d, fn in cases:
                r0 = L.run(L.load_agent(kind, L.state_path(kind, fold)), d, fn)
                r1 = L.run(Gated(L.load_agent(kind, L.state_path(kind, fold))), d, fn)
                rows.append(dict(agent=kind, fold=fold, case=name, packaged=r0["score"], gated=r1["score"],
                                 gain=r1["score"] - r0["score"], infr=r1["infractions"], exc=r1["exceptions"],
                                 fallbacks_packaged=r0["fallbacks"], fallbacks_gated=r1["fallbacks"]))
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
            pd.DataFrame(rows).to_csv(L.HERE / "out" / "fix_endrule.csv", index=False)
