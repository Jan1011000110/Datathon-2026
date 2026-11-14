"""
experiments/29_stress_hard/hol_patch.py
Candidate fix for holidays in h2: a CALENDAR prior for the load forecaster's holiday index.

LoadFC blends the weekday and the Sunday reference with a weight ("holiday index") that it estimates from
today's readings and carries to the next weekdays. On a public holiday that follows a normal working day the
index starts at 0, so everything planned before about 08:00 assumes a normal working day. The patch sets the
prior of the index to HOL_PRIOR on dates listed in HOL (today and future days of the horizon); the readings of
the day still overrule it within a few hours (same shrinkage as before).

Writes a patched COPY of the packaged agent to fix/h2_hol/agent.py (the targets are not touched) and compares
it with the packaged one on folds A, B, C (fold-out states).
    hol_patch.py [prior]
"""
import sys

import numpy as np
import pandas as pd

import lib as L

OFFICIAL = ["2025-09-11", "2025-12-08", "2025-12-25", "2025-12-26", "2026-01-01", "2026-01-06", "2026-04-03",
            "2026-04-06", "2026-05-01", "2026-06-24"]                    # Catalonia-wide public holidays on weekdays
BARCELONA = ["2025-09-24", "2026-05-25"]                                 # local holidays of the city of Barcelona
CLOSURE = ["2025-12-24", "2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"]   # Christmas closure


def days(dates):
    return sorted(int(pd.Timestamp(d).value // (86400 * 10 ** 9)) for d in dates)


def patched_source(dates, prior):
    src = (L.HERE / "pkg" / "h2" / "agent.py").read_text(encoding="utf-8")
    a = ("            prior = self.rho * lp if lp == lp else 0.0\n"
         "            if etype(i) != 0:\n")
    a2 = ("            prior = self.rho * lp if lp == lp else 0.0\n"
          "            if (d0 // 24) in _HOL_DAYS:\n"
          "                prior = max(prior, _HOL_PRIOR)\n"
          "            if etype(i) != 0:\n")
    b = ("                def ref(j):\n"
         "                    if etype(j) != 0 or lam0 == 0.0:\n"
         "                        return S(j, known)\n"
         "                    dj = day0(j)\n"
         "                    lam = lam_d.get(dj)\n"
         "                    if lam is None:\n"
         "                        steps = sum(1 for d in range(dref + 24, dj + 1, 24) if etype(d) == 0)\n"
         "                        lam = lam0 * rho ** steps\n")
    b2 = ("                def ref(j):\n"
          "                    dj = day0(j)\n"
          "                    hol = etype(j) == 0 and dj != dref and (dj // 24) in _HOL_DAYS\n"
          "                    if etype(j) != 0 or (lam0 == 0.0 and not hol):\n"
          "                        return S(j, known)\n"
          "                    lam = lam_d.get(dj)\n"
          "                    if lam is None:\n"
          "                        steps = sum(1 for d in range(dref + 24, dj + 1, 24) if etype(d) == 0)\n"
          "                        lam = lam0 * rho ** steps\n"
          "                        if hol:\n"
          "                            lam = max(lam, _HOL_PRIOR)\n")
    assert src.count(a) == 1 and src.count(b) == 1, (src.count(a), src.count(b))
    src = src.replace(a, a2).replace(b, b2)
    head = f"_HOL_DAYS = frozenset({days(dates)!r})\n_HOL_PRIOR = {float(prior)!r}\n"
    # module-level constants (the bundled modules are functions defined in this file, so they see them)
    k = src.index("\n_H1_TABLES_JSON = ")
    return src[:k + 1] + head + src[k + 1:]


def write_patch(name, dates, prior):
    d = L.HERE / "fix" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "agent.py").write_text(patched_source(dates, prior), encoding="utf-8")
    return d


if __name__ == "__main__":
    prior = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    folds = L.clean_folds()
    variants = {"packaged": None,
                "official": write_patch(f"h2_hol_official_{prior}", OFFICIAL + BARCELONA, prior),
                "official+closure": write_patch(f"h2_hol_closure_{prior}", OFFICIAL + BARCELONA + CLOSURE, prior)}
    rows = []
    for fold, df in folds.items():
        for name, pkg in variants.items():
            a = L.load_agent("h2", L.state_path("h2", fold), pkg=pkg)
            r = L.run(a, df)
            rows.append(dict(prior=prior, fold=fold, variant=name, score=r["score"], infractions=r["infractions"],
                             fallbacks=r["fallbacks"], spill_kwh=r["spill_kwh"], last_error=r["last_error"]))
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(L.HERE / "out" / f"hol_patch_{prior}.csv", index=False)
    print(d.pivot_table(index="variant", columns="fold", values="score").round(4))
