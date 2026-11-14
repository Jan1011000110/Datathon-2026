"""Logger glitches (lead's priority no. 2): what do the real train glitches cost, and what do a
"sudden drop" detector and a "charge the pv surplus" rule buy?

    .venv/Scripts/python.exe experiments/02_stress/glitch.py
Tables are printed and saved to out/glitch_*.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *          # noqa: F401,F403
from guard_lab import LabAgent, v1
from src.sim import get_folds

pd.set_option("display.width", 250)
VAR = {
    "v1": {},
    "drop.60": dict(drop_ratio=0.60),
    "drop.65": dict(drop_ratio=0.65),
    "drop.70": dict(drop_ratio=0.70),
    "drop.75": dict(drop_ratio=0.75),
    "abs430": dict(drop_abs=430.0),
    "drop.65+abs430": dict(drop_ratio=0.65, drop_abs=430.0),
    "surplus": dict(surplus="persist"),
    "surplus.shape": dict(surplus="shape"),
    "drop.65+surplus": dict(drop_ratio=0.65, surplus="persist"),
}


def table(cases, variants, label):
    rows = []
    for name, df in cases.items():
        orc = oracle_score(df)
        r = {"case": name, "oracle": orc}
        for vn, kw in variants.items():
            res = run(LabAgent(**kw), df)
            assert res["infractions"] == 0, (name, vn)
            r[vn] = orc - res["raw"]
        rows.append(r)
    d = pd.DataFrame(rows).set_index("case")
    d.to_csv(OUT / ("glitch_%s.csv" % label))
    return d


def main():
    F = folds("test")
    raw = get_folds("filled")
    off = get_folds("official")
    # ---- 1. real train data ---------------------------------------------------------------
    cases = {"fold A (2026-08-05/06 glitches)": F["A"], "fold B (no glitch)": F["B"],
             "fold C (2026-01-16, 01-20, 06-17/18)": F["C"], "ALL train days, filled (6,744 h)": raw["ALL"][CORE],
             "ALL train days, official table (5,993 h)": off["ALL"][CORE],
             "official fold A": off["A"][CORE], "official fold C": off["C"][CORE]}
    d = table(cases, VAR, "real")
    print("1. REAL TRAIN DATA: regret vs oracle (points)")
    print(d.round(4).to_string())

    # ---- 2. injected whole-logger glitches with sub-hour timing (load and pv -> 0) -----------
    # durations seen in train (5-min zero runs): 5, 10, 25, 110 min and 13.9 h
    cases = {}
    for fold in "AB":
        for dur, lab in ((10 / 60, "10min"), (25 / 60, "25min"), (50 / 60, "50min"), (110 / 60, "110min"), (13.9, "13.9h")):
            for st in (7.58, 13.25, 19.4, 20.75):
                cases["%s %s from %05.2f x12" % (fold, lab, st)] = glitches(F[fold], dur, st)
    d2 = table(cases, {k: VAR[k] for k in ("v1", "drop.60", "drop.65", "drop.70", "drop.75", "abs430", "drop.65+abs430")}, "injected")
    d2["dur"] = [c.split()[1] for c in d2.index]
    g = d2.drop(columns="oracle").groupby("dur", sort=False).mean()
    clean = np.mean([oracle_score(F[f]) - run(LabAgent(), F[f])["raw"] for f in "AB"])
    print("\n2. INJECTED LOGGER GLITCHES (load and pv scaled by the uncovered share of each hour), 12 per run,")
    print("   4 start times x folds A, B. Extra regret PER GLITCH vs oracle (points):")
    print(((g - clean) / 12).round(5).to_string())

    # ---- 3. meter-only glitches: load -> 0 but the inverters keep reporting (pv surplus) ----
    cases = {}
    for fold in "AB":
        for dur, lab in ((1.0, "1h"), (3.0, "3h"), (14.0, "14h")):
            for st in (8.0, 11.0, 14.0):
                cases["%s %s from %02d:00 x12" % (fold, lab, st)] = glitches(F[fold], dur, st, pv_too=False)
    d3 = table(cases, {k: VAR[k] for k in ("v1", "surplus", "surplus.shape", "drop.65+surplus")} | {
        "surplus x0.8": dict(surplus="persist", surplus_factor=0.8)}, "meter_only")
    d3["dur"] = [c.split()[1] for c in d3.index]
    g = d3.drop(columns="oracle").groupby("dur", sort=False).mean()
    print("\n3. METER-ONLY GLITCHES (load = 0, pv alive: NOT seen in train), daytime, 12 per run.")
    print("   Extra regret PER GLITCH vs oracle (points):")
    print(((g - clean) / 12).round(5).to_string())
    print("\n   by start hour, gain of 'surplus' over v1 per glitch:")
    d3["start"] = [c.split()[3] for c in d3.index]
    print(((d3["v1"] - d3["surplus"]) / 12).groupby([d3["dur"], d3["start"]], sort=False).mean().round(5).to_string())


if __name__ == "__main__":
    main()
