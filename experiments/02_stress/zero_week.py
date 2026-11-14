"""Zero-load week study (lead's priority): regret of v1 and of calendar-aware guard variants
against the perfect-foresight oracle, for a dead week (load = 0, pv = 0) at every position of
the run, plus the downside cases of each rule.

    .venv/Scripts/python.exe experiments/02_stress/zero_week.py
Writes out/zero_week.csv, out/zero_week_downside.csv and prints the tables used in
reports/02_robustness/report.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *          # noqa: F401,F403
from guard_lab import LabAgent, v1

MONTHS = [(2025, 9), (2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4),
          (2026, 5), (2026, 6), (2026, 7), (2026, 8)]
EE = dict(cal_exit=True, entry_week=2)


def variants(i):
    return {
        "v1": {},
        "exit": dict(cal_exit=True),
        "entry": dict(entry_week=2),
        "exit+entry": EE,
        "exit+entry+buy48": dict(EE, dead_buy=48),
        "prior": dict(EE, dead_months=(MONTHS[i],)),          # the dead week is known in advance
    }


def main():
    NG = make_noguard(v1)
    F = folds("test")
    pd.set_option("display.width", 250)

    # ---- 1. one dead week, every position ------------------------------------------------
    rows = []
    for fold in "ABC":
        for i in range(12):
            df = zero_week(F[fold], i, True)
            orc = oracle_score(df)
            r = {"fold": fold, "week": i, "oracle": orc, "noguard": run(NG(), df)["raw"]}
            for name, kw in variants(i).items():
                res = run(LabAgent(**kw), df)
                assert res["infractions"] == 0
                r[name] = res["raw"]
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "zero_week.csv", index=False)
    cols = ["noguard"] + list(variants(0))
    reg = out.copy()
    for c in cols:
        reg[c] = reg["oracle"] - reg[c]
    print("Regret vs oracle (points). One week with load = 0 and pv = 0. Mean of folds A, B, C.")
    t = reg.groupby("week")[cols].mean()
    t.loc["mean"] = t.mean()
    t.loc["mean w/o last"] = reg[reg.week < 11][cols].mean()
    print(t.round(4).to_string())
    print("\nNovember position (week 2), by fold:")
    print(reg[reg.week == 2].set_index("fold")[cols].round(4).to_string())
    print("\nworst case over the 36 runs:")
    print(reg[cols].max().round(4).to_string())

    # ---- 2. downside cases ---------------------------------------------------------------
    rows = []
    for fold in "ABC":
        base = F[fold]
        cases = {
            "clean (no dead week)": (base, None),
            "two dead weeks in a row (Nov+Dec)": (zero_week(zero_week(base, 2), 3), None),
            "dead LAST week (Aug)": (zero_week(base, 11), None),
            "dead FIRST week (Sep)": (zero_week(base, 0), None),
            "Nov dropped (H1), agent told Nov is dead": (drop_weeks(base, [2]), (2025, 11)),
            "Nov dead (H2), agent told Nov is dead": (zero_week(base, 2), (2025, 11)),
            "Nov present and alive, agent told Nov is dead": (base, (2025, 11)),
        }
        for name, (df, dead) in cases.items():
            orc = oracle_score(df)
            r = {"fold": fold, "case": name, "oracle": orc}
            vs = {"v1": {}, "exit": dict(cal_exit=True), "exit+entry": EE, "exit+entry+buy48": dict(EE, dead_buy=48)}
            if dead:
                vs["prior"] = dict(EE, dead_months=(dead,))
            for vn, kw in vs.items():
                res = run(LabAgent(**kw), df)
                assert res["infractions"] == 0
                r[vn] = orc - res["raw"]
            rows.append(r)
    dd = pd.DataFrame(rows)
    dd.to_csv(OUT / "zero_week_downside.csv", index=False)
    print("\nDownside cases: regret vs oracle (points), mean of folds A, B, C")
    print(dd.groupby("case", sort=False)[["v1", "exit", "exit+entry", "exit+entry+buy48", "prior"]].mean().round(4).to_string())
    print("\n... and by fold")
    print(dd.set_index(["case", "fold"])[["v1", "exit", "exit+entry", "exit+entry+buy48", "prior"]].round(4).to_string())


if __name__ == "__main__":
    main()
