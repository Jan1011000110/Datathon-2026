"""
experiments/02_stress/tune.py  --  tune the guard on scenario families (regret vs the oracle).

    .venv/Scripts/python.exe experiments/02_stress/tune.py <group> [<group> ...]
    groups: drop  pers  margin  future  calendar  final

Scenario families (folds on the hidden test's calendar):
    clean      folds A, B, C as they are (A and C contain the real logger glitches)
    glitch     realistic logger glitches: sub-hour timing, load AND pv read 0, night freeze of the
               meter honoured; 6 durations x 6 start times, 12 per run, folds A and B
    outage     hour-aligned outages of 1, 3, 14 h (x12) and 72 h (x4), 4 start hours, fold B
    zeroweek   one dead week (load = pv = 0) at positions 0..10, folds A, B, C
    lowload    load x0.6 / x0.45 / x0.3 for the Dec week, the Aug week and the whole run, folds A, B
Output: mean regret per family (points) and, for glitch/outage, extra regret per event.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *          # noqa: F401,F403

GL_DUR = ((10 / 60, "10min"), (25 / 60, "25min"), (50 / 60, "50min"), (110 / 60, "110min"), (5.0, "5h"), (13.9, "13.9h"))
GL_START = (7.58, 9.2, 13.25, 17.42, 19.4, 20.75)
_S = None


def scenarios():
    global _S
    if _S is not None:
        return _S
    F = folds("test")
    S = []
    for f in "ABC":
        S.append(("clean", "clean " + f, f, F[f], 0))
    for f in "AB":
        for dur, lab in GL_DUR:
            for st in GL_START:
                S.append(("glitch", "%s from %05.2f" % (lab, st), f, glitches_frozen(F[f], dur, st), 12))
    for dur, weeks in ((1, range(12)), (3, range(12)), (14, range(12)), (72, (1, 4, 7, 10))):
        for h in (20, 14, 17, 7):
            S.append(("outage", "%dh from %02d:00" % (dur, h), "B", outages(F["B"], dur, h, weeks), len(weeks)))
    for f in "ABC":
        for i in range(11):
            S.append(("zeroweek", "dead week #%d" % i, f, zero_week(F[f], i, True), 1))
    for f in "AB":
        for sc in (0.6, 0.45, 0.3):
            S.append(("lowload", "x%.2f Dec week" % sc, f, scale_load(F[f], sc, [3]), 0))
            S.append(("lowload", "x%.2f Aug week" % sc, f, scale_load(F[f], sc, [11]), 0))
            S.append(("lowload", "x%.2f whole run" % sc, f, scale_load(F[f], sc), 0))
    _S = S
    return S


def _work(task):
    vname, kw, i = task
    from guard_lab import LabAgent
    S = scenarios()
    kw = dict(kw)
    if kw.get("fut") == "truth":
        kw["truth"] = S[i][3]
    res = run(LabAgent(**kw), S[i][3])
    return vname, i, res["raw"], res["infractions"]


GROUPS = {
    "drop": {
        "v1": {},
        "drop.55": dict(drop_ratio=0.55), "drop.60": dict(drop_ratio=0.60), "drop.65": dict(drop_ratio=0.65),
        "drop.70": dict(drop_ratio=0.70), "drop.75": dict(drop_ratio=0.75), "drop.80": dict(drop_ratio=0.80),
        "drop.70 skip06": dict(drop_ratio=0.70, drop_skip6=True),
        "drop.80 skip06": dict(drop_ratio=0.80, drop_skip6=True),
        "drop.65 block1": dict(drop_ratio=0.65, drop_block=1),
        "drop.65 block2": dict(drop_ratio=0.65, drop_block=2),
        "drop.70 block1": dict(drop_ratio=0.70, drop_block=1),
        "drop.65 night": dict(drop_ratio=0.65, night=True),
        "night": dict(night=True),
    },
    "pers": {
        "v1 (a4 b2)": {},
        "a1 b0": dict(pers_a=1, pers_b=0), "a1 b1": dict(pers_a=1, pers_b=1), "a2 b1": dict(pers_a=2, pers_b=1),
        "a2 b2": dict(pers_a=2, pers_b=2), "a3 b1": dict(pers_a=3, pers_b=1), "a4 b1": dict(pers_a=4, pers_b=1),
        "a4 b3": dict(pers_a=4, pers_b=3), "a6 b2": dict(pers_a=6, pers_b=2), "a8 b2": dict(pers_a=8, pers_b=2),
        "a12 b2": dict(pers_a=12, pers_b=2), "a24 b0": dict(pers_a=24, pers_b=0),
        "a2 b2 night": dict(pers_a=2, pers_b=2, night=True), "a4 b2 night": dict(night=True),
        "thr 20": dict(outage_kw=20.0), "thr 100": dict(outage_kw=100.0), "thr 200": dict(outage_kw=200.0),
        "thr 300": dict(outage_kw=300.0),
    },
    "margin": {
        "v1 (5%)": {},
        "m 0%": dict(margin=0.0), "m 2%": dict(margin=0.02), "m 10%": dict(margin=0.10), "m 15%": dict(margin=0.15),
        "m 20%": dict(margin=0.20), "m 5% night": dict(night=True), "m 10% night": dict(margin=0.10, night=True),
        "q05": dict(profile="q05", margin=0.0), "q01": dict(profile="q01", margin=0.0),
        "q05 +5%": dict(profile="q05", margin=0.05), "q01 +5%": dict(profile="q01", margin=0.05),
    },
    "future": {
        "v1": {},
        "truth caps (cheat)": dict(fut="truth"),
        "truth caps +exact now": dict(fut="truth", margin=0.0),
        "persist <600": dict(fut="persist", fut_thresh=600.0),
        "persist <550": dict(fut="persist", fut_thresh=550.0),
        "persist <500": dict(fut="persist", fut_thresh=500.0),
        "persist <700": dict(fut="persist", fut_thresh=700.0),
        "persist <600 nearest": dict(fut="persist", fut_thresh=600.0, fut_round="nearest"),
        "persist <600 m10": dict(fut="persist", fut_thresh=600.0, fut_margin=0.10),
        "persist <600 + q05": dict(fut="persist", fut_thresh=600.0, profile="q05", margin=0.0),
    },
    "calendar": {
        "v1": {},
        "exit": dict(cal_exit=True), "entry": dict(entry_week=2), "exit+entry": dict(cal_exit=True, entry_week=2),
        "exit+entry1": dict(cal_exit=True, entry_week=1),
        "exit+entry+buy48": dict(cal_exit=True, entry_week=2, dead_buy=48),
    },
}
V11 = dict(drop_ratio=0.65, night=True, cal_exit=True)
GROUPS["final"] = {
    "v1": {},
    "v1.1 = drop.65 + night + exit": V11,
    "v1.1 + entry": dict(V11, entry_week=2),
    "v1.1 + q05": dict(V11, profile="q05", margin=0.0),
    "v1.1 + persist<600": dict(V11, fut="persist", fut_thresh=600.0),
    "v1.1 + q05 + persist<600": dict(V11, profile="q05", margin=0.0, fut="persist", fut_thresh=600.0),
    "v1.1 - drop": dict(night=True, cal_exit=True),
    "v1.1 - night": dict(drop_ratio=0.65, cal_exit=True),
    "v1.1 - exit": dict(drop_ratio=0.65, night=True),
}


def main():
    groups = sys.argv[1:] or ["final"]
    S = scenarios()
    t0 = time.time()
    orc = np.array([oracle_score(s[3]) for s in S])
    print("%d scenario runs per variant (oracle ready in %.0f s)" % (len(S), time.time() - t0), flush=True)
    fam = np.array([s[0] for s in S])
    ev = np.array([s[4] for s in S])
    pd.set_option("display.width", 250)
    for gname in groups:
        V = GROUPS[gname]
        tasks = [(vn, kw, i) for vn, kw in V.items() for i in range(len(S))]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(_work, tasks, chunksize=8))
        raw = {vn: np.full(len(S), np.nan) for vn in V}
        ninf = 0
        for vn, i, sc, k in res:
            raw[vn][i] = sc
            ninf += k
        reg = pd.DataFrame({vn: orc - raw[vn] for vn in V})
        reg.insert(0, "family", fam)
        reg.insert(1, "scenario", [s[1] for s in S])
        reg.insert(2, "fold", [s[2] for s in S])
        reg.to_csv(OUT / ("tune_%s.csv" % gname), index=False)
        m = reg.groupby("family", sort=False)[list(V)].mean().T
        m["clean A"] = [reg[vn][0] for vn in V]
        m["clean B"] = [reg[vn][1] for vn in V]
        m["clean C"] = [reg[vn][2] for vn in V]
        m["lowload max"] = reg[fam == "lowload"][list(V)].max()
        print("\n=== group %s: mean regret vs oracle (points)   [%d infractions, %.0f s] ===" % (gname, ninf, time.time() - t0))
        print(m.round(4).to_string())
        # per-event extra regret for glitch (by duration) and outage (by duration)
        cl = {"A": reg.loc[0, list(V)], "B": reg.loc[1, list(V)], "C": reg.loc[2, list(V)]}
        ext = reg[(fam == "glitch") | (fam == "outage")].copy()
        for vn in V:
            ext[vn] = [(r[vn] - cl[r["fold"]][vn]) / ev[idx] for idx, r in ext.iterrows()]
        ext["dur"] = ext["family"] + " " + ext["scenario"].str.split(" from").str[0]
        print("\nextra regret per event (1e-3 points), by duration:")
        print((1e3 * ext.groupby("dur", sort=False)[list(V)].mean().T).round(2).to_string())
        if gname in ("drop", "final"):
            ext["start"] = ext["scenario"].str.split("from ").str[1]
            print("\nglitches by start time, extra regret per event (1e-3 points):")
            print((1e3 * ext[ext.family == "glitch"].groupby("start", sort=False)[list(V)].mean().T).round(2).to_string())


if __name__ == "__main__":
    main()
