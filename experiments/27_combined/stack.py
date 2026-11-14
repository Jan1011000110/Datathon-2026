# Usage: .venv/Scripts/python.exe experiments/27_combined/stack.py build | run | paired | live
"""Do the round-2 signal forecaster and the signal-path planner stack on the combined agent?
  build   record the round-2 signal forecast on folds A, B, C (cache/tape2_<fold>.npz) and the cross-fitted
          error vectors of the round-2 forecaster (cache/errors2_<fold>.npz, fold = A, B, C, ALL)
  run     closed loop on folds A, B, C: {round 1, round 2} x {plan trusts the forecast, SigAgent} and variants,
          each with truncate_end only and with the calendar end. Writes results_stack.csv and the hourly grid
          import of every run (cache/stack_grid.npz) for the paired test
  paired  gain of each configuration over the shipped one by test-like week (sign test, bootstrap)
  live    untaped, single process: seconds per run for the shipped configuration and the stack"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined as cb  # noqa: E402
import combined3 as c3  # noqa: E402
from src.hard import get_folds, score_weights, simulate  # noqa: E402

HERE = Path(__file__).resolve().parent
PATHS = dict(planner="paths", lam=0.5, ar=1.0, K=20, m=12)
CONFIGS = [  # label, kwargs of make_agent3
    ("round 1 | trust (shipped h2)", dict(signal="fc", planner="trust")),
    ("round 2 | trust", dict(signal="fc2", planner="trust")),
    ("round 1 | SigAgent lam=0.5 ar=1", dict(signal="fc", **PATHS)),
    ("round 2 | SigAgent lam=0.5 ar=1", dict(signal="fc2", **PATHS)),
    ("round 2 | SigAgent lam=0.5 ar=0 (paths only)", dict(signal="fc2", **dict(PATHS, ar=0.0))),
    ("round 2 | SigAgent lam=0 ar=1 (mean correction only)", dict(signal="fc2", **dict(PATHS, lam=0.0))),
    ("round 2 | SigAgent lam=0.5 ar=1, paths rebuilt on round 2", dict(signal="fc2", paths_from="round2", **PATHS)),
    ("round 2 | SigAgent lam=0.5 ar=0, paths rebuilt on round 2", dict(signal="fc2", paths_from="round2", **dict(PATHS, ar=0.0))),
    ("round 2 | SigAgent lam=0 ar=1, slope refitted on round 2", dict(signal="fc2", paths_from="round2", **dict(PATHS, lam=0.0))),
    ("round 1 | SigAgent lam=0.5 ar=0 (paths only)", dict(signal="fc", **dict(PATHS, ar=0.0))),
    ("round 1 | SigAgent lam=0 ar=1 (mean correction only)", dict(signal="fc", **dict(PATHS, lam=0.0))),
]
ENDS = ("truncate_end", "calendar end")
SHIPPED = "round 1 | trust (shipped h2)"


def cal_end(fold):
    return pd.Timestamp(2026, 8, cb.FOLD_DAYS[fold][1], 23)


def tape(fold):
    z = np.load(cb.CACHE / f"tape_{fold}.npz")
    t = {k: z[k] for k in z.files}
    t["w_sig2"] = np.load(cb.CACHE / f"tape2_{fold}.npz")["w_sig2"]
    return t


# ------------------------------------------------------------------ build
def build_tape(fold):
    t0 = time.perf_counter()
    df = get_folds()[fold]
    fc = c3.SignalForecaster2(cb.fit_tables(df)[2], c3.KAPPA)
    t1 = time.perf_counter()
    out, sec = cb.record(df, {"w_sig": fc}, quantiles=False)
    np.savez_compressed(cb.CACHE / f"tape2_{fold}.npz", w_sig2=out["w_sig"])
    return f"tape2 {fold}: fit {t1 - t0:.1f} s, {1e3 * sec['w_sig'] / len(df):.2f} ms per call"


def build_err(fold):
    t0 = time.perf_counter()
    z = c3.signal_errors2(None if fold == "ALL" else get_folds()[fold])
    return f"errors2 {fold}: {len(z['hod'])} origins, {time.perf_counter() - t0:.1f} s"


# ------------------------------------------------------------------ run
def run_fold(fold):
    df = get_folds()[fold]
    tp = tape(fold)
    rows, grids = [], {}
    for label, kw in CONFIGS:
        for end in ENDS:
            ag = c3.make_agent3(df, tape=tp, end_ts=cal_end(fold) if end == "calendar end" else None, **kw)
            t0 = time.perf_counter()
            r = simulate(ag, df, return_trace=True)
            tr = r.pop("trace")
            grids[f"{label}|{end}|{fold}"] = tr["grid_kw"].to_numpy()
            rows.append({"config": label, "end": end, "fold": fold, "score": r["score"], "infractions": int(r["infractions"]),
                         "spill_mwh": float(tr["spill_kw"].sum()) / 1000.0, "seconds": time.perf_counter() - t0})
    return rows, grids


def run():
    res = Parallel(n_jobs=3)(delayed(run_fold)(f) for f in "ABC")
    out = pd.DataFrame([r for rr, _ in res for r in rr])
    out.to_csv(HERE / "results_stack.csv", index=False)
    np.savez_compressed(cb.CACHE / "stack_grid.npz", **{k: v for _, g in res for k, v in g.items()})
    show()


def show():
    out = pd.read_csv(HERE / "results_stack.csv")
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    for end in ENDS:
        d = out[out.end == end]
        piv = d.pivot(index="config", columns="fold", values="score")
        piv["mean"] = piv.mean(axis=1)
        piv = piv.loc[[c for c, _ in CONFIGS]]
        ref = piv.loc[SHIPPED]
        for f in "ABC":
            piv["d" + f] = piv[f] - ref[f]
        piv["gain"] = piv["mean"] - ref["mean"]
        piv["infr"] = d.groupby("config")["infractions"].sum()
        piv["sec"] = d.groupby("config")["seconds"].mean()
        print(f"--- {end}")
        print(piv.round(3).to_string())


# ------------------------------------------------------------------ paired test by week
def weekly_gains(label, end="calendar end", ref=SHIPPED):
    """Per test-like week: score points gained over the reference (they sum to the fold's score difference)."""
    g = np.load(cb.CACHE / "stack_grid.npz")
    rows = []
    for f in "ABC":
        df = get_folds()[f]
        w = score_weights(df) * 100.0
        d = w * (g[f"{ref}|{end}|{f}"] - g[f"{label}|{end}|{f}"])
        wk = df.index.strftime("%Y-%m")
        s = pd.Series(d).groupby(wk.to_numpy()).sum()
        for k, v in s.items():
            rows.append({"fold": f, "week": k, "gain": v})
    return pd.DataFrame(rows)


def paired():
    from math import comb
    rng = np.random.default_rng(0)
    out = []
    for label, _ in CONFIGS[1:]:
        for end in ENDS:
            wk = weekly_gains(label, end)
            x = wk["gain"].to_numpy()
            n, pos = len(x), int((x > 0).sum())
            k = max(pos, n - pos)
            p_sign = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)
            by = [wk.loc[wk.fold == f, "gain"].to_numpy() for f in "ABC"]
            boot = np.array([np.mean([rng.choice(b, len(b)).sum() for b in by]) for _ in range(10000)])
            # days as the unit is not available from sums; leave-one-fold-out view instead
            out.append({"config": label, "end": end, "weeks": n, "improve": pos, "sign_p": p_sign,
                        "gain_mean_of_folds": float(np.mean([b.sum() for b in by])),
                        "ci95_lo": float(np.quantile(boot, 0.025)), "ci95_hi": float(np.quantile(boot, 0.975)),
                        "p_boot_gain_le_0": float(np.mean(boot <= 0)),
                        "median_week": float(np.median(x)), "worst_week": float(x.min()), "best_week": float(x.max()),
                        **{f"weeks_up_{f}": f"{int((b > 0).sum())}/{len(b)}" for f, b in zip("ABC", by)}})
            if label == CONFIGS[3][0] and end == "calendar end":
                wk.to_csv(HERE / "stack_weekly.csv", index=False)
                print(wk.pivot(index="week", columns="fold", values="gain").round(4).to_string())
    o = pd.DataFrame(out)
    o.to_csv(HERE / "stack_paired.csv", index=False)
    pd.set_option("display.width", 280, "display.max_colwidth", 70)
    print(o.round(4).to_string())


# ------------------------------------------------------------------ live timing
def live():
    folds = get_folds()
    rows = []
    for label, kw in (CONFIGS[0], CONFIGS[3]):
        for f in "ABC":
            df = folds[f]
            t0 = time.perf_counter()
            ag = c3.make_agent3(df, end_ts=cal_end(f), **kw)
            t1 = time.perf_counter()
            r = simulate(ag, df)
            t2 = time.perf_counter()
            rows.append({"config": label, "fold": f, "score": r["score"], "infractions": int(r["infractions"]), "fit_s": t1 - t0,
                         "run_s": t2 - t1, "run_s_per_2016": (t2 - t1) * 2016 / len(df)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "stack_live.csv", index=False)
    t0 = time.perf_counter()
    ag = c3.make_agent3(None, end_ts="2026-08-31 23:00", **CONFIGS[3][1])
    t1 = time.perf_counter()
    r = simulate(ag, folds["ALL"].iloc[:2016])
    print(f"make_agent3(None): fit {t1 - t0:.1f} s, 2016 steps in {time.perf_counter() - t1:.2f} s, infractions {r['infractions']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        for line in Parallel(n_jobs=4)([delayed(build_tape)(f) for f in "ABC"] + [delayed(build_err)(f) for f in ("A", "B", "C", "ALL")]):
            print(line, flush=True)
    elif cmd == "run":
        run()
    elif cmd == "show":
        show()
    elif cmd == "paired":
        paired()
    elif cmd == "live":
        live()
