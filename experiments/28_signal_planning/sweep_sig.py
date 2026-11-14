# Usage: .venv/Scripts/python.exe experiments/28_signal_planning/sweep_sig.py <set>
"""Closed-loop scores on folds A, B, C of planning against signal scenario paths, from the recorded
forecast tapes of experiments/27_combined (cache/tape_<fold>.npz: every forecaster fitted outside
the fold) and the cross-fitted error paths (errors_<fold>.npz, xfit_signal.py).
  setting "iso"  : signal forecaster + perfect net load (isolates the signal; deterministic ref 20.69)
  setting "comb" : the combined agent (signal, load, solar forecasters, scen="conv"; ref 19.99)
Writes results_<set>.csv. One worker per fold (3)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "experiments" / "27_combined"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import combined as cb  # noqa: E402
import sigplan as sp  # noqa: E402
from src.hard import get_folds, simulate  # noqa: E402
from src.hard_mpc import PerfectN  # noqa: E402


def cfg(label, **kw):
    d = dict(label=label, K=10, m=12, lam=1.0, pmode="add", select="strat", centre=True, reveal=None,
             gamma=1.0, J=64, mode="value", ar=0.0, bias=0.0, exact=False, pool=True, seed=0)
    d.update(kw)
    return d


def configs(which):
    out = []
    if which == "check":
        out += [cfg("certainty-equivalent, exact planner", pool=False, exact=True),
                cfg("certainty-equivalent, threshold recursion J=64", pool=False),
                cfg("certainty-equivalent, threshold recursion J=8", pool=False, J=8),
                cfg("paths K=10, lam=1"),
                cfg("paths K=10, lam=0.5", lam=0.5)]
    if which == "main":
        out.append(cfg("certainty-equivalent, exact planner", pool=False, exact=True))
        for lam in (0.25, 0.5, 0.75, 1.0):
            out.append(cfg(f"paths K=20, lam={lam:.2f}", K=20, lam=lam))
        for K in (5, 10, 50):
            out.append(cfg(f"paths K={K}, lam=1.00", K=K))
            out.append(cfg(f"paths K={K}, lam=0.50", K=K, lam=0.5))
        for g in (0.5, 0.75, 1.25, 1.5):
            out.append(cfg(f"paths K=20, lam=1.00, gamma={g:.2f}", K=20, gamma=g))
        for m in (6, 9, 18):
            out.append(cfg(f"paths K=20, lam=1.00, m={m}", K=20, m=m))
        for r in (1, 3, 6):
            out.append(cfg(f"paths K=20, lam=1.00, reveal={r}", K=20, reveal=r))
        for J in (16, 32, 128):
            out.append(cfg(f"paths K=20, lam=1.00, J={J}", K=20, J=J))
        out.append(cfg("paths K=20, lam=1.00, multiplicative", K=20, pmode="mul"))
        out.append(cfg("paths K=20, lam=1.00, not centred", K=20, centre=False))
        out.append(cfg("paths K=20, lam=1.00, random draw", K=20, select="random"))
        out.append(cfg("paths K=20, lam=1.00, random draw seed 1", K=20, select="random", seed=1))
        out.append(cfg("paths K=20, average of first actions", K=20, mode="action"))
        out.append(cfg("paths K=20, median of first actions", K=20, mode="median"))
        out.append(cfg("certainty-equivalent + AR correction x1.0", pool=True, lam=0.0, ar=1.0, exact=True))
        out.append(cfg("certainty-equivalent + AR correction x0.5", pool=True, lam=0.0, ar=0.5, exact=True))
        out.append(cfg("paths K=20, lam=1.00 + AR correction x1.0", K=20, ar=1.0))
    if which == "grid":
        out.append(cfg("certainty-equivalent, exact planner", pool=False, exact=True))
        for K in (5, 10, 50):
            for lam in (0.25, 0.75):
                out.append(cfg(f"paths K={K}, lam={lam:.2f}", K=K, lam=lam))
        for K in (5, 10, 50):
            out.append(cfg(f"paths K={K}, lam=1.00, gamma=0.50", K=K, gamma=0.5))
        out.append(cfg("paths K=20, lam=1.00, gamma=0.25", K=20, gamma=0.25))
        out.append(cfg("paths K=20, lam=0.50, gamma=0.75", K=20, lam=0.5, gamma=0.75))
        for a in (0.75, 1.5, 2.0):
            out.append(cfg(f"certainty-equivalent + AR correction x{a}", lam=0.0, ar=a, exact=True))
        out.append(cfg("paths K=20, lam=0.50 + AR correction x1.0", K=20, lam=0.5, ar=1.0))
        out.append(cfg("paths K=20, lam=0.75 + AR correction x1.0", K=20, lam=0.75, ar=1.0))
        out.append(cfg("paths K=20, lam=1.00, gamma=0.50 + AR correction x1.0", K=20, gamma=0.5, ar=1.0))
        out.append(cfg("paths K=10, lam=0.50 + AR correction x1.0", K=10, lam=0.5, ar=1.0))
        out.append(cfg("paths K=20, lam=0.50 + AR correction x1.5", K=20, lam=0.5, ar=1.5))
    if which == "fix":
        out.append(cfg("certainty-equivalent, exact planner", pool=False, exact=True))
        out.append(cfg("certainty-equivalent + bias correction", lam=0.0, bias=1.0, exact=True))
        out.append(cfg("certainty-equivalent + bias + AR correction x1.0", lam=0.0, bias=1.0, ar=1.0, exact=True))
        out.append(cfg("paths K=20, lam=0.50 + bias + AR correction x1.0", K=20, lam=0.5, bias=1.0, ar=1.0))
        out.append(cfg("paths K=20, lam=0.50 + bias", K=20, lam=0.5, bias=1.0))
        out.append(cfg("paths K=20, lam=0.50, m=6 + AR correction x1.0", K=20, lam=0.5, m=6, ar=1.0))
        out.append(cfg("paths K=20, lam=0.50, m=9", K=20, lam=0.5, m=9))
        out.append(cfg("paths K=20, lam=0.50, m=6", K=20, lam=0.5, m=6))
        out.append(cfg("paths K=20, lam=0.50, reveal=1", K=20, lam=0.5, reveal=1))
        out.append(cfg("paths K=20, lam=0.50, random draw", K=20, lam=0.5, select="random"))
        out.append(cfg("paths K=20, lam=0.50, random draw seed 1", K=20, lam=0.5, select="random", seed=1))
        out.append(cfg("paths K=20, lam=0.50, random draw seed 2", K=20, lam=0.5, select="random", seed=2))
    if which == "final":
        out.append(cfg("certainty-equivalent, exact planner", pool=False, exact=True))
        out.append(cfg("certainty-equivalent + AR correction x1.0", lam=0.0, ar=1.0, exact=True))
        out.append(cfg("paths K=20, lam=0.50", K=20, lam=0.5))
        out.append(cfg("paths K=20, lam=0.50 + AR correction x1.0", K=20, lam=0.5, ar=1.0))
        out.append(cfg("paths K=20, lam=0.50, J=32 + AR correction x1.0", K=20, lam=0.5, ar=1.0, J=32))
        out.append(cfg("paths K=10, lam=0.50, J=32 + AR correction x1.0", K=10, lam=0.5, ar=1.0, J=32))
        out.append(cfg("paths K=50, lam=0.50 + AR correction x1.0", K=50, lam=0.5, ar=1.0))
        out.append(cfg("paths K=20, lam=0.75 + AR correction x1.0", K=20, lam=0.75, ar=1.0))
    return out


_POOL = {}


def pool_for(fold, c):
    key = (fold, c["K"], c["m"], c["pmode"], c["select"], c["centre"], c["reveal"], c["gamma"], c["seed"])
    if key not in _POOL:
        _POOL[key] = sp.Paths(fold, K=c["K"], m=c["m"], mode=c["pmode"], select=c["select"], centre=c["centre"],
                              reveal=c["reveal"], gamma=c["gamma"], seed=c["seed"])
    return _POOL[key]


def run_fold(fold, which, settings):
    df = get_folds()[fold]
    z = np.load(cb.CACHE / f"tape_{fold}.npz")
    tape = {k: z[k] for k in z.files}
    rows = []
    for c in configs(which):
        pool = pool_for(fold, c) if c["pool"] else None
        for setting in settings:
            if setting == "iso":
                ag = sp.SigAgent(cb.TapeW(tape, "w_sig"), PerfectN(df), pool=pool, lam=c["lam"], scen_hours=0,
                                 J=c["J"], mode=c["mode"], ar=c["ar"], bias=c["bias"], exact=c["exact"])
            else:
                b = cb.make_agent(df, tape=tape, signal="fc", load="fc", solar="fc", scen="conv")
                ag = sp.SigAgent(b.w_fc, b.net, pool=pool, lam=c["lam"], H=b.H, Q=b.Q, scen_hours=b.scen_hours,
                                 J=c["J"], mode=c["mode"], ar=c["ar"], bias=c["bias"], exact=c["exact"], v_end=b.v_end,
                                 truncate_end=b.truncate_end)
            t0 = time.perf_counter()
            r = simulate(ag, df)
            rows.append({"config": c["label"], "setting": setting, "fold": fold, "score": r["score"],
                         "infractions": int(r["infractions"]), "plan_ms": ag.sec / len(df) * 1e3,
                         "seconds": time.perf_counter() - t0})
    return rows


def main(which, settings=("iso", "comb")):
    res = Parallel(n_jobs=3)(delayed(run_fold)(f, which, settings) for f in "ABC")
    out = pd.DataFrame([r for rr in res for r in rr])
    out.to_csv(HERE / f"results_{which}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)
    order = [c["label"] for c in configs(which)]
    for s in settings:
        d = out[out.setting == s]
        piv = d.pivot(index="config", columns="fold", values="score")
        piv["mean"] = piv.mean(axis=1)
        piv["infr"] = d.groupby("config")["infractions"].sum()
        piv["plan_ms"] = d.groupby("config")["plan_ms"].mean()
        piv = piv.loc[order]
        piv["gain"] = piv["mean"] - piv["mean"].iloc[0]
        print(f"--- setting: {s}")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("iso", "comb"))
