# Usage: .venv/Scripts/python.exe experiments/27_combined/check_live.py
"""The recommended configuration run exactly as an agent would (live forecasters, no tapes), single process:
score against the taped result, where the seconds go, the organisers' environment on fold B, and a smoke test
of the shipped configuration (fitted on every train row). Writes live.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined as cb  # noqa: E402
from src.hard import get_folds, simulate, simulate_official  # noqa: E402

HERE = Path(__file__).resolve().parent
KW = dict(signal="fc", load="fc", solar="fc", scen="conv", H=36, Q=9, scen_hours=12, truncate_end=True)
SEC = {}


class Timed:
    """Proxy that adds up the seconds spent in each method of the wrapped object."""
    def __init__(self, obj, name):
        self._o, self._n = obj, name

    def __getattr__(self, attr):
        f = getattr(self._o, attr)
        if not callable(f):
            return f
        key = f"{self._n}.{attr}"

        def g(*a, **k):
            t0 = time.perf_counter()
            r = f(*a, **k)
            SEC[key] = SEC.get(key, 0.0) + time.perf_counter() - t0
            return r
        return g


def timed_plan(f):
    def g(*a, **k):
        t0 = time.perf_counter()
        r = f(*a, **k)
        SEC["planner.plan_stoch"] = SEC.get("planner.plan_stoch", 0.0) + time.perf_counter() - t0
        return r
    return g


if __name__ == "__main__":
    folds = get_folds()
    rows = []
    for fold in "ABC":
        df = folds[fold]
        end = pd.Timestamp(2026, 8, cb.FOLD_DAYS[fold][1], 23)
        t0 = time.perf_counter()
        ag = cb.make_agent(df, end_ts=end, **KW)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = simulate(ag, df)
        t_run = time.perf_counter() - t0
        z = np.load(cb.CACHE / f"tape_{fold}.npz")
        r2 = simulate(cb.make_agent(df, end_ts=end, tape={k: z[k] for k in z.files}, **KW), df)
        rows.append(dict(fold=fold, steps=len(df), live=r["score"], taped=r2["score"], infractions=int(r["infractions"]),
                         fit_s=t_fit, run_s=t_run, run_s_per_2016=t_run * 2016 / len(df)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "live.csv", index=False)

    # where the time goes (fold B, instrumented: the proxies add a little overhead)
    df = folds["B"]
    ag = cb.make_agent(df, end_ts=pd.Timestamp(2026, 8, 14, 23), **KW)
    ag.w_fc = Timed(ag.w_fc, "signal")
    ag.net.load_fc = Timed(ag.net.load_fc, "load")
    ag.net.pv_fc = Timed(ag.net.pv_fc, "solar")
    ag.net = Timed(ag.net, "net")
    cb.P.plan_stoch = timed_plan(cb.P.plan_stoch)
    t0 = time.perf_counter()
    simulate(ag, df)
    tot = time.perf_counter() - t0
    print(f"\ninstrumented run, fold B, {len(df)} steps: {tot:.2f} s")
    for k, v in sorted(SEC.items(), key=lambda kv: -kv[1]):
        print(f"  {k:36s} {v:6.2f} s   {1000 * v / len(df):.3f} ms/step")
    inner = sum(v for k, v in SEC.items() if k.split(".")[0] in ("load", "solar"))
    print(f"  scenario building alone (net.* minus load/solar calls): {SEC.get('net.predict_scen', 0) + SEC.get('net.predict', 0) + SEC.get('net.tail_bias', 0) - inner:.2f} s")

    # organisers' environment, fold B
    import importlib
    importlib.reload(cb.P)
    t0 = time.perf_counter()
    off = simulate_official(cb.make_agent(df, end_ts=pd.Timestamp(2026, 8, 14, 23), **KW), df)
    print("\nofficial environment, fold B:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in off.items() if not hasattr(v, "__len__")},
          f"{time.perf_counter() - t0:.1f} s")

    # shipped configuration: fitted on every train row (score is in-sample, only a smoke test and a timing)
    t0 = time.perf_counter()
    ag = cb.make_agent(None, end_ts="2026-08-31 23:00", **KW)
    t_fit = time.perf_counter() - t0
    allrows = folds["ALL"].iloc[:2016]
    t0 = time.perf_counter()
    r = simulate(ag, allrows)
    print(f"\nshipped configuration make_agent(None): fit {t_fit:.1f} s, 2016 steps in {time.perf_counter() - t0:.2f} s, "
          f"infractions {r['infractions']} (in-sample score {r['score']:.2f}, not a result)")
