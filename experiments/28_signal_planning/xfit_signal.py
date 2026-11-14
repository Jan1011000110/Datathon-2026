# Usage: .venv/Scripts/python.exe experiments/28_signal_planning/xfit_signal.py
"""Honest error paths of the signal forecaster. For each evaluation fold F (A, B, C, ALL = ship):
inside the rows a model may be fitted on (everything but F's days), take each remaining 7-day block
of days (1-7, 8-14, 15-21), fit SignalForecaster WITHOUT that block and replay the block as a run.
Stores, per origin, the forecast and the error vector (truth - forecast) over horizons 0..H-1.
Writes errors_<F>.npz. 3 workers."""
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "experiments" / "23_signal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.hard import load_hard  # noqa: E402
from src.hard_mpc import KAPPA  # noqa: E402
from common import FOLD_DAYS, replay, signal_fit_table, truth  # noqa: E402
from signal_fc import SignalForecaster  # noqa: E402

H = 36


def one(fold):
    t0 = time.perf_counter()
    fit = signal_fit_table(None if fold == "ALL" else fold)
    full = load_hard("filled")
    out = {"F": [], "E": [], "hod": [], "dow": [], "block": [], "pos": [], "ts": []}
    for b, (lo, hi) in FOLD_DAYS.items():
        if b == fold:
            continue
        d = fit.index.day
        fc = SignalForecaster(fit[~((d >= lo) & (d <= hi))], KAPPA)
        dfb = full[(full.index.day >= lo) & (full.index.day <= hi)]
        _, _, W = replay(fc, dfb, H=H, parts=False)
        T = truth(dfb, H=H)["w"]
        out["F"].append(W)
        out["E"].append(T - W)
        out["hod"].append(dfb.index.hour.to_numpy())
        out["dow"].append(dfb.index.dayofweek.to_numpy())
        out["block"].append(np.full(len(dfb), ord(b)))
        out["pos"].append(np.arange(len(dfb)))
        out["ts"].append(dfb.index.values.astype("datetime64[h]").astype(np.int64))
    z = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(HERE / f"errors_{fold}.npz", **z)
    return f"{fold}: {len(z['hod'])} origins, {time.perf_counter() - t0:.1f} s"


if __name__ == "__main__":
    for line in Parallel(n_jobs=3)(delayed(one)(f) for f in ("A", "B", "C", "ALL")):
        print(line, flush=True)
