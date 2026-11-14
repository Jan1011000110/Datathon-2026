# Usage: .venv/Scripts/python.exe experiments/27_combined/build_cache.py
"""Record once what every forecaster says at every step of folds A, B, C (cache/tape_<fold>.npz) and the
cross-fitted replays of the fit rows used for the error tables (cache/xfit_<fold>.npz, fold = A, B, C, ALL).
Everything is fitted outside the fold it is used on. 4 workers."""
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined as cb  # noqa: E402
from src.hard import get_folds  # noqa: E402


def tape(fold):
    t0 = time.perf_counter()
    df = get_folds()[fold]
    fcs = cb.build_forecasters(df)
    t1 = time.perf_counter()
    out, sec = cb.record(df, fcs)
    cb.CACHE.mkdir(exist_ok=True)
    np.savez_compressed(cb.CACHE / f"tape_{fold}.npz", **out)
    return f"tape {fold}: fit {t1 - t0:.1f} s, replay {time.perf_counter() - t1:.1f} s; " + \
        ", ".join(f"{k} {v:.2f}" for k, v in sec.items())


def xfit(fold):
    t0 = time.perf_counter()
    df = None if fold == "ALL" else get_folds()[fold]
    cb.crossfit(df)
    return f"xfit {fold}: {time.perf_counter() - t0:.1f} s"


if __name__ == "__main__":
    jobs = [delayed(tape)(f) for f in "ABC"] + [delayed(xfit)(f) for f in ("A", "B", "C", "ALL")]
    for line in Parallel(n_jobs=4)(jobs):
        print(line)
