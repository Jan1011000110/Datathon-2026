# Usage: .venv/Scripts/python.exe experiments/23_signal/strict_c.py
"""Test geometry check. In the real test no train day of the same month comes after the test week.
Fold C (days 15-21) fitted only on days 1-14 of every month reproduces that: the climatology is
two weeks stale and built from fewer days. Compares with the normal fold-C fit (days 1-14 and 22+)."""
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAPPA, get_folds, signal_fit_table  # noqa: E402
from signal_fc import SignalFC  # noqa: E402
from src.hard import simulate  # noqa: E402
from src.hard_mpc import ClimW, MPCHard, PerfectN  # noqa: E402

BEST = dict(variant="A48", use_lw=False, cross=True, ridge=0.01)


def run(kind, strict):
    df = get_folds()["C"]
    fit = signal_fit_table("C")
    if strict:
        fit = fit[fit.index.day <= 14]
    fc = ClimW(fit) if kind == "clim" else SignalFC(fit, KAPPA, **BEST, interp=(kind == "best interp"))
    r = simulate(MPCHard(fc, PerfectN(df), H=36), df)
    return {"forecaster": kind, "fit": "days 1-14 only (test geometry)" if strict else "days 1-14 and 22+", "score_C": r["score"]}


if __name__ == "__main__":
    rows = Parallel(n_jobs=3)(delayed(run)(k, s) for k in ("clim", "best", "best interp") for s in (False, True))
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).with_name("strict_c.csv"), index=False)
    print(out.round(3).to_string(index=False))
