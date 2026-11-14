# Usage: .venv/Scripts/python.exe experiments/24_planner/final_check.py
"""Recommended configuration, run exactly as an agent would (no forecast cache), folds A, B, C.
Writes final_check.csv."""
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, simulate, train_complement  # noqa: E402
from src.hard_mpc import ClimW  # noqa: E402
import planner as P  # noqa: E402


def run(fold):
    df = get_folds()[fold]
    fit = train_complement(df)
    rows = []
    for label, kw in (("stoch, hours 0-11, Q=9 (recommended)", dict(H=36, Q=9, scen_hours=12)),
                      ("stoch, all hours, Q=9", dict(H=36, Q=9, scen_hours=None)),
                      ("deterministic", dict(H=36, scen_hours=0))):
        ag = P.MPCStoch(ClimW(fit), P.QuantileN(fit, H=36), **kw)
        t0 = time.perf_counter()
        r = simulate(ag, df, return_trace=True)
        rows.append({"config": label, "fold": fold, "score": r["score"], "infractions": r["infractions"],
                     "spill_mwh": float(r["trace"]["spill_kw"].sum()) / 1000.0, "seconds": time.perf_counter() - t0})
    return rows


if __name__ == "__main__":
    out = pd.DataFrame([r for rr in Parallel(n_jobs=3)(delayed(run)(f) for f in "ABC") for r in rr])
    out.to_csv(Path(__file__).with_name("final_check.csv"), index=False)
    piv = out.pivot(index="config", columns="fold", values="score")
    piv["mean"] = piv.mean(axis=1)
    piv["spill_MWh"] = out.groupby("config")["spill_mwh"].mean()
    piv["infr"] = out.groupby("config")["infractions"].sum()
    piv["sec"] = out.groupby("config")["seconds"].mean()
    print(piv.round(3).to_string())
