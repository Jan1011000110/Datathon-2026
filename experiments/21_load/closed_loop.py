# Usage: .venv/Scripts/python.exe experiments/21_load/closed_loop.py <variant> [set]
"""Closed-loop score of a load-forecaster variant inside the MPC planner, folds A, B, C.
set = "core" (default): perfect signal with perfect solar / profile solar, and the legal agent at d_shade 0.9, 1.0
      "full": adds d_shade 0.85 and 0.95.   Appends to closed_loop.csv; saves the trace of the
      'perfect w, perfect pv' runs to traces/<variant>_<fold>.parquet."""
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, simulate, train_complement  # noqa: E402
from src.hard_mpc import ClimW, MPCHard, NetFromParts, PerfectLoad, PerfectPV, PerfectW, ProfileN  # noqa: E402
from variants import make  # noqa: E402

HERE = Path(__file__).parent


def run(variant, cfg, fold):
    df = get_folds()[fold]
    fit = train_complement(df)
    if variant == "ProfileN":
        fc = ProfileN(fit)
    elif variant == "perfect":
        fc = PerfectLoad(df)
    else:
        fc = make(variant, fit)
    P = ProfileN(fit)
    t0 = time.perf_counter()
    if cfg == "perfW_perfPV":
        ag = MPCHard(PerfectW(df), NetFromParts(fc, PerfectPV(df)), H=36)
    elif cfg == "perfW_profPV":
        ag = MPCHard(PerfectW(df), NetFromParts(fc, P), H=36)
    elif cfg.startswith("legal_s"):
        ag = MPCHard(ClimW(fit), NetFromParts(fc, P), H=36, d_shade=float(cfg[7:]))
    r = simulate(ag, df, return_trace=True)
    tr = r.pop("trace")
    if cfg == "perfW_perfPV":
        (HERE / "traces").mkdir(exist_ok=True)
        tr.to_parquet(HERE / "traces" / f"{variant}_{fold}.parquet")
    return {"variant": variant, "config": cfg, "fold": fold, "score": r["score"], "co2_pct": r.get("co2_pct"),
            "cost_pct": r.get("cost_pct"), "spill_kwh": float(tr["spill_kw"].sum()), "sec": time.perf_counter() - t0}


if __name__ == "__main__":
    variant = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "core"
    cfgs = {"core": ["perfW_perfPV", "perfW_profPV", "legal_s0.9", "legal_s1.0"],
            "full": ["perfW_perfPV", "perfW_profPV", "legal_s0.85", "legal_s0.9", "legal_s0.95", "legal_s1.0"],
            "q": ["perfW_perfPV", "perfW_profPV", "legal_s1.0"],
            "iso": ["perfW_perfPV"]}[which]
    rows = Parallel(n_jobs=3)(delayed(run)(variant, c, f) for c in cfgs for f in ("A", "B", "C"))
    out = pd.DataFrame(rows)
    p = HERE / "closed_loop.csv"
    out.to_csv(p, mode="a", header=not p.exists(), index=False)
    piv = out.pivot(index="config", columns="fold", values="score").round(3)
    piv["mean"] = piv.mean(axis=1).round(3)
    piv["spill_MWh"] = (out.groupby("config")["spill_kwh"].mean() / 1000).round(1)
    print(variant); print(piv.to_string())
