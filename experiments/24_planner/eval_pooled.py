# Usage: .venv/Scripts/python.exe experiments/24_planner/eval_pooled.py
"""How much does the SHAPE of the distribution matter? Same planner (stochastic, every hour,
Q=9), residual quantiles pooled over all hours of day (by horizon only) instead of by hour of
day. Writes closed_pooled.csv."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, simulate, train_complement  # noqa: E402
from src.hard_mpc import ClimW, PerfectW  # noqa: E402
import planner as P  # noqa: E402
from eval_closed import Cached  # noqa: E402

rows = []
for fold in ("A", "B", "C"):
    df = get_folds()[fold]
    fit = train_complement(df)
    qn = P.QuantileN(fit, H=36, by_hour=False)
    qn.point = Cached(qn.point)
    for wl, wf in (("clim w", Cached(ClimW(fit))), ("perfect w", PerfectW(df))):
        r = simulate(P.MPCStoch(wf, qn, H=36, Q=9, scen_hours=None), df, return_trace=True)
        rows.append({"config": f"{wl} + profile n | stoch all, Q=9, errors pooled over hours of day", "fold": fold,
                     "score": r["score"], "infractions": r["infractions"],
                     "spill_mwh": float(r["trace"]["spill_kw"].sum()) / 1000.0})
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("closed_pooled.csv"), index=False)
piv = out.pivot(index="config", columns="fold", values="score")
piv["mean"] = piv.mean(axis=1)
piv["spill_MWh"] = out.groupby("config")["spill_mwh"].mean()
print(piv.round(3).to_string())
