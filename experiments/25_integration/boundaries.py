# Usage: .venv/Scripts/python.exe experiments/25_integration/boundaries.py
"""Things no forecaster owns: (1) how much the start state and the week boundaries matter,
(2) what a kWh left in the battery at the end of a week / of the run is worth.
Uses the perfect-foresight LP, so the numbers are upper bounds on what these effects can cost."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hard import CAP, ETA, PMAX, SOC0, get_folds, score_weights  # noqa: E402
from src.sim import score_from_grid, solve_lp  # noqa: E402


def lp_score(df, soc0=SOC0, cap=CAP, pmax=PMAX, end_value=0.0):
    load, pv = df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float)
    price, co2 = df["precio_eur_per_kwh"].to_numpy(float), df["co2_kg_per_kwh"].to_numpy(float)
    w = score_weights(df)
    c, d, s, g, _ = solve_lp(w, load - pv, soc0=soc0, cap=cap, pmax=pmax, soc_end_value=end_value)
    return score_from_grid(g, load, pv, price, co2)["score"], s


if __name__ == "__main__":
    rows = []
    for f, df in get_folds().items():
        if f == "ALL":
            continue
        base, s = lp_score(df)
        # week boundaries = rows where the timestamp jumps by more than a day
        jump = np.r_[False, np.diff(df.index.values).astype("timedelta64[h]").astype(int) > 24]
        soc_at_boundary = s[np.where(jump)[0] - 1]
        # independent weeks: each week starts at 3,000 kWh and its leftover is worth nothing
        week_id = np.cumsum(jump)
        w = score_weights(df)
        load, pv = df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float)
        g_all = np.zeros(len(df))
        for k in np.unique(week_id):
            m = week_id == k
            c, d, s_k, g, _ = solve_lp(w[m], (load - pv)[m], soc0=SOC0, cap=CAP, pmax=PMAX)
            g_all[m] = g
        indep = score_from_grid(g_all, load, pv, df["precio_eur_per_kwh"].to_numpy(float), df["co2_kg_per_kwh"].to_numpy(float))["score"]
        rows.append({"fold": f, "oracle": base, "start empty": lp_score(df, soc0=0.0)[0], "start full": lp_score(df, soc0=CAP)[0],
                     "weeks planned separately": indep, "weeks": int(week_id.max() + 1),
                     "mean SOC at week end (kWh)": float(np.mean(soc_at_boundary)), "median": float(np.median(soc_at_boundary))})
    out = pd.DataFrame(rows).set_index("fold")
    print(out.round(3).T.to_string())
    out.to_csv(Path(__file__).with_name("boundaries.csv"))
