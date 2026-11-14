# Usage: .venv/Scripts/python.exe experiments/22_solar/eval_closed.py
"""Closed-loop value of the solar forecast on folds A, B, C. Writes closed.csv and traces/<config>_<fold>.csv."""
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, score_weights, simulate, train_complement  # noqa: E402
from src.hard_mpc import ClimW, MPCHard, NetFromParts, PerfectLoad, PerfectN, PerfectW, ProfileN  # noqa: E402
from solar_fc import QuantileH0, SolarFC  # noqa: E402

HERE = Path(__file__).parent
CONFIGS = ["perfect load + perfect solar", "perfect load + SolarFC", "perfect load + SolarFC no weather",
           "perfect load + profile solar", "profile load + SolarFC", "legal: clim w + profile load + SolarFC, shade 0.9",
           "perfect load + SolarFC q0.6 at h0", "perfect load + SolarFC q0.7 at h0", "perfect load + SolarFC q0.8 at h0",
           "legal: clim w + profile load + SolarFC q0.7 at h0, no shade", "legal: clim w + profile load + SolarFC q0.6 at h0, shade 0.9",
           "perfect load + SolarFC q0.9 at h0", "legal: clim w + profile load + SolarFC q0.7 at h0, shade 0.9",
           "legal: clim w + profile load + SolarFC q0.8 at h0, no shade", "profile load + SolarFC q0.7 at h0",
           # the three below were run with the final quantile scale (q_scale=1.1); the q runs above with 1.25
           "perfect load + SolarFC q0.75 at h0", "profile load + SolarFC q0.75 at h0",
           "legal: clim w + profile load + SolarFC q0.75 at h0, shade 0.9"]


def run(name, fold):
    df = get_folds()[fold]
    fit = train_complement(df)
    if name == "perfect load + perfect solar":
        ag = MPCHard(PerfectW(df), PerfectN(df), H=36)
    elif name == "perfect load + SolarFC":
        ag = MPCHard(PerfectW(df), NetFromParts(PerfectLoad(df), SolarFC(fit, fit_quantiles=False)), H=36)
    elif name == "perfect load + SolarFC no weather":
        ag = MPCHard(PerfectW(df), NetFromParts(PerfectLoad(df), SolarFC(fit, fit_quantiles=False, use_weather=False)), H=36)
    elif name == "perfect load + profile solar":
        ag = MPCHard(PerfectW(df), NetFromParts(PerfectLoad(df), ProfileN(fit)), H=36)
    elif name == "profile load + SolarFC":
        ag = MPCHard(PerfectW(df), NetFromParts(ProfileN(fit), SolarFC(fit, fit_quantiles=False)), H=36)
    elif name.startswith("perfect load + SolarFC q"):
        q = float(name.split(" q")[1].split(" ")[0])
        ag = MPCHard(PerfectW(df), NetFromParts(PerfectLoad(df), QuantileH0(SolarFC(fit), q)), H=36)
    elif name.startswith("profile load + SolarFC q"):
        q = float(name.split(" q")[1].split(" ")[0])
        ag = MPCHard(PerfectW(df), NetFromParts(ProfileN(fit), QuantileH0(SolarFC(fit), q)), H=36)
    elif name.startswith("legal: clim w + profile load + SolarFC q"):
        q = float(name.split(" q")[1].split(" ")[0])
        ag = MPCHard(ClimW(fit), NetFromParts(ProfileN(fit), QuantileH0(SolarFC(fit), q)), H=36,
                     d_shade=0.9 if name.endswith("shade 0.9") else 1.0)
    else:
        ag = MPCHard(ClimW(fit), NetFromParts(ProfileN(fit), SolarFC(fit, fit_quantiles=False)), H=36, d_shade=0.9)
    r = simulate(ag, df, return_trace=True)
    tr = r.pop("trace")
    tr["w"] = score_weights(df)
    (HERE / "traces").mkdir(exist_ok=True)
    tr.to_csv(HERE / "traces" / f"{CONFIGS.index(name)}_{fold}.csv")
    return {"config": name, "fold": fold, "score": r["score"], "co2_pct": r.get("co2_pct"), "cost_pct": r.get("cost_pct"),
            "infractions": r.get("infractions"), "spill_kwh": float(tr["spill_kw"].sum())}


if __name__ == "__main__":
    names = sys.argv[1:] or CONFIGS
    rows = Parallel(n_jobs=3)(delayed(run)(c, f) for c in names for f in ("A", "B", "C"))
    out = pd.DataFrame(rows)
    p = HERE / "closed.csv"
    if p.exists() and sys.argv[1:]:
        old = pd.read_csv(p)
        out = pd.concat([old[~old.config.isin(names)], out])
    out.to_csv(p, index=False)
    piv = out.pivot(index="config", columns="fold", values="score").round(3)
    piv["mean"] = piv.mean(axis=1).round(3)
    piv["spill_MWh"] = (out.groupby("config")["spill_kwh"].mean() / 1000).round(1)
    print(piv.sort_values("mean", ascending=False).to_string())
