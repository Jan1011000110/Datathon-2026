# Usage: .venv/Scripts/python.exe experiments/20_hard_baseline/decompose_parts.py
"""Is the net-load loss due to the load forecast or the solar forecast? Perfect price/CO2 signal.
Also: hour-0-only variants. Writes decompose_parts.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hard import get_folds, simulate, train_complement  # noqa: E402
from src.hard_mpc import MPCHard, NetFromParts, PerfectLoad, PerfectPV, PerfectW, ProfileN  # noqa: E402


def run(name, fold):
    df = get_folds()[fold]
    F = ProfileN(train_complement(df))
    n_fc = {"perfect load, profile solar": NetFromParts(PerfectLoad(df), F),
            "profile load, perfect solar": NetFromParts(F, PerfectPV(df)),
            "profile both": F}[name]
    r = simulate(MPCHard(PerfectW(df), n_fc, H=36), df, return_trace=True)
    tr = r.pop("trace")
    return {"config": name, "fold": fold, "score": r["score"], "spill_kwh": float(tr["spill_kw"].sum())}


def errors(fold):
    """1-step-ahead error of the profile forecast, by part, in kW."""
    df = get_folds()[fold]
    F = ProfileN(train_complement(df))
    rows = []
    load, pv, ts = df["load_kw"].to_numpy(), df["pv_kw"].to_numpy(), df.index
    for t in range(1, len(df)):
        obs = {"timestamp": ts[t], "noves_dades_5min": {"ultim_consum_kw": load[t - 1], "ultima_solar_kw": pv[t - 1]}}
        rows.append((ts[t].hour, F.predict_load(obs, None, 1)[0] - load[t], F.predict_pv(obs, None, 1)[0] - pv[t],
                     load[t - 1] - load[t], pv[t - 1] - pv[t]))
    e = pd.DataFrame(rows, columns=["hour", "load_profile", "pv_profile", "load_persist", "pv_persist"])
    return e


if __name__ == "__main__":
    names = ["perfect load, profile solar", "profile load, perfect solar", "profile both"]
    rows = Parallel(n_jobs=9)(delayed(run)(c, f) for c in names for f in ("A", "B", "C"))
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).with_name("decompose_parts.csv"), index=False)
    piv = out.pivot(index="config", columns="fold", values="score").round(3)
    piv["mean"] = piv.mean(axis=1).round(3)
    piv["spill_MWh"] = (out.groupby("config")["spill_kwh"].mean() / 1000).round(1)
    print(piv.sort_values("mean", ascending=False).to_string())
    e = pd.concat([errors(f) for f in ("B", "C")])
    print("1-step-ahead MAE (kW), folds B+C:", e.drop(columns="hour").abs().mean().round(1).to_dict())
    cols = ["load_profile", "pv_profile", "load_persist", "pv_persist"]
    day = (e.hour >= 8) & (e.hour <= 19)
    print("  daytime 08-19h:", e.loc[day, cols].abs().mean().round(1).to_dict())
    print("  night   20-07h:", e.loc[~day, cols].abs().mean().round(1).to_dict())
    print(e.groupby("hour")[["load_profile", "pv_profile", "load_persist", "pv_persist"]].apply(lambda g: g.abs().mean()).round(0).T.to_string())
