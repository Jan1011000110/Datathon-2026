# Usage: .venv/Scripts/python.exe experiments/20_hard_baseline/decompose_n.py
"""Which part of the net-load forecast costs score: the hour being decided (nowcast) or the
hours after it (profile)? Perfect price/CO2 signal throughout. Writes decompose_n.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hard import get_folds, simulate, train_complement  # noqa: E402
from src.hard_mpc import MPCHard, PerfectN, PerfectW, ProfileN  # noqa: E402


class MixN:
    """Hour 0 from one forecaster, hours 1.. from another."""
    def __init__(self, first, rest, k_first=1):
        self.first, self.rest, self.k = first, rest, k_first

    def predict(self, obs, hist, H):
        a = np.asarray(self.first.predict(obs, hist, H), float)
        b = np.asarray(self.rest.predict(obs, hist, H), float).copy()
        m = min(len(a), len(b))
        b = b[:m]
        b[:self.k] = a[:self.k]
        return b


class PersistN:
    """Net load of every future hour = last hour's net load (what the standard agent assumed)."""
    def predict(self, obs, hist, H):
        last = obs["noves_dades_5min"]
        return np.full(H, float(last["ultim_consum_kw"]) - float(last["ultima_solar_kw"]))


def run(name, fold):
    df = get_folds()[fold]
    fit = train_complement(df)
    P, F = PerfectN(df), ProfileN(fit)
    n_fc = {"perfect hour 0, profile after": MixN(P, F, 1),
            "perfect hours 0-2, profile after": MixN(P, F, 3),
            "profile hour 0, perfect after": MixN(F, P, 1),
            "persistence hour 0, perfect after": MixN(PersistN(), P, 1),
            "profile everywhere": F,
            "perfect everywhere": P}[name]
    r = simulate(MPCHard(PerfectW(df), n_fc, H=36), df, return_trace=True)
    tr = r.pop("trace")
    return {"config": name, "fold": fold, "score": r["score"], "spill_kwh": float(tr["spill_kw"].sum())}


if __name__ == "__main__":
    names = ["perfect everywhere", "perfect hour 0, profile after", "perfect hours 0-2, profile after",
             "profile hour 0, perfect after", "persistence hour 0, perfect after", "profile everywhere"]
    rows = Parallel(n_jobs=10)(delayed(run)(c, f) for c in names for f in ("A", "B", "C"))
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).with_name("decompose_n.csv"), index=False)
    piv = out.pivot(index="config", columns="fold", values="score").round(3)
    piv["mean"] = piv.mean(axis=1).round(3)
    piv["spill_MWh"] = (out.groupby("config")["spill_kwh"].mean() / 1000).round(1)
    print(piv.sort_values("mean", ascending=False).to_string())
