"""How leaky is a 'first half of each week = public' split of one continuous run?

We compare three controllers on test-like 12-week runs (one train week per month):
  * honest   : v1, optimises every hour the same way (what we submitted)
  * favour-A : same optimiser, but treats grid import in half B as free
  * favour-B : same optimiser, but treats grid import in half A as free
Half A = first 84 h of each week, half B = last 84 h. Scores are computed the way the official
`evaluate_results(mask=...)` does: one simulation, metrics restricted to the masked hours.

Purpose: quantify the flaw, not to build a submission. An agent would need to know the split
(it can read the calendar from the timestamps) and would be trading real performance for score.
Run from the repo root:  .venv/Scripts/python.exe experiments/13_fault_luck/split_leak.py
"""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.sim import get_folds, simulate, score_from_grid

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


class Favour(v1.BatteryAgent):
    """v1 whose signal is set to ~0 in the hours of the half it does not care about."""

    def __init__(self, free_half):
        super().__init__()
        self.free_half = free_half        # "A", "B" or None
        self.t = 0

    def _decide(self, obs):
        soc = float(obs["soc_kwh"])
        w = np.asarray(obs["co2_properes_24h"], float) + v1.KAPPA * np.asarray(obs["preus_properes_24h"], float)
        if self.free_half is not None:
            hw = (self.t + np.arange(len(w))) % 168
            free = (hw >= 84) if self.free_half == "B" else (hw < 84)
            w = np.where(free, 1e-6, w)
        self.t += 1
        d_cap, nb = self._discharge_limits(obs, len(w))
        return v1.first_step(soc, v1.marginal_values(w, self.v_end, nb), w[0], d_cap)


allf = get_folds("filled")
rows = []
for fold in ("A", "B"):                       # clean folds
    df = allf[fold]
    n = len(df)
    hw = np.arange(n) % 168
    A, B = hw < 84, hw >= 84
    load, pv = df.load_kw.to_numpy(), df.pv_kw.to_numpy()
    price, co2 = df.precio_eur_per_kwh.to_numpy(), df.co2_kg_per_kwh.to_numpy()
    for name, free in (("honest (v1)", None), ("favours half A (first 84 h)", "B"), ("favours half B (last 84 h)", "A")):
        r = simulate(Favour(free), df, return_trace=True)
        g = r["trace"].grid_kw.to_numpy()
        sc = lambda m=None: score_from_grid(g, load, pv, price, co2, mask=m)["score"]
        rows.append({"fold": fold, "controller": name, "score on half A": sc(A), "score on half B": sc(B),
                     "score on all hours": sc(), "infractions": r["infractions"]})
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).parent / "split_leak.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
