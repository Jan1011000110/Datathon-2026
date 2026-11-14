"""Do v1's safety nets cost score? Sweep the clamp margin EPS (kW left unused at the limits),
switch the load guard off, and try the starter kit's rounded-efficiency clamp.
Scores from the fast simulator; infractions also checked in the official environment.
Run from the repo root:  .venv/Scripts/python.exe experiments/11_safety_margin/run.py"""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate, simulate_official

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
CAP, PMAX, ETA = v1.CAP, v1.PMAX, v1.ETA

class Variant(v1.BatteryAgent):
    def __init__(self, eps=1e-4, guard=True, eta_clamp=ETA, soc_lo=0.0, soc_hi=CAP):
        super().__init__()
        self.eps, self.guard, self.eta_clamp, self.soc_lo, self.soc_hi = eps, guard, eta_clamp, soc_lo, soc_hi
    def _discharge_limits(self, obs, horizon):
        return super()._discharge_limits(obs, horizon) if self.guard else (PMAX, 0)
    def act(self, obs):
        try:
            p = float(self._decide(obs))
        except Exception:
            p = 0.0
        soc = float(obs["soc_kwh"])
        max_c = max(0.0, min(PMAX, (self.soc_hi - soc) / self.eta_clamp - self.eps))
        max_d = max(0.0, min(PMAX, (soc - self.soc_lo) * self.eta_clamp - self.eps))
        return float(min(max(p, -max_d), max_c))

variants = [
    ("v1 as submitted (margin 0.0001 kW)", dict(eps=1e-4)),
    ("no margin at all (0 kW)", dict(eps=0.0)),
    ("margin 0.000001 kW", dict(eps=1e-6)),
    ("margin 0.01 kW", dict(eps=1e-2)),
    ("margin 1 kW", dict(eps=1.0)),
    ("margin 10 kW", dict(eps=10.0)),
    ("keep battery between 1% and 99%", dict(eps=1e-4, soc_lo=20.0, soc_hi=1980.0)),
    ("keep battery between 5% and 95%", dict(eps=1e-4, soc_lo=100.0, soc_hi=1900.0)),
    ("starter-kit clamp (efficiency rounded to 0.9487, no margin)", dict(eps=0.0, eta_clamp=0.9487)),
    ("load guard switched off", dict(eps=1e-4, guard=False)),
]
folds = get_folds("filled")
rows = []
for name, kw in variants:
    r = {"variant": name}
    for f in ("A", "B", "C"):
        res = simulate(Variant(**kw), folds[f])
        r[f"score {f}"] = res["score"]; r[f"infr {f}"] = res["infractions"]
    off = simulate_official(Variant(**kw), folds["A"])
    r["official env A: score"] = off["score"]; r["official env A: infr"] = off["infraccions"]
    rows.append(r)
out = pd.DataFrame(rows)
base = out.iloc[0]
for f in "ABC":
    out[f"d{f} vs v1"] = out[f"score {f}"] - base[f"score {f}"]
out.to_csv(Path(__file__).parent / "results.csv", index=False)
pd.set_option("display.width", 250)
cols = ["variant", "score A", "dA vs v1", "infr A", "dB vs v1", "infr B", "dC vs v1", "infr C", "official env A: score", "official env A: infr"]
print(out[cols].round(5).to_string(index=False))
