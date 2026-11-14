"""Count what v1's two safety nets actually do during a run (fold A, 2,016 hours)."""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.sim import get_folds, simulate

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
CAP, PMAX, ETA, EPS = v1.CAP, v1.PMAX, v1.ETA, v1.EPS

class Counting(v1.BatteryAgent):
    def __init__(self):
        super().__init__()
        self.n = 0; self.trimmed = 0; self.kwh_withheld = 0.0; self.guard_active = 0; self.guard_changed = 0
        self.moved_kwh = 0.0
    def _discharge_limits(self, obs, horizon):
        cap, nb = super()._discharge_limits(obs, horizon)
        self._cap, self._nb = cap, nb
        if cap < PMAX or nb > 0:
            self.guard_active += 1
        return cap, nb
    def act(self, obs):
        self.n += 1
        soc = float(obs["soc_kwh"])
        p_plan = float(self._decide(obs))                      # what the optimiser wants (guard included)
        # what it would want with the guard off
        co2 = np.asarray(obs["co2_properes_24h"], float); price = np.asarray(obs["preus_properes_24h"], float)
        w = co2 + v1.KAPPA * price
        p_free = v1.first_step(soc, v1.marginal_values(w, self.v_end, 0), w[0], PMAX)
        if abs(p_free - p_plan) > 1e-9:
            self.guard_changed += 1
        max_c = max(0.0, min(PMAX, (CAP - soc) / ETA - EPS)); max_d = max(0.0, min(PMAX, soc * ETA - EPS))
        p = min(max(p_plan, -max_d), max_c)
        if abs(p - p_plan) > 0:
            self.trimmed += 1; self.kwh_withheld += abs(p - p_plan)
        self.moved_kwh += abs(p)
        return float(p)

for fold in ("A", "C"):
    a = Counting(); r = simulate(a, get_folds("filled")[fold])
    print(f"fold {fold}: {a.n} hours, score {r['score']:.5f}, infractions {r['infractions']}")
    print(f"   clamp margin: trimmed the command in {a.trimmed} hours, by {EPS} kW each time at most;"
          f" total energy withheld over the whole run = {a.kwh_withheld:.4f} kWh"
          f" (the battery moved {a.moved_kwh:,.0f} kWh in total -> {100 * a.kwh_withheld / a.moved_kwh:.6f} %)")
    print(f"   load guard:   was 'armed' (last net load < 526 kW) in {a.guard_active} hours; changed the action in {a.guard_changed} hours")
