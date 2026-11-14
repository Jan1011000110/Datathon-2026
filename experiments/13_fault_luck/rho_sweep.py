"""Can a different CO2/price weighting 'get lucky' around a meter-fault hour?
Sweep the weighting (rho) on a run that contains the 6 Aug partial fault (fold A) and on a
clean run (fold B). Report score minus the score at rho = 0.60 (v1), and the energy exported
(= wasted: discharged into a load that was not there).
Run from the repo root:  .venv/Scripts/python.exe experiments/13_fault_luck/rho_sweep.py"""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate, oracle

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

def make_agent(rho):
    a = v1.BatteryAgent(); kappa = (0.3 / 0.7) * rho
    def decide(obs, a=a, kappa=kappa):
        w = np.asarray(obs["co2_properes_24h"], float) + kappa * np.asarray(obs["preus_properes_24h"], float)
        d_cap, nb = a._discharge_limits(obs, len(w))
        return v1.first_step(float(obs["soc_kwh"]), v1.marginal_values(w, a.v_end, nb), w[0], d_cap)
    a._decide = decide
    return a

folds = get_folds("filled")
A = folds["A"]; hw = np.arange(len(A)) % 168
runs = {"A, full weeks (has 6 Aug fault)": A, "A, last 84 h of each week (has fault)": A[hw >= 84],
        "B, full weeks (clean)": folds["B"]}
rhos = [0.30, 0.40, 0.45, 0.50, 0.52, 0.54, 0.56, 0.58, 0.59, 0.60, 0.61, 0.62, 0.64, 0.66, 0.68, 0.70, 0.75, 0.80, 0.90, 1.00, 1.20]
rows = []
for name, df in runs.items():
    net = (df.load_kw - df.pv_kw).to_numpy()
    base = simulate(make_agent(0.60), df)["score"]; orc = oracle(df)["score"]
    for rho in rhos:
        r = simulate(make_agent(rho), df, return_trace=True)
        exported = float(np.maximum(-(net + r["trace"].p_kw.to_numpy()), 0).sum())
        rows.append({"run": name, "rho": rho, "score - v1": r["score"] - base, "exported kWh": exported,
                     "co2 kg": r["co2_saved_kg"], "eur": r["cost_saved_eur"]})
    print(f"{name}: v1 {base:.4f}, oracle {orc:.4f} (oracle - v1 = {orc - base:+.4f})")
out = pd.DataFrame(rows); out.to_csv(Path(__file__).parent / "rho_sweep.csv", index=False)
piv = out.pivot(index="rho", columns="run", values="score - v1").round(4)
exp = out.pivot(index="rho", columns="run", values="exported kWh").round(0)
pd.set_option("display.width", 250)
print("\nscore minus v1's score (rho = 0.60):"); print(piv.to_string())
print("\nenergy exported / wasted (kWh):"); print(exp.to_string())
