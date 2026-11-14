"""Variant of run.py: what if the public half is WHOLE WEEKS (6 of the 12)?  Then a cut only
happens at a calendar jump. Compare how much two near-identical weightings (rho 0.57 vs 0.60)
and two different-but-equally-good controllers (oracle vs v1) differ on the public part."""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate, oracle, score_from_grid

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

allf = get_folds("filled"); weeks = {}
for f in "ABC":
    df = allf[f]
    for i, (ym, g) in enumerate(df.groupby([df.index.year, df.index.month], sort=True)):
        weeks[(i, f)] = g
rng = np.random.default_rng(3)
rows = []
for it in range(120):
    pick = [rng.choice(list("ABC")) for _ in range(12)]
    df = pd.concat([weeks[(i, f)] for i, f in enumerate(pick)])
    if (df.load_kw < 50).any():
        continue
    n = len(df); wk = np.arange(n) // 168; hw = np.arange(n) % 168
    load, pv = df.load_kw.to_numpy(), df.pv_kw.to_numpy()
    price, co2 = df.precio_eur_per_kwh.to_numpy(), df.co2_kg_per_kwh.to_numpy()
    g = {"v1": simulate(make_agent(0.60), df, return_trace=True)["trace"].grid_kw.to_numpy(),
         "rho0.57": simulate(make_agent(0.57), df, return_trace=True)["trace"].grid_kw.to_numpy(),
         "oracle": oracle(df, return_trace=True)["trace"].grid_kw.to_numpy()}
    sc = lambda k, m=None: score_from_grid(g[k], load, pv, price, co2, mask=m)["score"]
    masks = {"whole test": None, "odd weeks": wk % 2 == 0, "even weeks": wk % 2 == 1,
             "first half of each week": hw < 84, "whole days, alternate": (hw // 24) % 2 == 0}
    r = {}
    for name, m in masks.items():
        r[f"{name} | rho0.57 - v1"] = sc("rho0.57", m) - sc("v1", m)
        r[f"{name} | oracle - v1"] = sc("oracle", m) - sc("v1", m)
    rows.append(r)
out = pd.DataFrame(rows); out.to_csv(Path(__file__).parent / "results_weekmask.csv", index=False)
print("runs:", len(out))
s = pd.DataFrame({"mean": out.mean(), "std": out.std(), "median |diff|": out.abs().median(),
                  "95th pct |diff|": out.abs().quantile(0.95), "max |diff|": out.abs().max()})
print(s.round(4).to_string())
