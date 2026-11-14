"""How many usable weeks does the hidden test have?
The battery's savings per hour depend only on the market signals (not on campus load), so the
plain optimum's savings on the public leaderboard (4,289.4 kg, 8,907.87 EUR on the public half)
can be compared with what train data predicts for 12 / 11 / 10 weeks. Baselines are compared too
(those do depend on load, e.g. holidays)."""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

df = get_folds("filled")["ALL"]
tr = simulate(v1.BatteryAgent(), df, return_trace=True)["trace"]
base = (df.load_kw - df.pv_kw).clip(lower=0)
d = pd.DataFrame({"co2_saved": (base - tr.grid_kw) * df.co2_kg_per_kwh, "eur_saved": (base - tr.grid_kw) * df.precio_eur_per_kwh,
                  "co2_base": base * df.co2_kg_per_kwh, "eur_base": base * df.precio_eur_per_kwh})
m = d.groupby([d.index.year, d.index.month]).mean()          # per-hour rates by month (train days)
m.index = ["%d-%02d" % k for k in m.index]
print((m * 168).round(0).rename(columns=lambda c: c + " /week").to_string())

PUB = {"co2_saved": 4289.4, "eur_saved": 8907.87, "co2_base": 71814.0, "eur_base": 118741.0}
print("\nPredicted FULL-test totals from train monthly rates vs 2 x public leaderboard:")
rows = []
for label, drop in (("12 weeks", []), ("11 weeks (no Nov)", ["2025-11"]), ("11 weeks (no Aug)", ["2026-08"]),
                    ("10 weeks (no Nov, no Aug)", ["2025-11", "2026-08"])):
    tot = (m.drop(index=drop) * 168).sum()
    rows.append(pd.Series({k: tot[k] / (2 * PUB[k]) for k in PUB}, name=label))
out = pd.DataFrame(rows)
print("ratio  predicted / (2 x public)   [1.00 = matches]")
print(out.round(3).to_string())
out.to_csv(Path(__file__).parent / "results.csv")
