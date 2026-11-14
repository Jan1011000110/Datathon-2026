"""Where does the oracle beat v1? Print the stretches where their state of charge differs,
with the per-stretch contribution to the score gap."""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.sim import get_folds, simulate, oracle, score_weights

spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

def analyse(df, label, show=3):
    tr = simulate(v1.BatteryAgent(), df, return_trace=True)["trace"]
    orc = oracle(df, return_trace=True)["trace"]
    w = score_weights(df) * 100                      # points per kWh of grid import
    gain = (tr.grid_kw.to_numpy() - orc.grid_kw.to_numpy()) * w      # oracle's gain per hour
    diff = np.abs(tr.soc_kwh.to_numpy() - orc.soc_kwh.to_numpy()) > 1.0
    # contiguous stretches where SOC differs
    idx = np.flatnonzero(diff)
    print(f"\n=== {label}: total gap {gain.sum():.4f} points; hours with different SOC: {diff.sum()} of {len(df)}")
    if len(idx) == 0:
        return
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]; ends = np.r_[idx[breaks], idx[-1]]
    segs = sorted([(gain[max(s - 1, 0):e + 2].sum(), s, e) for s, e in zip(starts, ends)], reverse=True)
    for g, s, e in segs[:show]:
        print(f"  stretch steps {s}..{e} ({df.index[s]} -> {df.index[e]}), length {e - s + 1} h, oracle gains {g:.4f} points")
        lo, hi = max(s - 3, 0), min(e + 4, len(df))
        kappa = v1.KAPPA
        t = pd.DataFrame({"w": (df.co2_kg_per_kwh + kappa * df.precio_eur_per_kwh).to_numpy()[lo:hi],
                          "p_v1": tr.p_kw.to_numpy()[lo:hi], "p_orc": orc.p_kw.to_numpy()[lo:hi],
                          "soc_v1": tr.soc_kwh.to_numpy()[lo:hi], "soc_orc": orc.soc_kwh.to_numpy()[lo:hi],
                          "net_load": (df.load_kw - df.pv_kw).to_numpy()[lo:hi]}, index=df.index[lo:hi])
        t.insert(0, "step", np.arange(lo, hi))
        print(t.round(3).to_string())

folds = get_folds("filled")
A = folds["A"]
hw = np.arange(len(A)) % 168
analyse(A, "fold A, full weeks", show=2)
analyse(A[hw >= 84], "fold A, last 84 h of each week", show=2)
