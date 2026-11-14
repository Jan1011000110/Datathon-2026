# Usage: .venv/Scripts/python.exe experiments/23_signal/timing2.py
"""SignalForecaster2: per-call time, npz round trip, dry run on the test layout, bridge weights."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402
from common import KAPPA, signal_fit_table  # noqa: E402
from signal_fc import SignalForecaster  # noqa: E402
from signal_fc2 import SignalForecaster2  # noqa: E402

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    full = signal_fit_table(None)
    t0 = time.perf_counter()
    fc = SignalForecaster2(full, KAPPA)
    print(f"fit on the full train table: {time.perf_counter() - t0:.2f} s")
    for nm in "cp":
        r = fc.rho[nm]; r2 = fc.bridge_r2[nm]
        print(f"bridge weights {nm}: rho[k=3] at g=1,3,6,12,24: " + ", ".join(f"{r[3, g]:.2f}" for g in (1, 3, 6, 12, 24)) +
              " | rho[k=8]: " + ", ".join(f"{r[8, g]:.2f}" for g in (1, 3, 6, 12, 24)) +
              " | share of error variance removed at k=6: " + ", ".join(f"{100 * r2[6, g]:.0f}%" for g in (1, 3, 6, 12, 24)))
    df = fast.fold("B")
    price = df["precio_eur_per_kwh"].to_numpy(float); co2 = df["co2_kg_per_kwh"].to_numpy(float)
    for label, f in (("round 1", SignalForecaster(full, KAPPA)), ("round 2 (bridge)", fc)):
        hist, dts = [], []
        for t in range(len(df)):
            obs = {"timestamp": df.index[t], "preus_properes_24h": price[t:t + 3].tolist(), "co2_properes_24h": co2[t:t + 3].tolist()}
            hist.append({"ts": df.index[t]})
            a = time.perf_counter(); f.predict(obs, hist, 36); dts.append(time.perf_counter() - a)
        dts = np.array(dts) * 1e3
        print(f"{label}: mean {dts.mean():.3f} ms, median {np.median(dts):.3f}, p99 {np.percentile(dts, 99):.3f}, max {dts.max():.2f}")
    p = HERE / "signal_fc2_full.npz"
    fc.to_npz(p)
    fc2 = SignalForecaster2.from_npz(p)
    a = fast.replay_full(fc, df.iloc[:400]); b = fast.replay_full(fc2, df.iloc[:400])
    print(f"npz: {p.stat().st_size / 1024:.0f} kB, max abs difference after reload {np.abs(a[2] - b[2]).max():.2e}")
    # dry run on the test layout with made-up values
    rng = np.random.default_rng(0)
    idx = []
    for per in pd.period_range("2025-09", "2026-08", freq="M"):
        end = per.end_time.normalize()
        idx.append(pd.date_range(end - pd.Timedelta(days=6), periods=168, freq="h"))
    idx = idx[0].append(idx[1:])
    pos = ((idx.values.astype("datetime64[ns]").astype("int64") - fc.t0_ns) // 3_600_000_000_000).astype(int)
    fake = pd.DataFrame({"co2_kg_per_kwh": fc.CL["c"][pos] * np.exp(0.2 * rng.standard_normal(len(idx))),
                         "precio_eur_per_kwh": fc.CL["p"][pos] * np.exp(0.2 * rng.standard_normal(len(idx)))}, index=idx)
    fc.reset()
    base = SignalForecaster(full, KAPPA)
    W2 = fast.replay_full(fc, fake)[2]; W1 = fast.replay_full(base, fake)[2]
    changed = np.abs(W2 - W1).max(axis=1) > 1e-12
    hin = np.arange(len(idx)) % 168
    print(f"test layout dry run: finite {np.isfinite(W2).all()}, rows changed by the bridge {changed.sum()} "
          f"(hours of the week {hin[changed].min()}..{hin[changed].max()}), blocks with a bridge {len(set(np.arange(len(idx))[changed] // 168))} of 12")
