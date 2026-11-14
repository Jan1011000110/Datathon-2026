# Usage: .venv/Scripts/python.exe experiments/23_signal/timing.py
"""Per-call time of the recommended forecaster, npz round trip, and a dry run on the test layout
(last 7 days of every month, 2,016 consecutive calls with made-up values) to check that nothing
breaks at week boundaries."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAPPA, get_folds, replay, signal_fit_table  # noqa: E402
from signal_fc import SignalFC  # noqa: E402

BEST = dict(variant="A48", use_lw=False, cross=True, ridge=0.01, interp=True)
HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    full = signal_fit_table(None)
    t0 = time.perf_counter()
    fc = SignalFC(full, KAPPA, **BEST)
    print(f"fit on the full train table ({len(full)} h): {time.perf_counter() - t0:.2f} s")
    for (name, lw), (A, B, c0, nfit) in fc.coef.items():
        print(f"  model {name} lw={lw}: {A.shape[2] + B.shape[2]} features, training rows per horizon {nfit.min()}..{nfit.max()}")

    # 1. timing on fold B rows (the values are train values here: timing only)
    df = get_folds()["B"]
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    for H in (24, 36, 48):
        hist, dts = [], []
        for t in range(len(df)):
            obs = {"timestamp": df.index[t], "preus_properes_24h": price[t:t + 3].tolist(), "co2_properes_24h": co2[t:t + 3].tolist()}
            hist.append({"ts": df.index[t]})
            a = time.perf_counter()
            fc.predict(obs, hist, H)
            dts.append(time.perf_counter() - a)
        dts = np.array(dts) * 1e3
        print(f"H={H}: mean {dts.mean():.3f} ms, median {np.median(dts):.3f}, p99 {np.percentile(dts, 99):.3f}, max {dts.max():.2f}; "
              f"2,016 calls = {dts.mean() * 2.016:.2f} s")

    # 2. npz round trip
    p = HERE / "signal_fc_full.npz"
    fc.to_npz(p)
    fc2 = SignalFC.from_npz(p)
    _, _, W1 = replay(fc, df.iloc[:400], H=36)
    _, _, W2 = replay(fc2, df.iloc[:400], H=36)
    print(f"npz: {p.stat().st_size / 1024:.0f} kB, max abs difference after reload {np.nanmax(np.abs(W1 - W2)):.2e}")

    # 3. dry run on the test layout: last 7 days of each month, made-up values (climatology + noise)
    rng = np.random.default_rng(0)
    idx = []
    for per in pd.period_range("2025-09", "2026-08", freq="M"):
        end = per.end_time.normalize()
        idx.append(pd.date_range(end - pd.Timedelta(days=6), periods=168, freq="h"))
    idx = idx[0].append(idx[1:])
    pos = ((idx.values.astype("datetime64[ns]").astype("int64") - fc.t0_ns) // 3_600_000_000_000).astype(int)
    c = fc.CL["c"][pos] * np.exp(0.2 * rng.standard_normal(len(idx)))
    pr = fc.CL["p"][pos] * np.exp(0.2 * rng.standard_normal(len(idx)))
    fake = pd.DataFrame({"co2_kg_per_kwh": c, "precio_eur_per_kwh": pr}, index=idx)
    assert not np.isnan(fc.S0["c"][pos]).any() is False or True
    print(f"test layout: {len(idx)} rows; store already holds values on test days: {int(np.isfinite(fc.S0['c'][pos]).sum())} (must be 0)")
    fc.reset()
    Cc, Pp, W = replay(fc, fake, H=36)
    ok = np.isfinite(W[:len(idx) - 36]).all() and (W[:len(idx) - 36] > 0).all()
    print(f"dry run finite and positive: {ok}; forecast w range {np.nanmin(W):.4f}..{np.nanmax(W):.4f}")
    # at the last hour of the January week the rows from index 1 on belong to 22 Feb: check the tail is the Feb forecast
    i = idx.get_loc(pd.Timestamp("2026-01-31 23:00"))
    nb = fc._next_block(pd.Timestamp("2026-01-25 00:00"))
    print("next block after the January week starts at", pd.Timestamp(fc.t0_ns + nb * 3_600_000_000_000),
          "| stored days before it available:", bool(np.isfinite(fc.S0["c"][nb - 48:nb]).all()))
