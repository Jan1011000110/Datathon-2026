# Usage: .venv/Scripts/python.exe experiments/22_solar/timing.py
"""Per-call time of SolarFC on fold B (1,848 steps, H=36): wall clock and CPU time of this process
(CPU time is not inflated by other jobs sharing the machine)."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, train_complement
from eval_offline import replay
from solar_fc import SolarFC

df = get_folds()["B"]
fit = train_complement(df)
t0 = time.perf_counter(); fc = SolarFC(fit); print(f"fit with quantile tables: {time.perf_counter() - t0:.2f} s")
t0 = time.perf_counter(); SolarFC(fit, fit_quantiles=False); print(f"fit without: {time.perf_counter() - t0:.2f} s")
steps = [(obs, hist[-1]) for t, obs, hist in replay(None, df)]

def loop(with_hist, quant):
    fc.reset()
    hist = []
    w0, c0 = time.perf_counter(), time.process_time()
    for obs, rec in steps:
        hist.append(rec)
        fc.predict_pv(obs, hist if with_hist else None, 36)
        if quant:
            fc.predict_pv_quantiles(obs, hist if with_hist else None, [0.1, 0.5, 0.7, 0.9])
    n = len(steps)
    return 1000 * (time.perf_counter() - w0) / n, 1000 * (time.process_time() - c0) / n

for rep in range(4):
    a, b, c = loop(True, False), loop(False, False), loop(True, True)
    print(f"ms/call (wall, cpu)  predict_pv with hist: {a[0]:.3f}, {a[1]:.3f} | obs only: {b[0]:.3f}, {b[1]:.3f} | "
          f"predict_pv + quantiles: {c[0]:.3f}, {c[1]:.3f}")
