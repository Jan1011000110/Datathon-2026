# Usage: .venv/Scripts/python.exe experiments/23_signal/closed_loop.py <config> [<config> ...]
"""Closed-loop score of signal forecasters on folds A, B, C (3 workers).
  n = perfect : MPCHard(w_fc, PerfectN(df))                    isolates the signal (refs 20.47 / 21.26)
  n = profile : MPCHard(w_fc, ProfileN(fit), d_shade=0.9)      fully legal (ref 18.96)
Appends to closed_loop.csv and saves per-hour traces under traces/ when asked."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAPPA, get_folds, signal_fit_table  # noqa: E402
from signal_fc import SignalFC  # noqa: E402
from src.hard import simulate, train_complement  # noqa: E402
from src.hard_mpc import ClimW, MPCHard, PerfectN, PerfectW, ProfileN  # noqa: E402

HERE = Path(__file__).resolve().parent


class Hybrid:
    """Diagnostic: the true signal for the first K hours, the forecaster beyond."""
    def __init__(self, fc, df, K):
        self.fc, self.pw, self.K = fc, PerfectW(df), K

    def predict(self, obs, hist, H):
        w = np.asarray(self.fc.predict(obs, hist, H), float).copy()
        t = self.pw.predict(obs, hist, H)
        k = min(self.K, len(t), len(w))
        w[:k] = t[:k]
        return w


class Inflate:
    """Stretch the forecast hours around the mean of the next 24 h by a factor gamma, shift by `shift` (relative)."""
    def __init__(self, fc, gamma=1.0, shift=0.0):
        self.fc, self.gamma, self.shift = fc, gamma, shift

    def predict(self, obs, hist, H):
        w = np.asarray(self.fc.predict(obs, hist, H), float).copy()
        m = w[:24].mean()
        w[3:] = np.maximum(m + self.gamma * (w[3:] - m), 1e-4) * (1.0 + self.shift)
        return w

BEST = dict(variant="A48", use_lw=False, cross=True, ridge=0.01)      # recommended configuration

FINAL = dict(BEST, interp=True)                                         # + month-interpolated climatology

# name: (w forecaster, kwargs, n forecaster, H, lam, trace)
CONFIGS = {
    "FINAL|perfectN|H36": ("fc", FINAL, "perfect", 36, 0.0, True),
    "FINAL|perfectN|H18": ("fc", FINAL, "perfect", 18, 0.0, False),
    "FINAL|perfectN|H24": ("fc", FINAL, "perfect", 24, 0.0, False),
    "FINAL|profileN|H36": ("fc", FINAL, "profile", 36, 0.0, False),
    "FINAL|profileN|H24": ("fc", FINAL, "profile", 24, 0.0, False),
    "FINAL lw|perfectN|H36": ("fc", dict(FINAL, use_lw=True), "perfect", 36, 0.0, False),
    "FINAL nohol|perfectN|H36": ("fc", dict(FINAL, holidays=False), "perfect", 36, 0.0, False),
    "BEST|perfectN|H36": ("fc", BEST, "perfect", 36, 0.0, True),
    "BEST|perfectN|H12": ("fc", BEST, "perfect", 12, 0.0, False),
    "BEST|perfectN|H18": ("fc", BEST, "perfect", 18, 0.0, False),
    "BEST|perfectN|H24": ("fc", BEST, "perfect", 24, 0.0, False),
    "BEST|perfectN|H48": ("fc", BEST, "perfect", 48, 0.0, False),
    "BEST|perfectN|H36|lam.06": ("fc", BEST, "perfect", 36, 0.06, False),
    "BEST|perfectN|H24|lam.06": ("fc", BEST, "perfect", 24, 0.06, False),
    "BEST|perfectN|H24|lam.09": ("fc", BEST, "perfect", 24, 0.09, False),
    "BEST|perfectN|H24|lam.03": ("fc", BEST, "perfect", 24, 0.03, False),
    "BEST|profileN|H36": ("fc", BEST, "profile", 36, 0.0, False),
    "BEST|profileN|H24": ("fc", BEST, "profile", 24, 0.0, False),
    "BEST cold12|perfectN|H36": ("fc", dict(BEST, cold_min=12), "perfect", 36, 0.0, False),
    "BEST cold24|perfectN|H36": ("fc", dict(BEST, cold_min=24), "perfect", 36, 0.0, False),
    "BEST cold4|perfectN|H36": ("fc", dict(BEST, cold_min=4), "perfect", 36, 0.0, False),
    "BEST interp|perfectN|H36": ("fc", dict(BEST, interp=True), "perfect", 36, 0.0, False),
    "BEST lw|perfectN|H36": ("fc", dict(BEST, use_lw=True), "perfect", 36, 0.0, False),
    "BEST g1.1|perfectN|H36": ("fc", dict(BEST, _gamma=1.1), "perfect", 36, 0.0, False),
    "BEST+truth<6|perfectN|H36": ("fc", dict(BEST, _hyb=6), "perfect", 36, 0.0, False),
    "perfect|perfectN|H36": ("perfect", {}, "perfect", 36, 0.0, True),
    "clim|perfectN|H36": ("clim", {}, "perfect", 36, 0.0, True),
    "L48|perfectN|H36": ("fc", dict(variant="L48"), "perfect", 36, 0.0, True),
    "L48nolw|perfectN|H36": ("fc", dict(variant="L48", use_lw=False), "perfect", 36, 0.0, False),
    "compact|perfectN|H36": ("fc", dict(variant="compact"), "perfect", 36, 0.0, False),
    "L48cross|perfectN|H36": ("fc", dict(variant="L48", cross=True), "perfect", 36, 0.0, False),
    "L48r0.3|perfectN|H36": ("fc", dict(variant="L48", ridge=0.3), "perfect", 36, 0.0, False),
    "L48r0.003|perfectN|H36": ("fc", dict(variant="L48", ridge=0.003), "perfect", 36, 0.0, False),
    "L48|perfectN|H24": ("fc", dict(variant="L48"), "perfect", 24, 0.0, False),
    "L48|perfectN|H48": ("fc", dict(variant="L48"), "perfect", 48, 0.0, False),
    "L48|perfectN|H24|lam.05": ("fc", dict(variant="L48"), "perfect", 24, 0.05, False),
    "L48|perfectN|H36|lam.05": ("fc", dict(variant="L48"), "perfect", 36, 0.05, False),
    "L48|perfectN|H36|lam.08": ("fc", dict(variant="L48"), "perfect", 36, 0.08, False),
    "L48|perfectN|H48|lam.05": ("fc", dict(variant="L48"), "perfect", 48, 0.05, False),
    "L48+truth<6|perfectN|H36": ("fc", dict(variant="L48", _hyb=6), "perfect", 36, 0.0, False),
    "L48+truth<12|perfectN|H36": ("fc", dict(variant="L48", _hyb=12), "perfect", 36, 0.0, False),
    "clim+truth<12|perfectN|H36": ("clim", dict(_hyb=12), "perfect", 36, 0.0, False),
    "L48nolw g1.2|perfectN|H36": ("fc", dict(variant="L48", use_lw=False, _gamma=1.2), "perfect", 36, 0.0, False),
    "L48nolw g1.4|perfectN|H36": ("fc", dict(variant="L48", use_lw=False, _gamma=1.4), "perfect", 36, 0.0, False),
    "L48nolw s+.05|perfectN|H36": ("fc", dict(variant="L48", use_lw=False, _shift=0.05), "perfect", 36, 0.0, False),
    "L48nolw s-.05|perfectN|H36": ("fc", dict(variant="L48", use_lw=False, _shift=-0.05), "perfect", 36, 0.0, False),
    "A48|perfectN|H36": ("fc", dict(variant="A48"), "perfect", 36, 0.0, True),
    "A48nolw|perfectN|H36": ("fc", dict(variant="A48", use_lw=False), "perfect", 36, 0.0, False),
    "A24|perfectN|H36": ("fc", dict(variant="A24"), "perfect", 36, 0.0, False),
    "A24nolw|perfectN|H36": ("fc", dict(variant="A24", use_lw=False), "perfect", 36, 0.0, False),
    "A48r0.3|perfectN|H36": ("fc", dict(variant="A48", ridge=0.3), "perfect", 36, 0.0, False),
    "A48r0.1|perfectN|H36": ("fc", dict(variant="A48", ridge=0.1), "perfect", 36, 0.0, False),
    "AH8nolw|perfectN|H36": ("fc", dict(variant="AH8", use_lw=False), "perfect", 36, 0.0, False),
    "AH8nolw r0.1|perfectN|H36": ("fc", dict(variant="AH8", use_lw=False, ridge=0.1), "perfect", 36, 0.0, False),
    "AH4nolw|perfectN|H36": ("fc", dict(variant="AH4", use_lw=False), "perfect", 36, 0.0, False),
    "AH8|perfectN|H36": ("fc", dict(variant="AH8"), "perfect", 36, 0.0, False),
    "A48nolw nohol|perfectN|H36": ("fc", dict(variant="A48", use_lw=False, holidays=False), "perfect", 36, 0.0, True),
    "A48nolw r0.1|perfectN|H36": ("fc", dict(variant="A48", use_lw=False, ridge=0.1), "perfect", 36, 0.0, False),
    "A48nolw r0.01|perfectN|H36": ("fc", dict(variant="A48", use_lw=False, ridge=0.01), "perfect", 36, 0.0, False),
    "A48nolw cross|perfectN|H36": ("fc", dict(variant="A48", use_lw=False, cross=True), "perfect", 36, 0.0, False),
    "A48nolw g1.2|perfectN|H36": ("fc", dict(variant="A48", use_lw=False, _gamma=1.2), "perfect", 36, 0.0, False),
    "A48nolw g0.85|perfectN|H36": ("fc", dict(variant="A48", use_lw=False, _gamma=0.85), "perfect", 36, 0.0, False),
    "A48nolw|perfectN|H24": ("fc", dict(variant="A48", use_lw=False), "perfect", 24, 0.0, False),
    "A48nolw|perfectN|H48": ("fc", dict(variant="A48", use_lw=False), "perfect", 48, 0.0, False),
    "A48nolw|perfectN|H36|lam.06": ("fc", dict(variant="A48", use_lw=False), "perfect", 36, 0.06, False),
    "A48nolw|perfectN|H24|lam.06": ("fc", dict(variant="A48", use_lw=False), "perfect", 24, 0.06, False),
    "A48nolw|perfectN|H24|lam.09": ("fc", dict(variant="A48", use_lw=False), "perfect", 24, 0.09, False),
    "A48nolw|perfectN|H18": ("fc", dict(variant="A48", use_lw=False), "perfect", 18, 0.0, False),
    "A48nolw|profileN|H36": ("fc", dict(variant="A48", use_lw=False), "profile", 36, 0.0, False),
    "A48|profileN|H36": ("fc", dict(variant="A48"), "profile", 36, 0.0, False),
    "L48|profileN|H36": ("fc", dict(variant="L48"), "profile", 36, 0.0, False),
    "clim|profileN|H36": ("clim", {}, "profile", 36, 0.0, False),
    "perfect|profileN|H36": ("perfect", {}, "profile", 36, 0.0, False),
}


def run(name, fold):
    wk, kw, nk, H, lam, trace = CONFIGS[name]
    df = get_folds()[fold]
    fit_sig = signal_fit_table(fold)
    fit = train_complement(df)
    kw = dict(kw)
    hyb, gam, shift = kw.pop("_hyb", None), kw.pop("_gamma", None), kw.pop("_shift", 0.0)
    w_fc = PerfectW(df) if wk == "perfect" else ClimW(fit_sig) if wk == "clim" else SignalFC(fit_sig, KAPPA, **kw)
    if hyb:
        w_fc = Hybrid(w_fc, df, hyb)
    if gam or shift:
        w_fc = Inflate(w_fc, gam or 1.0, shift)
    if nk == "perfect":
        agent = MPCHard(w_fc, PerfectN(df), H=H, lam=lam)
    else:
        agent = MPCHard(w_fc, ProfileN(fit), H=H, lam=lam, d_shade=0.9)
    t0 = time.time()
    r = simulate(agent, df, return_trace=True)
    tr = r.pop("trace")
    if trace:
        (HERE / "traces").mkdir(exist_ok=True)
        tr.to_pickle(HERE / "traces" / f"{name.replace('|', '_')}_{fold}.pkl")
    return {"config": name, "fold": fold, "score": r["score"], "co2_pct": r["co2_pct"], "cost_pct": r["cost_pct"],
            "infractions": r["infractions"], "seconds": time.time() - t0}


if __name__ == "__main__":
    names = sys.argv[1:]
    jobs = [(c, f) for c in names for f in ("A", "B", "C")]
    rows = Parallel(n_jobs=3)(delayed(run)(c, f) for c, f in jobs)
    out = pd.DataFrame(rows)
    p = HERE / "closed_loop.csv"
    if p.exists():
        old = pd.read_csv(p)
        out = pd.concat([old[~old.config.isin(out.config)], out])
    out.to_csv(p, index=False)
    piv = out.pivot(index="config", columns="fold", values="score")
    piv["mean"] = piv.mean(axis=1)
    piv["mean_BC"] = piv[["B", "C"]].mean(axis=1)
    print(piv.round(3).sort_values("mean", ascending=False).to_string())
