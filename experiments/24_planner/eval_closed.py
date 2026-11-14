# Usage: .venv/Scripts/python.exe experiments/24_planner/eval_closed.py [set]
"""Closed-loop evaluation of the planner on folds A, B, C (src.hard.simulate).
Forecasts do not depend on what the battery does, so each fold computes them once and every
configuration reuses them. Writes closed_<set>.csv (one row per configuration and fold)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import get_folds, simulate, train_complement  # noqa: E402
from src.hard_mpc import ClimW, PerfectN, PerfectW, ProfileN  # noqa: E402
import planner as P  # noqa: E402

HMAX = 48


class Cached:
    """Memoise predict() by step of the run (forecasts are prefix-consistent in H)."""
    def __init__(self, fc):
        self.fc, self.c = fc, {}

    def predict(self, obs, hist, H):
        i = len(hist)
        if i not in self.c:
            self.c[i] = np.asarray(self.fc.predict(obs, hist, HMAX), float)
        return self.c[i][:H]


def configs(which):
    """(label, w forecaster, n forecaster, agent kwargs)"""
    out = []
    if which == "main":
        out += [("perfect w + perfect n | det", "PW", "PN", dict(scen_hours=0)),
                ("perfect w + profile n | det", "PW", "QN", dict(scen_hours=0)),
                ("clim w + profile n | det", "CW", "QN", dict(scen_hours=0))]
        for wf in ("PW", "CW"):
            wl = "perfect w" if wf == "PW" else "clim w"
            for sh in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
                out.append((f"{wl} + profile n | det, d_shade={sh:.2f}", wf, "QN", dict(scen_hours=0, d_shade=sh)))
            for Q in (3, 5, 9, 20, 40):
                out.append((f"{wl} + profile n | stoch h0, Q={Q}", wf, "QN", dict(scen_hours=1, Q=Q)))
            for Q in (3, 5, 9, 20):
                out.append((f"{wl} + profile n | stoch h0-5, Q={Q}", wf, "QN", dict(scen_hours=6, Q=Q)))
                out.append((f"{wl} + profile n | stoch all, Q={Q}", wf, "QN", dict(scen_hours=None, Q=Q)))
            out.append((f"{wl} + profile n | stoch all, Q=9, nowcast spread", wf, "QN",
                        dict(scen_hours=None, Q=9, flat_horizon=0)))
            out.append((f"{wl} + profile n | stoch h0-5, Q=9, nowcast spread", wf, "QN",
                        dict(scen_hours=6, Q=9, flat_horizon=0)))
    if which == "sens":
        for wf in ("PW", "CW"):
            wl = "perfect w" if wf == "PW" else "clim w"
            for H in (24, 36, 48):
                out.append((f"{wl} + profile n | det, H={H}", wf, "QN", dict(scen_hours=0, H=H)))
                out.append((f"{wl} + profile n | stoch h0, Q=20, H={H}", wf, "QN", dict(scen_hours=1, Q=20, H=H)))
                out.append((f"{wl} + profile n | stoch all, Q=9, H={H}", wf, "QN", dict(scen_hours=None, Q=9, H=H)))
            for ve in (0.25, 0.5, 0.75, 1.0):
                for H in (24, 36):
                    out.append((f"{wl} + profile n | stoch h0, Q=20, H={H}, v_end={ve:.2f}*mean(w)*eta", wf, "QN",
                                dict(scen_hours=1, Q=20, H=H, v_end_rel=ve)))
        for H in (24, 36, 48):
            out.append((f"perfect w + perfect n | det, H={H}", "PW", "PN", dict(scen_hours=0, H=H)))
    if which == "ctrl":
        for wf in ("PW", "CW"):
            wl = "perfect w" if wf == "PW" else "clim w"
            for sh in (0.925, 0.975):
                out.append((f"{wl} + profile n | det, d_shade={sh:.3f}", wf, "QN", dict(scen_hours=0, d_shade=sh)))
            # Q = 1: the single scenario is the point forecast + mean residual = bias correction only
            out.append((f"{wl} + profile n | bias-corrected point, h0 (Q=1)", wf, "QN", dict(scen_hours=1, Q=1)))
            out.append((f"{wl} + profile n | bias-corrected point, all (Q=1)", wf, "QN", dict(scen_hours=None, Q=1)))
            for sp in (0.5, 0.75, 1.25, 1.5, 2.0):
                out.append((f"{wl} + profile n | stoch all, Q=9, spread x{sp:.2f}", wf, "QN",
                            dict(scen_hours=None, Q=9, spread=sp)))
            for K in (2, 3, 12, 24):
                out.append((f"{wl} + profile n | stoch h0-{K - 1}, Q=9", wf, "QN", dict(scen_hours=K, Q=9)))
            out.append((f"{wl} + profile n | stoch all, Q=60", wf, "QN", dict(scen_hours=None, Q=60)))
    return out


def run_fold(fold, which):
    df = get_folds()[fold]
    fit = train_complement(df)
    qn = P.QuantileN(fit, H=HMAX)
    qn.point = Cached(qn.point)
    fcs = {"PW": PerfectW(df), "PN": PerfectN(df), "CW": Cached(ClimW(fit)), "QN": qn}
    w_mean = float((fit["co2_kg_per_kwh"] + P.KAPPA * fit["precio_eur_per_kwh"]).mean())
    rows = []
    for label, wf, nf, kw in configs(which):
        kw = dict(kw)
        ve = kw.pop("v_end_rel", None)
        if ve is not None:
            kw["v_end"] = ve * w_mean * P.ETA
        ag = P.MPCStoch(fcs[wf], fcs[nf], **kw)
        t0 = time.perf_counter()
        r = simulate(ag, df, return_trace=True)
        dt = time.perf_counter() - t0
        tr = r.pop("trace")
        rows.append({"config": label, "fold": fold, "score": r["score"], "co2_pct": r["co2_pct"],
                     "cost_pct": r["cost_pct"], "infractions": r["infractions"],
                     "spill_mwh": float(tr["spill_kw"].sum()) / 1000.0,
                     "cycles": float(np.abs(tr["soc_kwh"].diff()).sum() / 2 / 6000.0),
                     "seconds": dt})
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    res = Parallel(n_jobs=3)(delayed(run_fold)(f, which) for f in ("A", "B", "C"))
    out = pd.DataFrame([r for rr in res for r in rr])
    out.to_csv(Path(__file__).with_name(f"closed_{which}.csv"), index=False)
    piv = out.pivot(index="config", columns="fold", values="score")
    piv["mean"] = piv.mean(axis=1)
    piv["spill_MWh"] = out.groupby("config")["spill_mwh"].mean()
    piv["infr"] = out.groupby("config")["infractions"].sum()
    piv["sec"] = out.groupby("config")["seconds"].mean()
    order = [c[0] for c in configs(which)]
    pd.set_option("display.width", 250)
    print(piv.loc[order].round(3).to_string())
