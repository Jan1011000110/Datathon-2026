"""Round 2 helpers: forecasts do not depend on what the battery does, so a forecaster is replayed once
per fold into a tape (n rows x H hours) and the tape is scored in closed loop with the fast planner of
experiments/24_planner:
    score_pn   : deterministic plan, net load known (reference: SignalForecaster 20.69, perfect 21.26)
    score_comb : the combined legal agent of experiments/27_combined (reference 19.99)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "experiments" / "27_combined"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import combined as cb  # noqa: E402  (adds the other experiment folders to sys.path)
import planner as P  # noqa: E402
from common import KAPPA, get_folds, replay, signal_fit_table, truth  # noqa: E402
from signal_fc import SignalForecaster  # noqa: E402
from src.hard import simulate  # noqa: E402
from src.hard_mpc import PerfectN  # noqa: E402

HT = 36
CACHE = HERE / "cache2"
_F, _CT = {}, {}


def fold(f):
    if not _F:
        _F.update(get_folds())
    return _F[f]


class Tape:
    def __init__(self, W):
        self.W = W

    def predict(self, obs, hist, H):
        return self.W[len(hist) - 1, :H].copy()


def replay_full(fc, df, H=HT):
    """(co2, price, w) said by the forecaster at every row of df, H hours each. Unlike common.replay it
    keeps what the forecaster says beyond the end of the run, as an agent would receive it (no NaN)."""
    n = len(df)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    ts = df.index
    hist = []
    Cc, Pp = np.empty((n, H)), np.empty((n, H))
    for t in range(n):
        obs = {"timestamp": ts[t], "preu_actual": price[t], "co2_actual": co2[t],
               "preus_properes_24h": price[t:t + 3].tolist(), "co2_properes_24h": co2[t:t + 3].tolist(),
               "noves_dades_5min": {}}
        hist.append({"ts": ts[t]})
        Cc[t], Pp[t] = fc.predict_parts(obs, hist, H)
    return Cc, Pp, Cc + KAPPA * Pp


def base_tape(f, strict=False):
    """(co2, price) tapes of the round-1 forecaster on fold f, and the truth; cached."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"base_{f}{'_strict' if strict else ''}.npz"
    if not p.exists():
        df = fold(f)
        fit = signal_fit_table(f)
        if strict:
            fit = fit[fit.index.day <= 14]
        fc = SignalForecaster(fit, KAPPA)
        Cc, Pp, _ = replay_full(fc, df, H=HT)
        T = truth(df, H=HT)
        np.savez_compressed(p, Cc=Cc, Pp=Pp, Tc=T["c"], Tp=T["p"])
    z = np.load(p)
    return {k: z[k] for k in z.files}


def w_of(c, p):
    return c + KAPPA * p


def fill(W, base):
    """A tape must be finite everywhere the planner may read: fall back to the base forecast."""
    return np.where(np.isfinite(W), W, base)


def score_pn(W, f, H=36):
    df = fold(f)
    return simulate(P.MPCStoch(Tape(W), PerfectN(df), H=H, scen_hours=0), df)["score"]


def score_comb(W, f, **kw):
    assert np.isfinite(W).all(), "a tape must be finite everywhere (the combined agent plans past the end of the run)"
    df = fold(f)
    if f not in _CT:
        z = np.load(cb.CACHE / f"tape_{f}.npz")
        _CT[f] = {k: z[k] for k in z.files}
    ag = cb.make_agent(df, signal=Tape(W), load="fc", solar="fc", scen="conv", tape=_CT[f], **kw)
    return simulate(ag, df)["score"]


def both(W, f):
    return score_pn(W, f), score_comb(W, f)
