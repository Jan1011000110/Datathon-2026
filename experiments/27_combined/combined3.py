"""
experiments/27_combined/combined3.py
The stack on top of combined.py (which is left untouched: h2 is built from it):

    round-2 signal forecaster   experiments/23_signal/signal_fc2.py          SignalForecaster2
    signal-path planner         experiments/28_signal_planning/sigplan.py    SigAgent + Paths

    from combined3 import make_agent3
    agent = make_agent3(fold_df)                                   # evaluated on fold_df, fitted outside it
    agent = make_agent3(None, end_ts="2026-08-31 23:00")           # fitted on every train row (what would ship)

make_agent3(fold_df, signal=, planner=, ...):
    signal   "fc2" round-2 forecaster (bridge + night level) | "fc" round-1 forecaster (as h2)
    planner  "paths" SigAgent: value of stored energy = (1 - lam) * value on the forecast + lam * mean over K
             error paths of the signal over hours 3..m-1, plus `ar` x the fitted correction of the forecast
             from the last revealed error | "trust" combined.CombinedAgent (the plan trusts the forecast: h2)
    paths_from  "round1": the author's cross-fitted errors of the round-1 forecaster
                (28_signal_planning/errors_<fold>.npz) | "round2": the same recipe replayed with the round-2
                forecaster (signal_errors2(), cached in cache/errors2_<fold>.npz)
    end_ts   calendar end of the run. SigAgent has no such option, so the signal forecaster is wrapped
             (EndCut): it returns no values beyond the last hour, which shortens the plan the same way.
Everything else (load, solar, net-load scenarios, H, Q, scen_hours) is combined.make_agent's.
"""
import sys

import numpy as np
import pandas as pd

import combined as cb

_SIGPLAN = cb.ROOT / "experiments" / "28_signal_planning"
if str(_SIGPLAN) not in sys.path:
    sys.path.append(str(_SIGPLAN))
import sigplan as sp  # noqa: E402
from signal_fc2 import SignalForecaster2  # noqa: E402

KAPPA = cb.KAPPA


class EndCut:
    """Signal forecaster that returns nothing beyond the last hour of the run (calendar end)."""
    def __init__(self, fc, end_ts):
        self.fc, self.end = fc, pd.Timestamp(end_ts)

    def predict(self, obs, hist, H):
        w = np.asarray(self.fc.predict(obs, hist, H), float)
        left = int((self.end - obs["timestamp"]) / pd.Timedelta(hours=1)) + 1
        return w[:left] if left >= 1 else w


def signal_errors2(fold_df=None):
    """Cross-fitted error vectors of the ROUND-2 signal forecaster: the recipe of
    28_signal_planning/xfit_signal.py (fit without one 7-day block of the fit rows, replay that block as a
    run) with SignalForecaster2 in place of SignalForecaster. Returns the dict Paths(errors=...) expects."""
    name = cb.fold_name(fold_df)
    path = cb.CACHE / f"errors2_{name}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    from common import replay, truth                      # experiments/23_signal/common.py
    fit = cb.fit_tables(fold_df)[2]
    full = cb.load_hard("filled")
    out = {"F": [], "E": [], "hod": [], "dow": [], "block": [], "pos": [], "ts": []}
    for b, (lo, hi) in cb.FOLD_DAYS.items():
        if b == name:
            continue
        d = fit.index.day
        fc = SignalForecaster2(fit[~((d >= lo) & (d <= hi))], KAPPA)
        dfb = full[(full.index.day >= lo) & (full.index.day <= hi)]
        _, _, W = replay(fc, dfb, H=cb.HT, parts=False)
        T = truth(dfb, H=cb.HT)["w"]
        out["F"].append(W)
        out["E"].append(T - W)
        out["hod"].append(dfb.index.hour.to_numpy())
        out["dow"].append(dfb.index.day_of_week.to_numpy())
        out["block"].append(np.full(len(dfb), ord(b)))
        out["pos"].append(np.arange(len(dfb)))
        out["ts"].append(dfb.index.values.astype("datetime64[h]").astype(np.int64))
    z = {k: np.concatenate(v) for k, v in out.items()}
    cb.CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **z)
    return z


def make_agent3(fold_df=None, signal="fc2", planner="paths", lam=0.5, ar=1.0, K=20, m=12, J=64,
                paths_from="round1", end_ts=None, truncate_end=True, tape=None, **kw):
    """See the module docstring. tape: recorded forecasts of the run (evaluation only), with the round-2
    signal under "w_sig2". **kw goes to combined.make_agent (load, solar, scen, H, Q, scen_hours, ...)."""
    if signal == "fc2":
        sig = cb.TapeW(tape, "w_sig2") if tape is not None else SignalForecaster2(cb.fit_tables(fold_df)[2], KAPPA)
    else:
        sig = "fc"
    base = cb.make_agent(fold_df, signal=sig, tape=tape, truncate_end=truncate_end,
                         end_ts=end_ts if planner == "trust" else None, **kw)
    if planner == "trust":
        return base
    errors = signal_errors2(fold_df) if paths_from == "round2" else None
    pool = sp.Paths(cb.fold_name(fold_df), K=K, m=m, errors=errors)
    w_fc = base.w_fc if end_ts is None else EndCut(base.w_fc, end_ts)
    return sp.SigAgent(w_fc, base.net, pool=pool, lam=lam, ar=ar, H=base.H, Q=base.Q, scen_hours=base.scen_hours,
                       J=J, exact=(lam == 0), v_end=base.v_end, truncate_end=truncate_end)
