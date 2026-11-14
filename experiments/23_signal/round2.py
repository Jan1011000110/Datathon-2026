# Usage: .venv/Scripts/python.exe experiments/23_signal/round2.py [strict] [names...]
"""Round 2 candidates built on the real forecaster classes: replay each on folds A, B, C into a tape and
score the tape in closed loop (net load known; combined agent). `strict`: fold C fitted on days 1-14
only (test geometry). Appends to round2.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402
from common import KAPPA, replay, signal_fit_table, truth  # noqa: E402
from signal_fc import SignalForecaster  # noqa: E402
from signal_fc2 import SignalForecaster2  # noqa: E402

CANDS = {
    "round 1": lambda fit: SignalForecaster(fit, KAPPA),
    "bridge": lambda fit: SignalForecaster2(fit, KAPPA, night=False),
    "bridge x0.7": lambda fit: SignalForecaster2(fit, KAPPA, night=False, rho_scale=0.7),
    "up 3%": lambda fit: SignalForecaster2(fit, KAPPA, bridge=False, night=False, up=0.03),
    "up 1.5%": lambda fit: SignalForecaster2(fit, KAPPA, bridge=False, night=False, up=0.015),
    "night level": lambda fit: SignalForecaster2(fit, KAPPA, bridge=False),
    "night level w=1": lambda fit: SignalForecaster2(fit, KAPPA, bridge=False, night_w=1.0),
    "bridge + night level (round 2)": lambda fit: SignalForecaster2(fit, KAPPA),
}


def run(name, f, strict):
    df = fast.fold(f)
    fit = signal_fit_table(f)
    if strict:
        fit = fit[fit.index.day <= 14]
    fc = CANDS[name](fit)
    t0 = time.perf_counter()
    Cc, Pp, W = fast.replay_full(fc, df, H=fast.HT)
    ms = (time.perf_counter() - t0) / len(df) * 1e3
    T = truth(df, H=fast.HT)["w"]
    lo = {"A": 1, "B": 8, "C": 15}[f]
    hin = (df.index.day.values - lo) * 24 + df.index.hour.values
    last = hin >= 168 - 30                                    # origins in the last 30 hours of a week
    e = np.abs(W - T)[:, 3:12]
    pn, cm = fast.both(W, f)
    return {"cand": name, "fold": f, "strict": strict, "pn": pn, "comb": cm, "ms_per_call": ms,
            "mae_3_11": np.nanmean(e) * 1e3, "mae_3_11_last30h": np.nanmean(e[last]) * 1e3,
            "lvl_last30h": np.nanmean(np.abs(np.nanmean((W - T)[last, 3:9], axis=1))) * 1e3}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "strict"]
    strict = "strict" in sys.argv
    names = args or list(CANDS)
    folds = ["C"] if strict else list("ABC")
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(n, f, strict) for n in names for f in folds))
    p = Path(__file__).with_name("round2.csv")
    if p.exists():
        old = pd.read_csv(p)
        key = old.cand.astype(str) + old.fold + old.strict.astype(str)
        out = pd.concat([old[~key.isin(out.cand + out.fold + out.strict.astype(str))], out])
    out.to_csv(p, index=False)
    cur = out[(out.strict == strict) & out.cand.isin(names)]
    pd.set_option("display.width", 220)
    for col in ("pn", "comb", "mae_3_11_last30h", "lvl_last30h", "ms_per_call"):
        piv = cur.pivot(index="cand", columns="fold", values=col)
        piv["mean"] = piv.mean(axis=1)
        print(f"== {col} {'(strict C)' if strict else ''} ==")
        print(piv.loc[names].round(3).to_string())
