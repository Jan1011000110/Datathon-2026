# Usage: .venv/Scripts/python.exe experiments/23_signal/hourspec.py
"""Do models fitted per hour of day predict the level of hours +3..+8 better than the single model?
(mechanism: an anomaly seen at solar hours says little about the evening, one seen at night does.)
Level MAE by hour of origin, and closed loop, per fold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402
from common import KAPPA, replay, signal_fit_table, truth  # noqa: E402
from signal_fc import SignalFC  # noqa: E402

CANDS = {
    "A48 (round 1)": dict(variant="A48", use_lw=False, cross=True, ridge=0.01, interp=True),
    "AH8 per hour": dict(variant="AH8", use_lw=False, ridge=0.03, interp=True),
    "AH8 per hour, ridge 0.1": dict(variant="AH8", use_lw=False, ridge=0.1, interp=True),
    "AH12 per hour": dict(variant="AH12", use_lw=False, ridge=0.03, interp=True),
}


def run(name, f):
    df = fast.fold(f)
    fc = SignalFC(signal_fit_table(f), KAPPA, **CANDS[name])
    Cc, Pp, W = fast.replay_full(fc, df, H=fast.HT)
    np.save(fast.CACHE / f"hs_{name.replace(' ', '_').replace(',', '')}_{f}.npy", W)
    T = truth(df, H=fast.HT)["w"]
    lvl = np.abs(np.nanmean((W - T)[:, 3:9], axis=1))
    h = df.index.hour.values
    pn, cm = fast.both(W, f)
    return {"cand": name, "fold": f, "pn": pn, "comb": cm, "lvl": np.nanmean(lvl) * 1e3,
            "lvl_h05_06": np.nanmean(lvl[(h >= 5) & (h <= 6)]) * 1e3, "lvl_h08_11": np.nanmean(lvl[(h >= 8) & (h <= 11)]) * 1e3,
            "lvl_h13_16": np.nanmean(lvl[(h >= 13) & (h <= 16)]) * 1e3, "lvl_h18_23": np.nanmean(lvl[h >= 18]) * 1e3,
            "mae_3_11": np.nanmean(np.abs(W - T)[:, 3:12]) * 1e3}


if __name__ == "__main__":
    out = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(n, f) for n in CANDS for f in "ABC"))
    out.to_csv(Path(__file__).with_name("hourspec.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out.groupby("cand")[["lvl", "lvl_h05_06", "lvl_h08_11", "lvl_h13_16", "lvl_h18_23", "mae_3_11"]].mean().loc[list(CANDS)].round(2).to_string())
    for col in ("pn", "comb", "lvl"):
        piv = out.pivot(index="cand", columns="fold", values=col); piv["mean"] = piv.mean(axis=1)
        print(f"== {col} =="); print(piv.loc[list(CANDS)].round(3).to_string())
