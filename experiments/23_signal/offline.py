# Usage: .venv/Scripts/python.exe experiments/23_signal/offline.py
"""Offline accuracy by horizon of signal forecasters on folds A, B, C: MAE of co2, price and w,
rank correlation over the 24 forecast hours (+3..+26). Writes offline.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAPPA, get_folds, rank_corr_rows, replay, signal_fit_table, truth  # noqa: E402
from signal_fc import SignalFC  # noqa: E402
from src.hard_mpc import ClimW  # noqa: E402

HS = (3, 6, 12, 18, 24, 36, 47)
VARIANTS = {
    "L48": dict(variant="L48"),
    "L48 no last week": dict(variant="L48", use_lw=False),
    "compact": dict(variant="compact"),
    "L48 ridge 0.3": dict(variant="L48", ridge=0.3),
    "L48 ridge 0.003": dict(variant="L48", ridge=0.003),
    "L48 cross": dict(variant="L48", cross=True),
    "A48": dict(variant="A48"),
    "A48 no last week": dict(variant="A48", use_lw=False),
    "A24": dict(variant="A24"),
    "A48 ridge 0.3": dict(variant="A48", ridge=0.3),
    "BEST": dict(variant="A48", use_lw=False, cross=True, ridge=0.01),
    "FINAL": dict(variant="A48", use_lw=False, cross=True, ridge=0.01, interp=True),
    "AH8 no last week": dict(variant="AH8", use_lw=False),
    "AH4 no last week": dict(variant="AH4", use_lw=False),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    folds = get_folds()
    rows = []
    for f in ("A", "B", "C"):
        df = folds[f]
        fit = signal_fit_table(f)
        T = truth(df)
        cands = {"ClimW": None}
        cands.update({k: VARIANTS[k] for k in names})
        for name, kw in cands.items():
            t0 = time.perf_counter()
            if kw is None:
                _, _, W = replay(ClimW(fit), df, parts=False)
                _, _, Cc = replay(ClimW(fit, kappa=0.0), df, kappa=0.0, parts=False)
                _, _, Pp = replay(ClimW(fit, kappa=1e6), df, kappa=1e6, parts=False)
                Pp = Pp / 1e6
                ms = np.nan
            else:
                fc = SignalFC(fit, KAPPA, **kw)
                t0 = time.perf_counter()
                Cc, Pp, W = replay(fc, df)
                ms = (time.perf_counter() - t0) / len(df) * 1e3
            rc = rank_corr_rows(W, T["w"])
            row = {"fold": f, "model": name, "ms_per_call": ms, "rank_corr_3_26": np.nanmean(rc),
                   "mae_w_3_26": np.nanmean(np.abs(W - T["w"])[:, 3:27]),
                   "bias_w_3_26": np.nanmean((W - T["w"])[:, 3:27])}
            day = df.index.normalize()
            st = np.zeros(len(df), bool)
            for lo, hi in (("2026-01-01", "2026-01-07"), ("2026-04-02", "2026-04-06"), ("2025-12-06", "2025-12-08")):
                st |= (day >= lo) & (day <= hi)
            err = np.abs(W - T["w"])[:, 3:27]
            row.update({"n_stretch_h": int(st.sum()), "rank_corr_stretch": np.nanmean(rc[st]) if st.any() else np.nan,
                        "mae_w_stretch": np.nanmean(err[st]) if st.any() else np.nan,
                        "rank_corr_ordinary": np.nanmean(rc[~st]), "mae_w_ordinary": np.nanmean(err[~st]),
                        "mae_w_3_11": np.nanmean(np.abs(W - T["w"])[:, 3:12]),
                        "rank_corr_3_11": np.nanmean(rank_corr_rows(W, T["w"], 0, 12))})
            for h in HS:
                row[f"co2_{h}"] = np.nanmean(np.abs(Cc - T["c"])[:, h])
                row[f"price_{h}"] = np.nanmean(np.abs(Pp - T["p"])[:, h])
                row[f"w_{h}"] = np.nanmean(np.abs(W - T["w"])[:, h])
            rows.append(row)
            print(f"{f} {name:20s} rank {row['rank_corr_3_26']:.3f}  MAE w {row['mae_w_3_26']*1e3:.2f}  "
                  f"bias {row['bias_w_3_26']*1e3:+.2f}  " +
                  " ".join(f"w{h}={row[f'w_{h}']*1e3:.1f}" for h in HS) + f"  {ms:.3f} ms", flush=True)
    out = pd.DataFrame(rows)
    p = Path(__file__).with_name("offline.csv")
    if p.exists() and sys.argv[1:]:
        old = pd.read_csv(p)
        out = pd.concat([old[~old.model.isin(out.model)], out])
    out.to_csv(p, index=False)
