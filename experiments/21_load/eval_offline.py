# Usage: .venv/Scripts/python.exe experiments/21_load/eval_offline.py [tag]
"""Offline accuracy of LoadFC against ProfileN and persistence on folds A, B, C (fit on the
complement each time). MAE and bias by horizon, and for horizon 0 by hour of day. Also quantile
coverage and per-call time. Writes offline_<tag>_*.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.hard import METEO_COLS, get_folds, train_complement  # noqa: E402
from src.hard_mpc import ProfileN  # noqa: E402
from load_fc import LoadFC  # noqa: E402
from variants import make  # noqa: E402

HS = [0, 1, 2, 3, 6, 12, 24]
QS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


def run_fold(fold, df, fcs, H=25):
    """fcs: dict name -> forecaster with predict_load. Returns long frame of errors."""
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_kw"].to_numpy(float)
    met = {c: df[c].to_numpy(float) for c in METEO_COLS}
    ts = df.index
    hi = (ts.values.astype("datetime64[ns]").astype(np.int64) // 3_600_000_000_000)
    n = len(df)
    rows = []
    qrows = []
    hist = []
    tcall = {k: 0.0 for k in fcs}
    for t in range(n):
        tl = t - 1 if t > 0 else t
        last = {"ultim_consum_kw": float(load[tl]), "ultima_solar_kw": float(pv[tl])}
        for c in METEO_COLS:
            last[c] = float(met[c][tl])
        obs = {"timestamp": ts[t], "noves_dades_5min": last}
        hist.append({"ts": ts[t], "load_last": last["ultim_consum_kw"]})
        for name, fc in fcs.items():
            t0 = time.perf_counter()
            p = fc.predict_load(obs, hist, H)
            tcall[name] += time.perf_counter() - t0
            for k in HS:
                if t + k < n and hi[t + k] - hi[t] == k:
                    rows.append((fold, name, t, ts[t].hour, k, p[k] - load[t + k], load[t + k]))
            if hasattr(fc, "predict_load_quantiles"):
                q = fc.predict_load_quantiles(obs, hist, QS)
                qrows.append((fold, name, t, ts[t].hour, load[t], *q))
        rows.append((fold, "persistence", t, ts[t].hour, 0, last["ultim_consum_kw"] - load[t], load[t]))
    e = pd.DataFrame(rows, columns=["fold", "model", "t", "hour", "k", "err", "y"])
    q = pd.DataFrame(qrows, columns=["fold", "model", "t", "hour", "y"] + [f"q{int(x * 100):02d}" for x in QS])
    return e, q, {k: v / n * 1000 for k, v in tcall.items()}


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["v2", "v1", "nomodel"]
    main = variants[0]
    folds = get_folds()
    E, Q = [], []
    for f in ("A", "B", "C"):
        df = folds[f]
        fit = train_complement(df)
        t0 = time.perf_counter()
        fcs = {v: make(v, fit) for v in variants}
        tfit = time.perf_counter() - t0
        fcs["ProfileN"] = ProfileN(fit)
        e, q, tc = run_fold(f, df, fcs)
        print(f"fold {f}: fit {tfit:.1f}s, ms/call:", {k: round(v, 3) for k, v in tc.items()},
              "oof MAE", round(fcs[main].oof_mae, 1), "n_train", fcs[main].n_train, flush=True)
        E.append(e); Q.append(q)
    e = pd.concat(E); q = pd.concat(Q)
    here = Path(__file__).parent
    e.to_csv(here / f"offline_{tag}_errors.csv", index=False)
    q.to_csv(here / f"offline_{tag}_quantiles.csv", index=False)
    e["ae"] = e["err"].abs()
    for grp, lab in ((["B", "C"], "B+C"), (["A"], "A")):
        x = e[e.fold.isin(grp)]
        print(f"\n== MAE by horizon, folds {lab} ==")
        print(x.pivot_table(index="model", columns="k", values="ae").round(1).to_string())
        print(f"== bias by horizon, folds {lab} ==")
        print(x.pivot_table(index="model", columns="k", values="err").round(1).to_string())
    x = e[e.fold.isin(["B", "C"]) & (e.k == 0)]
    print("\n== horizon-0 MAE by hour, folds B+C ==")
    print(x.pivot_table(index="model", columns="hour", values="ae").round(0).astype(int).to_string())
    print("== horizon-0 bias by hour, folds B+C ==")
    print(x.pivot_table(index="model", columns="hour", values="err").round(0).astype(int).to_string())
    qq = q[q.fold.isin(["B", "C"]) & (q.model == main)]
    day = (qq.hour >= 6) & (qq.hour <= 22)
    print("\n== quantile coverage (share of truths below q), main variant, B+C, hours 6-22 ==")
    print({c: round(float((qq.loc[day, "y"] < qq.loc[day, c]).mean()), 3) for c in qq.columns if c.startswith("q")})
