# Usage: .venv/Scripts/python.exe experiments/22_solar/eval_offline.py [lgb|linear]
"""Offline accuracy of the solar forecast on folds B and C (fit on train_complement), by horizon.
Compares SolarFC with ProfileN.predict_pv and persistence. Writes offline_<model>.csv (one row per
(fold, hour, horizon)) and prints MAE / bias tables. Also measures the time per call."""
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
from solar_fc import SolarFC  # noqa: E402

HS = [0, 1, 2, 3, 6, 12, 24]
H = 36


def replay(fc_list, df):
    """Yield (t, obs, hist) exactly as src.hard.simulate + MPCHard.record would."""
    load, pv = df["load_kw"].to_numpy(float), df["pv_kw"].to_numpy(float)
    met = {c: df[c].to_numpy(float) for c in METEO_COLS}
    ts = df.index
    hist = []
    for t in range(len(df)):
        tl = t - 1 if t > 0 else t
        last = {"ultim_consum_kw": float(load[tl]), "ultima_solar_kw": float(pv[tl])}
        for c, a in met.items():
            last[c] = float(a[tl])
        obs = {"timestamp": ts[t], "hour": int(ts[t].hour), "noves_dades_5min": last}
        rec = {"ts": ts[t], "load_last": last["ultim_consum_kw"], "pv_last": last["ultima_solar_kw"]}
        for c in METEO_COLS:
            rec[c] = last[c]
        hist.append(rec)
        yield t, obs, hist


def run_fold(fold, model):
    df = get_folds()[fold]
    fit = train_complement(df)
    t0 = time.perf_counter()
    fc = SolarFC(fit, use_weather=(model != "noweather"))
    t_fit = time.perf_counter() - t0
    base = ProfileN(fit)
    truth = df["pv_kw"]
    tmap = dict(zip(df.index, truth.to_numpy()))
    rows = []
    t_call = 0.0
    t_q = 0.0
    for t, obs, hist in replay([fc], df):
        t0 = time.perf_counter()
        p = fc.predict_pv(obs, hist, H)
        t_call += time.perf_counter() - t0
        t0 = time.perf_counter()
        q = fc.predict_pv_quantiles(obs, hist, [0.1, 0.25, 0.5, 0.75, 0.9])
        t_q += time.perf_counter() - t0
        b = base.predict_pv(obs, hist, H)
        ts = obs["timestamp"]
        pl = obs["noves_dades_5min"]["ultima_solar_kw"]
        for h in HS:
            tt = ts + pd.Timedelta(hours=h)
            y = tmap.get(tt)
            if y is None:
                continue
            r = {"fold": fold, "ts": ts, "h": h, "hour_t": tt.hour, "y": y, "fc": p[h], "base": b[h], "pers": pl}
            if h == 0:
                r.update({"q10": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q90": q[4]})
            rows.append(r)
    n = len(df)
    print(f"fold {fold}: fit {t_fit:.1f}s, predict_pv {1000 * t_call / n:.3f} ms/call, "
          f"quantiles(+cached) {1000 * t_q / n:.3f} ms/call", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "final"
    out = pd.concat([run_fold(f, model) for f in ("B", "C")])
    out.to_csv(Path(__file__).with_name(f"offline_{model}.csv"), index=False)
    out["day"] = (out.hour_t >= 8) & (out.hour_t <= 19)
    for lab, d in (("daytime targets 08-19h", out[out.day]), ("all hours", out)):
        tab = pd.DataFrame({
            "n": d.groupby("h").size(),
            "MAE fc": (d.fc - d.y).abs().groupby(d.h).mean(),
            "MAE base": (d.base - d.y).abs().groupby(d.h).mean(),
            "MAE pers": (d.pers - d.y).abs().groupby(d.h).mean(),
            "bias fc": (d.fc - d.y).groupby(d.h).mean(),
            "bias base": (d.base - d.y).groupby(d.h).mean(),
            "RMSE fc": ((d.fc - d.y) ** 2).groupby(d.h).mean() ** 0.5,
            "RMSE base": ((d.base - d.y) ** 2).groupby(d.h).mean() ** 0.5,
        })
        print(f"\n{lab} (folds B+C, kW)")
        print(tab.round(1).to_string())
    h0 = out[(out.h == 0) & out.day]
    print("\nhorizon 0, MAE by target hour:")
    print(pd.DataFrame({"fc": (h0.fc - h0.y).abs().groupby(h0.hour_t).mean(),
                        "base": (h0.base - h0.y).abs().groupby(h0.hour_t).mean()}).round(0).T.to_string())
    print("\nhorizon 0 quantile coverage (share of y below q), daytime:")
    print({q: round(float((h0.y <= h0[q]).mean()), 3) for q in ("q10", "q25", "q50", "q75", "q90")})
    print("by fold, h=0 daytime MAE:", (h0.fc - h0.y).abs().groupby(h0.fold).mean().round(1).to_dict(),
          "base:", (h0.base - h0.y).abs().groupby(h0.fold).mean().round(1).to_dict())
