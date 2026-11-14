"""
agents/h1/build.py  --  fit the tables of BatteryAgent h1 and write its data file.

Two parts:
  * the profile tables of h0 (those of src.hard_mpc.ClimW and ProfileN, same group-bys and
    fallbacks), stored as dense arrays;
  * the scenario offsets of experiments/24_planner/planner.QuantileN: the agent's own net-load
    forecast (agent.net_forecast) is replayed on the fit rows exactly as the simulator presents
    them, and its errors by (hour of day of the target hour, horizon) are summarised as the mean
    error in each of Q equal-probability bins  ->  offsets (24, H, Q), probs (Q,).

Validation (fit only on rows outside the fold):
    from agents.h1.build import write_tables            # or load this file with importlib
    write_tables(train_complement(fold_df), "some/dir/h1_tables.json")

The real thing (fit on every train row, then package submissions/h1/):
    .venv/Scripts/python.exe agents/h1/build.py
Variant with the tables of months that have no train rows (November) interpolated from the
neighbouring months instead of the all-month profile (-> submissions/h1/nov_fill/):
    .venv/Scripts/python.exe agents/h1/build.py neighbours
"""
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA_NAME = "h1_tables.json"

RHO_HARD = 0.634                       # kg CO2 per EUR of baseline import (public hard leaderboard)
KAPPA = (0.3 / 0.7) * RHO_HARD         # same expression as src.hard_mpc.KAPPA
FINE = 360                             # residual quantile levels kept before binning (as QuantileN)


def _agent_module():
    spec = importlib.util.spec_from_file_location("h1_agent_for_build", str(HERE / "agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_profiles(df, kappa=KAPPA, fill_missing="mean") -> dict:
    """df: hourly table with a DatetimeIndex and columns load_kw, pv_kw (already x12),
    precio_eur_per_kwh, co2_kg_per_kwh  (src.hard.load_hard or a subset of its rows).
    fill_missing: what a month without fit rows gets (November in the train set):
        "mean"       the all-month profile, as the harness does (default);
        "neighbours" the profiles of the nearest months with data on each side, interpolated."""
    idx = df.index
    we = (idx.day_of_week >= 5).astype(int)
    w = df["co2_kg_per_kwh"] + kappa * df["precio_eur_per_kwh"]
    tab = w.groupby([idx.month, we, idx.hour]).mean()
    tab_h = w.groupby([we, idx.hour]).mean()
    load = df["load_kw"].groupby([we, idx.hour]).mean()
    pv = df["pv_kw"].groupby([idx.month, idx.hour]).mean()
    pv_h = df["pv_kw"].groupby(idx.hour).mean()

    w_any = np.array([[float(tab_h.get((e, h))) for h in range(24)] for e in range(2)])
    load_tab = np.array([[float(load.get((e, h))) for h in range(24)] for e in range(2)])
    pv_any = np.array([float(pv_h.get(h)) for h in range(24)])
    w_tab = np.empty((12, 2, 24))
    pv_tab = np.empty((12, 24))
    missing_w, missing_pv = [], []
    for m in range(1, 13):
        for h in range(24):
            v = pv.get((m, h))
            if v is None:
                missing_pv.append((m, h))
            pv_tab[m - 1, h] = float(v) if v is not None else pv_any[h]
            for e in range(2):
                v = tab.get((m, e, h))
                if v is None:
                    missing_w.append((m, e, h))
                w_tab[m - 1, e, h] = float(v) if v is not None else w_any[e, h]
    no_w = sorted({m for m, _, _ in missing_w})
    no_pv = sorted({m for m, _ in missing_pv})
    if fill_missing == "neighbours":
        for tab_, gone in ((w_tab, no_w), (pv_tab, no_pv)):
            have = [m for m in range(1, 13) if m not in gone]
            for m in gone if have else []:
                a = next(k for k in range(1, 13) if ((m - k - 1) % 12) + 1 in have)      # months back to data
                b = next(k for k in range(1, 13) if ((m + k - 1) % 12) + 1 in have)      # months forward to data
                lo, hi = tab_[(m - a - 1) % 12], tab_[(m + b - 1) % 12]
                tab_[m - 1] = (b * lo + a * hi) / (a + b)
    elif fill_missing != "mean":
        raise ValueError("fill_missing must be 'mean' or 'neighbours'")
    return {"w_tab": w_tab, "w_any": w_any, "load_tab": load_tab, "pv_tab": pv_tab, "pv_any": pv_any,
            "months_without_w": no_w, "months_without_pv": no_pv, "fill_missing": fill_missing}


def fit_offsets(df, prof, phi, phi_pv, H, Q) -> np.ndarray:
    """(24, H, Q) scenario offsets (kW) of the agent's net-load forecast on the rows of df."""
    assert FINE % Q == 0, f"Q must divide {FINE}"
    ag = _agent_module()
    idx = df.index
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_kw"].to_numpy(float)
    net = load - pv
    hrs = idx.values.astype("datetime64[h]").astype(np.int64)
    month = idx.month.to_numpy()
    dow = idx.day_of_week.to_numpy()
    hod0 = idx.hour.to_numpy()
    pv_tab = np.concatenate([prof["pv_tab"], prof["pv_any"][None]])
    T = len(df)
    res = np.full((T, H), np.nan)
    ar = np.arange(H)
    for t in range(1, T):
        if hrs[t] - hrs[t - 1] != 1:
            continue                                         # the last reading is not the previous hour
        m, d, h = int(month[t]), int(dow[t]), int(hod0[t])
        mi, we, hh = ag.horizon_calendar(None, m, None, d, h, H, None)       # consecutive hours
        lh = (h - 1) % 24
        lwe = int((d if h > 0 else (d - 1) % 7) >= 5)
        f = ag.net_forecast(prof["load_tab"], pv_tab, phi, phi_pv, mi, we, hh, lwe, lh,
                            float(load[t - 1]), float(pv[t - 1]))
        k = min(H, T - t)
        ok = (hrs[t:t + k] - hrs[t]) == ar[:k]               # target rows that really are t + k hours
        res[t, :k] = np.where(ok, net[t:t + k] - f[:k], np.nan)
    lev = (np.arange(FINE) + 0.5) / FINE
    tab = np.zeros((24, H, FINE))
    for k in range(H):
        hod_k = (hod0 + k) % 24
        col = res[:, k]
        good = ~np.isnan(col)
        for h in range(24):
            x = col[good & (hod_k == h)]
            tab[h, k] = np.quantile(x, lev) if len(x) >= 20 else np.quantile(col[good], lev)
    return tab.reshape(24, H, Q, FINE // Q).mean(axis=3)


def fit_tables(df, kappa=KAPPA, half=24.0, phi=0.9, phi_pv=0.8, H=36, Q=9, scen_hours=12,
               fill_missing="mean") -> dict:
    prof = fit_profiles(df, kappa, fill_missing)
    off = fit_offsets(df, prof, phi, phi_pv, H, Q)
    for name, a in [("w_tab", prof["w_tab"]), ("w_any", prof["w_any"]), ("load_tab", prof["load_tab"]),
                    ("pv_tab", prof["pv_tab"]), ("pv_any", prof["pv_any"]), ("offsets", off)]:
        if not np.isfinite(a).all():
            raise ValueError(f"{name} has non-finite cells")
    idx = df.index
    return {
        "name": "h1", "n_fit_rows": int(len(df)),
        "fit_first": str(idx.min()), "fit_last": str(idx.max()),
        "months_without_w": prof["months_without_w"], "months_without_pv": prof["months_without_pv"],
        "fill_missing": prof["fill_missing"],
        "kappa": float(kappa), "half": float(half), "phi": float(phi), "phi_pv": float(phi_pv),
        "H": int(H), "Q": int(Q), "scen_hours": int(scen_hours),
        "w_tab": prof["w_tab"].tolist(), "w_any": prof["w_any"].tolist(), "load_tab": prof["load_tab"].tolist(),
        "pv_tab": prof["pv_tab"].tolist(), "pv_any": prof["pv_any"].tolist(),
        "probs": [1.0 / Q] * Q, "offsets": off.tolist(),
    }


def write_tables(df, path, **kw) -> dict:
    t = fit_tables(df, **kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(t, f)
    return t


def package(fill_missing="mean", out_dir=None, **kw):
    """Fit on every official train row and build a submission folder: agent.py + data file +
    submission.zip with both at the ZIP root.
        fill_missing="mean"        -> submissions/h1/           (and the data file next to agent.py)
        fill_missing="neighbours"  -> submissions/h1/nov_fill/  (November tables from October / December)"""
    sys.path.insert(0, str(ROOT))
    from src.hard import load_hard
    std = fill_missing == "mean"
    if out_dir is None:
        out_dir = ROOT / "submissions" / "h1" if std else ROOT / "submissions" / "h1" / "nov_fill"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = write_tables(load_hard("official"), out_dir / DATA_NAME, fill_missing=fill_missing, **kw)
    shutil.copyfile(HERE / "agent.py", out_dir / "agent.py")
    if std:
        shutil.copyfile(out_dir / DATA_NAME, HERE / DATA_NAME)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("agent.py", DATA_NAME):
            z.write(out_dir / name, arcname=name)
    return t, out_dir


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    t, out = package(sys.argv[1] if len(sys.argv) > 1 else "mean")       # argument: mean | neighbours
    print(f"fitted on {t['n_fit_rows']} rows ({t['fit_first']} .. {t['fit_last']}), H={t['H']}, Q={t['Q']}, "
          f"scen_hours={t['scen_hours']}, months without signal rows: {t['months_without_w']}, "
          f"without solar rows: {t['months_without_pv']}, filled with: {t['fill_missing']}")
    print("wrote", out)
