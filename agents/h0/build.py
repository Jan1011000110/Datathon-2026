"""
agents/h0/build.py  --  fit the tables of BatteryAgent h0 and write its data file.

The tables are those of src.hard_mpc.ClimW and ProfileN (same group-bys, same fallbacks), stored
as dense arrays so the agent needs no pandas.

Validation (fit only on rows outside the fold):
    from agents.h0.build import write_tables
    write_tables(train_complement(fold_df), "some/dir/h0_tables.json")

The real thing (fit on every train row, then package submissions/h0/; optional argument = horizon H):
    .venv/Scripts/python.exe agents/h0/build.py
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA_NAME = "h0_tables.json"

RHO_HARD = 0.634                       # kg CO2 per EUR of baseline import (public hard leaderboard)
KAPPA = (0.3 / 0.7) * RHO_HARD         # same expression as src.hard_mpc.KAPPA
H_PACKAGED = 18                        # horizon of the packaged agent: the shortest that stays within 0.02
                                       # points of the harness's H=36 on folds A, B and C (it is within
                                       # 0.004; H=12 is not) -- the LP time is what the 60 s limit is about


def fit_tables(df, kappa=KAPPA, half=24.0, phi=0.9, phi_pv=0.8, H=36, d_shade=0.9, lam=0.0) -> dict:
    """df: hourly table with a DatetimeIndex and columns load_kw, pv_kw (already x12),
    precio_eur_per_kwh, co2_kg_per_kwh  (src.hard.load_hard or a subset of its rows)."""
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
    for name, a in [("w_tab", w_tab), ("w_any", w_any), ("load_tab", load_tab), ("pv_tab", pv_tab), ("pv_any", pv_any)]:
        if not np.isfinite(a).all():
            raise ValueError(f"{name} has non-finite cells")
    return {
        "name": "h0", "n_fit_rows": int(len(df)),
        "fit_first": str(idx.min()), "fit_last": str(idx.max()),
        "months_without_w": sorted({m for m, _, _ in missing_w}),
        "months_without_pv": sorted({m for m, _ in missing_pv}),
        "kappa": float(kappa), "half": float(half), "phi": float(phi), "phi_pv": float(phi_pv),
        "H": int(H), "d_shade": float(d_shade), "lam": float(lam),
        "w_tab": w_tab.tolist(), "w_any": w_any.tolist(), "load_tab": load_tab.tolist(),
        "pv_tab": pv_tab.tolist(), "pv_any": pv_any.tolist(),
    }


def write_tables(df, path, **kw) -> dict:
    t = fit_tables(df, **kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(t, f)
    return t


def package(out_dir=ROOT / "submissions" / "h0", **kw):
    """Fit on every official train row, write the data file next to agent.py and build
    submissions/h0/ (agent.py + data file + submission.zip with both at the ZIP root)."""
    sys.path.insert(0, str(ROOT))
    from src.hard import load_hard
    t = write_tables(load_hard("official"), HERE / DATA_NAME, **kw)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("agent.py", DATA_NAME):
        shutil.copyfile(HERE / name, out_dir / name)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("agent.py", DATA_NAME):
            z.write(out_dir / name, arcname=name)
    return t


if __name__ == "__main__":
    kw = {"H": H_PACKAGED}
    if len(sys.argv) > 1:
        kw["H"] = int(sys.argv[1])          # e.g. `build.py 36` for the harness horizon
    t = package(**kw)
    print(f"fitted on {t['n_fit_rows']} rows ({t['fit_first']} .. {t['fit_last']}), H={t['H']}, "
          f"months without signal rows: {t['months_without_w']}, without solar rows: {t['months_without_pv']}")
    print("wrote", HERE / DATA_NAME, "and", ROOT / "submissions" / "h0")
