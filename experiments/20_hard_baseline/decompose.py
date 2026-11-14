# Usage: .venv/Scripts/python.exe experiments/20_hard_baseline/decompose.py
"""Hard mode, first numbers: perfect-foresight ceiling per fold, and where a rolling optimiser
loses score: (a) only 3 h of price/CO2 known, (b) load and solar not known.
Writes experiments/20_hard_baseline/decompose.csv."""
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hard import get_folds, oracle, simulate, simulate_official, train_complement, rho  # noqa: E402
from src.hard_mpc import ClimW, MPCHard, PerfectN, PerfectW, ProfileN  # noqa: E402

CONFIGS = {
    # name: (w forecaster, n forecaster, H, d_shade)
    "perfect w + perfect n, H=24": ("perfect", "perfect", 24, 1.0),
    "perfect w + perfect n, H=48": ("perfect", "perfect", 48, 1.0),
    "clim w   + perfect n, H=36": ("clim", "perfect", 36, 1.0),
    "perfect w + profile n, H=36": ("perfect", "profile", 36, 1.0),
    "clim w   + profile n, H=36 (legal)": ("clim", "profile", 36, 1.0),
    "clim w   + profile n, H=36, shade 0.9 (legal)": ("clim", "profile", 36, 0.9),
}


def run(name, fold):
    df = get_folds()[fold]
    fit = train_complement(df)
    wk, nk, H, shade = CONFIGS[name]
    w_fc = PerfectW(df) if wk == "perfect" else ClimW(fit)
    n_fc = PerfectN(df) if nk == "perfect" else ProfileN(fit)
    t0 = time.time()
    r = simulate(MPCHard(w_fc, n_fc, H=H, d_shade=shade), df, return_trace=True)
    tr = r.pop("trace")
    return {"config": name, "fold": fold, "score": r["score"], "co2_pct": r["co2_pct"], "cost_pct": r["cost_pct"],
            "infractions": r["infractions"], "spill_kwh": float(tr["spill_kw"].sum()),
            "cycles_per_day": float(tr["p_kw"].clip(lower=0).sum() * 0.9 ** 0.5 / 6000.0 / (len(df) / 24)),
            "seconds": time.time() - t0}


if __name__ == "__main__":
    folds = get_folds()
    rows = []
    for f in ("A", "B", "C"):
        df = folds[f]
        o = oracle(df)
        rows.append({"config": "ORACLE (whole run known)", "fold": f, "score": o["score"], "co2_pct": o["co2_pct"],
                     "cost_pct": o["cost_pct"], "infractions": 0})
        print(f"fold {f}: {len(df)} h, rho={rho(df):.3f}, baseline {o['co2_base_kg']:.0f} kg / {o['cost_base_eur']:.0f} EUR, "
              f"oracle {o['score']:.3f} (CO2 {o['co2_pct']:.2f}%, cost {o['cost_pct']:.2f}%)", flush=True)
    # cross-check the fast simulator against the organisers' environment (one legal config, fold B)
    dfb = folds["B"]
    fit = train_complement(dfb)
    a = simulate(MPCHard(ClimW(fit), ProfileN(fit), H=24), dfb)
    b = simulate_official(MPCHard(ClimW(fit), ProfileN(fit), H=24), dfb)
    print(f"cross-check fold B: fast {a['score']:.4f} / official {b['score']:.2f}; infr {a['infractions']} / {b['infraccions']}", flush=True)
    rows += Parallel(n_jobs=10)(delayed(run)(c, f) for c in CONFIGS for f in ("A", "B", "C"))
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).with_name("decompose.csv"), index=False)
    piv = out.pivot(index="config", columns="fold", values="score").round(3)
    piv["mean"] = piv.mean(axis=1).round(3)
    print(piv.sort_values("mean", ascending=False).to_string())
    print(out[out.config.str.contains("legal")][["config", "fold", "score", "co2_pct", "cost_pct", "infractions", "spill_kwh", "cycles_per_day", "seconds"]].round(3).to_string())
