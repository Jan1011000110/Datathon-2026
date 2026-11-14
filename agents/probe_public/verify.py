"""Checks of submissions/probe_public (the public-max probe). Usage: verify_public.py <repo> <workdir> [folds, e.g. BC]"""
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

repo, work = Path(sys.argv[1]), Path(sys.argv[2])
leak_folds = sys.argv[3] if len(sys.argv) > 3 else ""
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "starter_kit"))
from src.hard import get_folds, simulate, simulate_official  # noqa: E402
import check_submission as cs  # noqa: E402

FOLD_DAYS = {"A": (1, 7), "B": (8, 14), "C": (15, 21)}


def load(p, n):
    spec = importlib.util.spec_from_file_location(n, str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def unzip(name):
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(repo / "submissions" / name / "submission.zip") as z:
        z.extractall(d)
    return d


def layout(df):
    """Per row: position in its week (0..167) and whether it is the last row of a week."""
    gap = np.diff(df.index.values).astype("timedelta64[h]").astype(int) > 24
    wk = np.cumsum(np.r_[False, gap])
    pos = np.zeros(len(df), int)
    for k in np.unique(wk):
        pos[wk == k] = np.arange((wk == k).sum())
    return pos, np.r_[gap, False], wk


def run(make, df, label):
    pos, last, wk = layout(df)
    a = make()
    r = simulate(a, df, return_trace=True)
    tr = r.pop("trace")
    soc = tr.soc_kwh.to_numpy()
    p = tr.p_kw.to_numpy() if "p_kw" in tr else np.zeros(len(df))
    eta = 0.9 ** 0.5
    after = soc + np.where(p > 0, p * eta, p / eta)               # charge after the hour's command
    rf = simulate(make(), df, mask=pos < 84)["score"]
    rl = simulate(make(), df, mask=pos >= 84)["score"]
    print(f"{label}: whole {r['score']:.3f} | first 84 h {rf:.3f} | last 84 h {rl:.3f} | infractions {r['infractions']} | "
          f"answered {getattr(a, 'answered', None)} | charge after the last hour of a week {after[last].mean():.0f} kWh | "
          f"charge after hour 83 of a week {after[pos == 83].mean():.0f} kWh")
    return r["score"], rf, rl


pp = unzip("probe_public")
print("AST check:", cs.check_ast(pp / "agent.py") or "clean")
mods = {"h3j": load(unzip("h3j") / "agent.py", "m_h3j"), "h3w": load(unzip("h3w") / "agent.py", "m_h3w"),
        "cut probe": load(unzip("probe_cut") / "agent.py", "m_cut"), "public-max": load(pp / "agent.py", "m_pub")}

print("=== behaviour on a run laid out like the test (fold B relabelled to month ends, cut after July; shipped state, so in-sample: not a score forecast)")
dB = get_folds()["B"]
shifted = [(ts + pd.offsets.MonthEnd(0)).normalize() - pd.Timedelta(days=14 - ts.day) + pd.Timedelta(hours=ts.hour) for ts in dB.index]
d2 = dB.copy()
d2.index = pd.DatetimeIndex(shifted, name=dB.index.name)
d2 = d2[d2.index <= pd.Timestamp("2026-07-31 23:00")]
for name, m in mods.items():
    run(m.BatteryAgent, d2, name)
ro = simulate_official(mods["public-max"].BatteryAgent(), d2)
print(f"organisers' environment, public-max, same run: {ro['score']:.2f}, infractions {ro['infraccions']}")

a = mods["public-max"].BatteryAgent()
odd = [{}, {"timestamp": None, "soc_kwh": 100.0}, {"timestamp": "garbage", "soc_kwh": 5999.999, "co2_properes_24h": [0.1, 0.1, 0.1], "preus_properes_24h": [0.1, 0.1, 0.1]},
       {"timestamp": "2025-12-30 10:00:00", "soc_kwh": 5999.995, "co2_properes_24h": [0.1, 0.1, 0.1], "preus_properes_24h": [0.1, 0.1, 0.1]},
       {"timestamp": "2025-12-30 10:00:00", "soc_kwh": 0.0, "co2_properes_24h": [0.1, 0.1, 0.1], "preus_properes_24h": [0.1, 0.1, 0.1]},
       {"timestamp": "2026-07-30 10:00:00", "soc_kwh": 3000.0, "co2_properes_24h": [0.1, 0.1, 0.1], "preus_properes_24h": [0.1, 0.1, 0.1]},
       {"timestamp": "2025-12-30 10:00:00+01:00", "soc_kwh": float("nan")}]
print("odd observations ->", [round(float(a.act(o)), 3) for o in odd], "(4th: nearly full in a second half; 5th: empty in a second half; 6th: second half of the LAST week)")

if leak_folds:
    spec = importlib.util.spec_from_file_location("pp_make", str(repo / "agents/probe_public/make.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    bld = mk._build_module()
    M = mods["public-max"]
    print("=== leak-free: state fitted outside the fold, fold weeks as they are (rule = fold days), August dropped; one code base, three configurations")
    tot = {}
    for f in leak_folds:
        df = get_folds()[f]
        df = df[df.index < "2026-08-01"]
        lo, hi = FOLD_DAYS[f]
        t0 = time.time()
        arrays = bld.build_state(get_folds()[f], config={**bld.CONFIG, "week_rule": [lo, hi], "week_cut": False})[0]
        print(f"fold {f}: state built in {time.time() - t0:.0f} s, {len(df)} h")
        for label, extra in (("h3 (no cut)", {"public_max": False, "week_cut": False}), ("h3w (week cut)", {"public_max": False, "week_cut": True}),
                             ("public-max", {"public_max": True, "week_cut": True})):
            path = work / f"state_{f}_{label.split()[0]}.npz"
            bld.write_state(mk.patch_config(arrays, **extra), path)

            def make(path=path):
                M.BatteryAgent.state_path = path
                try:
                    return M.BatteryAgent()
                finally:
                    M.BatteryAgent.state_path = None
            tot.setdefault(label, []).append(run(make, df, f"  fold {f} | {label}"))
    print("--- mean over folds", leak_folds)
    base = np.mean(tot["h3 (no cut)"], axis=0)
    for label, v in tot.items():
        m = np.mean(v, axis=0)
        print(f"{label}: whole {m[0]:.3f} ({m[0] - base[0]:+.3f}) | first 84 h {m[1]:.3f} ({m[1] - base[1]:+.3f}) | last 84 h {m[2]:.3f} ({m[2] - base[2]:+.3f})")
