"""Which mechanism produces the leaderboard pattern?

Pattern on the public board: weightings 0.587-0.615 (the pack) all score the same within 0.0005,
while about 0.575 scores +0.0066 and about 0.625 scores +0.011.

Two candidate mechanisms, simulated on train weeks:
  (a) GLITCH: the public run is self-contained (6 weeks, 1,008 h) and contains one short meter
      glitch on an hour where the balanced weighting discharges.
  (b) MASK:   one joint 12-week run is simulated and the public score counts only 6 whole weeks.

For each simulated case we score the weightings 0.575 / 0.59 / 0.60 / 0.61 / 0.625 and ask how
often the pack (0.59, 0.61) is flat while an outlier weighting gains.
Run from the repo root:  .venv/Scripts/python.exe experiments/13_fault_luck/which_mechanism.py
"""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.sim import get_folds, simulate, score_from_grid

RHOS = [0.575, 0.59, 0.60, 0.61, 0.625]


def load_v1():
    spec = importlib.util.spec_from_file_location("agent_v1", ROOT / "agents" / "v1" / "agent.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_agent(v1, rho):
    a = v1.BatteryAgent()
    kappa = (0.3 / 0.7) * rho

    def decide(obs, a=a, kappa=kappa):
        w = np.asarray(obs["co2_properes_24h"], float) + kappa * np.asarray(obs["preus_properes_24h"], float)
        d_cap, nb = a._discharge_limits(obs, len(w))
        return v1.first_step(float(obs["soc_kwh"]), v1.marginal_values(w, a.v_end, nb), w[0], d_cap)

    a._decide = decide
    return a


def clean_weeks():
    allf = get_folds("filled")
    weeks = {}
    for f in "ABC":
        df = allf[f]
        for i, (_, g) in enumerate(df.groupby([df.index.year, df.index.month], sort=True)):
            if (g.load_kw < 600).any() and (g.load_kw < 50).any():
                continue                                    # weeks with a real meter outage
            if g.index[0].month == 8 and f == "A":
                continue                                    # 6 Aug partial glitch
            weeks[(i, f)] = g
    return weeks


def one_case(seed):
    v1 = load_v1()
    rng = np.random.default_rng(seed)
    weeks = clean_weeks()
    months = sorted({k[0] for k in weeks})
    out = {}

    # ---- (a) self-contained 6-week run with one injected short glitch -------------------------
    ms = sorted(rng.choice(months, size=6, replace=False))
    df = pd.concat([weeks[(m, rng.choice([f for (mm, f) in weeks if mm == m]))] for m in ms]).copy()
    tr = simulate(make_agent(v1, 0.60), df, return_trace=True)["trace"]
    net = (df.load_kw - df.pv_kw).to_numpy()
    cand = np.flatnonzero((tr.p_kw.to_numpy() <= -400) & (net < 1000))
    if len(cand):
        t = int(rng.choice(cand))
        frac = float(rng.choice([0.0, 0.45, 0.55, 0.6]))    # share of the hour's load that is still recorded
        df.iloc[t, df.columns.get_loc("load_kw")] = df.load_kw.iloc[t] * frac
        df.iloc[t, df.columns.get_loc("pv_kw")] = df.pv_kw.iloc[t] * frac
        sc = {r: simulate(make_agent(v1, r), df)["score"] for r in RHOS}
        for r in RHOS:
            out[f"glitch | {r}"] = sc[r] - sc[0.60]

    # ---- (b) joint 12-week run, public = 6 whole weeks ----------------------------------------
    df = pd.concat([weeks[(m, rng.choice([f for (mm, f) in weeks if mm == m]))] for m in months])
    n = len(df)
    wk = np.arange(n) // 168
    pub_weeks = rng.choice(np.unique(wk), size=len(np.unique(wk)) // 2, replace=False)
    alt = wk % 2 == int(rng.integers(0, 2))
    hw = np.arange(n) % 168
    masks = {"mask random weeks": np.isin(wk, pub_weeks), "mask alternate weeks": alt,
             "mask first 84 h of each week": hw < 84, "mask last 84 h of each week": hw >= 84,
             "mask alternate days": (hw // 24) % 2 == 0,
             "mask first 3 days + 12 h, cut at midnight-ish (days 1-3 and day 4 a.m.)": hw < 84,
             "mask first 4 days of each week": hw < 96, "mask last 3 days of each week": hw >= 96}
    load, pv = df.load_kw.to_numpy(), df.pv_kw.to_numpy()
    price, co2 = df.precio_eur_per_kwh.to_numpy(), df.co2_kg_per_kwh.to_numpy()
    grids = {r: simulate(make_agent(v1, r), df, return_trace=True)["trace"].grid_kw.to_numpy() for r in RHOS}
    for name, m in masks.items():
        sc = {r: score_from_grid(grids[r], load, pv, price, co2, mask=m)["score"] for r in RHOS}
        for r in RHOS:
            out[f"{name} | {r}"] = sc[r] - sc[0.60]
    return out


if __name__ == "__main__":
    res = Parallel(n_jobs=6)(delayed(one_case)(s) for s in range(240))
    d = pd.DataFrame(res)
    d.to_csv(Path(__file__).parent / "which_mechanism.csv", index=False)
    print("cases:", len(d))
    for mech in ("glitch", "mask random weeks", "mask alternate weeks", "mask first 84 h of each week",
                 "mask last 84 h of each week", "mask alternate days", "mask first 4 days of each week",
                 "mask last 3 days of each week"):
        g = d[[c for c in d.columns if c.startswith(mech + " |")]].dropna()
        g.columns = [float(c.split("| ")[1]) for c in g.columns]
        pack_flat = (g[0.59].abs() < 0.001) & (g[0.61].abs() < 0.001)
        lo_up, hi_up = g[0.575] > 0.004, g[0.625] > 0.004
        lo_dn, hi_dn = g[0.575] < -0.004, g[0.625] < -0.004
        print(f"\n{mech}  ({len(g)} cases)")
        print("  typical |change| vs 0.60:  " + ",  ".join(f"{r}: median {g[r].abs().median():.4f}, 95th pct {g[r].abs().quantile(.95):.4f}" for r in (0.575, 0.59, 0.61, 0.625)))
        print(f"  pack flat (0.59 and 0.61 within 0.001 of 0.60): {pack_flat.mean():.0%}")
        print(f"  0.575 gains more than 0.004: {lo_up.mean():.1%}    loses more than 0.004: {lo_dn.mean():.1%}")
        print(f"  0.625 gains more than 0.004: {hi_up.mean():.1%}    loses more than 0.004: {hi_dn.mean():.1%}")
        print(f"  pack flat AND at least one outlier weighting gains > 0.004: {(pack_flat & (lo_up | hi_up)).mean():.1%}")
        print(f"  pack flat AND both outlier weightings gain > 0.004:         {(pack_flat & lo_up & hi_up).mean():.1%}")
