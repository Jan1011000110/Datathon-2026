"""experiments/03_gap/lib.py -- shared helpers for the gap study.

* loads agents/v1/agent.py as module `v1` (never modified)
* GapAgent: research twin of v1 (same decisions when all options are default) with hooks for
  RHO, guard margin, window extension, terminal marginal values and "cheats" (true future)
* weekly paired differences that add up exactly to the score difference
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim import get_folds, simulate, oracle, score_weights, train_complement, solve_lp  # noqa: E402


def load_agent_module(path, name=None):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name or ("agent_" + path.parent.name), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v1 = load_agent_module(ROOT / "agents" / "v1" / "agent.py", "agent_v1")
NI, ETA, CAP, PMAX, GRID = v1.NI, v1.ETA, v1.CAP, v1.PMAX, v1.GRID
ZERO = np.zeros(NI)
FOLDS = ("A", "B", "C", "ALL")


def true_rho(df):
    base = np.maximum(df["load_kw"].to_numpy(float) - df["pv_kw"].to_numpy(float), 0.0)
    return float(np.sum(base * df["co2_kg_per_kwh"].to_numpy(float)) /
                 np.sum(base * df["precio_eur_per_kwh"].to_numpy(float)))


def block_ids(index):
    """Integer id of each contiguous block of hours (a 'week' in folds A/B/C)."""
    gap = np.diff(index.values).astype("timedelta64[s]").astype(float)
    return np.concatenate([[0], np.cumsum(np.abs(gap - 3600.0) > 1.0)])


def contributions(df, trace):
    """Per-hour score contribution in points: score = sum(contrib) (with 0 infractions)."""
    w = score_weights(df)
    base = np.maximum(df["load_kw"].to_numpy(float) - df["pv_kw"].to_numpy(float), 0.0)
    return 100.0 * w * (base - trace["grid_kw"].to_numpy(float))


def week_ids(df, days=7):
    """Blocks split further into chunks of `days` days (so ALL gets ~weekly chunks too)."""
    b = block_ids(df.index)
    out = np.zeros(len(df), dtype=int)
    k = -1
    for bid in np.unique(b):
        m = np.where(b == bid)[0]
        chunk = (np.arange(len(m)) // (24 * days))
        # merge a short tail (< 3 days) into the previous chunk
        if chunk.max() > 0 and np.sum(chunk == chunk.max()) < 72:
            chunk[chunk == chunk.max()] -= 1
        out[m] = k + 1 + chunk
        k = out[m].max()
    return out


def full_backward(w):
    """V[j] = marginal-value function after hour j given every later hour (zero after the end)."""
    n = len(w)
    V = np.zeros((n, NI))
    for j in range(n - 2, -1, -1):
        V[j] = v1.marginal_values(w[j:j + 2], V[j + 1])
    return V


class GapAgent:
    """v1 logic with hooks. Defaults reproduce agents/v1 exactly.

    rho         : float, or object with .update(agent, obs) and .value() (online estimate)
    guard_margin: fraction kept below the last measured net load when discharging
    ext         : callable(agent, obs, w_known) -> array of forecast w appended to the window
    vend        : callable(agent, obs, w_all) -> terminal marginal values (len NI, non-increasing)
    cheat_h     : int >0 -> extend the window with the TRUE next cheat_h hours (needs df)
    cheat_guard : True -> discharge cap uses the TRUE net load of the current hour (needs df)
    outage      : False -> disable the outage heuristic of v1 (guard stays)
    guard       : False -> no discharge guard at all
    """

    def __init__(self, rho=0.60, guard_margin=0.05, ext=None, vend=None, cheat_h=0,
                 cheat_guard=False, outage=True, df=None, guard=True):
        self.rho = rho
        self.guard_margin = guard_margin
        self.ext = ext
        self.vend = vend
        self.cheat_h = cheat_h
        self.cheat_guard = cheat_guard
        self.outage = outage
        self.guard = guard
        self.t = -1
        self.prev_ts = None
        self.outage_run = 0
        self.jump = False
        if df is not None:
            self.co2_all = df["co2_kg_per_kwh"].to_numpy(float)
            self.price_all = df["precio_eur_per_kwh"].to_numpy(float)
            self.net_all = (df["load_kw"] - df["pv_kw"]).to_numpy(float)
            self.n_all = len(df)
        self.v_full = None
        if cheat_h == "full":                       # true w for the whole rest of the run
            kap = self.kappa()
            self.w_full = np.maximum(self.co2_all + kap * self.price_all, 1e-9)
            self.v_full = full_backward(self.w_full)

    def kappa(self):
        r = self.rho if isinstance(self.rho, (int, float)) else self.rho.value()
        return (0.3 / 0.7) * r

    def _time_jump(self, obs):
        ts = obs.get("timestamp")
        jump = False
        if self.prev_ts is not None:
            jump = abs((ts - self.prev_ts).total_seconds() - 3600.0) > 1.0
        self.prev_ts = ts
        return jump

    def _discharge_limits(self, obs, horizon):
        jump = self._time_jump(obs)
        self.jump = jump
        last = obs["noves_dades_5min"]
        net = float(last["ultim_consum_kw"]) - float(last["ultima_solar_kw"])
        if self.cheat_guard:
            return min(PMAX, max(0.0, self.net_all[self.t])), 0
        if not self.guard:
            return PMAX, 0
        if jump or net != net:
            self.outage_run = 0
            return PMAX, 0
        if net < v1.OUTAGE_KW and self.outage:
            self.outage_run += 1
            return 0.0, min(horizon, max(4, 2 * self.outage_run))
        self.outage_run = 0
        return min(PMAX, max(0.0, net * (1.0 - self.guard_margin))), 0

    def act(self, obs):
        self.t += 1
        if not isinstance(self.rho, (int, float)):
            self.rho.update(self, obs)
        soc = float(obs["soc_kwh"])
        co2 = np.asarray(obs["co2_properes_24h"], float)
        price = np.asarray(obs["preus_properes_24h"], float)
        n = len(co2)
        kap = self.kappa()
        w = np.maximum(co2 + kap * price, 1e-9)
        d_cap, n_blocked = self._discharge_limits(obs, n)
        v_end = ZERO
        if self.v_full is not None:
            L = min(max(n_blocked, 1), self.n_all - self.t)
            w = self.w_full[self.t:self.t + L]
            v_end = self.v_full[self.t + L - 1]
        elif n == 24:                                # not in the end game
            if self.cheat_h:
                lo, hi = self.t + 24, min(self.t + 24 + self.cheat_h, self.n_all)
                w = np.concatenate([w, np.maximum(self.co2_all[lo:hi] + kap * self.price_all[lo:hi], 1e-9)])
            if self.ext is not None:
                e = self.ext(self, obs, w)
                if e is not None and len(e):
                    w = np.concatenate([w, np.maximum(np.asarray(e, float), 1e-9)])
            if self.vend is not None:
                v_end = self.vend(self, obs, w)
        v = v1.marginal_values(w, v_end, n_blocked)
        p = float(v1.first_step(soc, v, w[0], d_cap))
        if p != p:
            p = 0.0
        max_c = max(0.0, min(PMAX, (CAP - soc) / ETA - v1.EPS))
        max_d = max(0.0, min(PMAX, soc * ETA - v1.EPS))
        return float(min(max(p, -max_d), max_c))


def run(agent, df):
    """simulate + per-hour contributions (points)."""
    r = simulate(agent, df, return_trace=True)
    r["contrib"] = contributions(df, r["trace"])
    return r


_VCACHE = {}


def boundary_values(df):
    """V[t] = marginal value (points per stored kWh) of energy in the battery after hour t, under
    full foresight of the true score signal. Used only to attribute carried-over energy to weeks."""
    key = (len(df), str(df.index[0]), str(df.index[-1]))
    if key not in _VCACHE:
        _VCACHE[key] = full_backward(score_weights(df) * 100.0)
    return _VCACHE[key]


def stored_value(v, soc):
    """Value of holding `soc` kWh when the marginal-value step function is v."""
    F = np.concatenate([[0.0], np.cumsum(v * np.diff(GRID))])
    return float(np.interp(soc, GRID, F))


def paired(df, res_new, res_ref, days=7, carry=True):
    """Per-week paired gain in points (sums exactly to the score difference).

    carry=True: energy one agent carries over a week boundary that the other does not is credited
    to the week that stored it (valued with the full-foresight marginal value), otherwise a single
    shifted charge shows up as -x in one week and +x in the next."""
    wk = week_ids(df, days)
    g = np.bincount(wk, weights=res_new["contrib"] - res_ref["contrib"])
    if carry:
        V = boundary_values(df)
        sn = res_new["trace"]["soc_kwh"].to_numpy(float)
        sr = res_ref["trace"]["soc_kwh"].to_numpy(float)
        for t in np.where(np.diff(wk) != 0)[0]:
            dv = stored_value(V[t], sn[t]) - stored_value(V[t], sr[t])
            g[wk[t]] += dv
            g[wk[t + 1]] -= dv
    return g


def summarize(g):
    """mean/std/t-stat helper for a vector of weekly gains."""
    g = np.asarray(g, float)
    n = len(g)
    sd = g.std(ddof=1) if n > 1 else float("nan")
    return {"sum": g.sum(), "n": n, "pos": int((g > 1e-9).sum()), "neg": int((g < -1e-9).sum()),
            "min": g.min(), "max": g.max(), "t": g.mean() / (sd / np.sqrt(n)) if sd > 0 else float("nan")}
