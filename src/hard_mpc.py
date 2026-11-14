"""
src/hard_mpc.py
Experiment harness for hard mode: a rolling-horizon optimiser (MPC) with pluggable forecasters.
Not a submission file (it uses scipy's LP and pandas); packaged agents live in agents/.

Each hour the agent
  1. asks `w_fc` for the score signal  w = co2 + kappa * price  of the next H hours
     (the first 3 are known from obs and always overwrite the forecast),
  2. asks `n_fc` for the campus net load (load - solar, kW; negative = solar surplus) of the
     next H hours, hour 0 being the hour it is deciding (NOT known: obs only has the last hour),
  3. solves the LP "minimise sum w_k * grid_import_k" and applies the first step.

Forecaster interface (duck-typed):
    predict(obs, hist, H) -> np.ndarray of length <= H      (shorter = the run ends earlier)
`hist` is the list of per-hour records the agent has kept so far, hist[-1] being this hour:
    {"ts", "co2", "price", "w", "load_last", "pv_last", + the 4 weather keys of the last hour}
"""
import numpy as np
from scipy.optimize import linprog

from src.hard import CAP, ETA, HORIZON, METEO_COLS, PMAX

RHO_HARD = 0.634             # kg CO2 per EUR of baseline import implied by the public hard leaderboard
KAPPA = (0.3 / 0.7) * RHO_HARD


# --------------------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------------------
def plan_lp(w, n, soc0, lam=0.0, cap=CAP, pmax=PMAX):
    """x = [c_0..c_{H-1}, d_0.., g_0..]
        min  sum_k w_k g_k  -  lam * s_H
        s.t. g_k >= n_k + c_k - d_k,  g_k >= 0          (grid import; surplus is spilled for free)
             0 <= soc0 + sum_{j<=k} (ETA c_j - d_j / ETA) <= cap
             0 <= c_k, d_k <= pmax
    Returns (c, d, g) or (None, None, None)."""
    w = np.asarray(w, float)
    n = np.asarray(n, float)
    H = len(w)
    tri = np.tril(np.ones((H, H)))
    Z = np.zeros((H, H))
    I = np.eye(H)
    S = np.hstack([ETA * tri, -tri / ETA, Z])                    # soc_k - soc0
    A_ub = np.vstack([S, -S, np.hstack([I, -I, -I])])
    b_ub = np.concatenate([np.full(H, cap - soc0), np.full(H, soc0), -n])
    cost = np.concatenate([np.full(H, -lam * ETA), np.full(H, lam / ETA), w])
    bounds = [(0, pmax)] * (2 * H) + [(0, None)] * H
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if r.status != 0:
        return None, None, None
    return r.x[:H], r.x[H:2 * H], r.x[2 * H:]


def safe_clamp(p, soc, eps=1e-4):
    max_c = max(0.0, min(PMAX, (CAP - soc) / ETA - eps))
    max_d = max(0.0, min(PMAX, soc * ETA - eps))
    return float(min(max(p, -max_d), max_c))


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
class MPCHard:
    def __init__(self, w_fc, n_fc, H=36, kappa=KAPPA, lam=0.0, d_shade=1.0, c_extra=1.0):
        """d_shade: discharge at most d_shade * (forecast net load of this hour).
        lam: value (in w units) of one kWh left in the battery at the end of the horizon."""
        self.w_fc, self.n_fc = w_fc, n_fc
        self.H, self.kappa, self.lam, self.d_shade = H, kappa, lam, d_shade
        self.hist = []

    def record(self, obs):
        last = obs["noves_dades_5min"]
        rec = {"ts": obs["timestamp"], "co2": obs["co2_actual"], "price": obs["preu_actual"],
               "w": obs["co2_actual"] + self.kappa * obs["preu_actual"],
               "load_last": float(last["ultim_consum_kw"]), "pv_last": float(last["ultima_solar_kw"])}
        for c in METEO_COLS:
            rec[c] = float(last.get(c, np.nan))
        self.hist.append(rec)

    def act(self, obs):
        self.record(obs)
        soc = float(obs["soc_kwh"])
        w_known = (np.asarray(obs["co2_properes_24h"], float)
                   + self.kappa * np.asarray(obs["preus_properes_24h"], float))
        w = np.asarray(self.w_fc.predict(obs, self.hist, self.H), float).copy()
        n = np.asarray(self.n_fc.predict(obs, self.hist, self.H), float).copy()
        H = min(len(w), len(n))
        if H == 0:
            return 0.0
        w, n = w[:H], n[:H]
        k = min(len(w_known), H)
        w[:k] = w_known[:k]
        n_plan = n.copy()
        if self.d_shade != 1.0:
            n_plan[0] = n[0] * self.d_shade if n[0] > 0 else n[0]
        c, d, g = plan_lp(np.maximum(w, 1e-9), n_plan, soc, lam=self.lam)
        if c is None:
            return 0.0
        return safe_clamp(c[0] - d[0], soc)


# --------------------------------------------------------------------------------------
# Perfect-information forecasters (for decomposing where the score is lost; not legal agents)
# --------------------------------------------------------------------------------------
class PerfectW:
    """The true signal of the next H rows of the run."""
    def __init__(self, df, kappa=KAPPA):
        self.w = (df["co2_kg_per_kwh"] + kappa * df["precio_eur_per_kwh"]).to_numpy(float)

    def predict(self, obs, hist, H):
        i = len(hist) - 1
        return self.w[i:i + H]


class PerfectN:
    """The true net load of the next H rows of the run."""
    def __init__(self, df):
        self.n = (df["load_kw"] - df["pv_kw"]).to_numpy(float)

    def predict(self, obs, hist, H):
        i = len(hist) - 1
        return self.n[i:i + H]


# --------------------------------------------------------------------------------------
# Naive forecasters (baseline; fitted on train rows outside the evaluation fold)
# --------------------------------------------------------------------------------------
def _future_index(ts, H):
    """Calendar of the next H hours assuming consecutive hours (ignores week boundaries)."""
    h = (ts.hour + np.arange(H)) % 24
    days = (ts.hour + np.arange(H)) // 24
    dow = (ts.dayofweek + days) % 7
    return h, dow


class ClimW:
    """Signal forecast = mean by (month, weekend, hour) on the fit rows, rescaled so that it
    matches the 3 known hours; the correction fades with the horizon (half-life `half` hours)."""
    def __init__(self, fit_df, kappa=KAPPA, half=24.0):
        w = fit_df["co2_kg_per_kwh"] + kappa * fit_df["precio_eur_per_kwh"]
        idx = fit_df.index
        self.tab = w.groupby([idx.month, (idx.dayofweek >= 5).astype(int), idx.hour]).mean()
        self.tab_h = w.groupby([(idx.dayofweek >= 5).astype(int), idx.hour]).mean()
        self.kappa, self.half = kappa, half

    def _clim(self, month, we, hour):
        v = self.tab.get((month, we, hour))
        return float(v) if v is not None else float(self.tab_h.get((we, hour)))

    def predict(self, obs, hist, H):
        ts = obs["timestamp"]
        h, dow = _future_index(ts, H)
        clim = np.array([self._clim(ts.month, int(d >= 5), int(hh)) for hh, d in zip(h, dow)])
        known = (np.asarray(obs["co2_properes_24h"], float) + self.kappa * np.asarray(obs["preus_properes_24h"], float))
        k = min(len(known), H)
        if k == 0:
            return clim
        ratio = float(np.clip(np.mean(known[:k]) / max(np.mean(clim[:k]), 1e-9), 0.2, 5.0))
        fade = 0.5 ** (np.arange(H) / self.half)
        return clim * (1.0 + (ratio - 1.0) * fade)


class ProfileN:
    """Net load forecast = load profile - solar profile, both anchored on the last hour's
    readings. Load: mean by (weekend, hour) plus the last anomaly, fading with factor `phi`
    per hour. Solar: mean by (month, hour) times the last hour's clearness ratio (fading)."""
    def __init__(self, fit_df, phi=0.9, phi_pv=0.8):
        idx = fit_df.index
        we = (idx.dayofweek >= 5).astype(int)
        self.load = fit_df["load_kw"].groupby([we, idx.hour]).mean()
        self.pv = fit_df["pv_kw"].groupby([idx.month, idx.hour]).mean()
        self.pv_h = fit_df["pv_kw"].groupby(idx.hour).mean()
        self.phi, self.phi_pv = phi, phi_pv

    def _pv(self, month, hour):
        v = self.pv.get((month, hour))
        return float(v) if v is not None else float(self.pv_h.get(hour))

    def predict_load(self, obs, hist, H):
        ts = obs["timestamp"]
        h, dow = _future_index(ts, H)
        last = obs["noves_dades_5min"]
        lh = (ts.hour - 1) % 24
        ldow = ts.dayofweek if ts.hour > 0 else (ts.dayofweek - 1) % 7
        k = np.arange(1, H + 1)
        load_c = np.array([float(self.load.get((int(d >= 5), int(hh)))) for hh, d in zip(h, dow)])
        anom = float(last["ultim_consum_kw"]) - float(self.load.get((int(ldow >= 5), lh)))
        return np.maximum(load_c + anom * self.phi ** k, 0.0)

    def predict_pv(self, obs, hist, H):
        ts = obs["timestamp"]
        h, _ = _future_index(ts, H)
        last = obs["noves_dades_5min"]
        lh = (ts.hour - 1) % 24
        k = np.arange(1, H + 1)
        pv_c = np.array([self._pv(ts.month, int(hh)) for hh in h])
        pv_last_c = self._pv(ts.month, lh)
        ratio = float(np.clip(float(last["ultima_solar_kw"]) / pv_last_c, 0.0, 3.0)) if pv_last_c > 20.0 else 1.0
        return pv_c * (1.0 + (ratio - 1.0) * self.phi_pv ** k)

    def predict(self, obs, hist, H):
        return self.predict_load(obs, hist, H) - self.predict_pv(obs, hist, H)


class PerfectLoad:
    """True campus load of the next H rows (kW). Diagnostic only."""
    def __init__(self, df):
        self.x = df["load_kw"].to_numpy(float)

    def predict_load(self, obs, hist, H):
        i = len(hist) - 1
        return self.x[i:i + H]


class PerfectPV:
    """True solar output (already x12) of the next H rows (kW). Diagnostic only."""
    def __init__(self, df):
        self.x = df["pv_kw"].to_numpy(float)

    def predict_pv(self, obs, hist, H):
        i = len(hist) - 1
        return self.x[i:i + H]


class NetFromParts:
    """Net-load forecaster built from a load forecaster (.predict_load) and a solar forecaster
    (.predict_pv), so the two can be developed and scored separately."""
    def __init__(self, load_fc, pv_fc):
        self.load_fc, self.pv_fc = load_fc, pv_fc

    def predict(self, obs, hist, H):
        a = np.asarray(self.load_fc.predict_load(obs, hist, H), float)
        b = np.asarray(self.pv_fc.predict_pv(obs, hist, H), float)
        m = min(len(a), len(b))
        return a[:m] - b[:m]
