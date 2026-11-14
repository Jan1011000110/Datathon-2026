"""
experiments/28_signal_planning/sigplan.py
Planning that does not trust the signal forecast blindly (hard mode).

The plan of experiments/24_planner treats the forecast of w = co2 + kappa * price beyond the 3
known hours as certain. Here the first step is taken against a value of stored energy AVERAGED
over scenario paths of w (forecast + error paths drawn from honest, cross-fitted forecast errors),
optionally blended with the value on the forecast itself:

    v(soc) = (1 - lam) * v_forecast(soc) + lam * mean_s v_path_s(soc)

lam = 0 is the certainty-equivalent plan (what we had); lam = 1 is the perfect-information
relaxation (each path is planned as if it were known: optimistic about what we will know later).

How it is computed (numpy, vectorised over paths)
-------------------------------------------------
For a threshold tau, let L_k(tau) = kWh of battery whose stored energy is worth more than tau from
hour k on. The exact backward pass of planner.py (merge the hour's segments into the sorted value
list, cut pmax*eta off the top and pmax/eta off the bottom) reads, threshold by threshold,

    L_k(tau) = clip( L_{k+1}(tau) + [kWh of hour k's segments worth more than tau] - pmax*eta, 0, cap )

which is a scalar recursion: it runs for all thresholds and all K paths at once as one 2-D array.
The value function of a path is recovered from its L by the layer-cake formula
v(y) = integral of [L(tau) > y] d tau (trapezoid rule on the thresholds); the average over paths is
a step function handed to planner.first_step together with the exact segments of hour 0. The
thresholds are J evenly spaced levels plus the values of hour 0's own segments: with those, the
first step against a single path is exactly planner.py's; J only matters for averages of paths.

Hours beyond `m` use the forecast for every path (one common pass); only hours 1..m-1 are done per path.

    Paths(fit_set, K=..., m=...)                         error paths, by hour of day of the origin
    plan_wpaths(w, paths, omega, soc, sv1, sl, ...)      -> p0
    SigAgent(w_fc, net, pool=Paths(...), lam=...)        drop-in agent (same interface as CombinedAgent)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "experiments" / "24_planner"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import planner as P  # noqa: E402

CAP, PMAX, ETA = P.CAP, P.PMAX, P.ETA
KAPPA = P.KAPPA
METEO = ("irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms")


# --------------------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------------------
class Paths:
    """Scenario paths of the signal error over horizons 3..m-1, per hour of day of the origin.
    Whole error vectors are taken from cross-fitted origins at the same hour of day, so the
    correlation along the horizon is kept.

    mode    "add": w + e          "mul": w * (1 + e / forecast)   (errors grow with the level)
    select  "strat": the K vectors at evenly spaced ranks of their mean over the first 6 horizons
            "random": K vectors drawn at random (seed)
    centre  subtract the mean of the K vectors (the paths then average to the forecast)
    reveal  r: keep the first r errors, replace the later ones by their linear expectation given
            those r ("in r hours we will know this much"); None = whole vectors
    gamma   stretch of the errors (1 = as measured)
    Also fits `beta`: slope of the error at horizon k on the last revealed error (the surprise of
    the newest known hour against the forecast made one hour earlier)."""
    def __init__(self, fit_set="ALL", K=10, m=12, mode="add", select="strat", centre=True, reveal=None,
                 gamma=1.0, seed=0, errors=None):
        z = np.load(HERE / f"errors_{fit_set}.npz") if errors is None else errors
        E, F, hod = z["E"], z["F"], z["hod"]
        self.K, self.m, self.mode, self.gamma = K, m, mode, gamma
        ok = np.isfinite(E[:, 3:m]).all(1) & np.isfinite(F[:, 3:m]).all(1)
        X = E[ok, 3:m] if mode == "add" else E[ok, 3:m] / np.maximum(F[ok, 3:m], 1e-6)
        hod_ok = hod[ok]
        if reveal is not None and reveal < m - 3:
            A = X[:, :reveal]
            B = np.linalg.lstsq(A, X, rcond=None)[0]
            X = A @ B
        rng = np.random.default_rng(seed)
        self.sets = []
        # mean cross-fitted error by (hour of day of the origin, horizon): a bias correction of the forecast
        self.bias = np.array([E[ok & (hod == h), 3:m].mean(0) for h in range(24)])
        for h in range(24):
            Xh = X[hod_ok == h]
            if select == "strat":
                o = np.argsort(Xh[:, :6].mean(1))
                idx = o[((np.arange(K) + 0.5) / K * len(o)).astype(int)]
            else:
                idx = rng.choice(len(Xh), K, replace=len(Xh) < K)
            S = Xh[idx]
            if centre:
                S = S - S.mean(0)
            self.sets.append(S * gamma)
        # last revealed error -> next errors (additive), same run and consecutive origins only
        pos, blk = z["pos"], z["block"]
        same = (pos[1:] == pos[:-1] + 1) & (blk[1:] == blk[:-1])
        prev = np.full(len(E), np.nan)
        prev[1:] = np.where(same, E[:-1, 3], np.nan)
        self.beta = np.zeros(m)
        for k in range(3, m):
            g = np.isfinite(prev) & np.isfinite(E[:, k])
            self.beta[k] = float(prev[g] @ E[g, k] / (prev[g] @ prev[g]))

    def paths(self, w, hod):
        """(K, min(m, H)) scenario paths of w for hours 0..m-1 (hours 0..2 are the known values)."""
        mm = min(self.m, len(w))
        out = np.tile(w[:mm], (self.K, 1))
        if mm > 3:
            S = self.sets[hod][:, :mm - 3]
            out[:, 3:] = w[3:mm] + S if self.mode == "add" else w[3:mm] * (1.0 + S)
        return np.maximum(out, 1e-6)


# --------------------------------------------------------------------------------------
# Planner against scenario paths of w
# --------------------------------------------------------------------------------------
def _nodes(w, paths, s0, v_end, J, eta):
    """Thresholds t (ascending, t[0] = 0) and trapezoid weights c for the layer-cake integral. The
    values of hour 0's own segments are among the thresholds, so every comparison the first step
    makes against a single path is exact; J only sets how finely an AVERAGE of paths is resolved."""
    top = max(float(w.max()), float(paths.max())) / eta * 1.0001
    if v_end is not None:
        top = max(top, float(v_end) * 1.0001)
    t = np.unique(np.concatenate((np.linspace(0.0, top, J + 1), s0[s0 > 0.0])))
    d = np.diff(t)
    c = np.zeros(len(t))
    c[1:] += 0.5 * d
    c[:-1] += 0.5 * d
    return t, c


def path_levels(w, paths, sv1, sl, v_end=None, J=64, cap=CAP, pmax=PMAX, eta=ETA):
    """L (K, len(t)): kWh of battery worth more than t_j after hour 0, for each path; and the weights
    c. sv1 / sl: per-hour segment values for w = 1 (non-increasing) and widths (planner.stage_segments)."""
    H = len(w)
    K, mm = paths.shape
    tau, c = _nodes(w, paths, w[0] * sv1[0], v_end, J, eta)
    top = pmax * eta
    if mm < H:                                               # common tail (the forecast itself): exact list pass
        tv = [None] + [w[k] * sv1[k] for k in range(mm, H)]
        tl = [None] + [sl[k] for k in range(mm, H)]
        vv, vl = P.backward(tv, tl, v_end, cap, pmax, eta)
        L = np.concatenate(((0.0,), np.cumsum(vl)))[np.searchsorted(-vv, -tau)]      # kWh worth more than tau
    else:
        L = np.zeros(len(tau)) if v_end is None else cap * (float(v_end) > tau)
    L = np.tile(L, (K, 1))
    if mm > 1:
        # hours 1..mm-1 for all paths at once. Each segment of each hour is worth (path's w) x (its
        # value for w = 1); it counts for every threshold below that worth. One searchsorted of the
        # K x N worths into the thresholds, a bincount and a cumulative sum give, for every path, hour
        # and threshold, the kWh worth more than the threshold.
        nst = mm - 1
        sizes = [len(sv1[j + 1]) for j in range(nst)]
        v1 = np.concatenate(sv1[1:mm])
        wd = np.concatenate(sl[1:mm])
        hr = np.repeat(np.arange(nst), sizes)
        N, JJ = len(v1), len(tau)
        idx = np.searchsorted(tau, (paths[:, 1:mm][:, hr] * v1).ravel())          # thresholds below the worth
        flat = np.repeat(np.arange(K) * nst, N) * (JJ + 1) + np.tile(hr * (JJ + 1), K) + idx
        A = np.bincount(flat, weights=np.tile(wd, K), minlength=K * nst * (JJ + 1)).reshape(K, nst, JJ + 1)
        lam_ = pmax / eta - np.cumsum(A, axis=2)[:, :, :JJ]                        # kWh above threshold - top
        for j in range(nst - 1, -1, -1):
            L = np.clip(L + lam_[:, j, :], 0.0, cap)
    return L, c


def _path_levels_loop(w, paths, sv1, sl, v_end=None, J=64, cap=CAP, pmax=PMAX, eta=ETA):
    """Reference implementation of path_levels (one searchsorted per hour); used in tests only."""
    H = len(w)
    K, mm = paths.shape
    tau, c = _nodes(w, paths, w[0] * sv1[0], v_end, J, eta)
    top = pmax * eta
    L = np.zeros(len(tau)) if v_end is None else cap * (float(v_end) > tau)
    for k in range(H - 1, mm - 1, -1):
        cw = np.concatenate(((0.0,), np.cumsum(sl[k])))
        L = np.clip(L + cw[np.searchsorted(-sv1[k], -(tau / w[k]))] - top, 0.0, cap)
    L = np.tile(L, (K, 1))
    for k in range(mm - 1, 0, -1):
        cw = np.concatenate(((0.0,), np.cumsum(sl[k])))
        L = np.clip(L + cw[np.searchsorted(-sv1[k], -(tau[None, :] / paths[:, k, None]))] - top, 0.0, cap)
    return L, c


def value_from_levels(L, omega, c, cap=CAP):
    """Weighted average over paths of the marginal value functions, as (values descending, widths)."""
    pos = L.ravel()
    drop = np.outer(omega, c).ravel()
    o = np.argsort(pos, kind="stable")
    vv = float(drop.sum()) - np.concatenate(((0.0,), np.cumsum(drop[o])))
    vl = np.diff(np.concatenate(((0.0,), pos[o], (cap,))))
    return vv, vl


def plan_wpaths(w, paths, omega, soc, sv1, sl, v_end=None, J=64, mode="value", cap=CAP, pmax=PMAX, eta=ETA):
    """First command (kW, + charge). w: (H,) signal (hours 0..2 known). paths: (K, m) scenario paths of
    hours 0..m-1 with weights omega (K,) (the forecast itself may be one of them: that is the blend).
    mode "value": act against the averaged value of stored energy; "action" / "median": average
    (or weighted median) of the first actions of the individual path plans."""
    L, c = path_levels(w, paths, sv1, sl, v_end, J, cap, pmax, eta)
    s0 = w[0] * sv1[0]
    if mode == "value":
        vv, vl = value_from_levels(L, omega, c, cap)
        return P.first_step(soc, s0, sl[0], vv, vl, cap, pmax, eta)
    one = np.ones(1)
    acts = np.array([P.first_step(soc, s0, sl[0], *value_from_levels(L[i:i + 1], one, c, cap), cap, pmax, eta)
                     for i in range(len(L))])
    if mode == "action":
        return float(acts @ omega / omega.sum())
    o = np.argsort(acts)
    c = np.cumsum(omega[o]) / omega.sum()
    return float(acts[o][np.searchsorted(c, 0.5)])


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
def make_rec(obs, kappa=KAPPA):
    """History record of one step, weather readings included (the solar forecaster reads them)."""
    last = obs["noves_dades_5min"]
    rec = {"ts": obs["timestamp"], "co2": obs["co2_actual"], "price": obs["preu_actual"],
           "w": obs["co2_actual"] + kappa * obs["preu_actual"],
           "load_last": float(last["ultim_consum_kw"]), "pv_last": float(last["ultima_solar_kw"])}
    for c in METEO:
        rec[c] = float(last.get(c, np.nan))
    return rec


class SigAgent:
    def __init__(self, w_fc, net, pool=None, lam=1.0, H=36, Q=9, scen_hours=12, J=64, mode="value",
                 ar=0.0, bias=0.0, v_end=None, kappa=KAPPA, truncate_end=True, exact=False):
        """Same contract as experiments/27_combined CombinedAgent (w_fc.predict, net.predict,
        net.predict_scen), plus:
        pool   Paths object (None = plan on the forecast only)
        lam    weight of the path-averaged value against the value on the forecast (0..1)
        ar     multiplier of the fitted correction 'last revealed error -> next errors' (0 = off)
        bias   multiplier of the mean cross-fitted error by (hour of day, horizon) added to the forecast
        exact  lam = 0 only: use planner.plan_stoch itself (no value-axis grid)"""
        self.w_fc, self.net, self.pool, self.lam = w_fc, net, pool, lam
        self.H, self.Q, self.scen_hours, self.J, self.mode = H, Q, scen_hours, J, mode
        self.ar, self.v_end, self.kappa, self.truncate_end, self.exact = ar, v_end, kappa, truncate_end, exact
        self.bias = bias
        self.hist = []
        self._prev = None                                    # (timestamp, forecast of hour +3) of the last step
        self.sec = 0.0                                       # time spent in the planner (s)

    def act(self, obs):
        import time
        self.hist.append(make_rec(obs, self.kappa))
        soc = float(obs["soc_kwh"])
        w_known = (np.asarray(obs["co2_properes_24h"], float)
                   + self.kappa * np.asarray(obs["preus_properes_24h"], float))
        w = np.asarray(self.w_fc.predict(obs, self.hist, self.H), float).copy()
        n = np.asarray(self.net.predict(obs, self.hist, self.H), float)
        H = min(len(w), len(n))
        v_end = self.v_end
        if self.truncate_end and len(w_known) < 3:
            H = min(H, len(w_known))
            v_end = None
        if H == 0:
            return 0.0
        w, n = w[:H], n[:H]
        k = min(len(w_known), H)
        f3 = float(w[3]) if H > 3 else None
        w[:k] = w_known[:k]
        ts = obs["timestamp"]
        if self.ar and self.pool is not None and self._prev is not None and k == 3 and H > 3:
            pts, pf3 = self._prev
            if pf3 is not None and (ts - pts).total_seconds() == 3600.0:
                mm = min(self.pool.m, H)
                w[3:mm] += self.ar * self.pool.beta[3:mm] * (w_known[2] - pf3)
        self._prev = (ts, f3)
        if self.bias and self.pool is not None and H > 3:
            mm = min(self.pool.m, H)
            w[3:mm] += self.bias * self.pool.bias[ts.hour][:mm - 3]
        w = np.maximum(w, 1e-9)
        Kn = H if self.scen_hours is None else min(self.scen_hours, H)
        t0 = time.perf_counter()
        if Kn == 0:
            scen = probs = None
            a, b = P.stage_segments_det(np.ones(H), n)
            sv1, sl = list(a), list(b)
        else:
            scen, probs = self.net.predict_scen(obs, self.hist, Kn, self.Q)
            scen = scen[:Kn]
            t0 = time.perf_counter()
            sv1, sl = P._segments(np.ones(H), scen, probs, n[Kn:H], PMAX, ETA)
        use = self.pool is not None and self.lam > 0 and H > 3
        if not use and self.exact:
            p = P.plan_det(w, n, soc, v_end=v_end) if Kn == 0 else \
                P.plan_stoch(w, scen, probs, soc, v_end=v_end, n_tail=n[Kn:H])
        else:
            if use:
                paths = self.pool.paths(w, ts.hour)
                omega = np.full(len(paths), self.lam / len(paths))
                if self.lam < 1.0:
                    paths = np.vstack((w[None, :paths.shape[1]], paths))
                    omega = np.concatenate(((1.0 - self.lam,), omega))
            else:
                paths, omega = w[None, :min(H, 3)], np.ones(1)
            p = plan_wpaths(w, paths, omega, soc, sv1, sl, v_end=v_end, J=self.J, mode=self.mode)
        self.sec += time.perf_counter() - t0
        return P.safe_clamp(p, soc)
