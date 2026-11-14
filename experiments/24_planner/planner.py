"""
experiments/24_planner/planner.py
Fast exact planner for hard mode. numpy only (no scipy at run time).

Problem (one call per hour)
---------------------------
    min  sum_k  E[ w_k * max(n_k + p_k, 0) ]  -  v_end * soc_H
    soc_{k+1} = soc_k + ETA * p_k   (p_k >= 0, charge)      |p_k| <= pmax
              = soc_k + p_k / ETA   (p_k <  0, discharge)   0 <= soc <= cap
w_k = score signal (co2 + kappa * price), n_k = campus net load (load - solar, kW; < 0 = free
solar surplus), p_k = battery command. The expectation is over net-load scenarios of hour k
(one scenario = the deterministic problem = src.hard_mpc.plan_lp).

Method: exact dynamic programme on the MARGINAL VALUE of stored energy
---------------------------------------------------------------------
The cost-to-go V_k(soc) is convex and piecewise linear, so it is fully described by its slope
v_k(soc) = -dV_k/dsoc >= 0: "what one more stored kWh is worth from hour k on", a decreasing
step function of soc. We keep it as a short sorted list of (value, width in kWh) segments.

One hour of the campus is also a sorted list of segments, in the same units (per kWh of SOC):
    charging   the kW between p and p+dp costs    (w / ETA) * P(n > -p)   (free while solar surplus covers it)
    discharge  the kW between d and d+dd earns    (w * ETA) * P(n >  d)   (nothing once it exceeds the load)
Going one hour back is then just: merge the two sorted lists, and cut pmax*ETA kWh off the
top (the most we can add in an hour) and pmax/ETA kWh off the bottom (the most we can take
out). This is the infimal convolution of two convex functions and it is exact: no SOC grid,
no discretisation error, cost O(H * (segments)) with a handful of numpy calls per hour.

First step: discharge the d-th kW only if  w0 * ETA * P(n0 > d)  >  v_1(soc after),
charge the p-th kW only if  v_1(soc after)  >  (w0 / ETA) * P(n0 > -p).  With a continuous
forecast distribution this is the newsvendor rule: discharge up to the quantile of the net
load at level  1 - v / (w0 * ETA).

Public functions
----------------
    plan_det(w, n, soc, ...)                          -> p0    (same optimum as plan_lp)
    plan_stoch(w, n_scen, probs, soc, n_tail=None)    -> p0    (expected import cost)
    MPCStoch(w_fc, n_fc, ...)                         drop-in agent for src.hard.simulate
    QuantileN(fit_df, ...)                            ProfileN + empirical residual quantiles
"""
from bisect import bisect_left as _bisect

import numpy as np

CAP = 6000.0
PMAX = 1500.0
ETA = 0.9 ** 0.5
RHO_HARD = 0.634
KAPPA = (0.3 / 0.7) * RHO_HARD


# --------------------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------------------
def stage_segments(w, n_scen, probs, pmax=PMAX, eta=ETA):
    """Marginal value / cost of battery power in each hour, per kWh of SOC.
    w: (H,), n_scen: (H, Q), probs: (Q,). Returns (val, wid), both (H, Q + 2): the segments of
    hour k ordered from full charge (+pmax) to full discharge (-pmax); val is non-increasing
    along that order, wid is the SOC width (kWh) of each segment (sums to pmax*eta + pmax/eta)."""
    H, Q = n_scen.shape
    y = np.empty((H, Q + 3))
    y[:, 0] = -pmax
    y[:, 1] = 0.0
    y[:, 2] = pmax
    np.clip(n_scen, -pmax, pmax, out=y[:, 3:])
    y.sort(axis=1)
    mid = 0.5 * (y[:, 1:] + y[:, :-1])
    surv = ((n_scen[:, None, :] > mid[:, :, None]) * probs).sum(axis=2)      # P(n > mid)
    neg = mid < 0.0
    wid = (y[:, 1:] - y[:, :-1]) * np.where(neg, eta, 1.0 / eta)
    val = w[:, None] * surv * np.where(neg, 1.0 / eta, eta)
    return val, wid


def stage_segments_det(w, n, pmax=PMAX, eta=ETA):
    """Same for a single scenario per hour (3 segments per hour); cheaper than the general one."""
    H = len(w)
    y = np.clip(n, -pmax, pmax)
    val = np.zeros((H, 3))
    wid = np.empty((H, 3))
    pos = y > 0.0
    # charge side: [-pmax, min(y, 0)] paid at w/eta, [min(y, 0), 0] free
    ym = np.minimum(y, 0.0)
    wid[:, 0] = (ym + pmax) * eta
    val[:, 0] = w / eta
    # middle: free charging (surplus) if y < 0, useful discharge if y > 0
    wid[:, 1] = np.where(pos, y / eta, -ym * eta)
    val[:, 1] = np.where(pos, w * eta, 0.0)
    # discharge beyond the load: worth nothing
    wid[:, 2] = (pmax - np.maximum(y, 0.0)) / eta
    return val, wid


def _v_end(v_end, cap):
    if v_end is None:
        return np.zeros(1), np.array([cap])
    if np.ndim(v_end) == 0:
        return np.array([float(v_end)]), np.array([cap])
    vv, vl = v_end                                    # (values descending, widths summing to cap)
    return np.asarray(vv, float), np.asarray(vl, float)


def backward(seg_val, seg_wid, v_end=None, cap=CAP, pmax=PMAX, eta=ETA, keep=False):
    """Backward pass over hours len-1 .. 1. seg_val / seg_wid: per-hour arrays (2-D array or list
    of 1-D arrays). Returns (vv, vl): marginal value of stored energy right after hour 0 as
    segments over [0, cap] (vv descending). keep=True returns the list for every hour
    (index k = value after hour k-1).

    Plain Python lists + bisect: with a few hundred segments at most this is several times
    faster than numpy (the cost of a numpy call dominates at this size)."""
    vv, vl = _v_end(v_end, cap)
    vals = vv[::-1].tolist()                          # ascending values = from SOC cap down to 0
    wids = vl[::-1].tolist()
    top = pmax * eta                                  # most SOC we can add in one hour
    bot = pmax / eta                                  # most SOC we can take out in one hour
    H = len(seg_val)
    sv = seg_val.tolist() if isinstance(seg_val, np.ndarray) else [np.asarray(a).tolist() for a in seg_val]
    sw = seg_wid.tolist() if isinstance(seg_wid, np.ndarray) else [np.asarray(a).tolist() for a in seg_wid]
    out = [None] * (H + 1) if keep else None
    if keep:
        out[H] = (vv, vl)
    for k in range(H - 1, 0, -1):
        for m, ln in zip(sv[k], sw[k]):               # merge this hour's segments into the sorted list
            if ln > 1e-9:
                i = _bisect(vals, m)
                vals.insert(i, m)
                wids.insert(i, ln)
        r = top                                       # cut `top` kWh off the high-value end
        while wids:
            ln = wids[-1]
            if ln <= r:
                r -= ln
                wids.pop()
                vals.pop()
            else:
                wids[-1] = ln - r
                break
        r = bot                                       # cut `bot` kWh off the low-value end
        i = 0
        nw = len(wids)
        while i < nw:
            ln = wids[i]
            if ln <= r:
                r -= ln
                i += 1
            else:
                wids[i] = ln - r
                break
        if i:
            del vals[:i]
            del wids[:i]
        if keep:
            out[k] = (np.array(vals[::-1]), np.array(wids[::-1]))
    if keep:
        return out
    return np.array(vals[::-1]), np.array(wids[::-1])


def first_step(soc, sv, sl, vv, vl, cap=CAP, pmax=PMAX, eta=ETA):
    """Optimal command (kW, + charge) for this hour. sv, sl: this hour's segments
    (stage_segments row); vv, vl: marginal value after this hour. Ties are broken towards the
    highest SOC: never spill, always take free solar."""
    xlo = max(-pmax * eta, soc - cap)                 # x = SOC taken out of the battery (kWh)
    xhi = min(pmax / eta, soc)
    if xhi <= xlo:
        return 0.0
    bx = np.cumsum(sl) - pmax * eta                   # right end of each stage segment, in x
    bs = np.cumsum(vl)                                # right end of each value segment, in SOC
    cand = np.concatenate((bx, soc - bs, (xlo, xhi)))
    np.clip(cand, xlo, xhi, out=cand)
    cand.sort()
    mid = 0.5 * (cand[1:] + cand[:-1])
    i0 = np.minimum(np.searchsorted(bx, mid, side="right"), len(sv) - 1)
    i1 = np.minimum(np.searchsorted(bs, soc - mid, side="right"), len(vv) - 1)
    g = sv[i0] - vv[i1]                               # gain of taking one more kWh out; non-increasing in x
    tol = 1e-12 * (1.0 + sv[0])
    nz = np.flatnonzero(g > tol)
    x = cand[nz[-1] + 1] if len(nz) else xlo
    return float(-x * eta if x > 0 else -x / eta)


# --------------------------------------------------------------------------------------
# Planners
# --------------------------------------------------------------------------------------
def plan_det(w, n, soc, cap=CAP, pmax=PMAX, eta=ETA, v_end=None, full=False):
    """Deterministic plan: first command p0 (kW, + charge), same optimum as plan_lp(w, n, soc).
    full=True returns the whole plan (array of H commands)."""
    w = np.asarray(w, float)
    n = np.asarray(n, float)
    sv, sl = stage_segments_det(w, n, pmax, eta)
    if not full:
        vv, vl = backward(sv, sl, v_end, cap, pmax, eta)
        return first_step(soc, sv[0], sl[0], vv, vl, cap, pmax, eta)
    return _forward(soc, sv, sl, v_end, cap, pmax, eta)


def _segments(w, n_scen, probs, n_tail, pmax, eta):
    w = np.asarray(w, float)
    n_scen = np.asarray(n_scen, float)
    if n_scen.ndim == 1:
        n_scen = n_scen[None, :]
    probs = np.asarray(probs, float)
    K = len(n_scen)
    sv, sl = stage_segments(w[:K], n_scen, probs, pmax, eta)
    sv, sl = list(sv), list(sl)
    if n_tail is not None and len(n_tail):
        tv, tl = stage_segments_det(w[K:K + len(n_tail)], np.asarray(n_tail, float), pmax, eta)
        sv += list(tv)
        sl += list(tl)
    return sv, sl


def plan_stoch(w, n_scen, probs, soc, cap=CAP, pmax=PMAX, eta=ETA, v_end=None, n_tail=None, full=False):
    """Plan against the EXPECTED import cost. n_scen: (K, Q) net-load scenarios of the first K
    hours (K >= 1; hour 0 = the hour being decided), probs: (Q,). n_tail: point forecast for
    hours K .. H-1 (optional). Hours are treated as independent. Returns p0 (kW, + charge)."""
    sv, sl = _segments(w, n_scen, probs, n_tail, pmax, eta)
    if not full:
        vv, vl = backward(sv, sl, v_end, cap, pmax, eta)
        return first_step(soc, sv[0], sl[0], vv, vl, cap, pmax, eta)
    return _forward(soc, sv, sl, v_end, cap, pmax, eta)


def _forward(soc, sv, sl, v_end, cap, pmax, eta):
    vs = backward(sv, sl, v_end, cap, pmax, eta, keep=True)
    H = len(sv)
    p = np.zeros(H)
    s = float(soc)
    for k in range(H):
        vv, vl = vs[k + 1]
        p[k] = first_step(s, sv[k], sl[k], vv, vl, cap, pmax, eta)
        s = s + p[k] * eta if p[k] > 0 else s + p[k] / eta
        s = min(max(s, 0.0), cap)
    return p


def marginal_value(w, n, soc_grid, cap=CAP, pmax=PMAX, eta=ETA, v_end=None):
    """v_1 on a grid of SOC values (for plots / diagnostics)."""
    sv, sl = stage_segments_det(np.asarray(w, float), np.asarray(n, float), pmax, eta)
    vv, vl = backward(sv, sl, v_end, cap, pmax, eta)
    i = np.minimum(np.searchsorted(np.cumsum(vl), soc_grid, side="right"), len(vv) - 1)
    return vv[i]


def expected_cost(w, n_scen, probs, p):
    """sum_k E[w_k max(n_k + p_k, 0)] for a plan p (testing)."""
    return float(np.sum(w * (np.maximum(n_scen + p[:, None], 0.0) * probs).sum(axis=1)))


def safe_clamp(p, soc, cap=CAP, pmax=PMAX, eta=ETA, eps=1e-4):
    max_c = max(0.0, min(pmax, (cap - soc) / eta - eps))
    max_d = max(0.0, min(pmax, soc * eta - eps))
    return float(min(max(p, -max_d), max_c))


# --------------------------------------------------------------------------------------
# Net-load forecaster with a distribution: point forecast + empirical residual quantiles
# --------------------------------------------------------------------------------------
class QuantileN:
    """Wraps a point forecaster of the net load (default: src.hard_mpc.ProfileN) and adds the
    empirical distribution of its errors by (hour of day of the target hour, horizon), measured
    on the fit rows (replayed exactly as the simulator would present them).

        predict(obs, hist, H)            -> point forecast, (H,)
        predict_scen(obs, hist, H, Q)    -> (scen (H, Q), probs (Q,)): scenario j = point +
                                            mean of the residuals in the j-th of Q equal-probability bins
    """
    FINE = 360

    def __init__(self, fit_df, point=None, H=48, by_hour=True):
        if point is None:
            from src.hard_mpc import ProfileN
            point = ProfileN(fit_df)
        self.point, self.Hfit = point, H
        self.table = self._fit(fit_df, H, by_hour)          # (24, H, FINE) residual quantiles
        self._cache = {}
        self._last = (None, None)

    def _fit(self, df, H, by_hour):
        from src.hard import METEO_COLS
        ts = df.index
        load = df["load_kw"].to_numpy(float)
        pv = df["pv_kw"].to_numpy(float)
        net = load - pv
        met = {c: df[c].to_numpy(float) for c in METEO_COLS if c in df.columns}
        hrs = (ts.values.astype("datetime64[h]").astype(np.int64))
        T = len(df)
        res = np.full((T, H), np.nan)
        for t in range(1, T):
            if hrs[t] - hrs[t - 1] != 1:
                continue                                    # last reading is not the previous hour
            last = {"ultim_consum_kw": load[t - 1], "ultima_solar_kw": pv[t - 1]}
            for c, a in met.items():
                last[c] = a[t - 1]
            obs = {"timestamp": ts[t], "noves_dades_5min": last}
            f = np.asarray(self.point.predict(obs, None, H), float)
            m = min(H, T - t)
            ok = (hrs[t:t + m] - hrs[t]) == np.arange(m)    # target hours that really are t + k
            r = np.full(H, np.nan)
            r[:m] = np.where(ok, net[t:t + m] - f[:m], np.nan)
            res[t] = r
        self.resid = res
        hod0 = ts.hour.to_numpy()
        lev = (np.arange(self.FINE) + 0.5) / self.FINE
        tab = np.zeros((24, H, self.FINE))
        for k in range(H):
            hod_k = (hod0 + k) % 24
            col = res[:, k]
            good = ~np.isnan(col)
            for h in range(24):
                x = col[good & (hod_k == h)] if by_hour else col[good]
                tab[h, k] = np.quantile(x, lev) if len(x) >= 20 else np.quantile(col[good], lev)
        return tab

    def offsets(self, Q):
        """(24, H, Q) scenario offsets: mean residual in each of Q equal-probability bins."""
        if Q not in self._cache:
            assert self.FINE % Q == 0, "Q must divide %d" % self.FINE
            self._cache[Q] = self.table.reshape(24, self.Hfit, Q, self.FINE // Q).mean(axis=3)
        return self._cache[Q]

    def predict(self, obs, hist, H):
        last = obs["noves_dades_5min"]                      # the agent asks twice per step: compute once
        key = (obs["timestamp"], H, float(last["ultim_consum_kw"]), float(last["ultima_solar_kw"]))
        if self._last[0] != key:
            self._last = (key, np.asarray(self.point.predict(obs, hist, H), float))
        return self._last[1]

    def predict_scen(self, obs, hist, H, Q=9, flat_horizon=None):
        """flat_horizon = k: use the spread of horizon k for every hour (e.g. 0 = 'in each future
        hour I will know as much as I know now about this hour')."""
        f = self.predict(obs, hist, H)
        H = len(f)
        off = self.offsets(Q)
        hod = (obs["timestamp"].hour + np.arange(H)) % 24
        kk = np.minimum(np.arange(H), self.Hfit - 1) if flat_horizon is None else np.full(H, flat_horizon)
        return f[:, None] + off[hod, kk], np.full(Q, 1.0 / Q)


# --------------------------------------------------------------------------------------
# Agent (harness version: same forecaster interface as src.hard_mpc.MPCHard)
# --------------------------------------------------------------------------------------
class MPCStoch:
    def __init__(self, w_fc, n_fc, H=36, kappa=KAPPA, v_end=None, Q=9, scen_hours=1,
                 flat_horizon=None, d_shade=1.0, spread=1.0):
        """n_fc must offer predict(obs, hist, H) and, if scen_hours > 0,
        predict_scen(obs, hist, H, Q) -> (scen (H, Q), probs (Q,)).
        scen_hours: number of leading hours planned against scenarios (0 = deterministic,
        1 = this hour only, None = every hour); the rest use the point forecast.
        d_shade: deterministic haircut on hour 0 (only meaningful with scen_hours = 0).
        spread: stretch the scenarios around their mean (1.0 = as given; to test calibration)."""
        self.w_fc, self.n_fc = w_fc, n_fc
        self.H, self.kappa, self.v_end, self.Q = H, kappa, v_end, Q
        self.scen_hours, self.flat_horizon, self.d_shade = scen_hours, flat_horizon, d_shade
        self.spread = spread
        self.hist = []

    def record(self, obs):
        last = obs["noves_dades_5min"]
        self.hist.append({"ts": obs["timestamp"], "co2": obs["co2_actual"], "price": obs["preu_actual"],
                          "w": obs["co2_actual"] + self.kappa * obs["preu_actual"],
                          "load_last": float(last["ultim_consum_kw"]), "pv_last": float(last["ultima_solar_kw"])})

    def act(self, obs):
        self.record(obs)
        soc = float(obs["soc_kwh"])
        w_known = (np.asarray(obs["co2_properes_24h"], float)
                   + self.kappa * np.asarray(obs["preus_properes_24h"], float))
        w = np.asarray(self.w_fc.predict(obs, self.hist, self.H), float).copy()
        n = np.asarray(self.n_fc.predict(obs, self.hist, self.H), float)
        H = min(len(w), len(n))
        if H == 0:
            return 0.0
        w, n = w[:H], n[:H]
        k = min(len(w_known), H)
        w[:k] = w_known[:k]
        w = np.maximum(w, 1e-9)
        K = H if self.scen_hours is None else min(self.scen_hours, H)
        if K == 0:
            if self.d_shade != 1.0 and n[0] > 0:
                n = n.copy()
                n[0] *= self.d_shade
            p = plan_det(w, n, soc, v_end=self.v_end)
        else:
            if self.flat_horizon is None:
                scen, probs = self.n_fc.predict_scen(obs, self.hist, H, self.Q)
            else:
                scen, probs = self.n_fc.predict_scen(obs, self.hist, H, self.Q, self.flat_horizon)
            scen = scen[:K]
            if self.spread != 1.0:
                m = (scen @ probs)[:, None]
                scen = m + self.spread * (scen - m)
            p = plan_stoch(w, scen, probs, soc, v_end=self.v_end, n_tail=n[K:H])
        return safe_clamp(p, soc)
