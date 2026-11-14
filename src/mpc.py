"""
src/mpc.py
Experiment-oriented rolling-horizon optimiser (MPC). Not the submission file: this one is
configurable and uses scipy's HiGHS; the packaged agent lives in agents/.

Each hour:
  1. Build the per-kWh score signal for the known window (current hour + up to 23 more):
         w_k = co2_k + kappa * price_k          with kappa = (0.3/0.7) * rho,
     rho = CO2_baseline / Cost_baseline (kg per EUR) -- the score normalises both terms by
     their baseline totals, so only this ratio matters.
  2. Optionally extend the window with a forecast of w for the hours after the lookahead
     (`extend_fn`), and/or attach a value per kWh left in the battery at the end (`lam_fn`).
  3. Solve the LP (pure arbitrage; discharge capped by expected net load) and apply hour 0.
"""
import numpy as np
from scipy.optimize import linprog

CAP, PMAX, ETA = 2000.0, 500.0, 0.9 ** 0.5
RHO_TRAIN = 0.587            # kg CO2 per EUR of baseline import, measured on the train set
EPS = 1e-4


def plan_lp(w, soc0, d_cap=None, lam=0.0):
    """Arbitrage LP over len(w) hours.  x = [c_0..c_{H-1}, d_0..d_{H-1}]
        min  sum_k w_k (c_k - d_k)  -  lam * s_H
        s.t. 0 <= soc0 + sum_{j<=k} (ETA c_j - d_j/ETA) <= CAP      for every k
             0 <= c_k <= PMAX,  0 <= d_k <= d_cap_k
    Returns (c, d)."""
    H = len(w)
    w = np.asarray(w, float)
    cost = np.concatenate([w - lam * ETA, -w + lam / ETA])
    tri = np.tril(np.ones((H, H)))
    A = np.hstack([ETA * tri, -tri / ETA])
    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([np.full(H, CAP - soc0), np.full(H, soc0)])
    ub_d = np.full(H, PMAX) if d_cap is None else np.minimum(PMAX, np.asarray(d_cap, float))
    bounds = np.column_stack([np.zeros(2 * H), np.concatenate([np.full(H, PMAX), ub_d])])
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if r.status != 0:
        return None, None
    return r.x[:H], r.x[H:]


def safe_clamp(p, soc):
    max_c = max(0.0, min(PMAX, (CAP - soc) / ETA - EPS))
    max_d = max(0.0, min(PMAX, soc * ETA - EPS))
    return float(min(max(p, -max_d), max_c))


def extend_periodic(known_w, n_ext, hist=None, obs=None):
    """Forecast for the hours after the lookahead: repeat the known window (24 h period)."""
    k = np.asarray(known_w, float)
    reps = int(np.ceil(n_ext / len(k)))
    return np.tile(k, reps)[:n_ext]


class MPCAgent:
    def __init__(self, rho=RHO_TRAIN, n_ext=0, extend_fn=extend_periodic, lam_fn=None,
                 load_guard=True, guard_margin=0.0):
        self.kappa = (0.3 / 0.7) * rho
        self.n_ext = n_ext
        self.extend_fn = extend_fn
        self.lam_fn = lam_fn
        self.load_guard = load_guard
        self.guard_margin = guard_margin
        self.hist = []            # per-hour dicts of what has been observed so far

    def signal(self, obs):
        co2 = np.asarray(obs["co2_properes_24h"], float)
        price = np.asarray(obs["preus_properes_24h"], float)
        return co2 + self.kappa * price

    def act(self, obs):
        w_known = self.signal(obs)
        last = obs["noves_dades_5min"]
        last_net = float(last["ultim_consum_kw"]) - float(last["ultima_solar_kw"])
        self.hist.append({"ts": obs["timestamp"], "w": w_known[0], "co2": obs["co2_actual"],
                          "price": obs["preu_actual"], "last_net": last_net})
        soc = float(obs["soc_kwh"])

        end_game = len(w_known) < 24          # the simulation ends when the lookahead shrinks
        w = w_known
        lam = 0.0
        if not end_game:
            if self.n_ext > 0:
                w = np.concatenate([w_known, self.extend_fn(w_known, self.n_ext, self.hist, obs)])
            if self.lam_fn is not None:
                lam = float(self.lam_fn(w_known, self.hist, obs))

        d_cap = None
        if self.load_guard:
            # only this hour's discharge is capped: by last hour's net load (persistence)
            d_cap = np.full(len(w), PMAX)
            d_cap[0] = min(PMAX, max(0.0, last_net - self.guard_margin))

        c, d = plan_lp(w, soc, d_cap=d_cap, lam=lam)
        if c is None:
            return 0.0
        return safe_clamp(c[0] - d[0], soc)
