"""
agent.py  --  BatteryAgent v1, weighting probe RHO = 0.575 (exact rolling-horizon optimiser, numpy only)

Idea
----
The campus always draws far more than the battery's 500 kW, so every kWh the battery delivers
displaces a kWh of grid import. The score is then linear in the battery schedule:

    score  ~  sum_t  -P_t * w_t ,        w_t = co2_t + KAPPA * price_t

(KAPPA converts EUR into "kg-equivalents": the score divides CO2 and cost savings by their
baseline totals and weighs them 70/30.) The signal w is known for the next 24 hours, so each
hour we solve the optimal charge/discharge plan for the known window *exactly* and apply its
first step.

Solver
------
Dynamic programming on the marginal value of stored energy v(s) ("how much is one more kWh in
the battery worth?"). Going backwards from the end of the window, for an hour with signal w:

    b = w / eta   (cost of adding 1 kWh to the battery)     a = w * eta   (value of removing 1 kWh)
    v_new(s) = min( v(s - D), max( a, max( v(s + C), min(b, v(s)) ) ) )

with C = 500*eta and D = 500/eta the largest state-of-charge moves in one hour. Because
C/D = 0.9 = 9/10 exactly, every breakpoint of v lies on a small lattice (76 points), so the
recursion is exact with plain index shifts -- same answer as a linear programme, ~0.1 ms.

Safety
------
* Every command is clamped so it can never exceed the power or capacity limits (0 infractions).
* Discharge is capped by the last measured net load: energy sent to the grid earns nothing.
* If the meter reads ~0 (sensor outage / campus disconnected), the plan assumes the outage
  continues for a while instead of cycling the battery for no benefit.
"""
import numpy as np

CAP = 2000.0                 # kWh
PMAX = 500.0                 # kW
ETA = 0.9 ** 0.5             # one-way efficiency
EPS = 1e-4                   # safety margin on the physical limits

RHO = 0.575                 # diagnostic probe: v1 with a different CO2/price weighting (v1 uses 0.60)
KAPPA = (0.3 / 0.7) * RHO    # weight of price relative to CO2 in the score signal

# ---- lattice of state-of-charge breakpoints: C = 9U, D = 10U ---------------------------------
U = PMAX / ETA / 10.0
_K = int(CAP // U)
_R = CAP - _K * U
GRID = np.empty(2 * _K + 2)
GRID[0::2] = np.arange(_K + 1) * U
GRID[1::2] = np.arange(_K + 1) * U + _R
GRID[-1] = CAP
NI = len(GRID) - 1           # number of intervals; v is constant on each
SH_C, SH_D = 18, 20          # index shifts for a full-power charge / discharge hour
C_STEP = PMAX * ETA          # max SOC gain in one hour (kWh)
D_STEP = PMAX / ETA          # max SOC drop in one hour (kWh)
INF = float("inf")

GUARD_MARGIN = 0.05          # discharge at most 95% of the last measured net load
OUTAGE_KW = 50.0             # net load below this = meter outage / no demand to serve


def marginal_values(w, v_end, n_blocked=0):
    """Backward pass over hours len(w)-1 .. 1. Returns v for the state right after hour 0.
    Hours with index < n_blocked are assumed unable to discharge (outage)."""
    v = v_end
    g = np.empty(NI)
    h = np.empty(NI)
    for k in range(len(w) - 1, 0, -1):
        b = w[k] / ETA
        a = w[k] * ETA
        g[:NI - SH_C] = v[SH_C:]
        g[NI - SH_C:] = -INF
        x = np.maximum(g, np.minimum(b, v))
        if k < n_blocked:
            v = x
        else:
            h[SH_D:] = v[:NI - SH_D]
            h[:SH_D] = INF
            v = np.minimum(h, np.maximum(a, x))
    return v


def first_step(soc, v, w0, d_cap_kw):
    """Optimal power (kW) for the current hour given v = marginal value after this hour."""
    b = w0 / ETA
    a = w0 * ETA
    i_up = int(np.searchsorted(GRID, soc, side="right")) - 1      # interval just above soc
    i_dn = int(np.searchsorted(GRID, soc, side="left")) - 1       # interval just below soc
    if 0 <= i_up < NI and v[i_up] > b:
        target = GRID[int(np.count_nonzero(v > b))]               # charge while a kWh is worth more than it costs
        return (min(target, soc + C_STEP) - soc) / ETA
    if 0 <= i_dn < NI and v[i_dn] < a:
        target = GRID[NI - int(np.count_nonzero(v < a))]          # discharge while a kWh is worth less than it earns
        return -(soc - max(target, soc - d_cap_kw / ETA)) * ETA
    return 0.0


class BatteryAgent:
    def __init__(self):
        self.v_end = np.zeros(NI)        # energy left at the end of the known window: valued at 0
        self.prev_ts = None
        self.outage_run = 0              # consecutive hours with (almost) no net load

    # -- helpers ---------------------------------------------------------------------------
    def _time_jump(self, obs):
        """True when this step does not follow the previous one by exactly one hour."""
        ts = obs.get("timestamp")
        jump = False
        try:
            if self.prev_ts is not None:
                gap = (ts - self.prev_ts).total_seconds()
                jump = abs(gap - 3600.0) > 1.0
        except Exception:
            jump = False
        self.prev_ts = ts
        return jump

    def _discharge_limits(self, obs, horizon):
        """(cap for this hour in kW, number of leading hours assumed blocked by an outage)."""
        jump = self._time_jump(obs)
        try:
            last = obs["noves_dades_5min"]
            net = float(last["ultim_consum_kw"]) - float(last["ultima_solar_kw"])
        except Exception:
            net = float("nan")
        if jump or net != net:               # after a calendar jump the last reading is a month old
            self.outage_run = 0
            return PMAX, 0
        if net < OUTAGE_KW:
            self.outage_run += 1
            n_blocked = min(horizon, max(4, 2 * self.outage_run))
            return 0.0, n_blocked
        self.outage_run = 0
        return min(PMAX, max(0.0, net * (1.0 - GUARD_MARGIN))), 0

    def _decide(self, obs):
        soc = float(obs["soc_kwh"])
        co2 = np.asarray(obs["co2_properes_24h"], float)
        price = np.asarray(obs["preus_properes_24h"], float)
        n = min(len(co2), len(price))
        if n == 0:
            co2 = np.array([float(obs["co2_actual"])])
            price = np.array([float(obs["preu_actual"])])
            n = 1
        w = np.maximum(co2[:n] + KAPPA * price[:n], 1e-9)
        d_cap, n_blocked = self._discharge_limits(obs, n)
        v = marginal_values(w, self.v_end, n_blocked)
        return first_step(soc, v, w[0], d_cap)

    # -- API -------------------------------------------------------------------------------
    def act(self, obs: dict) -> float:
        try:
            p = float(self._decide(obs))
        except Exception:
            p = 0.0
        if p != p:
            p = 0.0
        soc = float(obs["soc_kwh"])
        max_c = max(0.0, min(PMAX, (CAP - soc) / ETA - EPS))
        max_d = max(0.0, min(PMAX, soc * ETA - EPS))
        return float(min(max(p, -max_d), max_c))
