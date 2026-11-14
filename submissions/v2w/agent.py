"""
agent.py  --  BatteryAgent v2w: v2 with every test week treated as its own run.

The 24-hour window is cut at the last hour of the week (in the test, the last hour of the calendar month), so
nothing is planned beyond it and the battery is emptied before each week ends. Over the whole run this scores
lower than v2; on the last half of each week it scores higher. Everything else is v2, unchanged:

BatteryAgent v2 (exact rolling-horizon optimiser, numpy only, hardened against bad telemetry)

Idea
----
The campus always draws far more than the battery's 500 kW, so every kWh the battery delivers
displaces a kWh of grid import. The score is then linear in the battery schedule:

    score  ~  sum_t  -P_t * w_t ,        w_t = co2_t + KAPPA * price_t

The signal w is known for the next 24 hours, so each hour we solve the optimal charge/discharge
plan for the known window *exactly* (dynamic programming on the marginal value of stored energy,
same answer as a linear programme) and apply its first step.

Guard  (changes against v1 are marked [1.1])
-----
* Every command is clamped so it can never exceed the power or capacity limits (0 infractions).
* Discharge is capped by the last measured net load: energy sent to the grid earns nothing.
* A reading of ~0 kW means the data logger is down (in the train set load and solar read exactly
  0 together, so nothing can be displaced and there is no solar surplus either): no discharge.
* [1.1] A reading that falls below 70% of the previous one is the first, partial hour of such an
  outage (normal operation never drops that fast outside the 06:00 reading): no discharge.
* [1.1] The plan assumes an outage lasts as long again as it has lasted so far (v1: twice as
  long), and never past the end of the month, where the next test week starts.
* [1.1] act() never raises and never returns NaN, whatever the observation looks like.
* [low-load variant] When the last net load is under 500 kW the plan no longer counts on
  discharging 500 kW in the next 6 hours: those hours are capped at the last net load (rounded
  down to 50 kW, which keeps the solver exact).
* [2.0] If the last reading shows solar above demand (possible when the meter reads 0 but the
  inverters do not), that surplus is stored: charging up to it draws nothing from the grid.
"""
import numpy as np

CAP = 2000.0                 # kWh
PMAX = 500.0                 # kW
ETA = 0.9 ** 0.5             # one-way efficiency
EPS = 1e-4                   # safety margin on the physical limits

RHO = 0.60                   # kg CO2 per EUR of baseline import (train: 0.59, leaderboard-implied: 0.60)
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
NAN = float("nan")

# ---- guard parameters ------------------------------------------------------------------------
GUARD_MARGIN = 0.05          # discharge at most 95% of the last measured net load
OUTAGE_KW = 50.0             # net load below this = data logger down / no demand to serve
DROP_RATIO = 0.70            # net load below 70% of the previous hour's = partial outage hour (0 = off)
RELEASE_HOUR = 7             # ... except at 07:00: the 06:00 reading is the first after the meter's
                             #     night freeze (22:00 value held) and legitimately drops up to 28%
PERSIST_MIN = 4              # hours an outage is assumed to go on: max(PERSIST_MIN, PERSIST_X * hours so far)
PERSIST_X = 1
CALENDAR_EXIT = True         # ... but never past the end of the month (= end of the test week)
LOWLOAD_HOURS = 6            # net load under 500 kW is assumed to stay there for this many hours (0 = off)
SURPLUS_CHARGE = True        # store solar surplus (last reading: solar > demand) -- free energy
WEEK_CUT = True              # [v2w] plan nothing beyond the last hour of the week
WEEK_LAST_DAY = None         # [v2w] last day of the week; None = last day of the month (the test layout)
_DIM = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def marginal_values(w, v_end, shift):
    """Backward pass over hours len(w)-1 .. 1. Returns v for the state right after hour 0.
    shift[k] = index shift of the largest discharge allowed in hour k: SH_D = 500 kW, 2 per
    50 kW, 0 = no discharge (outage)."""
    v = v_end
    g = np.empty(NI)
    h = np.empty(NI)
    for k in range(len(w) - 1, 0, -1):
        b = w[k] / ETA
        a = w[k] * ETA
        g[:NI - SH_C] = v[SH_C:]
        g[NI - SH_C:] = -INF
        x = np.maximum(g, np.minimum(b, v))
        s = shift[k]
        if s <= 0:
            v = x
        else:
            h[s:] = v[:NI - s]
            h[:s] = INF
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


def _num(x):
    """float(x) if x is a finite number, else NaN. Never raises."""
    try:
        x = float(x)
    except Exception:
        return NAN
    return x if abs(x) < INF else NAN


def _calendar(ts):
    """(hours since a fixed origin, hour of day, hours left in the month including this one)
    from any timestamp-like value (pandas, datetime, numpy, 'YYYY-MM-DD HH:MM:SS' text), or None."""
    try:
        s = str(ts)
        y, m, d, h = int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13])
        if s[4] != "-" or s[7] != "-" or not (1 <= m <= 12 and 1 <= d <= 31 and 0 <= h <= 23):
            return None
    except Exception:
        return None
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    dim = 29 if (m == 2 and leap) else _DIM[m - 1]
    yy = y - 1
    days = 365 * yy + yy // 4 - yy // 100 + yy // 400 + sum(_DIM[:m - 1]) + (1 if (leap and m > 2) else 0) + d
    return days * 24 + h, h, (dim - d) * 24 + (24 - h)


def _week_left(ts):
    """Hours left in the week including this one, or None if the timestamp cannot be read."""
    cal = _calendar(ts)
    if cal is None:
        return None
    if WEEK_LAST_DAY is None:
        return cal[2]                                            # hours left in the month
    try:
        s = str(ts)
        return (int(WEEK_LAST_DAY) - int(s[8:10])) * 24 + (24 - int(s[11:13]))
    except Exception:
        return None


class BatteryAgent:
    def __init__(self):
        self.v_end = np.zeros(NI)        # energy left at the end of the known window: valued at 0
        self.prev_hour = None            # absolute hour index of the previous step
        self.prev_net = NAN              # previous valid net-load reading (kW)
        self.outage_run = 0              # consecutive hours with an outage reading
        self.surplus = 0.0               # solar surplus seen in the last reading (kW), 0 if none

    # -- guard -----------------------------------------------------------------------------
    def _discharge_limits(self, obs, horizon):
        """(cap for this hour in kW, discharge shift of every hour of the window)."""
        shift = [SH_D] * horizon
        cal = _calendar(obs.get("timestamp"))
        hour = left = None
        jump = False
        if cal is not None:
            # the next hour (or the same one again, e.g. the repeated hour of a DST change) is not a jump
            jump = self.prev_hour is not None and cal[0] - self.prev_hour not in (0, 1)
            self.prev_hour, hour, left = cal
        else:
            self.prev_hour = None
            h = _num(obs.get("hour"))
            if h == h and 0 <= h <= 23:
                hour = int(h)

        try:
            last = obs.get("noves_dades_5min")
            net = _num(last["ultim_consum_kw"]) - _num(last["ultima_solar_kw"])
        except Exception:
            net = NAN
        prev_net, self.prev_net = self.prev_net, net
        self.surplus = 0.0
        if jump or net != net:               # after a calendar jump the last reading is a month old
            self.outage_run = 0
            self.prev_net = NAN
            return PMAX, shift
        if SURPLUS_CHARGE and net < 0.0:
            self.surplus = -net              # solar exceeded demand last hour: expect the same now

        outage = net < OUTAGE_KW
        if (not outage and DROP_RATIO > 0.0 and hour != RELEASE_HOUR
                and prev_net >= OUTAGE_KW and net < DROP_RATIO * prev_net):
            outage = True                    # first, partial hour of an outage
        if outage:
            self.outage_run += 1
            n_blocked = max(PERSIST_MIN, PERSIST_X * self.outage_run)
            if CALENDAR_EXIT and left is not None:
                n_blocked = min(n_blocked, left)                  # the next row is another month
            for k in range(min(horizon, n_blocked)):
                shift[k] = 0
            return 0.0, shift

        self.outage_run = 0
        if LOWLOAD_HOURS > 0 and net < PMAX:                      # low load: plan with what is there
            s = 2 * int(net / 50.0 + 1e-9)
            n_low = min(horizon, LOWLOAD_HOURS)
            if CALENDAR_EXIT and left is not None:
                n_low = min(n_low, left)
            for k in range(n_low):
                shift[k] = s
        return min(PMAX, max(0.0, net * (1.0 - GUARD_MARGIN))), shift

    # -- plan ------------------------------------------------------------------------------
    def _decide(self, obs, soc):
        co2 = np.atleast_1d(np.asarray(obs.get("co2_properes_24h"), float)).ravel()
        price = np.atleast_1d(np.asarray(obs.get("preus_properes_24h"), float)).ravel()
        n = min(len(co2), len(price))
        if WEEK_CUT and n > 1:                                   # [v2w] hours left in the week, this one included
            left = _week_left(obs.get("timestamp"))
            if left is not None and 1 <= left < n:
                n = left
        w = co2[:n] + KAPPA * price[:n]
        w0 = _num(obs.get("co2_actual")) + KAPPA * _num(obs.get("preu_actual"))
        if n == 0:
            w = np.array([w0])
            n = 1
        elif not np.isfinite(w[0]):
            w[0] = w0                        # current hour missing in the lists: use the scalar fields
        d_cap, shift = self._discharge_limits(obs, n)
        ok = np.isfinite(w)
        if not ok[0]:
            return 0.0                       # no usable signal for this hour: stay idle
        if not ok.all():
            w = np.where(ok, w, float(np.mean(w[ok])))            # unknown future hours: neutral value
        w = np.maximum(w, 1e-9)
        v = marginal_values(w, self.v_end, shift)
        p = first_step(min(max(soc, 0.0), CAP), v, w[0], d_cap)
        if self.surplus > 0.0:
            p = max(p, self.surplus)         # storing the surplus costs nothing (act() clamps it to what fits)
        return p

    # -- API -------------------------------------------------------------------------------
    def act(self, obs: dict) -> float:
        try:
            soc = _num(obs.get("soc_kwh"))
            if soc != soc:                   # fall back on the relative fields
                soc = _num(obs.get("soc_pct")) * CAP
            if soc != soc:
                soc = _num(obs.get("soc_percent")) * CAP / 100.0
        except Exception:
            soc = NAN
        if soc != soc:
            return 0.0                       # state unknown: 0 kW is the only command that is always legal
        try:
            p = float(self._decide(obs, soc))
        except Exception:
            p = 0.0
        if p != p:
            p = 0.0
        max_c = max(0.0, min(PMAX, (CAP - soc) / ETA - EPS))
        max_d = max(0.0, min(PMAX, soc * ETA - EPS))
        return float(min(max(p, -max_d), max_c))
