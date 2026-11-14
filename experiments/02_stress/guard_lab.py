"""
experiments/02_stress/guard_lab.py
A configurable version of v1's guard, used to tune it and to test alternative rules.
The optimiser is v1's exact lattice DP; the only generalisation is that every future hour may
carry its own discharge cap (a multiple of 50 kW, which keeps the lattice exact).

LabAgent(**params) -- see DEFAULTS. With the defaults it reproduces agents/v1 exactly.
"""
import datetime as _dt
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from lib import ROOT, load_agent_module  # noqa: E402

v1 = load_agent_module(ROOT / "agents" / "v1" / "agent.py", "agent_v1_lab")
CAP, PMAX, ETA, EPS, NI, GRID = v1.CAP, v1.PMAX, v1.ETA, v1.EPS, v1.NI, v1.GRID
SH_C, SH_D, INF = v1.SH_C, v1.SH_D, v1.INF

DEFAULTS = dict(
    margin=0.05,          # discharge now <= (1 - margin) * last net load
    outage_kw=50.0,       # last net load below this = outage
    pers_a=4, pers_b=2.0, # outage assumed to last max(pers_a, pers_b * run) more hours
    pers_max=10 ** 6,     # upper bound on that persistence
    cal_exit=False,       # cap the blocked hours at the end of the calendar month (week boundary)
    entry_week=0,         # N > 0: an outage whose first zero reading comes within N steps of a week
                          #        start is assumed to last the whole week
    first_hour="free",    # first step after a calendar jump (reading is stale): "free" | "block"
    nan="free",           # reading missing/NaN: "free" (no guard, v1) | "hold" (keep last state) | "outage"
    fut="none",           # caps for FUTURE hours: "none" | "persist" | "truth" (cheat, upper bound)
    fut_thresh=600.0,     # ... only when the last net load is below this
    fut_margin=0.0,
    fut_round="floor",    # how the future caps are put on the 50 kW lattice
    free_solar=False,     # (old name) = surplus="persist", surplus_factor=0.9, only when the plan idles
    surplus="off",        # last net load < 0 (pv > load): charge at least the expected surplus and
                          #   never discharge.  "persist" = last surplus; "shape" = last surplus scaled
                          #   by the typical pv change between the two hours of the day
    surplus_factor=1.0,
    drop_ratio=0.0,       # > 0: last net load below drop_ratio * the one before = logger glitch in
                          #   progress -> treated like a zero reading
    drop_abs=0.0,         # > 0: last net load below this many kW (but above outage_kw) = glitch too
    drop_block=None,      # hours blocked after a "drop" reading (None = same rule as a zero reading)
    drop_skip6=False,     # ignore the drop test for the 06:00 reading (night-freeze release)
    night=False,          # the meter is not polled 22:00-06:00 (value held): a fault seen at night
                          #   lasts at least until 06:00, and last hour's load is exact
    profile=None,         # None | "q05" | "q01": cap = last net * (that quantile of net_h/net_{h-1}
                          #   for this hour of the day, train) * (1 - margin)
    dead_months=(),       # (year, month) pairs known in advance to be dead (prior knowledge):
                          #   the week is treated as an outage from its first hour AND, on the last
                          #   day of the previous week, the hours beyond the boundary are blocked
    dead_buy=0,           # N > 0: in a dead stretch older than N hours, with the boundary not yet in
                          #   sight, buy at hours cheaper than the 3rd cheapest seen so far in it
    truth=None,           # DataFrame of the run (only for fut="truth")
)


# quantiles of net_h / net_{h-1} by hour of day h (official train table, consecutive hours)
RATIO = {
    "q05": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.777, 0.916, 0.922, 0.902, 0.891, 0.926, 0.913, 0.901, 0.923, 0.931,
            0.936, 0.951, 0.940, 0.930, 0.883, 0.836, 0.925, 0.999],
    "q01": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.749, 0.878, 0.880, 0.862, 0.855, 0.869, 0.843, 0.840, 0.867, 0.900,
            0.908, 0.934, 0.919, 0.914, 0.865, 0.823, 0.910, 0.998],
}
# mean PV (kW) by hour of day over the train set (all months) -- only ratios between hours are used
PV_SHAPE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 3.7, 17.6, 41.1, 65.0, 81.4, 89.2, 89.8, 82.4, 67.8, 49.2, 29.7, 13.5, 4.7, 1.4, 0.1, 0.0, 0.0]


def marginal_values(w, v_end, sd):
    """v1's backward pass with a per-hour discharge shift sd[k] in {0, 2, .., 20} (0 = blocked)."""
    v = v_end
    g = np.empty(NI)
    h = np.empty(NI)
    for k in range(len(w) - 1, 0, -1):
        b = w[k] / ETA
        a = w[k] * ETA
        g[:NI - SH_C] = v[SH_C:]
        g[NI - SH_C:] = -INF
        x = np.maximum(g, np.minimum(b, v))
        s = int(sd[k])
        if s <= 0:
            v = x
        else:
            h[s:] = v[:NI - s]
            h[:s] = INF
            v = np.minimum(h, np.maximum(a, x))
    return v


def as_datetime(ts):
    """Best-effort conversion of whatever the evaluator sends to a naive datetime (or None)."""
    try:
        if ts is None:
            return None
        if isinstance(ts, _dt.datetime):          # includes pandas.Timestamp
            return ts.replace(tzinfo=None)
        if isinstance(ts, np.datetime64):
            return ts.astype("datetime64[s]").item()
        if isinstance(ts, str):
            return _dt.datetime.fromisoformat(ts.strip().replace("T", " ").replace("Z", "")[:19])
    except Exception:
        return None
    return None


def hours_left_in_month(d):
    """Hourly steps left in the current calendar month, counting the current hour."""
    nxt = _dt.datetime(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return int(round((nxt - _dt.datetime(d.year, d.month, d.day, d.hour)).total_seconds() / 3600.0))


class LabAgent:
    def __init__(self, **params):
        self.p = dict(DEFAULTS)
        unknown = set(params) - set(DEFAULTS)
        assert not unknown, unknown
        self.p.update(params)
        self.v_end = np.zeros(NI)
        self.prev = None            # previous timestamp (datetime)
        self.run = 0                # consecutive outage readings
        self.since_jump = 0         # steps since the last week start (0 = first hour of a week)
        self.week_dead = False
        self.last = (PMAX, 0)       # last (cap, n_blocked) -- for nan="hold"
        self.seen = []              # signal values seen during the current outage
        self.prev_net = float("nan")
        self.surplus_kw = 0.0
        self.all_blocked = False
        self.t = -1

    # ------------------------------------------------------------------------------------------
    def _plan_limits(self, obs, n):
        """Returns (cap for this hour in kW, sd = discharge shift per hour of the window)."""
        cap, sd = self._plan_limits0(obs, n)
        now = self.prev
        self.all_blocked = bool(cap == 0.0 and not sd[1:].any()) and self.run > 0
        if self.p["dead_months"] and now is not None:
            left = hours_left_in_month(now)
            nm = (now.year + (now.month == 12), now.month % 12 + 1)
            if left < n and nm in self.p["dead_months"]:
                sd[left:] = 0
        return cap, sd

    def _plan_limits0(self, obs, n):
        P = self.p
        self.t += 1
        now = as_datetime(obs.get("timestamp"))
        jump = False
        if now is not None and self.prev is not None:
            jump = abs((now - self.prev).total_seconds() - 3600.0) > 1.0
        self.prev = now
        self.since_jump = 0 if (jump or self.t == 0) else self.since_jump + 1
        left = hours_left_in_month(now) if (now is not None and P["cal_exit"]) else None

        try:
            last = obs["noves_dades_5min"]
            net = float(last["ultim_consum_kw"]) - float(last["ultima_solar_kw"])
        except Exception:
            net = float("nan")

        sd = np.full(n, SH_D, dtype=int)
        known_dead = now is not None and (now.year, now.month) in P["dead_months"]

        self.surplus_kw = 0.0
        if jump or net != net:
            self.prev_net = float("nan")
        if jump and not known_dead:
            self.run = 0
            self.week_dead = False
            cap = 0.0 if P["first_hour"] == "block" else PMAX
            self.last = (PMAX, 0)
            return cap, sd
        if net != net and not known_dead:
            if P["nan"] == "free":
                self.run = 0
                self.week_dead = False
                return PMAX, sd
            if P["nan"] == "hold":
                cap, nb = self.last
                sd[:max(0, min(n, nb - 1))] = 0
                self.last = (cap, max(0, nb - 1))
                return cap, sd
            net = 0.0                               # "outage"

        prev_net, self.prev_net = self.prev_net, net
        hour = now.hour if now is not None else obs.get("hour")
        if not isinstance(hour, (int, np.integer)) or not 0 <= hour <= 23:
            hour = None
        glitch = False
        if net >= P["outage_kw"]:
            if P["drop_ratio"] and prev_net == prev_net and prev_net >= P["outage_kw"] and net < P["drop_ratio"] * prev_net:
                glitch = not (P["drop_skip6"] and hour == 7)
            if P["drop_abs"] and net < P["drop_abs"]:
                glitch = True
        self.surplus_kw = max(0.0, -net)
        if glitch and P["drop_block"] is not None and not known_dead:
            # a partial glitch hour: block this hour and the next drop_block - 1, keep the run counter
            nb = min(n, int(P["drop_block"]))
            sd[:nb] = 0
            self.last = (0.0, nb)
            return 0.0, sd
        if net < P["outage_kw"] or known_dead or glitch:
            self.run += 1
            if self.run == 1 and P["entry_week"] and self.since_jump <= P["entry_week"]:
                self.week_dead = True
            if known_dead:
                self.week_dead = True
            nb = n if self.week_dead else int(min(P["pers_max"], max(P["pers_a"], P["pers_b"] * self.run)))
            if P["night"] and hour is not None and (hour >= 23 or hour <= 5):
                nb = max(nb, (6 - hour) % 24)       # held value: zero until the 06:00 poll at least
            if left is not None:
                nb = min(nb, left)
            elif known_dead and now is not None:
                nb = min(nb, hours_left_in_month(now))
            nb = min(n, nb)
            sd[:nb] = 0
            self.last = (0.0, nb)
            return 0.0, sd

        self.run = 0
        self.week_dead = False
        m = P["margin"]
        f = 1.0
        if P["profile"] and hour is not None:
            f = RATIO[P["profile"]][hour]
        if P["night"] and hour is not None and hour <= 5:
            m, f = 0.0, 1.0                           # held value: this hour's load = last hour's load
        cap = min(PMAX, max(0.0, net * f * (1.0 - m)))
        self.last = (cap, 0)

        if P["fut"] == "persist" and net < P["fut_thresh"]:
            c = net * (1.0 - P["fut_margin"]) / 50.0
            m = np.floor(c + 1e-9) if P["fut_round"] == "floor" else np.round(c)
            s = int(2 * min(10, max(0, m)))
            hi = n if left is None else min(n, left)   # beyond the week boundary: unknown -> 500
            sd[:hi] = s
        elif P["fut"] == "truth":
            df = P["truth"]
            nl = (df["load_kw"].to_numpy(float) - df["pv_kw"].to_numpy(float))[self.t:self.t + n]
            c = nl * (1.0 - P["fut_margin"]) / 50.0
            m = np.floor(c + 1e-9) if P["fut_round"] == "floor" else np.round(c)
            sd[:len(nl)] = (2 * np.clip(m, 0, 10)).astype(int)
        return cap, sd

    def _decide(self, obs):
        soc = float(obs["soc_kwh"])
        co2 = np.asarray(obs["co2_properes_24h"], float)
        price = np.asarray(obs["preus_properes_24h"], float)
        n = min(len(co2), len(price))
        if n == 0:
            co2 = np.array([float(obs["co2_actual"])])
            price = np.array([float(obs["preu_actual"])])
            n = 1
        w = np.maximum(co2[:n] + v1.KAPPA * price[:n], 1e-9)
        cap, sd = self._plan_limits(obs, n)
        v_end = self.v_end
        if self.run <= 1:
            self.seen = []
        if self.run > 0:
            if self.p["dead_buy"] and self.all_blocked and n >= 24 and len(self.seen) >= max(3, self.p["dead_buy"]):
                ref = sorted(self.seen)[2]
                v_end = np.full(NI, ref / ETA)
            self.seen.append(float(w[0]))
        v = marginal_values(w, v_end, sd)
        p = v1.first_step(soc, v, w[0], cap)
        if self.p["surplus"] != "off" and self.surplus_kw > 0.0:
            est = self.surplus_kw * self.p["surplus_factor"]
            if self.p["surplus"] == "shape":
                now = self.prev
                if now is not None:
                    h = now.hour
                    a, b = PV_SHAPE[(h - 1) % 24], PV_SHAPE[h]
                    est = est * (b / a) if a > 1e-3 else est
            p = max(p, est, 0.0)
        if self.p["free_solar"] and p <= 0.0:
            try:
                last = obs["noves_dades_5min"]
                net = float(last["ultim_consum_kw"]) - float(last["ultima_solar_kw"])
                if net < 0.0:
                    p = 0.9 * (-net)
            except Exception:
                pass
        return p

    def act(self, obs):
        p = float(self._decide(obs))          # no try/except here: in the lab we want to see errors
        if p != p:
            p = 0.0
        soc = float(obs["soc_kwh"])
        max_c = max(0.0, min(PMAX, (CAP - soc) / ETA - EPS))
        max_d = max(0.0, min(PMAX, soc * ETA - EPS))
        return float(min(max(p, -max_d), max_c))
