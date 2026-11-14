"""
agent.py  --  BatteryAgent h0 (hard mode baseline: rolling-horizon LP on profile forecasts)

Hard mode: battery 6,000 kWh / +-1,500 kW, solar x12, obs carries only 3 hours of price / CO2
and the last hour's load and solar. Each hour the agent

  1. forecasts the score signal  w = co2 + KAPPA * price  for the next H hours: mean by
     (month, weekend, hour) of the train set, rescaled to match the hours known from obs
     (the correction fades with the horizon); the known hours overwrite the forecast,
  2. forecasts the campus net load (load - solar, kW) for the same hours: load = mean by
     (weekend, hour) plus the last hour's anomaly (fading); solar = mean by (month, hour) times
     the last hour's clearness ratio (fading),
  3. solves the linear programme "minimise sum w_k * grid_import_k" over those H hours (H = 18)
     (surplus solar is free, discharge beyond the net load earns nothing) and applies its
     first step. The net load of the current hour is shaded by D_SHADE when discharging, since
     overshooting it wastes stored energy.

Guard
-----
* Every command is clamped to the power and capacity limits read from obs (0 infractions).
* act() never raises and never returns NaN / inf: anything unusable gives 0.0 kW.
* When obs carries fewer than 3 price / CO2 values the run is ending: the plan covers only the
  hours that remain.

The fitted tables live in h0_tables.json next to this file (written by build.py).
"""
import datetime
import json
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linprog
except Exception:                                    # no scipy: the agent stays idle (always legal)
    linprog = None

ETA = 0.9 ** 0.5             # one-way efficiency
EPS = 1e-4                   # safety margin on the physical limits
CAP_DEFAULT = 6000.0         # kWh, used when obs has no "cap_max_kwh"
PMAX_DEFAULT = 1500.0        # kW, used when obs has no "power_max_kw"
LOOKAHEAD = 3                # price / CO2 values obs carries in hard mode
DATA_NAME = "h0_tables.json"
INF = float("inf")
NAN = float("nan")


def _num(x):
    """float(x) if x is a finite number, else NaN. Never raises."""
    try:
        x = float(x)
    except Exception:
        return NAN
    return x if abs(x) < INF else NAN


def _vec(x):
    """1-D float array from a list-like (NaN where a value is not a number). Never raises."""
    try:
        return np.atleast_1d(np.asarray(x, float)).ravel()
    except Exception:
        pass
    try:
        return np.array([_num(v) for v in x], float)
    except Exception:
        return np.zeros(0)


def _calendar(ts):
    """(month, day of week [Mon=0], hour) from any timestamp-like value (pandas, datetime,
    numpy, 'YYYY-MM-DD HH:MM:SS' text), or None."""
    try:
        s = str(ts)
        y, m, d, h = int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13])
        if s[4] != "-" or s[7] != "-" or not 0 <= h <= 23:
            return None
        return m, datetime.date(y, m, d).weekday(), h
    except Exception:
        return None


def _default_path():
    try:
        return Path(__file__).parent / DATA_NAME
    except Exception:
        return Path(DATA_NAME)


class BatteryAgent:
    data_path = None         # set on the class before construction to load other tables (validation)
    horizon = None           # planning horizon in hours; None = the value stored in the tables
    end_of_run = True        # plan only over the remaining hours when obs carries < 3 values

    def __init__(self):
        self.ready = False
        self.month = None                # last month seen (used if a timestamp cannot be read)
        self._lp = {}                    # constraint matrix per horizon length
        try:
            self._load(self.data_path if self.data_path is not None else _default_path())
            self.ready = linprog is not None
        except Exception:
            self.ready = False

    def _load(self, path):
        with open(str(path), "r", encoding="utf-8") as f:
            t = json.load(f)
        self.kappa = float(t["kappa"])               # weight of price relative to CO2 in the signal
        self.half = float(t["half"])                 # half-life (h) of the signal's level correction
        self.phi = float(t["phi"])                   # hourly decay of the load anomaly
        self.phi_pv = float(t["phi_pv"])             # hourly decay of the solar clearness correction
        self.d_shade = float(t["d_shade"])           # share of this hour's forecast net load we dare discharge
        self.lam = float(t["lam"])                   # value of a kWh left at the end of the horizon
        self.H = int(self.horizon if self.horizon is not None else t["H"])
        self.w_tab = np.asarray(t["w_tab"], float)           # [month-1, weekend, hour]
        self.w_any = np.asarray(t["w_any"], float)           # [weekend, hour]        (month unknown)
        self.load_tab = np.asarray(t["load_tab"], float)     # [weekend, hour]
        self.pv_tab = np.asarray(t["pv_tab"], float)         # [month-1, hour]
        self.pv_any = np.asarray(t["pv_any"], float)         # [hour]                 (month unknown)
        if (self.w_tab.shape != (12, 2, 24) or self.w_any.shape != (2, 24) or self.load_tab.shape != (2, 24)
                or self.pv_tab.shape != (12, 24) or self.pv_any.shape != (24,) or self.H < 1):
            raise ValueError("bad tables")
        for a in (self.w_tab, self.w_any, self.load_tab, self.pv_tab, self.pv_any):
            if not np.isfinite(a).all():
                raise ValueError("bad tables")

    # -- calendar ----------------------------------------------------------------------------
    def _when(self, obs):
        """(month or None, day of week, hour) of the hour being decided."""
        cal = _calendar(obs.get("timestamp"))
        if cal is not None:
            self.month = cal[0]
            return cal
        h = _num(obs.get("hour"))
        d = _num(obs.get("day_of_week"))
        hour = int(h) if (h == h and 0 <= h <= 23) else 12
        dow = int(d) if (d == d and 0 <= d <= 6) else 2
        return self.month, dow, hour

    # -- plan --------------------------------------------------------------------------------
    def _plan(self, w, n, soc, cap, pmax):
        """First step (kW, + = charge) of
            min  sum_k w_k g_k  -  lam * soc_H
            s.t. g_k >= n_k + c_k - d_k,  g_k >= 0
                 0 <= soc + sum_{j<=k} (ETA c_j - d_j / ETA) <= cap,   0 <= c_k, d_k <= pmax
        with x = [c, d, g]. None if the solver fails."""
        H = len(w)
        A = self._lp.get(H)
        if A is None:
            tri = np.tril(np.ones((H, H)))
            I = np.eye(H)
            S = np.hstack([ETA * tri, -tri / ETA, np.zeros((H, H))])
            A = np.vstack([S, -S, np.hstack([I, -I, -I])])
            self._lp[H] = A
        b = np.concatenate([np.full(H, cap - soc), np.full(H, soc), -n])
        cost = np.concatenate([np.full(H, -self.lam * ETA), np.full(H, self.lam / ETA), w])
        bounds = [(0, pmax)] * (2 * H) + [(0, None)] * H
        r = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if r.status != 0:
            return None
        return float(r.x[0] - r.x[H])

    def _decide(self, obs, soc, cap, pmax):
        month, dow, hour = self._when(obs)

        # ---- known part of the signal ----
        co2 = _vec(obs.get("co2_properes_24h"))
        price = _vec(obs.get("preus_properes_24h"))
        nk = min(len(co2), len(price))
        known = co2[:nk] + self.kappa * price[:nk]
        w0 = _num(obs.get("co2_actual")) + self.kappa * _num(obs.get("preu_actual"))
        H = self.H
        if nk == 0:
            known = np.array([w0])                   # lists missing: fall back on the scalar fields
        else:
            if known[0] != known[0]:
                known[0] = w0
            if self.end_of_run and nk < LOOKAHEAD:
                H = min(H, nk)                       # the run ends in nk hours: nothing to save for later
        k = min(len(known), H)
        known = known[:k]
        ok = np.isfinite(known)

        # ---- calendar of the next H hours (consecutive hours assumed) ----
        step = hour + np.arange(H)
        hh = step % 24
        we = (((dow + step // 24) % 7) >= 5).astype(int)
        lh = (hour - 1) % 24
        lwe = int((dow if hour > 0 else (dow - 1) % 7) >= 5)
        j = np.arange(1, H + 1)

        # ---- signal forecast ----
        clim = (self.w_tab[month - 1] if month is not None else self.w_any)[we, hh]
        w = clim
        if ok.any():
            ratio = float(np.mean(known[ok])) / max(float(np.mean(clim[:k][ok])), 1e-9)
            ratio = min(max(ratio, 0.2), 5.0)
            w = clim * (1.0 + (ratio - 1.0) * 0.5 ** (np.arange(H) / self.half))
        w = np.array(w, float)
        w[:k][ok] = known[ok]

        # ---- net load forecast ----
        try:
            last = obs.get("noves_dades_5min")
            load_last, pv_last = _num(last.get("ultim_consum_kw")), _num(last.get("ultima_solar_kw"))
        except Exception:
            load_last = pv_last = NAN
        anom = load_last - self.load_tab[lwe, lh]
        if anom != anom:
            anom = 0.0                               # no reading: plain profile
        load = np.maximum(self.load_tab[we, hh] + anom * self.phi ** j, 0.0)
        pv_prof = self.pv_tab[month - 1] if month is not None else self.pv_any
        ratio = 1.0
        if pv_prof[lh] > 20.0 and pv_last == pv_last:
            ratio = min(max(pv_last / pv_prof[lh], 0.0), 3.0)
        pv = pv_prof[hh] * (1.0 + (ratio - 1.0) * self.phi_pv ** j)
        n = load - pv
        if self.d_shade != 1.0 and n[0] > 0:
            n[0] = n[0] * self.d_shade               # discharging past the real net load is wasted

        return self._plan(np.maximum(w, 1e-9), n, min(max(soc, 0.0), cap), cap, pmax)

    # -- API ---------------------------------------------------------------------------------
    def act(self, obs: dict) -> float:
        try:
            cap = _num(obs.get("cap_max_kwh", CAP_DEFAULT))
            pmax = _num(obs.get("power_max_kw", PMAX_DEFAULT))
            if not cap > 0.0:
                cap = CAP_DEFAULT
            if not pmax > 0.0:
                pmax = PMAX_DEFAULT
            soc = _num(obs.get("soc_kwh"))
            if soc != soc:                           # fall back on the relative fields
                soc = _num(obs.get("soc_pct")) * cap
            if soc != soc:
                soc = _num(obs.get("soc_percent")) * cap / 100.0
        except Exception:
            return 0.0
        if soc != soc:
            return 0.0                               # state unknown: 0 kW is the only command that is always legal
        p = 0.0
        if self.ready:
            try:
                p = self._decide(obs, soc, cap, pmax)
                p = 0.0 if p is None else float(p)
            except Exception:
                p = 0.0
        if p != p or abs(p) == INF:
            p = 0.0
        max_c = max(0.0, min(pmax, (cap - soc) / ETA - EPS))
        max_d = max(0.0, min(pmax, soc * ETA - EPS))
        return float(min(max(p, -max_d), max_c)) + 0.0           # + 0.0: never -0.0
