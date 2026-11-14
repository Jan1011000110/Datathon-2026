"""
experiments/30_boundaries/rules.py
Boundary rules as wrappers around the forecasters of the combined agent (h2 = CombinedAgent, h3 = SigAgent).
Only the part of each forecast that lies BEYOND the end of the current 7-day block is touched.

    Calendar(fold_days)            rows left in the block and start of the next block, from the timestamp alone
    CrossW(fc, cal, fn)            signal forecaster: rows beyond the block end := fn(ctx)
    CrossNet(net, cal, fn, ...)    net-load forecaster: same for the point forecast; scenarios are shifted with it
    DayProfile(fit_off)            load / solar of a calendar day from the last same-type fit days before it
    install(agent, w=..., net=...) put the wrappers on an agent built by make_agent3

Legal at run time: the calendar (block = 7 days from the first midnight of the block, next block = same days of
the next month in the folds, last 7 days of the next month in the test) and the fit rows. The `truth_*`
functions are diagnostics only.
"""
import numpy as np
import pandas as pd

import lib as L
from lib import cb

HOLIDAYS = {(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)}
H1 = pd.Timedelta(hours=1)


def day_type(d):
    """0 weekday, 1 Saturday, 2 Sunday or fixed national holiday."""
    if d.dayofweek == 6 or (d.month, d.day) in HOLIDAYS:
        return 2
    return 1 if d.dayofweek == 5 else 0


class Calendar:
    """Block bookkeeping from timestamps only (same rule as SignalFC._next_block)."""
    def __init__(self):
        self.prev = None
        self.b0 = None

    def step(self, ts):
        if self.prev is None or (ts - self.prev) > pd.Timedelta(hours=48) or ts < self.prev:
            self.b0 = ts.normalize()
            first_next = self.b0.replace(day=1) + pd.offsets.MonthBegin(1)
            if self.b0.day == self.b0.days_in_month - 6:
                self.nb = first_next + pd.Timedelta(days=first_next.days_in_month - 7)
            else:
                self.nb = first_next + pd.Timedelta(days=self.b0.day - 1)
        self.prev = ts
        r = int((self.b0 + pd.Timedelta(hours=168) - ts) / H1)          # rows left in the block, this one included
        return r, self.nb


class Ctx:
    __slots__ = ("i", "r", "H", "nb", "ts", "old")


class _Cross:
    def _ctx(self, obs, hist, H, old):
        c = Ctx()
        c.i, c.H, c.ts, c.old = len(hist) - 1, H, obs["timestamp"], old
        if self._seen != c.i:
            self._r, self._nb = self.cal.step(c.ts)
            self._seen = c.i
        c.r, c.nb = self._r, self._nb
        return c


class CrossW(_Cross):
    def __init__(self, fc, fn, extend=0):
        self.fc, self.fn, self.cal, self._seen, self.extend = fc, fn, Calendar(), None, extend

    def predict(self, obs, hist, H):
        w = np.asarray(self.fc.predict(obs, hist, H), float).copy()
        c = self._ctx(obs, hist, len(w) + (self.extend if len(w) >= min(H, cb.HT) else 0), w)
        if 1 <= c.r < c.H:
            new = np.asarray(self.fn(c), float)                         # values for positions r .. H-1
            w = np.concatenate((w[:c.r], new))
        return w


class CrossNet(_Cross):
    def __init__(self, net, fn, width=1.0, zero_width=False, extend=0):
        self.net, self.fn, self.cal, self._seen = net, fn, Calendar(), None
        self.width, self.zero_width, self.extend = width, zero_width, extend

    def predict(self, obs, hist, H):
        f = np.asarray(self.net.predict(obs, hist, H), float).copy()
        c = self._ctx(obs, hist, len(f) + (self.extend if len(f) >= min(H, cb.HT) else 0), f)
        if 1 <= c.r < c.H:
            f = np.concatenate((f[:c.r], np.asarray(self.fn(c), float)))
        return f

    def tail_bias(self, obs, K, H):
        return 0.0 if not hasattr(self.net, "tail_bias") else self.net.tail_bias(obs, K, min(H, cb.HT))

    def predict_scen(self, obs, hist, K, Q):
        scen, probs = self.net.predict_scen(obs, hist, K, Q)
        scen = np.array(scen, float)
        old = np.asarray(self.net.predict(obs, hist, K), float)
        c = self._ctx(obs, hist, len(scen), old)
        if 1 <= c.r < len(scen):
            new = np.asarray(self.fn(c), float)[:len(scen) - c.r]
            m = scen[c.r:] @ probs
            if self.zero_width:
                scen[c.r:] = new[:, None]
            else:                                                        # keep the shape of the errors, move the centre
                scen[c.r:] = new[:, None] + self.width * (scen[c.r:] - m[:, None])
        return scen, probs


def install(agent, w=None, net=None):
    """w / net: functions (wrapper_class kwargs as dict with key 'fn') or None."""
    if w is not None:
        inner = agent.w_fc
        if hasattr(inner, "fc") and hasattr(inner, "end"):               # combined3.EndCut: keep the cut outermost
            inner.fc = CrossW(inner.fc, **w)
        else:
            agent.w_fc = CrossW(inner, **w)
    if net is not None:
        agent.net = CrossNet(agent.net, **net)
    return agent


# --------------------------------------------------------------------------------------
# Diagnostics: the truth beyond the jump
# --------------------------------------------------------------------------------------
def truth_fn(x):
    """Rows beyond the block end := the true values of the run rows (padded with the last value at the run's end)."""
    x = np.asarray(x, float)

    def fn(c):
        idx = np.minimum(np.arange(c.i + c.r, c.i + c.H), len(x) - 1)
        return x[idx]
    return fn


# --------------------------------------------------------------------------------------
# Legal replacements
# --------------------------------------------------------------------------------------
class DayProfile:
    """Net load of a calendar day from the fit rows: load = mean of the last n_load fit days of the same day type
    before the block start, solar = mean of the last n_pv fit days before it, hour by hour."""
    def __init__(self, fit_off, n_load=3, n_pv=7, max_age=None):
        d = fit_off[["load_kw", "pv_kw"]].copy()
        d["date"] = d.index.normalize()
        d["hour"] = d.index.hour
        self.load = d.pivot(index="date", columns="hour", values="load_kw").reindex(columns=range(24))
        self.pv = d.pivot(index="date", columns="hour", values="pv_kw").reindex(columns=range(24))
        self.types = np.array([day_type(x) for x in self.load.index])
        self.n_load, self.n_pv, self.max_age = n_load, n_pv, max_age
        self._memo = {}

    def day(self, nb, date):
        """(24,) net load expected for `date`, a day of the block that starts at nb."""
        key = (nb, date)
        if key not in self._memo:
            before = self.load.index < nb
            t = day_type(date)
            ok = before & (self.types == t) & self.load.notna().all(axis=1).to_numpy()
            if not ok.any():
                ok = before & self.load.notna().all(axis=1).to_numpy()
            ld = self.load[ok].tail(self.n_load).mean(axis=0).to_numpy()
            okp = (self.pv.index < nb) & self.pv.notna().all(axis=1).to_numpy()
            pv = self.pv[okp].tail(self.n_pv).mean(axis=0).to_numpy()
            self._memo[key] = (ld, pv)
        return self._memo[key]

    def fn(self, load_only=False, pv_only=False):
        def f(c):
            out = np.empty(c.H - c.r)
            for j in range(c.H - c.r):
                t = c.nb + j * H1
                ld, pv = self.day(c.nb, t.normalize())
                out[j] = ld[t.hour] - pv[t.hour]
            return out
        return f


class SignalStore:
    """True signal of the fit rows by timestamp (prices and CO2 have no gaps in train)."""
    def __init__(self, fit_sig):
        self.w = (fit_sig["co2_kg_per_kwh"] + cb.KAPPA * fit_sig["precio_eur_per_kwh"])

    def prev_day(self, nb):
        idx = pd.date_range(nb - pd.Timedelta(hours=24), periods=24, freq="h")
        return self.w.reindex(idx).to_numpy()

    def blend_fn(self, a=0.5, scale=1.0, days=1):
        """Forecast beyond the jump := scale * ((1 - a) * forecast + a * the stored day right before the next block)."""
        def f(c):
            old = np.asarray(c.old[c.r:c.H], float)
            if len(old) < c.H - c.r:
                old = np.concatenate((old, np.full(c.H - c.r - len(old), old[-1] if len(old) else np.nan)))
            if a > 0:
                pd_ = self.prev_day(c.nb)
                hh = (np.arange(c.H - c.r)) % 24
                p = pd_[hh]
                p = np.where(np.isfinite(p), p, old)
                old = (1 - a) * old + a * p
            return scale * old
        return f


# --------------------------------------------------------------------------------------
# Diagnostics: the truth for the target rows of a mask (for example every row of day d of a week)
# --------------------------------------------------------------------------------------
class TargetW:
    def __init__(self, fc, truth, tmask):
        self.fc, self.x, self.m = fc, np.asarray(truth, float), np.asarray(tmask, bool)

    def predict(self, obs, hist, H):
        w = np.asarray(self.fc.predict(obs, hist, H), float).copy()
        i = len(hist) - 1
        idx = i + np.arange(len(w))
        ok = idx < len(self.x)
        ok[ok] &= self.m[idx[ok]]
        w[ok] = self.x[idx[ok]]
        return w


class TargetNet:
    def __init__(self, net, truth, tmask):
        self.net, self.x, self.m = net, np.asarray(truth, float), np.asarray(tmask, bool)

    def _ok(self, i, n):
        idx = i + np.arange(n)
        ok = idx < len(self.x)
        ok[ok] &= self.m[idx[ok]]
        return idx, ok

    def predict(self, obs, hist, H):
        f = np.asarray(self.net.predict(obs, hist, H), float).copy()
        idx, ok = self._ok(len(hist) - 1, len(f))
        f[ok] = self.x[idx[ok]]
        return f

    def tail_bias(self, obs, K, H):
        return self.net.tail_bias(obs, K, H)

    def predict_scen(self, obs, hist, K, Q):
        scen, probs = self.net.predict_scen(obs, hist, K, Q)
        scen = np.array(scen, float)
        idx, ok = self._ok(len(hist) - 1, len(scen))
        scen[ok] = self.x[idx[ok]][:, None]
        return scen, probs
