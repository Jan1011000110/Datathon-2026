"""
experiments/21_load/cal_fc.py
LoadFC with an optional holiday calendar (subclass; load_fc.py is not modified).

    fc = CalLoadFC(fit_df)                         # no calendar: every forecast is LoadFC's, bit for bit
    fc = CalLoadFC(fit_df, calendar=CALENDAR)      # calendar: {date string: kind}

What the calendar changes (only on the dates it names, and on the 3 weekdays after them):
  * A calendar day starts with a holiday index (weight of the Sunday reference in the day's profile)
    equal to the prior of its kind instead of the value carried from the previous weekday. The prior
    of each kind is the mean index of the fit-table days of that kind. Today's readings then update
    the index exactly as in LoadFC (least squares, shrunk towards the prior with `kappa`).
  * The index is known at every horizon: at 18:00 the day before, tomorrow's hours already use it.
  * Calendar days are not used as weekday references, so the days after a holiday are not forecast
    low; and a normal weekday after a calendar day starts with index 0 instead of rho x yesterday's.
  * Hour being decided, 06:00-22:00 of a calendar day: the model's change is pulled towards the
    change of the blended reference with weight min(cal_blend x index, 1) (cal_blend = 0: off).

Diagnostic mode (not a legal forecaster): `lam_day` = {meter-day start: index} fixes the index of
those days (cal_upper_bound.py gives it the true index).

Runtime imports: numpy, pandas, lightgbm (through load_fc).
"""
import numpy as np
import pandas as pd

from load_fc import FROZEN, IDX_DSMIX, IDX_LAMN, IDX_SGMIX, NAN, NS_H, LoadFC, day0, etype

# kinds
HOLIDAY, CLOSURE, EVE, LOCAL, HOLYWEEK = "holiday", "closure", "eve", "local", "holyweek"

# Calendar of the competition's span (general knowledge, typed in; no data file).
CALENDAR = {
    "2025-09-11": HOLIDAY, "2025-12-08": HOLIDAY, "2025-12-25": HOLIDAY, "2025-12-26": HOLIDAY,
    "2026-01-01": HOLIDAY, "2026-01-06": HOLIDAY, "2026-04-03": HOLIDAY, "2026-04-06": HOLIDAY,
    "2026-05-01": HOLIDAY, "2026-06-24": HOLIDAY,
    "2025-09-24": LOCAL, "2026-05-25": LOCAL,
    "2025-12-24": CLOSURE, "2025-12-29": CLOSURE, "2025-12-30": CLOSURE, "2025-12-31": CLOSURE,
    "2026-01-02": CLOSURE, "2026-01-05": CLOSURE,
    "2026-03-30": HOLYWEEK, "2026-03-31": HOLYWEEK, "2026-04-01": HOLYWEEK, "2026-04-02": HOLYWEEK,
}


def d0_of(date) -> int:
    """Hour index of 06:00 of a calendar date (start of its meter day)."""
    return int(pd.Timestamp(date).normalize().value // NS_H) + 6


class CalLoadFC(LoadFC):
    def __init__(self, fit_df=None, calendar=None, cal_blend=0.0, prior=None, local_trust=1.0,
                 clean_ref=True, lam_day=None, active=None, **kw):
        """calendar: {date: kind} or None. prior: {kind: index} to force (default: estimated from the
        fit table by fit_priors()). local_trust: the prior of the "local" kind (never seen in the data)
        is local_trust x the prior of "holiday". A kind with no day in the fit table and no forced prior
        is dropped from the calendar. clean_ref: leave calendar days out of the weekday reference.
        lam_day: diagnostic override {meter-day start: index}. active: dates of the calendar that act at
        run time (default: all); the other entries only serve to estimate the priors."""
        self.cal = {}                                  # meter-day start -> kind
        self.cal_blend = cal_blend
        self.clean_ref = clean_ref
        self.prior_forced = dict(prior or {})
        self.prior = {}
        self.local_trust = local_trust
        self.prior_n = {}
        self.dropped = []
        self.lam_day = dict(lam_day or {})
        self.active = None if active is None else {d0_of(d) for d in active}
        self._calendar = dict(calendar or {})
        super().__init__(fit_df, **kw)
        if fit_df is None:
            self.set_calendar(self._calendar)
            self.prior = dict(self.prior_forced)

    # ------------------------------------------------------------------ calendar and priors
    def set_calendar(self, calendar):
        self._calendar = dict(calendar or {})
        self.cal = {d0_of(d): k for d, k in self._calendar.items() if pd.Timestamp(d).dayofweek < 5}
        self.reset()

    def fit(self, fit_df):
        cal, self.cal = self._calendar, {}             # the model is fitted exactly as LoadFC's
        super().fit(fit_df)
        self.set_calendar(cal)
        if self.cal:
            self.fit_priors()
        return self

    def fit_priors(self):
        """Prior index per kind = mean, over the fit-table days of that kind, of the least-squares index
        of the whole day against the reference the forecaster would have had that morning."""
        est = {}
        for d0, kind in sorted(self.cal.items()):
            num, den, n = self._lam_parts(d0, d0 + 16)
            if n >= 8 and den > 1e4 * n:
                est.setdefault(kind, []).append(min(max(num / den, -1.5), 1.3))
        self.prior_n = {k: len(v) for k, v in est.items()}
        self.prior_fit = {k: float(np.mean(v)) for k, v in est.items()}
        self.prior, self.dropped = {}, []
        for kind in sorted(set(self.cal.values())):
            if kind in self.prior_forced:
                self.prior[kind] = float(self.prior_forced[kind])
            elif kind in self.prior_fit:
                self.prior[kind] = self.prior_fit[kind]
            elif kind == LOCAL and (HOLIDAY in self.prior_fit or HOLIDAY in self.prior_forced):
                self.prior[kind] = self.local_trust * float(self.prior_forced.get(HOLIDAY, self.prior_fit.get(HOLIDAY)))
            else:                                      # nothing to estimate it from: the calendar ignores this kind
                self.dropped.append(kind)
        if self.dropped:
            self.cal = {d: k for d, k in self.cal.items() if k not in self.dropped}
        if self.active is not None:                    # the priors are fitted on every entry; only these act
            self.cal = {d: k for d, k in self.cal.items() if d in self.active}
        self.reset()
        return self.prior

    # ------------------------------------------------------------------ reference without calendar days
    def _S(self, j, known, t=-1):
        if not self.cal or not self.clean_ref:
            return super()._S(j, known, t)
        if t < 0:
            t = etype(j)
        if t != 0:
            return super()._S(j, known, t)
        m0 = -((known - j) // 24)
        if m0 < 1:
            m0 = 1
        key = (j, t, m0)
        v = self._sc.get(key)
        if v is not None:
            return v
        need = self.n_ref[0]
        tab, cal = self.tab, self.cal
        s, n = 0.0, 0
        for m in range(m0, m0 + 35):
            jj = j - 24 * m
            if etype(jj) != 0 or day0(jj) in cal:
                continue
            x = tab.get(jj)
            if x is not None:
                s += x
                n += 1
                if n == need:
                    break
        v = s / n if n else self.prof[0][j % 24]
        self._sc[key] = v
        return v

    # ------------------------------------------------------------------ holiday index
    def _prior_of(self, d0):
        """Index a weekday meter day starts with, or None when LoadFC's own carry-over applies."""
        v = self.lam_day.get(d0)
        if v is not None:
            return v
        k = self.cal.get(d0)
        if k is not None:
            return self.prior.get(k, 0.0)
        if self.cal:                                   # normal weekday right after a calendar weekday: no carry-over
            for m in range(1, 5):
                dp = d0 - 24 * m
                if etype(dp) == 0:
                    return 0.0 if dp in self.cal else None
        return None

    def _lam_now(self, i):
        if not self.cal and not self.lam_day:
            return super()._lam_now(i)
        if etype(i) != 0:
            return super()._lam_now(i)
        d0 = day0(i)
        if d0 in self.lam_day:
            return self.lam_day[d0], d0
        prior = self._prior_of(d0)
        if prior is None:
            return super()._lam_now(i)
        num, den, n = self._lam_parts(d0, i - 1)
        lam = (num + self.kappa * prior) / (den + self.kappa)
        return min(max(lam, -1.5), 1.3), d0

    # ------------------------------------------------------------------ predictions
    def _h0(self, i, obs):
        if (self.cal or self.lam_day) and self.cal_blend > 0.0 and etype(i) == 0:
            d0 = day0(i)
            if d0 in self.cal or d0 in self.lam_day:
                keep = self.hol_blend
                self.hol_blend = self.cal_blend
                try:
                    return super()._h0(i, obs)
                finally:
                    self.hol_blend = keep
        return super()._h0(i, obs)

    def _predict(self, obs, hist, H):
        if not self.cal and not self.lam_day:
            return super()._predict(obs, hist, H)
        i = self._sync(obs, hist)
        c = self._cache
        if c is not None and len(c[0]) >= H:
            return c
        p0, regime = self._h0(i, obs)
        out = np.empty(H)
        out[0] = p0
        if H > 1:
            known = i - 1
            S = self._S
            if self.mix:
                lam0, dref = self._lam_now(i)
            else:
                lam0, dref = 0.0, day0(i)
            rho = self.rho
            lam_d = {day0(i): lam0} if etype(i) == 0 else {}

            def lam_of(dj):
                lam = lam_d.get(dj)
                if lam is None:
                    pr = self._prior_of(dj) if self.mix else None
                    if pr is not None:
                        lam = pr                           # the calendar (or the override) knows this day
                    else:                                  # LoadFC's carry-over from the last known weekday
                        steps = sum(1 for d in range(dref + 24, dj + 1, 24) if etype(d) == 0)
                        lam = lam0 * rho ** steps
                    lam_d[dj] = lam
                return lam

            def ref(j):
                if etype(j) != 0:
                    return S(j, known)
                lam = lam_of(day0(j))
                if lam == 0.0:
                    return S(j, known)
                swd = S(j, known, 0)
                return swd - lam * (swd - S(j, known, 2))

            a = 0.0 if regime == "fault" else p0 - ref(i)
            phi = self.phi
            f = 1.0
            prevv = p0
            for k in range(1, H):
                j = i + k
                f *= phi
                if j % 24 in FROZEN and not (k == 1 and regime in ("fault", "cold")):
                    v = prevv
                else:
                    v = ref(j) + a * f
                    if v < 0.0:
                        v = 0.0
                out[k] = v
                prevv = v
        self._cache = (out, regime)
        return self._cache

    # ------------------------------------------------------------------ packaging
    def to_state(self) -> dict:
        st = super().to_state()
        st["cal"] = {"calendar": dict(self._calendar), "prior": dict(self.prior), "cal_blend": self.cal_blend,
                     "clean_ref": self.clean_ref, "local_trust": self.local_trust,
                     "active": None if self.active is None else sorted(self.active)}
        return st

    @classmethod
    def from_state(cls, st: dict):
        fc = super().from_state({k: v for k, v in st.items() if k != "cal"})
        c = st.get("cal")
        if c:
            fc.cal_blend, fc.clean_ref, fc.local_trust = c["cal_blend"], c["clean_ref"], c["local_trust"]
            fc.set_calendar(c["calendar"])
            fc.prior = dict(c["prior"])
            fc.cal = {d: k for d, k in fc.cal.items() if k in fc.prior}      # kinds dropped at fit time stay dropped
            if c.get("active") is not None:
                fc.active = set(c["active"])
                fc.cal = {d: k for d, k in fc.cal.items() if d in fc.active}
            fc.reset()
        return fc
