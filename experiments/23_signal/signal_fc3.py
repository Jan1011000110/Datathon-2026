"""
experiments/23_signal/signal_fc3.py
Round 3 of the signal forecaster: the round-2 forecaster (signal_fc2.SignalForecaster2, untouched) with
the option of fitting on OLDER public history of the two signals (REE, dated up to 2025-08-31, stored by
fetch_ree_history.py in data/external/ree_history/).

    hist = fetch_ree_history.load_history()            # reads the stored CSV, never the network
    fc = SignalForecaster3(fit_df, kappa, hist_df=hist)             # hist_df=None -> exactly round 2
    w = fc.predict(obs, hist_records, H)               # same interface: predict, predict_parts
    fc.to_npz(path); fc = SignalForecaster3.from_npz(path)

What the history changes
------------------------
* It only adds rows to FIT on. At run time the forecaster looks back 170 hours at most, and the first
  test week starts on 2025-09-24, so no external value is ever read during a run. `to_npz` therefore
  drops the external values from the stored calendar (trim=True): a packaged agent carries coefficients
  fitted with the history, not the history itself.
* The average day ("climatology") is kept per SEASON-YEAR (September to August) x month x day type x
  hour, so that each year's level and solar share stay out of the anomalies; the regressions on
  anomalies (48 + 12 inputs per horizon, the per-hour-of-day night model, the bridge weights) are then
  fitted on every season together, the older rows weighted by `hist_weight` (1.0 = like recent rows).
* The competition season's own cells are computed from the competition's train rows only, as in round 2.

The boundary (nothing dated after 2025-08-31 from outside data/train) is asserted again here on the
table that is passed in.
"""
import numpy as np
import pandas as pd

from signal_fc import BACK, BEST, COLS, HOLIDAYS, HOUR_NS, KAPPA_DEFAULT, SignalFC
from signal_fc2 import ALT, SignalForecaster2

COMP_START = pd.Timestamp("2025-09-01")              # first hour of the competition's data
HIST_LAST = pd.Timestamp("2025-08-31 23:00")         # last hour external history may contain


class _Fit3:
    """fit() of signal_fc.SignalFC generalised to several seasons and to weighted rows."""
    hist_df = None
    hist_weight = 1.0

    def _history(self):
        h = self.hist_df
        if h is None or len(h) == 0:
            return None
        assert h.index.max() <= HIST_LAST, f"external history contains {h.index.max()}, after the boundary {HIST_LAST}"
        h = h[[c for _, c in COLS]].dropna()
        return h[~h.index.duplicated()].sort_index()

    def fit(self, fit_df, kappa=None):
        if kappa is not None:
            self.kappa = kappa
        hist = self._history()
        both = fit_df[[c for _, c in COLS]] if hist is None else pd.concat([hist, fit_df[[c for _, c in COLS]]]).sort_index()
        both = both[~both.index.duplicated(keep="last")]
        idx = both.index
        t0 = idx.min().normalize() - pd.Timedelta(days=40)
        t1 = idx.max().normalize() + pd.Timedelta(days=75)
        grid = pd.date_range(t0, t1, freq="h")
        self.t0_ns, self.n = int(t0.value), len(grid)
        pos = ((idx.values.astype("datetime64[ns]").astype("int64") - self.t0_ns) // HOUR_NS).astype(int)
        hol = np.fromiter(((m, d) in HOLIDAYS for m, d in zip(grid.month, grid.day)), bool, len(grid))
        if not self.holidays:
            hol[:] = False
        dt = ((grid.dayofweek.values >= 5) | hol).astype(int)
        hour = grid.hour.values
        mon = grid.month.values - 1
        # season-year: September..August. Calendar rows outside the seasons that hold data use the nearest one,
        # so that without history everything falls in one season and the result is exactly round 2.
        season = np.where(grid.month.values >= 9, grid.year.values, grid.year.values - 1)
        s_of = lambda t: t.year if t.month >= 9 else t.year - 1          # noqa: E731
        s_lo, s_hi = s_of(idx.min()), s_of(fit_df.index.min())
        sidx = np.clip(season, s_lo, s_hi) - s_lo
        ns = int(sidx.max()) + 1
        sm_idx = sidx * 12 + mon                                          # (season, month)
        cell = sm_idx * 48 + dt * 24 + hour
        ncell = ns * 576
        cell2 = dt * 24 + hour
        self.hour_grid = hour.astype(int)
        self.nh = 24 if self.variant.startswith("AH") else 1
        self.S0, self.CL, CLT = {}, {}, {}
        dayf = grid.day.values + hour / 24.0
        for name, col in COLS:
            s = np.full(self.n, np.nan)
            s[pos] = both[col].to_numpy(float)
            ok = ~np.isnan(s)
            sm = np.bincount(cell[ok], weights=s[ok], minlength=ncell)
            cn = np.bincount(cell[ok], minlength=ncell)
            sm2 = np.bincount(cell2[ok], weights=s[ok], minlength=48)
            cn2 = np.bincount(cell2[ok], minlength=48)
            base = (sm2 / np.maximum(cn2, 1))[cell2]
            cl = np.where(cn[cell] > 0, sm[cell] / np.maximum(cn[cell], 1), base)
            clt = cl.copy()
            m = ok & (cn[cell] > 1)
            clt[m] = (sm[cell][m] - s[m]) / (cn[cell][m] - 1)
            if self.interp:
                dsum = np.bincount(sm_idx[ok], weights=dayf[ok], minlength=ns * 12)
                dcnt = np.bincount(sm_idx[ok], minlength=ns * 12)
                cen = np.where(dcnt > 0, dsum / np.maximum(dcnt, 1), 15.5)
                lam_t = (dayf - cen[sm_idx]) / 30.4
                nb_mon = np.where(lam_t >= 0, (mon + 1) % 12, (mon - 1) % 12)
                nb_cell = (sidx * 12 + nb_mon) * 48 + dt * 24 + hour
                wn = np.where(cn[nb_cell] > 0, np.clip(np.abs(lam_t), 0.0, 1.0), 0.0)
                cl_n = sm[nb_cell] / np.maximum(cn[nb_cell], 1)
                cl = (1 - wn) * cl + wn * cl_n
                clt = (1 - wn) * clt + wn * cl_n
            if self.variant.startswith("A"):
                cl = clt if self.loo else cl
            self.S0[name], self.CL[name], CLT[name] = s, cl, clt if self.loo else cl
        O = np.arange(BACK + 2, self.n - self.kmax - 2)
        i_comp = int((COMP_START.value - self.t0_ns) // HOUR_NS)
        wO = np.where(O < i_comp, float(self.hist_weight), 1.0)
        weighted = bool((wO != 1.0).any())
        if hist is not None and self.hist_weight <= 0.0:
            wO = (O >= i_comp).astype(float)
        self.coef = {}
        for name, _ in COLS:
            other = "p" if name == "c" else "c"
            for lw in ((True, False) if self.use_lw else (False,)):
                g, P = self._feats(self.S0[name], self.CL[name], CLT[name], O, lw,
                                   self.S0[other] if self.cross else None, self.CL[other])
                Y = self.S0[name][O[:, None] + self.ks[None, :]]
                if self.variant.startswith("A"):
                    Y = Y - CLT[name][O[:, None] + self.ks[None, :]]
                g_ok = np.isfinite(g).all(1) & (wO > 0)
                K, nh = len(self.ks), self.nh
                A = np.zeros((nh, K, g.shape[1]))
                B = np.zeros((nh, K, P.shape[2]))
                c0 = np.zeros((nh, K))
                nfit = np.zeros((nh, K), int)
                oh = self.hour_grid[O]
                for h in range(nh):
                    h_ok = g_ok & (oh == h) if nh > 1 else g_ok
                    for j in range(K):
                        okj = h_ok & np.isfinite(P[:, j, :]).all(1) & np.isfinite(Y[:, j])
                        X = np.hstack([g[okj], P[okj, j, :]])
                        y = Y[okj, j]
                        nfit[h, j] = len(y)
                        if len(y) < 40:
                            continue
                        if weighted:
                            w = wO[okj]
                            sw = w.sum()
                            mu = (X * w[:, None]).sum(0) / sw
                            sd = np.sqrt((((X - mu) ** 2) * w[:, None]).sum(0) / sw) + 1e-12
                            ym = float((y * w).sum() / sw)
                            Xs = (X - mu) / sd
                            b = np.linalg.solve((Xs * w[:, None]).T @ Xs + self.ridge * sw * np.eye(X.shape[1]),
                                                (Xs * w[:, None]).T @ (y - ym))
                        else:
                            mu, sd = X.mean(0), X.std(0) + 1e-12
                            ym = float(y.mean())
                            Xs = (X - mu) / sd
                            b = np.linalg.solve(Xs.T @ Xs + self.ridge * len(y) * np.eye(X.shape[1]), Xs.T @ (y - ym))
                        b = b / sd
                        A[h, j], B[h, j] = b[:g.shape[1]], b[g.shape[1]:]
                        c0[h, j] = ym - mu @ b
                self.coef[(name, lw)] = (A, B, c0, nfit)
        self.S = {k: v.copy() for k, v in self.S0.items()}
        self._last_i = None
        self.__dict__.pop("_lo", None)
        self.n_fit_rows = {"history": 0 if hist is None else len(hist), "competition": len(fit_df)}
        return self

    def _floor(self, name):
        """Lowest value a forecast may take: 0.9 x the lowest value in the competition's own fit rows."""
        if not hasattr(self, "_lo"):
            i = max(int((COMP_START.value - self.t0_ns) // HOUR_NS), 0)
            self._lo = {k: 0.9 * float(np.nanmin(v[i:])) for k, v in self.S0.items()}
        return self._lo[name]


class SignalFC3(_Fit3, SignalFC):
    """A plain SignalFC (any variant) fitted with optional history."""
    def __init__(self, fit_df=None, kappa=KAPPA_DEFAULT, hist_df=None, hist_weight=1.0, **kw):
        self.hist_df, self.hist_weight = hist_df, hist_weight
        SignalFC.__init__(self, fit_df, kappa, **kw)


class SignalForecaster3(_Fit3, SignalForecaster2):
    """Round 2 (bridge + night level) fitted with optional history. hist_df=None gives exactly round 2.
    alt: keyword overrides of the per-hour-of-day night model (signal_fc2.ALT)."""
    def __init__(self, fit_df=None, kappa=KAPPA_DEFAULT, hist_df=None, hist_weight=1.0, alt=None, **kw):
        self.hist_df, self.hist_weight = hist_df, hist_weight
        self.alt_kw = {**ALT, **(alt or {})}
        SignalForecaster2.__init__(self, fit_df, kappa, **kw)

    def fit(self, fit_df, kappa=None):
        _Fit3.fit(self, fit_df, kappa)
        if self.bridge:
            self._fit_bridge()
        if self.night:
            self.alt = SignalFC3(fit_df, self.kappa, hist_df=self.hist_df, hist_weight=self.hist_weight, **self.alt_kw)
        return self

    # ------------------------------------------------------------------ packaging
    def _trimmed(self, obj):
        """Calendar arrays of `obj` cut to the competition span, external values blanked."""
        i_comp = int((COMP_START.value - obj.t0_ns) // HOUR_NS)
        a = max(i_comp - 40 * 24, 0)
        S0 = {k: v[a:].copy() for k, v in obj.S0.items()}
        for v in S0.values():
            v[:i_comp - a] = np.nan                                     # nothing external is shipped
        return dict(t0_ns=obj.t0_ns + a * HOUR_NS, n=obj.n - a, hour_grid=obj.hour_grid[a:],
                    S0=S0, CL={k: v[a:].copy() for k, v in obj.CL.items()})

    def to_npz(self, path, trim=True):
        """trim=True: the stored calendar starts 40 days before 2025-09-01 and holds no external value."""
        objs = [self] + ([self.alt] if self.night and self.alt is not None else [])
        keep = [{k: getattr(o, k) for k in ("t0_ns", "n", "hour_grid", "S0", "CL")} for o in objs]
        lo = [o._floor(n) for o in objs for n in "cp"]                   # floors are fixed before trimming
        try:
            if trim:
                for o in objs:
                    o.__dict__.update(self._trimmed(o))
            SignalForecaster2.to_npz(self, path)
        finally:
            for o, k in zip(objs, keep):
                o.__dict__.update(k)
        assert lo == [o._floor(n) for o in objs for n in "cp"]

    @classmethod
    def from_npz(cls, path):
        base = SignalForecaster2.from_npz(path)
        self = cls(None, kappa=base.kappa)
        keep = {k: getattr(self, k) for k in ("hist_df", "hist_weight", "alt_kw")}
        self.__dict__.update(base.__dict__)
        self.__dict__.update(keep)
        self.reset()
        return self


def packaged_arrays(fc):
    """Names and shapes of the arrays a packaged agent needs (what to_npz writes)."""
    out = {"meta, meta2, meta3, t0_ns, variant": "scalars", "hour_grid": (fc.n,)}
    for name, _ in COLS:
        out[f"S0_{name} (stored calendar, competition span only when trimmed)"] = fc.S0[name].shape
        out[f"CL_{name} (average day on the calendar)"] = fc.CL[name].shape
    for (name, lw), (A, B, c0, _) in fc.coef.items():
        out[f"A_{name}_{int(lw)}, B_{name}_{int(lw)}, c_{name}_{int(lw)} (ridge coefficients)"] = (A.shape, B.shape, c0.shape)
    if fc.bridge:
        out["rho_c, rho_p (bridge weights)"] = fc.rho["c"].shape
    if fc.night and fc.alt is not None:
        out["second file .alt.npz: the same arrays for the per-hour-of-day model"] = fc.alt.coef[("c", False)][0].shape
    return out
