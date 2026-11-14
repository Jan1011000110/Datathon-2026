"""
experiments/22_solar/solar_fc.py
Solar forecaster for hard mode: output (kW, x12 scale) of the hour being decided and the next hours.

    fc = SolarFC(fit_df)                         # fit_df: hourly table (index = local time, column pv_kw
                                                 # already x12, weather columns named as in obs if available)
    pv = fc.predict_pv(obs, hist, 36)            # np.ndarray, index 0 = the hour being decided
    q  = fc.predict_pv_quantiles(obs, hist, [0.1, 0.5, 0.9])        # horizon 0, shape (3,)
    Q  = fc.predict_pv_quantiles(obs, hist, [0.1, 0.9], H=3)        # shape (len(qs), H)

How it works (numpy only at run time, about 0.1 ms per call)
  1. Clear-sky envelope E(day of year, UTC hour): 90th percentile of the fit rows in a +-12 day window,
     smoothed. Timestamps are local time (CET/CEST); the DST rule is hand-coded.
  2. Clearness k = output / envelope. Stage 1, one ridge per horizon: clearness of the target hour from the
     clearness of the last hour, of today so far, of yesterday, and of the same hour on the last day seen.
  3. Stage 2, one ridge per (horizon, current solar hour) on the stage-1 residual: panel minus ambient
     temperature of the last hour and of the last 3 hours, and ambient temperature. At night the panel is
     colder than the air under a clear sky, so this is a cloud sensor that works before sunrise.
  4. Quantiles: out-of-block residuals / envelope, by horizon group and predicted clearness.
  5. Inverter outage rule: sun on the sensor but (almost) no output -> forecast scaled towards zero.

State: the forecaster remembers what the run has revealed (from `hist` if given, else from successive obs).
It resets itself when the timestamp goes backwards (new run). Stored fit rows are looked up by timestamp,
so yesterday's curve is known in the first hours of a test week when the day before is in the fit rows.
"""
import datetime as _dt
import math

import numpy as np
import pandas as pd

HNS = 3_600_000_000_000
EPOCH_ORD = 719163                                   # date(1970, 1, 1).toordinal()
E_MIN = 60.0                                         # kW: below this envelope a clearness ratio is noise
E_T = 20.0                                           # kW: target hours below this envelope are "night"
NAN = math.nan

FEATS = ["h", "hs_now", "k_last", "k_today", "k_yday", "k_same", "dT", "dT3", "tamb"]
NF = len(FEATS)
IX = {f: i for i, f in enumerate(FEATS)}
K_COLS = [IX[f] for f in ("k_last", "k_today", "k_yday", "k_same")]
W_COLS = [IX[f] for f in ("dT", "dT3", "tamb")]
NHS = 25                                             # current solar hour -2..22
HGROUPS = np.array([0, 1, 2, 3, 6, 12, 24, 10 ** 6])   # horizon groups for the residual quantiles
KBINS = np.array([0.3, 0.5, 0.7, 0.85])                # predicted-clearness bins for the residual quantiles
QGRID = np.linspace(0.01, 0.99, 99)


def _last_sunday(year, month):
    d = _dt.date(year, month, 31)
    return 31 - ((d.weekday() + 1) % 7)


def utc_offset(d):
    """Hours that Spanish local time is ahead of UTC on date d (the switch day counts as already switched)."""
    m = d.month
    if m < 3 or m > 10:
        return 1
    if 3 < m < 10:
        return 2
    ls = _last_sunday(d.year, m)
    if m == 3:
        return 2 if d.day >= ls else 1
    return 1 if d.day >= ls else 2


def _doy_off(daynum):
    d = _dt.date.fromordinal(int(daynum) + EPOCH_ORD)
    return d.timetuple().tm_yday - 1, utc_offset(d)


def _key(ts):
    return pd.Timestamp(ts).value // HNS


class SolarFC:
    def __init__(self, fit_df=None, H=36, fit_quantiles=True, env_q=0.9, env_win=12, lam1=1e-3, lam2=0.3,
                 use_weather=True, outage_rule=True, q_scale=1.1):
        self.Hfit, self.fit_q = H, fit_quantiles
        self.env_q, self.env_win = env_q, env_win
        self.lam1, self.lam2 = lam1, lam2
        self.use_weather, self.outage_rule = use_weather, outage_rule
        self.q_scale = q_scale                       # widens the residual quantiles around their median (1.1 calibrates folds B+C)
        self.pv_fit, self.dT_fit, self.ta_fit = {}, {}, {}
        self.qtab = None
        self.c_irr = 1.65
        self.reset()
        if fit_df is not None:
            self.fit(fit_df)

    # ------------------------------------------------------------------ state
    def reset(self):
        self.pv_run, self.dT_run, self.ta_run = {}, {}, {}
        self._dstat = {}
        self._n_ing = 0
        self._last_key = None
        self._out = 0                                # sunlit hours of inverter outage seen in a row
        self._cache_key, self._cache = None, None

    def _get(self, kk):
        v = self.pv_run.get(kk)
        if v is None:
            v = self.pv_fit.get(kk)
        return v

    def _get_dT(self, kk):
        v = self.dT_run.get(kk)
        if v is None:
            v = self.dT_fit.get(kk)
        return v

    def _dayinfo(self, d):
        r = self._dcache.get(d)
        if r is None:
            doy, off = _doy_off(d)
            loc = np.zeros(24)
            loc[off:] = self.env_tab[doy, :24 - off]
            r = (doy, off, loc, loc.tolist())
            self._dcache[d] = r
        return r

    def _env(self, kk):
        return self._dayinfo(kk // 24)[3][kk % 24]

    def _daystat(self, d):
        """Mean clearness of day d from what is known about it (NaN if fewer than 4 sunlit hours known)."""
        r = self._dstat.get(d)
        if r is None:
            env = self._dayinfo(d)[3]
            base = d * 24
            sp = se = 0.0
            n = 0
            for hh in range(24):
                e = env[hh]
                if e >= E_MIN:
                    v = self._get(base + hh)
                    if v is not None:
                        sp += v
                        se += e
                        n += 1
            r = sp / se if n >= 4 else NAN
            self._dstat[d] = r
        return r

    def _ingest(self, k, pv_last, irr, tamb, tpan):
        if self._last_key is not None and k <= self._last_key:
            if k == self._last_key:
                return
            self.reset()                                           # time went backwards: a new run
        valid = self._last_key is not None and k == self._last_key + 1
        if self._last_key is not None and k > self._last_key + 12:
            self._out = 0                                          # long jump (new test week): outage state is stale
        if valid and pv_last == pv_last:
            self.pv_run[k - 1] = pv_last
            e1 = self._env(k - 1)
            if irr == irr and irr > 100.0:
                if pv_last < 0.15 * self.c_irr * irr:
                    self._out += 1
                elif pv_last > 0.5 * self.c_irr * irr:
                    self._out = 0
            elif e1 >= 200.0:
                if pv_last <= 0.5:
                    self._out += 1
                elif pv_last > 0.05 * e1:
                    self._out = 0
        if valid and tamb == tamb and tpan == tpan:
            self.dT_run[k - 1] = tpan - tamb
            self.ta_run[k - 1] = tamb
        self._last_key = k

    def _update(self, obs, hist):
        if hist is not None and len(hist) > 0:
            if len(hist) < self._n_ing:
                self.reset()
            for rec in hist[self._n_ing:]:
                self._ingest(_key(rec["ts"]), float(rec["pv_last"]), float(rec.get("irradiancia_wm2", NAN)),
                             float(rec.get("temp_ambient_c", NAN)), float(rec.get("temp_panel_c", NAN)))
            self._n_ing = len(hist)
            return self._last_key
        k = _key(obs["timestamp"])
        last = obs["noves_dades_5min"]
        self._ingest(k, float(last.get("ultima_solar_kw", NAN)), float(last.get("irradiancia_wm2", NAN)),
                     float(last.get("temp_ambient_c", NAN)), float(last.get("temp_panel_c", NAN)))
        return k

    # ------------------------------------------------------------------ features
    def _features(self, k, H):
        """Returns (Et: envelope of the H target hours, rows: indices of sunlit targets, X: features).
        Everything comes from hours before k (run history first, stored fit rows otherwise)."""
        d0, h0 = divmod(k, 24)
        doy, off, env0, env0l = self._dayinfo(d0)
        nd = (h0 + H - 1) // 24 + 1
        if nd == 1:
            Et = env0[h0:h0 + H]
        else:
            Et = np.concatenate([env0] + [self._dayinfo(d0 + j)[2] for j in range(1, nd)])[h0:h0 + H]
        rows = np.nonzero(Et >= E_T)[0]
        X = np.empty((len(rows), NF))
        if len(rows) == 0:
            return Et, rows, X
        get = self._get
        e1 = env0l[h0 - 1] if h0 >= 1 else 0.0
        pv1 = get(k - 1) if e1 >= E_MIN else None
        k_last = pv1 / e1 if pv1 is not None else NAN
        sp = se = 0.0
        base = d0 * 24
        for hh in range(h0):
            e = env0l[hh]
            if e >= E_MIN:
                v = get(base + hh)
                if v is not None:
                    sp += v
                    se += e
        k_today = sp / se if se > 0 else NAN
        ksame = []
        for r in rows.tolist():
            kk = k + r - (24 if r < 24 else 48)
            e = self._dayinfo(kk // 24)[3][kk % 24]
            v = get(kk) if e >= E_MIN else None
            ksame.append(v / e if v is not None else NAN)
        d1 = self._get_dT(k - 1)
        if d1 is None:
            d1 = d3 = ta = NAN
        else:
            s, n = d1, 1
            for j in (2, 3):
                v = self._get_dT(k - j)
                if v is not None:
                    s += v
                    n += 1
            d3 = s / n
            ta = self.ta_run.get(k - 1)
            if ta is None:
                ta = self.ta_fit.get(k - 1, NAN)
        X[:, 0] = rows
        X[:, 1] = h0 - off
        X[:, 2] = k_last
        X[:, 3] = k_today
        X[:, 4] = self._daystat(d0 - 1)
        X[:, 5] = ksame
        X[:, 6] = d1
        X[:, 7] = d3
        X[:, 8] = ta
        return Et, rows, X

    # ------------------------------------------------------------------ fit
    def _build_envelope(self, keys, pv):
        dn = keys // 24
        hl = keys % 24
        info = {d: _doy_off(d) for d in np.unique(dn).tolist()}
        doy = np.array([info[d][0] for d in dn.tolist()])
        hu = hl - np.array([info[d][1] for d in dn.tolist()])
        tab = np.zeros((366, 24))
        x = np.arange(366)
        for u in range(24):
            m = hu == u
            if m.sum() < 20 or pv[m].max() < 5.0:
                continue
            d, v = doy[m], pv[m]
            col = np.full(366, np.nan)
            for dd in range(366):
                dist = np.abs(d - dd)
                dist = np.minimum(dist, 365 - dist)
                sel = v[dist <= self.env_win]
                if len(sel) >= 5:
                    col[dd] = np.quantile(sel, self.env_q)
            ok = ~np.isnan(col)
            col = np.interp(x, np.concatenate([x[ok] - 366, x[ok], x[ok] + 366]), np.tile(col[ok], 3))
            ext = np.concatenate([col[-7:], col, col[:7]])
            tab[:, u] = np.convolve(ext, np.ones(15) / 15, mode="valid")
        return tab

    def _replay(self, keys, H):
        """Features and targets for every fit hour as if it were a run (looks backwards only)."""
        Xs, ys, es, ds, dts = [], [], [], [], []
        get = self.pv_fit.get
        for k in keys.tolist():
            Et, rows, X = self._features(k, H)
            if len(rows) == 0:
                continue
            y = np.array([get(k + r, NAN) for r in rows.tolist()])
            ok = y == y
            Xs.append(X[ok])
            ys.append(y[ok])
            es.append(Et[rows][ok])
            ds.append(np.full(int(ok.sum()), k // 24))
            dts.append((k + rows[ok]) // 24)
        return np.vstack(Xs), np.concatenate(ys), np.concatenate(es), np.concatenate(ds), np.concatenate(dts)

    def fit(self, fit_df, verbose=False):
        df = fit_df[fit_df["pv_kw"].notna()].sort_index()
        keys = df.index.as_unit("ns").asi8 // HNS
        pv = df["pv_kw"].to_numpy(float)
        self.pv_fit = dict(zip(keys.tolist(), pv.tolist()))
        self.dT_fit, self.ta_fit = {}, {}
        if self.use_weather and "temp_panel_c" in df.columns and "temp_ambient_c" in df.columns:
            ta = df["temp_ambient_c"].to_numpy(float)
            dT = df["temp_panel_c"].to_numpy(float) - ta
            ok = ~np.isnan(dT)
            self.dT_fit = dict(zip(keys[ok].tolist(), dT[ok].tolist()))
            self.ta_fit = dict(zip(keys[ok].tolist(), ta[ok].tolist()))
        if "irradiancia_wm2" in df.columns:
            ir = df["irradiancia_wm2"].to_numpy(float)
            m = (ir > 200) & (pv > 0)
            if m.sum() > 50:
                self.c_irr = float(np.median(pv[m] / ir[m]))
        self.env_tab = self._build_envelope(keys, pv)
        self._dcache = {}
        self.reset()
        X, y, E, D, Dt = self._replay(keys, self.Hfit)
        self.reset()
        r = np.clip(y / E, 0.0, 1.6)
        w = (E / 1000.0) ** 2
        self.k_clim = float(np.sum(y) / np.sum(E))
        self.par = self._fit_model(X, r, w)
        self.n_train = len(y)
        if self.fit_q:
            # residuals from models that have not seen the target's 7-day block -> quantile tables
            bd, bt = (D // 7) % 3, (Dt // 7) % 3
            pred = np.full(len(y), np.nan)
            for b in range(3):
                tr, te = (bd != b) & (bt != b), (bd == b) & (bt == b)
                pred[te] = self._apply(X[te], self._fit_model(X[tr], r[tr], w[tr]))
            ok = ~np.isnan(pred)
            res = (y - pred * E) / E
            hg = np.searchsorted(HGROUPS, X[:, 0], side="right") - 1
            kb = np.searchsorted(KBINS, np.nan_to_num(pred))
            big = (E >= E_MIN) & ok
            self.qtab = np.zeros((len(HGROUPS) - 1, len(KBINS) + 1, len(QGRID)))
            for a in range(len(HGROUPS) - 1):
                for b in range(len(KBINS) + 1):
                    m = big & (hg == a) & (kb == b)
                    if m.sum() < 30:
                        m = big & (hg == a)
                    self.qtab[a, b] = np.quantile(res[m], QGRID)
            self.oof = {"h": X[ok, 0].astype(int), "hs_now": X[ok, 1].astype(int), "pred": pred[ok] * E[ok],
                        "y": y[ok], "E": E[ok], "X": X[ok]}
        if verbose:
            print(f"SolarFC fitted: {len(df)} rows, {self.n_train} (hour, horizon) pairs, c_irr={self.c_irr:.3f}, "
                  f"k_clim={self.k_clim:.3f}")
        return self

    def _design1(self, X):
        kc = self.k_clim
        cols = [np.ones(len(X))]
        for c in K_COLS:
            v = X[:, c]
            miss = np.isnan(v)
            cols.append(np.where(miss, 0.0, v - kc))
            cols.append(miss.astype(float))
        return np.column_stack(cols)

    def _fit_model(self, X, r, w):
        kc = self.k_clim
        Hn = self.Hfit
        h = X[:, 0].astype(int)
        hs = X[:, 1].astype(int) + 2
        Z = self._design1(X)
        c1 = np.zeros((Hn, Z.shape[1]))
        sw = np.sqrt(w)
        fitted = np.full(len(r), kc)
        for hh in range(Hn):
            m = h == hh
            if m.sum() < 50:
                continue
            A = Z[m] * sw[m][:, None]
            b = (r[m] - kc) * sw[m]
            c1[hh] = np.linalg.solve(A.T @ A + self.lam1 * w[m].sum() * np.eye(Z.shape[1]), A.T @ b)
            fitted[m] = kc + Z[m] @ c1[hh]
        nw = len(W_COLS)
        c2 = np.zeros((Hn, NHS, nw + 1))
        mu = np.zeros((Hn, NHS, nw))
        if self.use_weather:
            res = r - fitted
            cell = h * NHS + hs
            order = np.argsort(cell, kind="stable")
            cs = cell[order]
            starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
            ends = np.r_[starts[1:], len(cs)]
            for s, e in zip(starts.tolist(), ends.tolist()):
                idx = order[s:e]
                if len(idx) < 40:
                    continue
                V = X[idx][:, W_COLS]
                ok = ~np.isnan(V)
                if ok.sum(axis=0).min() < 30:
                    continue
                m_ = np.nanmean(V, axis=0)
                Vc = np.where(ok, V - m_, 0.0)
                A = np.column_stack([np.ones(len(idx)), Vc]) * sw[idx][:, None]
                b = res[idx] * sw[idx]
                # scale-aware ridge: penalise each slope by lam2 * (weighted variance of its feature)
                pen = self.lam2 * np.r_[0.0, (A[:, 1:] ** 2).sum(axis=0)]
                coef = np.linalg.solve(A.T @ A + np.diag(pen + 1e-9), A.T @ b)
                hh, ss = divmod(int(cs[s]), NHS)
                c2[hh, ss] = coef
                mu[hh, ss] = m_
        return {"c1": c1, "c2": c2, "mu": mu}

    def _apply(self, X, par):
        h = np.minimum(X[:, 0].astype(int), self.Hfit - 1)
        hs = np.clip(X[:, 1].astype(int) + 2, 0, NHS - 1)
        k = self.k_clim + np.einsum("ij,ij->i", self._design1(X), par["c1"][h])
        if self.use_weather:
            c2 = par["c2"][h, hs]
            V = X[:, W_COLS] - par["mu"][h, hs]
            k = k + c2[:, 0] + np.einsum("ij,ij->i", np.where(np.isnan(V), 0.0, V), c2[:, 1:])
        return np.clip(k, 0.0, 1.3)

    # ------------------------------------------------------------------ predict
    def _predict(self, obs, hist, H):
        k = self._update(obs, hist)
        if self._cache_key is not None and self._cache_key[0] == k and self._cache_key[1] >= H:
            return self._cache
        Et, rows, X = self._features(k, H)
        out = np.zeros(H)
        kp = np.zeros(H)
        if len(rows):
            kk = self._apply(X, self.par)
            if self.outage_rule and self._out > 0:
                kk = kk * (1.0 - 0.97 * (1.0 - 1.0 / (2.0 + 2.0 * self._out)) ** rows)
            kp[rows] = kk
            out[rows] = kk * Et[rows]
        self._cache_key, self._cache = (k, H), (out, kp, Et)
        return self._cache

    def predict_pv(self, obs, hist=None, H=36):
        return self._predict(obs, hist, H)[0][:H].copy()

    def predict_pv_quantiles(self, obs, hist=None, qs=(0.1, 0.5, 0.9), H=None):
        """Quantiles of the solar output. H=None: horizon 0 only, shape (len(qs),). H given: (len(qs), H)."""
        n = 1 if H is None else H
        out, kp, Et = self._predict(obs, hist, n)
        qs = np.atleast_1d(np.asarray(qs, float))
        res = np.zeros((len(qs), n))
        for j in range(n):
            if Et[j] < E_T or self.qtab is None:
                res[:, j] = out[j]
                continue
            a = int(np.searchsorted(HGROUPS, j, side="right") - 1)
            b = int(np.searchsorted(KBINS, kp[j]))
            tab = self.qtab[a, b]
            med = tab[len(QGRID) // 2]
            dq = med + self.q_scale * (np.interp(qs, QGRID, tab) - med)
            res[:, j] = np.clip(out[j] + Et[j] * dq, 0.0, 1.3 * Et[j])
        return res[:, 0] if H is None else res


class QuantileH0:
    """Solar forecaster for a planner that wants a cautious first hour: the mean forecast, with the first
    `n` hours replaced by the q-quantile (q > 0.5 = more sun = smaller net load = smaller discharge)."""
    def __init__(self, fc, q=0.7, n=1):
        self.fc, self.q, self.n = fc, q, n

    def predict_pv(self, obs, hist=None, H=36):
        p = self.fc.predict_pv(obs, hist, H)
        m = min(self.n, len(p))
        p[:m] = self.fc.predict_pv_quantiles(obs, hist, [self.q], H=m)[0]
        return p
