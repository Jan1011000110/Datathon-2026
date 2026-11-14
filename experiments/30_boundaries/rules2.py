"""
experiments/30_boundaries/rules2.py
Boundary rules that act on the VALUE of the energy held at the end of a week rather than on a forecast:

    MidnightValue(fit_off)                  marginal value of a kWh stored at midnight, by month, from the fit rows
    BoundaryAgent(h2_agent, mv, mode, ...)  h2 whose plan is split at the block end, with that value there
    JumpPaths(pool, gamma) + NoteTs         h3: wider signal error paths for the hours beyond the jump
"""
import numpy as np
import pandas as pd

from lib import cb
from rules import Calendar

P = cb.P
GRID = 25.0                                    # kWh per cell of the tabulated value functions
NCELL = int(P.CAP / GRID)
MID = (np.arange(NCELL) + 0.5) * GRID


def tabulate(vv, vl):
    """Step function (values descending, widths) -> values on the SOC cells."""
    return np.asarray(vv, float)[np.minimum(np.searchsorted(np.cumsum(vl), MID, side="right"), len(vv) - 1)]


class MidnightValue:
    """For every fit day with 36 complete hours ahead: the perfect-foresight value function of the energy held at
    midnight (planner.backward on the true signal and net load); averaged over the days of each month.
    Signal units, tabulated on SOC cells of 25 kWh."""
    def __init__(self, fit_off, hours=36):
        w = (fit_off["co2_kg_per_kwh"] + cb.KAPPA * fit_off["precio_eur_per_kwh"]).to_numpy(float)
        n = (fit_off["load_kw"] - fit_off["pv_kw"]).to_numpy(float)
        ts = fit_off.index
        hrs = ts.values.astype("datetime64[h]").astype(np.int64)
        acc = {}
        for i in np.flatnonzero(ts.hour == 0):
            if i + hours > len(ts) or hrs[i + hours - 1] - hrs[i] != hours - 1:
                continue
            sv, sl = P.stage_segments_det(w[i:i + hours], n[i:i + hours])
            vv, vl = P.backward([None] + list(sv), [None] + list(sl))
            acc.setdefault(ts[i].month, []).append(tabulate(vv, vl))
        self.by_month = {m: np.mean(v, axis=0) for m, v in acc.items()}
        self.n_days = {m: len(v) for m, v in acc.items()}
        self.all = np.mean([x for v in acc.values() for x in v], axis=0)

    def at(self, month):
        return self.by_month.get(month, self.all)


class BoundaryAgent(cb.CombinedAgent):
    """h2 with a modified value of the energy held at the end of the block. In the last hours of a block the plan is
    split there: value at the block end = f(value on the forecast beyond the jump, train value of that month).
        mode "same"   the value on the forecast (must reproduce h2 exactly: a check of the split)
             "clim"   (1 - lam) * value on the forecast + lam * gain * train value of the next block's month
             "floor"  the first `floor_kwh` are worth at least gain x the train value
             "max"    the larger of the forecast value and gain x the train value, cell by cell"""
    def __init__(self, base, mv=None, mode="same", lam=0.5, floor_kwh=3000.0, gain=1.0):
        self.__dict__.update(base.__dict__)
        self.mv, self.mode, self.lam, self.floor_kwh, self.gain = mv, mode, lam, floor_kwh, gain
        self.cal = Calendar()

    def act(self, obs):
        self.hist.append(cb.make_rec(obs, self.kappa))
        soc = float(obs["soc_kwh"])
        r, nb = self.cal.step(obs["timestamp"])
        w_known = (np.asarray(obs["co2_properes_24h"], float) + self.kappa * np.asarray(obs["preus_properes_24h"], float))
        w = np.asarray(self.w_fc.predict(obs, self.hist, self.H), float).copy()
        n = np.asarray(self.net.predict(obs, self.hist, self.H), float)
        H = min(len(w), len(n))
        v_end = self.v_end
        ended = False
        if self.truncate_end and len(w_known) < 3:
            H = min(H, len(w_known))
            v_end, ended = None, True
        if self.end_ts is not None:
            left = int((self.end_ts - obs["timestamp"]) / pd.Timedelta(hours=1)) + 1
            if 1 <= left < H:
                H = left
                v_end, ended = None, True
        if H == 0:
            return 0.0
        w, n = w[:H], n[:H]
        k = min(len(w_known), H)
        w[:k] = w_known[:k]
        w = np.maximum(w, 1e-9)
        K = H if self.scen_hours is None else min(self.scen_hours, H)
        scen, probs = self.net.predict_scen(obs, self.hist, K, self.Q)
        scen = scen[:K]
        sv, sl = P._segments(w, scen, probs, n[K:H] + self.net.tail_bias(obs, K, H), P.PMAX, P.ETA)
        if ended or not (1 <= r < H):
            vv, vl = P.backward(sv, sl, v_end)
            return P.safe_clamp(P.first_step(soc, sv[0], sl[0], vv, vl), soc)
        vf = tabulate(*P.backward([None] + sv[r:], [None] + sl[r:], v_end))      # value at the block end, on the forecast
        if self.mode == "same":
            vx = vf
        else:
            vc = self.mv.at(nb.month) * self.gain
            if self.mode == "clim":
                vx = (1.0 - self.lam) * vf + self.lam * vc
            elif self.mode == "max":
                vx = np.maximum(vf, vc)
            else:                                                             # floor
                vx = np.where(MID < self.floor_kwh, np.maximum(vf, vc), vf)
                vx = np.maximum.accumulate(vx[::-1])[::-1]                    # keep it non-increasing in SOC
        vend = (vx, np.full(NCELL, GRID))
        if r == 1:
            vv, vl = vend
        else:
            vv, vl = P.backward(sv[:r], sl[:r], vend)
        return P.safe_clamp(P.first_step(soc, sv[0], sl[0], np.asarray(vv), np.asarray(vl)), soc)


class JumpPaths:
    """Paths pool of SigAgent whose errors are stretched by `gamma` for the hours beyond the block end."""
    def __init__(self, pool, gamma):
        self.pool, self.gamma, self.cal, self.r = pool, gamma, Calendar(), 10 ** 6
        self.K, self.m, self.beta, self.bias = pool.K, pool.m, pool.beta, pool.bias

    def note(self, ts):
        self.r, _ = self.cal.step(ts)

    def paths(self, w, hod):
        out = self.pool.paths(w, hod)
        r = max(self.r, 3)
        if r < out.shape[1]:
            out[:, r:] = np.maximum(w[r:out.shape[1]] + self.gamma * (out[:, r:] - w[r:out.shape[1]]), 1e-6)
        return out


class NoteTs:
    """Signal forecaster wrapper that tells a JumpPaths pool the timestamp of the step."""
    def __init__(self, fc, pool):
        self.fc, self.pool = fc, pool

    def predict(self, obs, hist, H):
        self.pool.note(obs["timestamp"])
        return self.fc.predict(obs, hist, H)
