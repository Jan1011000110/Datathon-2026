"""
experiments/27_combined/combined.py
The combined hard-mode agent (harness version): stochastic planner + load, solar and signal forecasters.

    from combined import make_agent
    agent = make_agent(fold_df, signal="fc", load="fc", solar="fc")      # evaluated on fold_df, fitted outside it
    agent = make_agent(None)                                             # fitted on every train row (what we ship)
    res = simulate(agent, fold_df)

Pieces (none of them is modified here, they are only wrapped):
    planner   experiments/24_planner/planner.py     plan_stoch / plan_det
    load      experiments/21_load/load_fc.py        LoadFC          ("fc", "nomodel") or ProfileN ("naive")
    solar     experiments/22_solar/solar_fc.py      SolarFC         ("fc")            or ProfileN ("naive")
    signal    experiments/23_signal/signal_fc.py    SignalForecaster("fc"), ClimW ("clim"), PerfectW ("perfect")

Net-load scenarios (class NetScen), `scen=`:
    "emp"     point forecast (load - solar) + empirical errors of that same difference by (hour of day of the
              target hour, horizon). The errors come from blocked cross-fitting: each 7-day block of the fit rows
              is replayed as a run by forecasters fitted without it (crossfit()).
    "conv"    conditional: hour 0 = LoadFC's own quantiles (zero width when the meter is frozen) combined with
              SolarFC's quantiles (they depend on the predicted clearness), assuming independent errors: all
              45 x 45 differences, reduced to Q equal-probability scenarios. Later hours: load point forecast
              + cross-fitted load errors by (hour, horizon), combined the same way with SolarFC's quantiles.
    "hybrid"  "conv" for hour 0, "emp" for hours 1+.

Recommended (reports/18_combined/report.md): the defaults of make_agent, i.e. signal="fc", load="fc", solar="fc",
scen="conv", H=36, Q=9, scen_hours=12, truncate_end=True, plus end_ts="2026-08-31 23:00" for the real test if we
accept that the run ends there. Files: build_cache.py (forecast tapes + cross-fitted replays), sweep.py (closed-loop
sets), calibration.py, check_live.py (untaped run, timing, organisers' environment), make_figs.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "experiments" / "21_load", ROOT / "experiments" / "22_solar",
           ROOT / "experiments" / "23_signal", ROOT / "experiments" / "24_planner"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.hard import METEO_COLS, get_folds, load_hard, simulate  # noqa: E402
from src.hard_mpc import KAPPA, ClimW, PerfectW, ProfileN  # noqa: E402
import planner as P  # noqa: E402
from fit_table import load_fit_table  # noqa: E402
from load_fc import LoadFC  # noqa: E402
from solar_fc import SolarFC  # noqa: E402
from signal_fc import SignalForecaster  # noqa: E402

HT = 36                                   # longest horizon covered by the residual tables
FINE = 450                                # levels of the empirical residual tables (divisible by 5, 9, 15, 25)
NLEV = 45                                 # levels of the conditional quantile sets (45 * 45 divisible by 5, 9, 15, 25)
LEV = (np.arange(NLEV) + 0.5) / NLEV
LEV_FINE = (np.arange(FINE) + 0.5) / FINE
FOLD_DAYS = {"A": (1, 7), "B": (8, 14), "C": (15, 21)}
CACHE = HERE / "cache"
LOAD_KEYS = {"fc": "load_fc", "nomodel": "load_nm", "naive": "load_naive"}
PV_KEYS = {"fc": "pv_fc", "naive": "pv_naive"}


# --------------------------------------------------------------------------------------
# Fit tables and forecasters
# --------------------------------------------------------------------------------------
def fold_name(fold_df):
    if fold_df is None:
        return "ALL"
    lo, hi = int(fold_df.index.day.min()), int(fold_df.index.day.max())
    for k, v in FOLD_DAYS.items():
        if v == (lo, hi):
            return k
    return f"d{lo}-{hi}"


def fit_tables(fold_df=None, drop_days=None):
    """(official rows, load table, signal table) a model may be fitted on when it is evaluated on fold_df
    (None = every train row). drop_days=(lo, hi) also removes those days of every month (cross-fitting)."""
    off, lt, sg = load_hard(), load_fit_table(), load_hard("filled")
    if fold_df is not None:
        off = off[~off.index.isin(fold_df.index)]
        lt = lt[~lt.index.isin(fold_df.index)]                    # as experiments/21_load/closed_loop.py
        lo, hi = int(fold_df.index.day.min()), int(fold_df.index.day.max())
        sg = sg[~((sg.index.day >= lo) & (sg.index.day <= hi))]   # as experiments/23_signal/common.py
    if drop_days is not None:
        lo, hi = drop_days
        off = off[~((off.index.day >= lo) & (off.index.day <= hi))]
        lt = lt[~((lt.index.day >= lo) & (lt.index.day <= hi))]
    return off, lt, sg


def build_forecasters(fold_df=None, drop_days=None, quantiles=True, which=None):
    """Fresh forecasters fitted on the legal rows. Keys: load_fc, load_nm, load_naive, pv_fc, pv_naive,
    w_clim, w_sig. `which` restricts the set."""
    off, lt, sg = fit_tables(fold_df, drop_days)
    want = (lambda k: which is None or k in which)
    out = {}
    if want("load_naive") or want("pv_naive"):
        prof = ProfileN(off)
        out["load_naive"] = out["pv_naive"] = prof
    if want("load_fc"):
        out["load_fc"] = LoadFC(lt, oof=quantiles)
    if want("load_nm"):
        out["load_nm"] = LoadFC(lt, use_model=False)
    if want("pv_fc"):
        out["pv_fc"] = SolarFC(off, fit_quantiles=quantiles)
    if want("w_clim"):
        out["w_clim"] = ClimW(off)
    if want("w_sig"):
        out["w_sig"] = SignalForecaster(sg, KAPPA)
    return {k: v for k, v in out.items() if want(k)}


def make_rec(obs, kappa=KAPPA):
    """History record of one step. Unlike planner.MPCStoch.record it keeps the weather readings: SolarFC reads
    them from `hist` when hist is given, and silently loses its cloud sensor if they are missing."""
    last = obs["noves_dades_5min"]
    rec = {"ts": obs["timestamp"], "co2": obs["co2_actual"], "price": obs["preu_actual"],
           "w": obs["co2_actual"] + kappa * obs["preu_actual"],
           "load_last": float(last["ultim_consum_kw"]), "pv_last": float(last["ultima_solar_kw"])}
    for c in METEO_COLS:
        rec[c] = float(last.get(c, np.nan))
    return rec


# --------------------------------------------------------------------------------------
# Tapes: forecasts do not depend on what the battery does, so they are recorded once per run
# --------------------------------------------------------------------------------------
class Recorder:
    """Pseudo-agent: calls every forecaster each hour exactly as an agent would and stores the answers."""
    def __init__(self, fcs, quantiles=True, ht=HT):
        self.fcs, self.q, self.ht = fcs, quantiles, ht
        self.hist = []
        self.tape = {}
        self.sec = {}

    def _put(self, key, f):
        import time
        t0 = time.perf_counter()
        v = np.asarray(f(), float)
        self.sec[key] = self.sec.get(key, 0.0) + time.perf_counter() - t0
        self.tape.setdefault(key, []).append(v)

    def act(self, obs):
        self.hist.append(make_rec(obs))
        h, fcs, ht = self.hist, self.fcs, self.ht
        for k in ("load_fc", "load_nm", "load_naive"):
            if k in fcs:
                self._put(k, lambda: fcs[k].predict_load(obs, h, ht))
                if self.q and k != "load_naive":
                    self._put(k + "_q", lambda: fcs[k].predict_load_quantiles(obs, h, LEV))
        for k in ("pv_fc", "pv_naive"):
            if k in fcs:
                self._put(k, lambda: fcs[k].predict_pv(obs, h, ht))
                if self.q and k == "pv_fc":
                    self._put(k + "_q", lambda: fcs[k].predict_pv_quantiles(obs, h, LEV, H=ht))
        for k in ("w_clim", "w_sig"):
            if k in fcs:
                self._put(k, lambda: fcs[k].predict(obs, h, ht))
        return 0.0

    def arrays(self):
        return {k: np.asarray(v, np.float32 if k.endswith("_q") else float) for k, v in self.tape.items()}


def record(df, fcs, quantiles=True):
    rec = Recorder(fcs, quantiles)
    simulate(rec, df)
    out = rec.arrays()
    out["hrs"] = df.index.values.astype("datetime64[h]").astype(np.int64)
    out["load"] = df["load_kw"].to_numpy(float)
    out["pv"] = df["pv_kw"].to_numpy(float)
    return out, rec.sec


class TapeLoad:
    def __init__(self, tape, key):
        self.x, self.q = tape[key], tape.get(key + "_q")

    def predict_load(self, obs, hist, H):
        return self.x[len(hist) - 1, :H].copy()

    def predict_load_quantiles(self, obs, hist, qs):
        return self.q[len(hist) - 1].astype(float)             # recorded at LEV


class TapePV:
    def __init__(self, tape, key):
        self.x, self.q = tape[key], tape.get(key + "_q")

    def predict_pv(self, obs, hist, H):
        return self.x[len(hist) - 1, :H].copy()

    def predict_pv_quantiles(self, obs, hist, qs, H=None):
        q = self.q[len(hist) - 1].astype(float)                # (NLEV, HT), recorded at LEV
        return q[:, 0] if H is None else q[:, :H]


class TapeW:
    def __init__(self, tape, key):
        self.x = tape[key]

    def predict(self, obs, hist, H):
        return self.x[len(hist) - 1, :H].copy()


# --------------------------------------------------------------------------------------
# Honest forecast errors: blocked cross-fitting inside the fit rows
# --------------------------------------------------------------------------------------
def crossfit(fold_df=None, verbose=False):
    """For every 7-day block of the fit rows (the other folds' day ranges): fit the point forecasters on the fit
    rows minus that block and replay the block as a run. Returns {block: tape}. Cached on disk."""
    name = fold_name(fold_df)
    path = CACHE / f"xfit_{name}.npz"
    if path.exists():
        z = np.load(path)
        out = {}
        for k in z.files:
            b, key = k.split("|")
            out.setdefault(b, {})[key] = z[k]
        return out
    folds = get_folds()
    out = {}
    for b, days in FOLD_DAYS.items():
        if b == name:
            continue
        fcs = build_forecasters(fold_df, drop_days=days, quantiles=False,
                                which=("load_fc", "load_nm", "load_naive", "pv_fc", "pv_naive"))
        out[b], _ = record(folds[b], fcs, quantiles=False)
        if verbose:
            print("crossfit", name, "block", b, "done", flush=True)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **{f"{b}|{k}": v for b, t in out.items() for k, v in t.items()})
    return out


def residuals(tape, key=None, load_key=None, pv_key=None):
    """(n, HT) errors actual - forecast of the forecasts made at each row, NaN where row t+k is not hour t+k.
    key = "net" (load_key - pv_key), "load" or "pv"."""
    hrs = tape["hrs"]
    if key == "load":
        act, F = tape["load"], tape[load_key]
    elif key == "pv":
        act, F = tape["pv"], tape[pv_key]
    else:
        act, F = tape["load"] - tape["pv"], tape[load_key] - tape[pv_key]
    n, H = F.shape
    R = np.full((n, H), np.nan)
    for k in range(H):
        ok = (hrs[k:] - hrs[:n - k]) == k
        R[:n - k, k] = np.where(ok, act[k:] - F[:n - k, k], np.nan)
    return R


def quantile_table(hod0, R, lev, min_n=20):
    """(24, H, len(lev)) quantiles of the errors by (hour of day of the target hour, horizon)."""
    H = R.shape[1]
    tab = np.zeros((24, H, len(lev)))
    for k in range(H):
        hod_k = (hod0 + k) % 24
        col = R[:, k]
        good = ~np.isnan(col)
        for h in range(24):
            x = col[good & (hod_k == h)]
            tab[h, k] = np.quantile(x if len(x) >= min_n else col[good], lev)
    return tab


_TABLES = {}


def error_tables_for(fold_df, load="fc", solar="fc", blocks=None):
    """error_tables(crossfit(fold_df), ...) memoised per process."""
    key = (fold_name(fold_df), load, solar, None if blocks is None else tuple(blocks))
    if key not in _TABLES:
        _TABLES[key] = error_tables(crossfit(fold_df), load, solar, blocks)
    return _TABLES[key]


def error_tables(xfit, load="fc", solar="fc", blocks=None):
    """Cross-fitted error tables of a (load, solar) pair: net-load errors at FINE levels, load errors at LEV."""
    lk, pk = LOAD_KEYS[load], PV_KEYS[solar]
    bl = [b for b in xfit if blocks is None or b in blocks]
    hod = np.concatenate([xfit[b]["hrs"] % 24 for b in bl])
    Rn = np.vstack([residuals(xfit[b], "net", lk, pk) for b in bl])
    Rl = np.vstack([residuals(xfit[b], "load", lk, pk) for b in bl])
    Rp = np.vstack([residuals(xfit[b], "pv", lk, pk) for b in bl])
    return quantile_table(hod, Rn, LEV_FINE), quantile_table(hod, Rl, LEV), quantile_table(hod, Rp, LEV)


class EmpLoadQ:
    """Gives a point load forecaster the hour-0 quantile method of LoadFC, from cross-fitted errors by hour."""
    def __init__(self, fc, load_table):
        self.fc, self.t = fc, load_table

    def predict_load(self, obs, hist, H):
        return self.fc.predict_load(obs, hist, H)

    def predict_load_quantiles(self, obs, hist, qs):
        return np.maximum(self.fc.predict_load(obs, hist, 1)[0] + self.t[obs["timestamp"].hour, 0], 0.0)


class EmpPVQ:
    """Gives a point solar forecaster the quantile method of SolarFC, from cross-fitted errors by (hour, horizon)."""
    def __init__(self, fc, pv_table):
        self.fc, self.t = fc, pv_table

    def predict_pv(self, obs, hist, H):
        return self.fc.predict_pv(obs, hist, H)

    def predict_pv_quantiles(self, obs, hist, qs, H=None):
        n = 1 if H is None else H
        kk = np.arange(n)
        q = np.maximum(np.asarray(self.fc.predict_pv(obs, hist, n), float)[:, None]
                       + self.t[(obs["timestamp"].hour + kk) % 24, kk], 0.0).T
        return q[:, 0] if H is None else q


# --------------------------------------------------------------------------------------
# Net-load forecaster with scenarios
# --------------------------------------------------------------------------------------
def _reduce(v, Q):
    """(..., M) equally likely values -> (..., Q) means of Q equal-probability bins (M divisible by Q)."""
    v = np.sort(v, axis=-1)
    return v.reshape(v.shape[:-1] + (Q, v.shape[-1] // Q)).mean(axis=-1)


def _reduce_w(v, w, Q):
    """Same for values with unequal probabilities w (sum 1): (M,) -> (Q,)."""
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.concatenate(([0.0], np.cumsum(w)))
    c /= c[-1]
    e = np.arange(Q + 1) / Q
    ov = np.minimum(c[1:][None, :], e[1:][:, None]) - np.maximum(c[:-1][None, :], e[:-1][:, None])
    return (np.clip(ov, 0.0, None) @ v) * Q


class NetScen:
    def __init__(self, load_fc, pv_fc, mode="hybrid", table=None, load_table=None, dep=0.0, bias_tail=False):
        """table: (24, H, FINE) cross-fitted errors of (load - solar); load_table: (24, H, NLEV) of the load.
        dep: in "conv", share of probability given to the pairing "high load with low solar" (0 = independent;
        the rank correlation of the two errors is then -dep). bias_tail: add the mean cross-fitted error to the
        point forecast of the hours planned without scenarios."""
        assert mode in ("emp", "conv", "hybrid")
        self.load_fc, self.pv_fc, self.mode = load_fc, pv_fc, mode
        self.table, self.load_table = table, load_table
        self.dep, self.bias_tail = dep, bias_tail
        self._off = {}
        self._bias = table.mean(axis=2) if (bias_tail and table is not None) else None
        self._wdep = np.concatenate([np.full(NLEV * NLEV, (1.0 - dep) / (NLEV * NLEV)), np.full(NLEV, dep / NLEV)])

    def tail_bias(self, obs, K, H):
        """Mean error of the point forecast for hours K .. H-1 (zeros unless bias_tail)."""
        if self._bias is None or H <= K:
            return 0.0
        kk = np.arange(K, H)
        return self._bias[(obs["timestamp"].hour + kk) % 24, kk]

    def _combine(self, lq, sq, Q):
        """(NLEV,) load values and (NLEV,) solar values -> (Q,) net-load scenarios."""
        d = (lq[:, None] - sq[None, :]).ravel()
        if self.dep <= 0.0:
            return _reduce(d, Q)
        return _reduce_w(np.concatenate([d, lq - sq[::-1]]), self._wdep, Q)

    def offsets(self, Q):
        if Q not in self._off:
            t = self.table
            self._off[Q] = t.reshape(t.shape[0], t.shape[1], Q, FINE // Q).mean(axis=3)
        return self._off[Q]

    def predict(self, obs, hist, H):
        a = np.asarray(self.load_fc.predict_load(obs, hist, H), float)
        b = np.asarray(self.pv_fc.predict_pv(obs, hist, H), float)
        m = min(len(a), len(b))
        return a[:m] - b[:m]

    def h0_values(self, obs, hist):
        """All equally likely values of the hour-0 net load under this mode (for calibration checks)."""
        if self.mode == "emp":
            return self.predict(obs, hist, 1)[0] + self.table[obs["timestamp"].hour, 0]
        lq = np.asarray(self.load_fc.predict_load_quantiles(obs, hist, LEV), float)
        sq = np.asarray(self.pv_fc.predict_pv_quantiles(obs, hist, LEV, H=1), float)[:, 0]
        return (lq[:, None] - sq[None, :]).ravel()

    def predict_scen(self, obs, hist, K, Q):
        """(scen (K, Q), probs (Q,)) for the first K hours (hour 0 = the hour being decided)."""
        probs = np.full(Q, 1.0 / Q)
        hod = (obs["timestamp"].hour + np.arange(K)) % 24
        kk = np.arange(K)
        if self.mode == "emp":
            f = self.predict(obs, hist, K)
            return f[:, None] + self.offsets(Q)[hod[:len(f)], kk[:len(f)]], probs
        if self.mode == "hybrid":
            f = self.predict(obs, hist, K)
            scen = f[:, None] + self.offsets(Q)[hod[:len(f)], kk[:len(f)]]
            lq = np.asarray(self.load_fc.predict_load_quantiles(obs, hist, LEV), float)
            sq = np.asarray(self.pv_fc.predict_pv_quantiles(obs, hist, LEV, H=1), float)[:, 0]
            scen[0] = self._combine(lq, sq, Q)
            return scen, probs
        # conv: every hour = (load distribution) - (solar distribution), independent
        lp = np.asarray(self.load_fc.predict_load(obs, hist, K), float)
        lq = lp[:, None] + self.load_table[hod, kk] if K > 1 else np.empty((1, NLEV))
        lq[0] = self.load_fc.predict_load_quantiles(obs, hist, LEV)
        sq = np.asarray(self.pv_fc.predict_pv_quantiles(obs, hist, LEV, H=K), float).T        # (K, NLEV)
        if self.dep > 0.0:
            return np.array([self._combine(lq[k], sq[k], Q) for k in range(K)]), probs
        d = (lq[:, :, None] - sq[:, None, :]).reshape(K, NLEV * NLEV)
        return _reduce(d, Q), probs


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
class CombinedAgent:
    def __init__(self, w_fc, net, H=36, Q=9, scen_hours=12, spread=1.0, v_end=None, kappa=KAPPA,
                 truncate_end=False, end_ts=None):
        """w_fc: predict(obs, hist, H) -> signal. net: predict(obs, hist, H) -> net load point forecast and
        predict_scen(obs, hist, K, Q) -> (scen (K, Q), probs (Q,)).
        scen_hours: leading hours planned against scenarios (0 = deterministic plan, None = all).
        spread: width multiplier of the scenarios around their mean. v_end: value of a kWh left at the end of
        the horizon (signal units). truncate_end: when obs shows fewer than 3 prices the run is ending: plan
        only the hours that exist. end_ts: timestamp of the last hour of the run if it is known from the
        calendar (test: 2026-08-31 23:00); the plan then stops there instead of saving energy for later."""
        self.end_ts = pd.Timestamp(end_ts) if end_ts is not None else None
        self.w_fc, self.net = w_fc, net
        self.H, self.Q, self.scen_hours, self.spread = H, Q, scen_hours, spread
        self.v_end, self.kappa, self.truncate_end = v_end, kappa, truncate_end
        self.hist = []

    def act(self, obs):
        self.hist.append(make_rec(obs, self.kappa))
        soc = float(obs["soc_kwh"])
        w_known = (np.asarray(obs["co2_properes_24h"], float)
                   + self.kappa * np.asarray(obs["preus_properes_24h"], float))
        w = np.asarray(self.w_fc.predict(obs, self.hist, self.H), float).copy()
        n = np.asarray(self.net.predict(obs, self.hist, self.H), float)
        H = min(len(w), len(n))
        v_end = self.v_end
        if self.truncate_end and len(w_known) < 3:
            H = min(H, len(w_known))
            v_end = None
        if self.end_ts is not None:
            left = int((self.end_ts - obs["timestamp"]) / pd.Timedelta(hours=1)) + 1
            if 1 <= left < H:
                H = left
                v_end = None
        if H == 0:
            return 0.0
        w, n = w[:H], n[:H]
        k = min(len(w_known), H)
        w[:k] = w_known[:k]
        w = np.maximum(w, 1e-9)
        K = H if self.scen_hours is None else min(self.scen_hours, H)
        if K == 0:
            p = P.plan_det(w, n, soc, v_end=v_end)
        else:
            scen, probs = self.net.predict_scen(obs, self.hist, K, self.Q)
            scen = scen[:K]
            if self.spread != 1.0:
                m = (scen @ probs)[:, None]
                scen = m + self.spread * (scen - m)
            p = P.plan_stoch(w, scen, probs, soc, v_end=v_end, n_tail=n[K:H] + self.net.tail_bias(obs, K, H))
        return P.safe_clamp(p, soc)


def make_agent(fold_df=None, signal="fc", load="fc", solar="fc", scen="conv", H=36, Q=9, scen_hours=12,
               spread=1.0, v_end=None, truncate_end=True, blocks=None, tape=None, dep=0.0, bias_tail=False,
               end_ts=None):
    """The combined agent, fitted on the rows outside fold_df (None = every train row).
    signal: "fc" | "clim" | "perfect" (needs fold_df) or any object with predict(obs, hist, H)
    load:   "fc" | "nomodel" | "naive"         solar: "fc" | "naive"
    scen:   "hybrid" | "emp" | "conv"          (see the module docstring; a naive part has no quantiles of its
            own: in "conv" it gets cross-fitted errors by (hour, horizon), in "hybrid" the agent falls back to "emp")
    blocks: cross-fitting blocks used for the error tables (default: all available)
    truncate_end / end_ts: see CombinedAgent (end_ts for the real test: "2026-08-31 23:00")
    tape:   recorded forecasts of the run (evaluation only: same code path, no forecaster calls)."""
    if (load == "naive" or solar == "naive") and scen == "hybrid":
        scen = "emp"
    if tape is None:
        which = [LOAD_KEYS[load], PV_KEYS[solar]] + ([{"fc": "w_sig", "clim": "w_clim"}[signal]]
                                                    if signal in ("fc", "clim") else [])
        fcs = build_forecasters(fold_df, which=which)
        lf, pf = fcs[LOAD_KEYS[load]], fcs[PV_KEYS[solar]]
        wf = fcs.get("w_sig") or fcs.get("w_clim")
    else:
        lf, pf = TapeLoad(tape, LOAD_KEYS[load]), TapePV(tape, PV_KEYS[solar])
        wf = TapeW(tape, {"fc": "w_sig", "clim": "w_clim"}[signal]) if signal in ("fc", "clim") else None
    if signal == "perfect":
        wf = PerfectW(fold_df)
    elif not isinstance(signal, str):
        wf = signal
    table, load_table, pv_table = error_tables_for(fold_df, load, solar, blocks)
    if scen == "conv" and load == "naive":
        lf = EmpLoadQ(lf, load_table)
    if scen == "conv" and solar == "naive":
        pf = EmpPVQ(pf, pv_table)
    net = NetScen(lf, pf, scen, table, load_table, dep=dep, bias_tail=bias_tail)
    return CombinedAgent(wf, net, H=H, Q=Q, scen_hours=scen_hours, spread=spread, v_end=v_end,
                         truncate_end=truncate_end, end_ts=end_ts)
