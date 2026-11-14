# Usage: .venv/Scripts/python.exe experiments/27_combined/sweep.py <set> [--trace]
"""Closed-loop scores of named sets of configurations on folds A, B, C, from the recorded forecast tapes
(build_cache.py first). Writes results_<set>.csv; --trace also saves cache/trace_<set>_<i>_<fold>.pkl.
Sets: first, ladder, scen, sens, end, live."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined as cb  # noqa: E402
from src.hard import get_folds, simulate, train_complement  # noqa: E402
from src.hard_mpc import ClimW, KAPPA, PerfectN, PerfectW  # noqa: E402
import planner as P  # noqa: E402

HERE = Path(__file__).resolve().parent
_T = {}


def tape(fold):
    if fold not in _T:
        z = np.load(cb.CACHE / f"tape_{fold}.npz")
        _T[fold] = {k: z[k] for k in z.files}
    return _T[fold]


class QN450(P.QuantileN):
    FINE = cb.FINE


class TrueEnd:
    """Diagnostic: a signal forecaster that stops at the true end of the run (as PerfectW does)."""
    def __init__(self, fc, n):
        self.fc, self.n = fc, n

    def predict(self, obs, hist, H):
        return np.asarray(self.fc.predict(obs, hist, H), float)[:max(self.n - len(hist) + 1, 0)]


class OracleNet:
    """Diagnostic: the true net load (zero-width scenarios) for the first m hours, the wrapped forecaster beyond."""
    def __init__(self, net, df, m):
        self.net, self.m = net, m
        self.n = (df["load_kw"] - df["pv_kw"]).to_numpy(float)

    def _truth(self, hist, H):
        i = len(hist) - 1
        return self.n[i:i + min(self.m, H)]

    def predict(self, obs, hist, H):
        f = np.asarray(self.net.predict(obs, hist, H), float).copy()
        t = self._truth(hist, len(f))
        f[:len(t)] = t
        return f

    def tail_bias(self, obs, K, H):
        return 0.0

    def predict_scen(self, obs, hist, K, Q):
        scen, probs = self.net.predict_scen(obs, hist, K, Q)
        scen = np.array(scen, float)
        t = self._truth(hist, len(scen))
        scen[:len(t)] = t[:, None]
        return scen, probs


class OracleW:
    """Diagnostic: the true signal for the first m hours, the wrapped forecaster beyond."""
    def __init__(self, fc, df, m):
        self.fc, self.pw, self.m = fc, PerfectW(df), m

    def predict(self, obs, hist, H):
        w = np.asarray(self.fc.predict(obs, hist, H), float).copy()
        t = self.pw.predict(obs, hist, H)
        k = min(self.m, len(t), len(w))
        w[:k] = t[:k]
        return w


def true_parts(tp, df):
    """Tape entries for a perfect load / solar forecaster (run-indexed truth, zero-width quantiles)."""
    tp = dict(tp)
    n = len(df)
    idx = np.minimum(np.arange(n)[:, None] + np.arange(cb.HT)[None, :], n - 1)
    tp["load_true"] = tp["load"][idx]
    tp["pv_true"] = tp["pv"][idx]
    tp["load_true_q"] = np.repeat(tp["load"][:, None], cb.NLEV, axis=1)
    tp["pv_true_q"] = np.repeat(tp["pv_true"][:, None, :], cb.NLEV, axis=1)
    return tp


def base(**kw):
    d = dict(signal="clim", load="fc", solar="fc", scen="hybrid", H=36, Q=9, scen_hours=12, spread=1.0,
             truncate_end=False)
    d.update(kw)
    return d


def configs(which):
    out = []
    if which in ("first", "ladder"):
        out.append(dict(label="REF planner's own agent: ClimW + QuantileN(ProfileN), in-sample errors", kind="ref"))
        out.append(dict(label="ceiling: perfect signal + perfect net load, deterministic", kind="perfect"))
        for sig in ("clim", "perfect", "fc"):
            for ld, so in (("naive", "naive"), ("fc", "naive"), ("naive", "fc")):
                out.append(base(label=f"{sig} | load {ld} + solar {so} | emp", signal=sig, load=ld, solar=so, scen="emp"))
            for sc, K in (("emp", 12), ("hybrid", 12), ("conv", 12), ("conv", 1), ("emp", 1)):
                out.append(base(label=f"{sig} | load fc + solar fc | {sc}, K={K}", signal=sig, scen=sc, scen_hours=K))
            out.append(base(label=f"{sig} | load fc + solar fc | deterministic", signal=sig, scen_hours=0))
            for sc in ("emp", "hybrid", "conv"):
                out.append(base(label=f"{sig} | load nomodel + solar fc | {sc}, K=12", signal=sig, load="nomodel", scen=sc))
    if which == "ladder2":
        for sig in ("clim", "perfect", "fc"):
            for ld, so in (("naive", "naive"), ("fc", "naive"), ("naive", "fc"), ("fc", "fc"), ("nomodel", "fc"), ("nomodel", "naive")):
                out.append(base(label=f"{sig} | load {ld} + solar {so} | conv", signal=sig, load=ld, solar=so, scen="conv"))
    b = dict(signal="fc", scen="conv")
    if which == "sens":
        out.append(base(label="BEST: fc signal, LoadFC + SolarFC, conv, Q=9, K=12, H=36", **b))
        for Q in (5, 15, 25):
            out.append(base(label=f"Q={Q}", Q=Q, **b))
        for K in (1, 3, 6, 24, None):
            out.append(base(label=f"scenario hours K={K if K else 'all (36)'}", scen_hours=K, **b))
        out.append(base(label="H=24", H=24, **b))
        out.append(base(label="H=24, K=all (24)", H=24, scen_hours=None, **b))
        for sp in (0.5, 0.75, 0.9, 1.1, 1.25, 1.5):
            out.append(base(label=f"width x{sp:.2f}", spread=sp, **b))
        for ve in (0.25, 0.5, 0.75, 1.0):
            out.append(base(label=f"v_end = {ve:.2f} x mean(w) x eta, H=36", v_end_rel=ve, **b))
            out.append(base(label=f"v_end = {ve:.2f} x mean(w) x eta, H=24", v_end_rel=ve, H=24, **b))
        for sc in ("hybrid", "emp"):
            for sp in (0.75, 1.0, 1.25):
                out.append(base(label=f"{sc}, width x{sp:.2f}", signal="fc", scen=sc, spread=sp))
        for sp in (0.75, 1.0, 1.25):
            out.append(base(label=f"perfect signal, conv, width x{sp:.2f}", signal="perfect", scen="conv", spread=sp))
            out.append(base(label=f"clim signal, conv, width x{sp:.2f}", signal="clim", scen="conv", spread=sp))
    if which == "end":
        for sig in ("fc", "clim"):
            out.append(base(label=f"{sig} | horizon never truncated", signal=sig, scen="conv"))
            out.append(base(label=f"{sig} | truncated when obs shows < 3 prices (legal)", signal=sig, scen="conv", truncate_end=True))
            out.append(base(label=f"{sig} | truncated at the true end of the run (diagnostic)", signal=sig, scen="conv", true_end=True))
            out.append(base(label=f"{sig} | truncated at the calendar end of the fold (last fold day of Aug 2026, 23:00) + < 3 prices",
                            signal=sig, scen="conv", truncate_end=True, cal_end=True))
        out.append(base(label="fc | no-LightGBM load, calendar end + < 3 prices", signal="fc", scen="conv", load="nomodel",
                        truncate_end=True, cal_end=True))
        out.append(base(label="fc | no-LightGBM load, < 3 prices", signal="fc", scen="conv", load="nomodel", truncate_end=True))
        out.append(base(label="fc | hybrid, calendar end + < 3 prices", signal="fc", scen="hybrid", truncate_end=True, cal_end=True))
        out.append(base(label="fc | hybrid, < 3 prices", signal="fc", scen="hybrid", truncate_end=True))
    if which == "tried":
        out.append(base(label="BEST (conv, independent)", **b))
        for dep in (0.15, 0.3, 0.5):
            out.append(base(label=f"conv, negative dependence {dep:.2f}", dep=dep, **b))
        out.append(base(label="conv + mean-error correction of hours 12-35", bias_tail=True, **b))
        out.append(base(label="hybrid + mean-error correction of hours 12-35", signal="fc", scen="hybrid", bias_tail=True))
        out.append(base(label="conv, nomodel load", signal="fc", scen="conv", load="nomodel"))
        out.append(base(label="conv, nomodel load, negative dependence 0.30", signal="fc", scen="conv", load="nomodel", dep=0.3))
    if which == "tuned":
        e = dict(signal="fc", scen="conv", truncate_end=True, cal_end=True)
        for Q in (9, 15, 25, 45, 75):
            for K in (6, 12):
                out.append(base(label=f"Q={Q}, K={K} (calendar end + < 3 prices)", Q=Q, scen_hours=K, **e))
        out.append(base(label="Q=25, K=6, width x0.9", Q=25, scen_hours=6, spread=0.9, **e))
        out.append(base(label="Q=25, K=6, H=24", Q=25, scen_hours=6, H=24, **e))
        out.append(base(label="Q=25, K=6, no-LightGBM load", Q=25, scen_hours=6, load="nomodel", **e))
        out.append(base(label="Q=25, K=6, ClimW signal", Q=25, scen_hours=6, **dict(e, signal="clim")))
    if which == "gap":
        out.append(dict(label="ceiling: perfect signal + perfect net load, deterministic", kind="perfect"))
        for sig in ("fc", "perfect"):
            out.append(base(label=f"{sig} | best net-load side (LoadFC + SolarFC, conv)", signal=sig, scen="conv"))
            for m in (1, 2, 3, 6, 12, 36):
                out.append(base(label=f"{sig} | + true net load for hours 0..{m - 1}", signal=sig, scen="conv", oracle_net=m))
            out.append(base(label=f"{sig} | true load (all hours) + SolarFC", signal=sig, scen="conv", parts=("perfect", "fc")))
            out.append(base(label=f"{sig} | LoadFC + true solar (all hours)", signal=sig, scen="conv", parts=("fc", "perfect")))
        for m in (6, 12, 24):
            out.append(base(label=f"fc | best net-load side + true signal for hours 0..{m - 1}", signal="fc", scen="conv", oracle_w=m))
            out.append(base(label=f"fc | true net load + true signal for hours 0..{m - 1}", signal="fc", scen="conv", oracle_w=m, oracle_net=36))
        out.append(base(label="fc | true net load, all hours (signal forecast only)", signal="fc", scen="conv", oracle_net=36))
        out.append(base(label="fc | best net-load side, horizon cut at the true end of the run", signal="fc", scen="conv", true_end=True))
        out.append(base(label="fc | true net load, all hours, horizon cut at the true end of the run", signal="fc", scen="conv",
                        oracle_net=36, true_end=True))
        out.append(base(label="fc | + true net load for hours 0..0, horizon cut at the true end of the run", signal="fc", scen="conv",
                        oracle_net=1, true_end=True))
    return out


def run(i, cfg, fold, which, trace):
    df = get_folds()[fold]
    kw = dict(cfg)
    label = kw.pop("label")
    kind = kw.pop("kind", "combined")
    t0 = time.perf_counter()
    if kind == "ref":
        fit = train_complement(df)
        ag = P.MPCStoch(ClimW(fit), P.QuantileN(fit, H=36), H=36, Q=9, scen_hours=12)
    elif kind == "perfect":
        ag = P.MPCStoch(PerfectW(df), PerfectN(df), H=36, scen_hours=0)
    else:
        ve = kw.pop("v_end_rel", None)
        true_end = kw.pop("true_end", False)
        if ve:
            fit = train_complement(df)
            kw["v_end"] = ve * float((fit["co2_kg_per_kwh"] + KAPPA * fit["precio_eur_per_kwh"]).mean()) * P.ETA
        om, ow, parts = kw.pop("oracle_net", None), kw.pop("oracle_w", None), kw.pop("parts", None)
        if kw.pop("cal_end", False):
            kw["end_ts"] = pd.Timestamp(2026, 8, cb.FOLD_DAYS[fold][1], 23)
        ag = cb.make_agent(df, tape=tape(fold), **kw)
        if parts:
            tp = true_parts(tape(fold), df)
            lk = "load_true" if parts[0] == "perfect" else "load_fc"
            pk = "pv_true" if parts[1] == "perfect" else "pv_fc"
            lt = ag.net.load_table if parts[0] != "perfect" else np.zeros_like(ag.net.load_table)
            ag.net = cb.NetScen(cb.TapeLoad(tp, lk), cb.TapePV(tp, pk), "conv", None, lt)
        if om:
            ag.net = OracleNet(ag.net, df, om)
        if ow:
            ag.w_fc = OracleW(ag.w_fc, df, ow)
        if true_end:
            ag.w_fc = TrueEnd(ag.w_fc, len(df))
    r = simulate(ag, df, return_trace=True)
    tr = r.pop("trace")
    if trace:
        tr.to_pickle(cb.CACHE / f"trace_{which}_{i}_{fold}.pkl")
    return {"i": i, "config": label, "fold": fold, "score": r["score"], "co2_pct": r["co2_pct"],
            "cost_pct": r["cost_pct"], "infractions": int(r["infractions"]),
            "spill_mwh": float(tr["spill_kw"].sum()) / 1000.0, "seconds": time.perf_counter() - t0}


def main(which, trace=False):
    cfgs = configs(which)
    rows = Parallel(n_jobs=4)(delayed(run)(i, c, f, which, trace) for i, c in enumerate(cfgs) for f in "ABC")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"results_{which}.csv", index=False)
    piv = out.pivot(index="config", columns="fold", values="score")
    piv["mean"] = piv.mean(axis=1)
    piv["spill_MWh"] = out.groupby("config")["spill_mwh"].mean()
    piv["infr"] = out.groupby("config")["infractions"].sum()
    piv["sec"] = out.groupby("config")["seconds"].mean()
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 500)
    print(piv.loc[[c["label"] for c in cfgs]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], "--trace" in sys.argv)
