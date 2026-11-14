# Usage: .venv/Scripts/python.exe experiments/23_signal/round3.py [strict] [names...]
"""Round 3: does fitting on older REE history (<= 2025-08-31) help? Closed loop only, per fold:
  pn   deterministic plan, net load known
  h3   the shipped stack (experiments/27_combined/combined3.make_agent3, truncate_end only) with the
       candidate's forecasts in place of the round-2 signal (make_agent3 does not take a forecaster
       object, so the agent is built as shipped and its w_fc is replaced by the candidate's tape)
and a paired comparison by week against the round-2 signal. Candidates that need the history are skipped
when data/external/ree_history/ree_history_hourly.csv does not exist. `strict`: fold C fitted on days
1-14 only for the competition span (test geometry). Writes round3.csv and round3_weeks.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402
from common import KAPPA, signal_fit_table, truth  # noqa: E402
from fetch_ree_history import OUT as HIST_DIR, load_history  # noqa: E402
from signal_fc2 import SignalForecaster2  # noqa: E402
from signal_fc3 import SignalFC3, SignalForecaster3  # noqa: E402
from src.hard import score_weights, simulate  # noqa: E402
from src.hard_mpc import PerfectN  # noqa: E402

HERE = Path(__file__).resolve().parent
HAVE_HIST = (HIST_DIR / "ree_history_hourly.csv").exists()
REF = "round 2 (h3 signal)"


def hist(start=None):
    h = load_history()
    return h if start is None else h[h.index >= start]


# name: (needs history, builder(fit))
CANDS = {
    REF: (False, lambda fit: SignalForecaster2(fit, KAPPA)),
    "round 3 code, no history": (False, lambda fit: SignalForecaster3(fit, KAPPA)),
    "history 2 y, weight 1": (True, lambda fit: SignalForecaster3(fit, KAPPA, hist_df=hist())),
    "history 2 y, weight 0.5": (True, lambda fit: SignalForecaster3(fit, KAPPA, hist_df=hist(), hist_weight=0.5)),
    "history 2 y, weight 0.25": (True, lambda fit: SignalForecaster3(fit, KAPPA, hist_df=hist(), hist_weight=0.25)),
    "history 1 y, weight 1": (True, lambda fit: SignalForecaster3(fit, KAPPA, hist_df=hist("2024-09-01"))),
    "history in the store only (weight 0)": (True, lambda fit: SignalForecaster3(fit, KAPPA, hist_df=hist(), hist_weight=0.0)),
    "history 2 y, night level from the per-hour model alone": (True, lambda fit: SignalForecaster3(fit, KAPPA, hist_df=hist(), night_w=1.0)),
    "history 2 y, per-hour model everywhere": (True, lambda fit: SignalFC3(fit, KAPPA, hist_df=hist(), variant="AH12", use_lw=False,
                                                                           ridge=0.03, interp=True)),
}
_T2 = {}


def h3_agent(f, W):
    import combined3 as c3
    df = fast.fold(f)
    if f not in _T2:
        z = np.load(fast.cb.CACHE / f"tape_{f}.npz")                  # load, solar and round-1 signal
        _T2[f] = {k: z[k] for k in z.files}
        _T2[f]["w_sig2"] = np.load(fast.cb.CACHE / f"tape2_{f}.npz")["w_sig2"]     # the shipped round-2 signal
    ag = c3.make_agent3(df, signal="fc2", tape=_T2[f])
    if W is not None:
        ag.w_fc = fast.Tape(W)
    return ag


def week_costs(f, agent):
    """Score of the run and the cost of every week in score points (lower = better)."""
    df = fast.fold(f)
    r = simulate(agent, df, return_trace=True)
    c = r["trace"]["grid_kw"].to_numpy() * score_weights(df) * 100.0
    wk = pd.Series(c, index=df.index).groupby([df.index.year, df.index.month]).sum()
    return r["score"], wk


def run(name, f, strict):
    need, build = CANDS[name]
    df = fast.fold(f)
    fit = signal_fit_table(f)
    if strict:
        fit = fit[fit.index.day <= 14]
    t0 = time.perf_counter()
    fc = build(fit)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    Cc, Pp, W = fast.replay_full(fc, df, H=fast.HT)
    ms = (time.perf_counter() - t0) / len(df) * 1e3
    np.save(fast.CACHE / f"r3_{abs(hash(name)) % 10**8}_{f}{'_s' if strict else ''}.npy", W)
    T = truth(df, H=fast.HT)["w"]
    pn, wk_pn = week_costs(f, fast.P.MPCStoch(fast.Tape(W), PerfectN(df), H=36, scen_hours=0))
    h3, wk_h3 = week_costs(f, h3_agent(f, W))
    lvl = np.abs(np.nanmean((W - T)[:, 3:9], axis=1))
    row = {"cand": name, "fold": f, "strict": strict, "pn": pn, "h3": h3, "fit_s": t_fit, "ms_per_call": ms,
           "mae_3_11": np.nanmean(np.abs(W - T)[:, 3:12]) * 1e3, "level_mae": np.nanmean(lvl) * 1e3}
    weeks = pd.DataFrame({"cand": name, "fold": f, "strict": strict, "year": [k[0] for k in wk_pn.index],
                          "month": [k[1] for k in wk_pn.index], "cost_pn": wk_pn.values, "cost_h3": wk_h3.values})
    return row, weeks, W


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "strict"]
    strict = "strict" in sys.argv
    names = args or [n for n, (need, _) in CANDS.items() if HAVE_HIST or not need]
    skipped = [n for n in names if CANDS[n][0] and not HAVE_HIST]
    names = [n for n in names if n not in skipped]
    if REF not in names:
        names = [REF] + names
    print(f"history available: {HAVE_HIST}" + (f"; skipped (need it): {skipped}" if skipped else ""), flush=True)
    folds = ["C"] if strict else list("ABC")
    res = Parallel(n_jobs=3)(delayed(run)(n, f, strict) for n in names for f in folds)
    out = pd.DataFrame([r[0] for r in res])
    weeks = pd.concat([r[1] for r in res], ignore_index=True)
    tag = "_strict" if strict else ""
    out.to_csv(HERE / f"round3{tag}.csv", index=False)
    weeks.to_csv(HERE / f"round3_weeks{tag}.csv", index=False)
    tapes = {(r[0]["cand"], r[0]["fold"]): r[2] for r in res}
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    for col in ("pn", "h3", "level_mae", "mae_3_11", "ms_per_call", "fit_s"):
        piv = out.pivot(index="cand", columns="fold", values=col)
        piv["mean"] = piv.mean(axis=1)
        print(f"== {col}{' (strict C)' if strict else ''} =="); print(piv.loc[names].round(3).to_string())
    # h3 reference as shipped (the recorded round-2 tape) and identity checks
    for f in folds:
        if not strict:
            s, _ = week_costs(f, h3_agent(f, None))
            d2 = np.abs(tapes[(REF, f)] - _T2[f]["w_sig2"][:, :fast.HT]).max()
            line = f"fold {f}: h3 as shipped {s:.3f}; my replay of round 2 differs from the shipped tape by {d2:.2e}"
            if "round 3 code, no history" in names:
                line += f"; round-3 code without history differs from round 2 by {np.abs(tapes[('round 3 code, no history', f)] - tapes[(REF, f)]).max():.2e}"
            print(line)
    # paired by week against the round-2 signal: positive = the candidate is better
    ref = weeks[weeks.cand == REF].set_index(["fold", "year", "month"])
    rows = []
    for n in names:
        if n == REF:
            continue
        w = weeks[weeks.cand == n].set_index(["fold", "year", "month"])
        for col in ("cost_pn", "cost_h3"):
            d = (ref[col] - w[col]).dropna()
            rows.append({"cand": n, "harness": col[5:], "weeks": len(d), "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                         "mean_gain_per_week": d.mean(), "sd": d.std(), "t": d.mean() / (d.std() / np.sqrt(len(d))) if d.std() > 0 else 0.0})
    if rows:
        print("== paired by week against the round-2 signal (points of fold score per week) ==")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
