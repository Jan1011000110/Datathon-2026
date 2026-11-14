"""
experiments/29_stress_hard/shipped_check.py
The ZIPs exactly as shipped (state fitted on every train row), and h2's fallback paths.
    shipped_check.py        -> out/shipped_check.csv
Scores of the shipped state on train folds are in-sample (not an estimate of the test score); what matters
here: exceptions, infractions, the fallback counter, the load mode and the run time.
"""
import shutil
import sys
import time

import numpy as np
import pandas as pd

import lib as L
import suite as S

rows = []


def go(name, agent, df, obs_fn=None, kind="h2"):
    r = L.run(agent, df, obs_fn)
    rows.append(dict(agent=kind, case=name, n=r["n"], score=r["score"], infractions=r["infractions"],
                     exceptions=r["exceptions"], bad_returns=r["bad_returns"], fallbacks=r["fallbacks"],
                     load_mode=r["load_mode"], secs=r["secs"], max_step_ms=r["max_step_ms"],
                     build_error=str(getattr(agent, "build_error", ""))[:100], last_error=r["last_error"]))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    return r


if __name__ == "__main__":
    folds = L.clean_folds()
    B = folds["B"]
    d12 = L.add_week(B, S.wk_of(B, 10), 28)
    test12 = L.dst(L.relabel(d12, "last7"))                    # 12 weeks, test calendar, both clock changes
    print("test-calendar run:", test12.index[0], "..", test12.index[-1], len(test12), "hours", flush=True)

    # ---- as shipped
    for kind in ("h1", "h2"):
        for fname in ("B", "C"):
            go(f"shipped, fold {fname} (real dates)", L.load_agent(kind), folds[fname], kind=kind)
        t0 = time.perf_counter()
        a = L.load_agent(kind)
        t_init = time.perf_counter() - t0
        r = go("shipped, 12 weeks on the test calendar + clock changes", a, test12, kind=kind)
        rows[-1]["init_secs"] = t_init
        go("shipped, same run, text timestamps, no cap/power keys", L.load_agent(kind), test12,
           L.chain(L.o_ts("str"), L.o_drop("cap_max_kwh", "power_max_kw")), kind=kind)

    # ---- h2 fallback paths (fold-out state of fold B, so the scores compare with the suite)
    st = L.state_path("h2", "B")
    tmp = L.HERE / "states" / "tmp"
    tmp.mkdir(exist_ok=True)
    go("fold-out reference (fold B)", L.load_agent("h2", st), B)
    go("state file missing -> embedded h1", L.load_agent("h2", tmp / "does_not_exist.npz"), B)
    z = dict(np.load(st, allow_pickle=False))
    bad = dict(z)
    bad["tree_leaf"] = bad["tree_leaf"] + 5.0                  # the stored trees no longer pass their self-check
    np.savez_compressed(tmp / "bad_trees.npz", **bad)
    go("load trees fail their self-check -> rules mode", L.load_agent("h2", tmp / "bad_trees.npz"), B)
    bad = {k: v for k, v in z.items() if not k.startswith("tree_")}
    np.savez_compressed(tmp / "no_trees.npz", **bad)
    go("no trees in the state -> rules mode", L.load_agent("h2", tmp / "no_trees.npz"), B)
    bad = {k: v for k, v in z.items() if k != "sol_env_tab"}
    np.savez_compressed(tmp / "no_solar.npz", **bad)
    go("solar table missing from the state -> h1", L.load_agent("h2", tmp / "no_solar.npz"), B)
    raw = open(st, "rb").read()
    open(tmp / "truncated.npz", "wb").write(raw[:len(raw) // 2])
    go("state file truncated -> h1", L.load_agent("h2", tmp / "truncated.npz"), B)
    bad = dict(z)
    bad["sig_" + [k for k in z if k.startswith("sig_")][0][4:]] = np.zeros(3)
    np.savez_compressed(tmp / "bad_signal.npz", **bad)
    go("signal array of the wrong shape", L.load_agent("h2", tmp / "bad_signal.npz"), B)

    # pandas not importable on the server: the module is loaded again with pandas hidden
    d = L.HERE / "fix" / "h2_nopandas"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(L.HERE / "pkg" / "h2" / "agent.py", d / "agent.py")
    keep = sys.modules.get("pandas")
    sys.modules["pandas"] = None
    try:
        a = L.load_agent("h2", st, pkg=d)
    finally:
        sys.modules["pandas"] = keep
    go("pandas not importable -> h1", a, B)

    # mid-run failures: the readings dict turns into None for 100 hours, then comes back
    def blackout(obs, t, ctx):
        if 500 <= t < 600:
            obs["noves_dades_5min"] = None
            obs["timestamp"] = None
        return obs
    go("100 h without timestamp and readings mid-run, then normal", L.load_agent("h2", st), B, blackout)
    pd.DataFrame(rows).to_csv(L.HERE / "out" / "shipped_check.csv", index=False)
    shutil.rmtree(tmp, ignore_errors=True)
