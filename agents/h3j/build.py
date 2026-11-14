"""
agents/h3j/build.py  --  fit the combined agent and write its ONE state file (plain numpy arrays, no pickle).

Fitting is done by the specialists' own code (experiments/27_combined/combined.py and what it imports);
this script only turns the fitted objects into arrays. Every setting of the agent is in CONFIG below.

Validation (state fitted outside a fold; the returned objects let a test build the harness agent too):
    arrays, fcs, tabs = build_state(fold_df)
    write_state(arrays, "some/dir/h3_state.npz")

The real thing (fit on every train row, regenerate agent.py, package submissions/h3j/):
    .venv/Scripts/python.exe agents/h3j/build.py
"""
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STATE_NAME = "h3j_state.npz"

# ---- every setting of the agent (arguments of experiments/27_combined/combined.make_agent) ----------
CONFIG = dict(load="fc", solar="fc", scen="conv", H=36, Q=9, scen_hours=12, spread=1.0,
              v_end=None, truncate_end=True, dep=0.0, bias_tail=False,
              end_ts=["2026-07-31 23:00", "2026-08-31 23:00"],     # calendar ends of the run, one date or a list: the plan stops
                                                                   # at the next one ahead (None = off; "2026-08-31 23:00" alone = h3)
              # ---- h3 (arguments of experiments/27_combined/combined3.make_agent3)
              signal="fc2", planner="paths", lam=0.5, ar=1.0, K=20, m=12, J=64, paths_from="round1",
              # ---- what happens at the end of each test week (experiments/30_boundaries)
              week_cut=False,                    # True: the plan stops at the last hour of every week ("each week is its own run")
              beyond_scale=1.0,                 # factor on the signal forecast for the hours after the current week (the cut probe: 1.2)
              week_rule="month_end")            # the week ends with the calendar month (test); [lo, hi] = fold days (validation)
H1_FILL = "neighbours"       # tables of the embedded h1 fallback: November from October / December


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def combined():
    for p in (ROOT, ROOT / "experiments" / "27_combined"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import combined as cb
    return cb


def combined3():
    combined()
    import combined3 as c3
    return c3


def _u8(text):
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def _kv(arrays, name, d, vtype=float):
    arrays[name + "_k"] = np.fromiter(d.keys(), np.int64, len(d))
    arrays[name + "_v"] = np.fromiter(d.values(), vtype, len(d))


def parse_lgb_model(text):
    """LightGBM model text -> padded arrays {feat, thr, dtype, left, right (T, M), leaf (T, L), nint (T,)}."""
    head, *blocks = text.split("\nTree=")
    if "average_output" in head or not blocks:
        raise ValueError("unexpected LightGBM model")
    trees = []
    for b in blocks:
        b = b.split("\nend of trees")[0]
        kv = dict(line.split("=", 1) for line in b.split("\n") if "=" in line)
        if int(kv.get("num_cat", "0")) != 0 or kv.get("is_linear", "0").strip() != "0":
            raise ValueError("categorical or linear trees are not supported")
        nl = int(kv["num_leaves"])
        leaf = [float(x) for x in kv["leaf_value"].split()]
        ints = (lambda k: [int(x) for x in kv[k].split()]) if nl > 1 else (lambda k: [])
        thr = [float(x) for x in kv["threshold"].split()] if nl > 1 else []
        t = dict(feat=ints("split_feature"), thr=thr, dtype=ints("decision_type"), left=ints("left_child"),
                 right=ints("right_child"), leaf=leaf)
        if len(leaf) != nl or any(len(t[k]) != nl - 1 for k in ("feat", "thr", "dtype", "left", "right")):
            raise ValueError("tree with inconsistent sizes")
        trees.append(t)
    T, M, L = len(trees), max(len(t["feat"]) for t in trees), max(len(t["leaf"]) for t in trees)
    M = max(M, 1)
    out = {"feat": np.zeros((T, M), np.int64), "thr": np.zeros((T, M)), "dtype": np.zeros((T, M), np.int64),
           "left": np.full((T, M), -1, np.int64), "right": np.full((T, M), -1, np.int64), "leaf": np.zeros((T, L)),
           "nint": np.array([len(t["feat"]) for t in trees], np.int64)}
    for i, t in enumerate(trees):
        for k in ("feat", "thr", "dtype", "left", "right"):
            out[k][i, :len(t[k])] = t[k]
        out["leaf"][i, :len(t["leaf"])] = t["leaf"]
    return out


def h1_tables(fold_df=None):
    """Tables of the embedded h1 agent, fitted outside fold_df (None = every official train row)."""
    from src.hard import load_hard, train_complement
    h1b = _load("h1_build_for_h2", ROOT / "agents" / "h1" / "build.py")
    return h1b.fit_tables(load_hard() if fold_df is None else train_complement(fold_df), fill_missing=H1_FILL)


def build_state(fold_df=None, config=None, verbose=False):
    """Returns (arrays, fcs, tabs): the state as {name: ndarray}, the fitted forecasters
    {"load_fc", "load_nm", "pv_fc", "w_sig"} and the cross-fitted error tables {"fc": (net, load, pv), "nm": ...}
    (combined.error_tables_for: read from experiments/27_combined/cache/, rebuilt by build_cache.py)."""
    cfg = dict(CONFIG if config is None else config)
    if (cfg["signal"], cfg["planner"], cfg["load"], cfg["solar"]) != ("fc2", "paths", "fc", "fc"):
        raise ValueError("h3 packages round-2 signal + path planner + the load and solar specialists only")
    cb = combined()
    from src.hard import METEO_COLS
    rt = _load("rt_combined_check", HERE / "rt_combined.py") if False else None      # (constants checked below)
    assert abs(cb.KAPPA - (0.3 / 0.7) * 0.634) < 1e-15 and list(METEO_COLS) == ["irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms"]
    fcs = cb.build_forecasters(fold_df, which=["load_fc", "load_nm", "pv_fc", "w_sig"])
    tabs = {"fc": cb.error_tables_for(fold_df, "fc", "fc"), "nm": cb.error_tables_for(fold_df, "nomodel", "fc")}
    A = {}
    info = {}
    cfg_rt = {k: cfg[k] for k in ("scen", "H", "Q", "scen_hours", "spread", "v_end", "truncate_end", "dep", "bias_tail", "end_ts",
                                  "lam", "ar", "K", "m", "J", "beyond_scale", "week_rule", "week_cut")}
    cfg_rt["kappa"] = float(cb.KAPPA)
    A["config"] = _u8(json.dumps(cfg_rt))

    # ---- load: LoadFC with the model ("fc") and rules-only ("nm"); same fit rows, different residuals
    lf, ln = fcs["load_fc"], fcs["load_nm"]
    par = dict(phi=lf.phi, rho=lf.rho, kappa=lf.kappa, mix=lf.mix, n_ref=list(lf.n_ref), cols=list(lf.cols),
               base_mix=lf.base_mix, hol_blend=lf.hol_blend)
    par_n = dict(phi=ln.phi, rho=ln.rho, kappa=ln.kappa, mix=ln.mix, n_ref=list(ln.n_ref), cols=list(ln.cols),
                 base_mix=ln.base_mix, hol_blend=ln.hol_blend)
    assert par == par_n and lf.base == ln.base and lf.tbase == ln.tbase and lf.prof == ln.prof
    assert lf.model is not None and ln.model is None
    A["load_params"] = _u8(json.dumps(par))
    _kv(A, "load_base", lf.base)
    _kv(A, "load_tbase", lf.tbase)
    A["load_prof"] = np.asarray(lf.prof, float)
    for tag, fc in (("fc", lf), ("nm", ln)):
        seq = [np.asarray(fc.res[(h, we)], float) for h in range(24) for we in (0, 1)]
        A[f"load_res_{tag}"] = np.concatenate(seq)
        A[f"load_res_{tag}_len"] = np.array([len(s) for s in seq], np.int64)
        A[f"load_res_cold_{tag}"] = np.asarray(fc.res_cold, float)
    # the LightGBM model as arrays (evaluated by rt_trees.TreeModel) + rows to re-check it at load time
    trees = parse_lgb_model(lf.model.model_to_string())
    X = lf._Xall[:, lf.ci]
    want = lf.model.predict(X, num_threads=1)
    rt_trees = _load("rt_trees_for_build", HERE / "rt_trees.py")
    got = rt_trees.TreeModel(*(trees[k] for k in ("feat", "thr", "dtype", "left", "right", "leaf", "nint"))).predict(X)
    err = float(np.abs(got - want).max())
    if not err <= 1e-9:
        raise ValueError(f"numpy trees differ from LightGBM by up to {err} kW on the training rows")
    for k, v in trees.items():
        A["tree_" + k] = v
    pick = np.unique(np.concatenate([np.linspace(0, len(X) - 1, 40).astype(int),
                                     np.flatnonzero(np.isnan(X).any(axis=1))[:20]]))
    A["tree_check_x"], A["tree_check_y"] = X[pick], want[pick]
    info["trees_max_abs_diff_vs_lightgbm_kw"] = err
    info["trees_rows_checked"] = int(len(X))
    info["trees_identical_rows"] = int((got == want).sum())

    # ---- solar
    sf = fcs["pv_fc"]
    A["sol_meta"] = _u8(json.dumps(dict(Hfit=int(sf.Hfit), fit_q=bool(sf.fit_q), use_weather=bool(sf.use_weather),
                                        outage_rule=bool(sf.outage_rule), q_scale=float(sf.q_scale),
                                        c_irr=float(sf.c_irr), k_clim=float(sf.k_clim))))
    _kv(A, "sol_pv", sf.pv_fit)
    _kv(A, "sol_dT", sf.dT_fit)
    _kv(A, "sol_ta", sf.ta_fit)
    A["sol_env_tab"] = np.asarray(sf.env_tab, float)
    A["sol_c1"], A["sol_c2"], A["sol_mu"] = (np.asarray(sf.par[k], float) for k in ("c1", "c2", "mu"))
    if sf.qtab is not None:
        A["sol_qtab"] = np.asarray(sf.qtab, float)

    # ---- signal: its own packaging format (to_npz), read back as arrays
    buf = io.BytesIO()
    fcs["w_sig"].to_npz(buf)
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as f:
        for k in f.files:
            A["sig_" + k] = f[k]

    # ---- h3: round-2 signal forecaster. Its own packaging writes two files (base + per-hour "alt" model); only
    #      the arrays that differ from the round-1 arrays above are stored (the base model is the same fit).
    c3 = combined3()
    sig2 = c3.SignalForecaster2(cb.fit_tables(fold_df)[2], cb.KAPPA)                 # as make_agent3 does
    with tempfile.TemporaryDirectory() as td:
        f2 = str(Path(td) / "sig2.npz")
        sig2.to_npz(f2)
        with np.load(f2, allow_pickle=False) as f:
            z2 = {k: f[k] for k in f.files}
        with np.load(f2 + ".alt.npz", allow_pickle=False) as f:
            za = {k: f[k] for k in f.files}
    same = lambda a, b: a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    r1 = {k[4:]: v for k, v in A.items() if k.startswith("sig_")}
    shared2 = [k for k in z2 if k in r1 and same(z2[k], r1[k])]
    for k in z2:
        if k not in shared2:
            A["sig2_" + k] = z2[k]
    shareda = [k for k in za if k in z2 and same(za[k], z2[k])]
    for k in za:
        if k not in shareda:
            A["alt_" + k] = za[k]
    A["sig2_keys"], A["alt_keys"] = _u8(json.dumps(sorted(z2))), _u8(json.dumps(sorted(za)))
    info["sig2_same_as_round1"], info["sig2_own"] = sorted(shared2), sorted(k for k in z2 if k not in shared2)
    info["alt_same_as_round2"], info["alt_own"] = sorted(shareda), sorted(k for k in za if k not in shareda)
    info["round1_not_in_round2"] = sorted(k for k in r1 if k not in z2 or not same(z2[k], r1[k]))

    # ---- h3: error paths of the signal and the slopes of the mean correction (sigplan.Paths, built by its own code)
    errors = c3.signal_errors2(fold_df) if cfg["paths_from"] == "round2" else None
    pool = c3.sp.Paths(cb.fold_name(fold_df), K=cfg["K"], m=cfg["m"], errors=errors)
    assert pool.mode == "add" and len(pool.sets) == 24
    A["paths_sets"], A["paths_beta"], A["paths_bias"] = np.asarray(pool.sets, float), np.asarray(pool.beta, float), np.asarray(pool.bias, float)
    fcs["w_sig2"], fcs["pool"] = sig2, pool

    # ---- cross-fitted error tables (the net-load table only when the scenario mode uses it)
    need_net = cfg["scen"] != "conv" or cfg["bias_tail"]
    for tag in ("fc", "nm"):
        A[f"err_load_{tag}"] = np.asarray(tabs[tag][1], float)
        if need_net:
            A[f"err_net_{tag}"] = np.asarray(tabs[tag][0], float)

    # ---- h1 fallback tables: only for validation states (the shipped agent.py has its own copy inside)
    if fold_df is not None:
        A["h1_json"] = _u8(json.dumps(h1_tables(fold_df)))
    for k, v in A.items():
        if v.dtype == object:
            raise TypeError(f"{k}: object array in the state")
        if v.dtype.kind == "f" and not np.isfinite(v).all() and not k.startswith(("sig_S0", "sig_CL", "sig2_S0", "sig2_CL", "alt_S0", "alt_CL", "tree_check_x", "tree_thr")):
            raise ValueError(f"{k}: non-finite values in the state")
    if verbose:
        print(f"round-2 signal: arrays identical to round 1: {info['sig2_same_as_round1']}; own: {info['sig2_own']}; "
              f"round-1 arrays not reused: {info['round1_not_in_round2']}")
        print(f"per-hour (alt) model: identical to round 2: {info['alt_same_as_round2']}; own: {info['alt_own']}")
        print(f"paths: sets {A['paths_sets'].shape}, beta {np.round(A['paths_beta'], 3).tolist()}")
        print(f"state: {len(A)} arrays, {sum(v.nbytes for v in A.values()) / 1e6:.2f} MB uncompressed; trees vs LightGBM on "
              f"{info['trees_rows_checked']} training rows: max abs diff {info['trees_max_abs_diff_vs_lightgbm_kw']:.2e} kW, "
              f"{info['trees_identical_rows']} rows bit-identical")
    build_state.info = info
    return A, fcs, tabs


def write_state(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _ends(cfg):
    import pandas as pd
    e = cfg.get("end_ts")
    return [] if e is None else sorted(pd.Timestamp(x) for x in ([e] if isinstance(e, str) else e))


def harness_agent3(fcs, tabs, config=None, load="fc"):
    """The integrator's h3 agent built from the SAME fitted objects as the state (what make_agent3 builds)."""
    cfg = dict(CONFIG if config is None else config)
    cb, c3 = combined(), combined3()
    tag = "fc" if load == "fc" else "nm"
    net = cb.NetScen(fcs["load_fc" if load == "fc" else "load_nm"], fcs["pv_fc"], cfg["scen"], tabs[tag][0], tabs[tag][1],
                     dep=cfg["dep"], bias_tail=cfg["bias_tail"])
    w_fc = fcs["w_sig2"]
    for e in sorted(_ends(cfg), reverse=True):                       # nested calendar cuts, the earliest outermost
        w_fc = c3.EndCut(w_fc, e)
    return c3.sp.SigAgent(w_fc, net, pool=fcs["pool"], lam=cfg["lam"], ar=cfg["ar"], H=cfg["H"], Q=cfg["Q"],
                          scen_hours=cfg["scen_hours"], J=cfg["J"], exact=(cfg["lam"] == 0), v_end=cfg["v_end"],
                          truncate_end=cfg["truncate_end"])


def harness_agent(fcs, tabs, config=None, load="fc"):
    """The h2 agent (round-1 signal, plan trusts the forecast) from the SAME fitted objects: what the h2 level must equal."""
    cfg = dict(CONFIG if config is None else config)
    cb = combined()
    tag = "fc" if load == "fc" else "nm"
    net = cb.NetScen(fcs["load_fc" if load == "fc" else "load_nm"], fcs["pv_fc"], cfg["scen"], tabs[tag][0], tabs[tag][1],
                     dep=cfg["dep"], bias_tail=cfg["bias_tail"])
    return cb.CombinedAgent(fcs["w_sig"], net, H=cfg["H"], Q=cfg["Q"], scen_hours=cfg["scen_hours"], spread=cfg["spread"],
                            v_end=cfg["v_end"], truncate_end=cfg["truncate_end"], end_ts=(_ends(cfg) or [None])[0])


def package(out_dir=ROOT / "submissions" / "h3j", verbose=True):
    """Fit on every train row, regenerate the run-time copies and agent.py, build submissions/h3j/."""
    bundle = _load("h3j_bundle", HERE / "bundle.py")
    bundle.sync(verbose)
    arrays, _, _ = build_state(None, verbose=verbose)
    write_state(arrays, HERE / STATE_NAME)
    bundle.bundle(h1_tables(None), verbose=verbose)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("agent.py", STATE_NAME):
        shutil.copyfile(HERE / name, out_dir / name)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("agent.py", STATE_NAME):
            z.write(out_dir / name, arcname=name)
    if verbose:
        for name in ("agent.py", STATE_NAME, "submission.zip"):
            print(f"  {out_dir / name}: {(out_dir / name).stat().st_size / 1024:.0f} KB")
    return out_dir


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    import warnings
    warnings.simplefilter("ignore")
    package()
