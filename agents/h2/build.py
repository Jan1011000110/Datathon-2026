"""
agents/h2/build.py  --  fit the combined agent and write its ONE state file (plain numpy arrays, no pickle).

Fitting is done by the specialists' own code (experiments/27_combined/combined.py and what it imports);
this script only turns the fitted objects into arrays. Every setting of the agent is in CONFIG below.

Validation (state fitted outside a fold; the returned objects let a test build the harness agent too):
    arrays, fcs, tabs = build_state(fold_df)
    write_state(arrays, "some/dir/h2_state.npz")

The real thing (fit on every train row, regenerate agent.py, package submissions/h2/):
    .venv/Scripts/python.exe agents/h2/build.py
"""
import importlib.util
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STATE_NAME = "h2_state.npz"

# ---- every setting of the agent (arguments of experiments/27_combined/combined.make_agent) ----------
CONFIG = dict(signal="fc", load="fc", solar="fc", scen="conv", H=36, Q=9, scen_hours=12, spread=1.0,
              v_end=None, truncate_end=True, dep=0.0, bias_tail=False,
              end_ts="2026-08-31 23:00")        # last hour the test can contain: the plan stops there (None = off)
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
    if (cfg["signal"], cfg["load"], cfg["solar"]) != ("fc", "fc", "fc"):
        raise ValueError("h2 packages the three specialist forecasters only")
    cb = combined()
    from src.hard import METEO_COLS
    rt = _load("rt_combined_check", HERE / "rt_combined.py") if False else None      # (constants checked below)
    assert abs(cb.KAPPA - (0.3 / 0.7) * 0.634) < 1e-15 and list(METEO_COLS) == ["irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms"]
    fcs = cb.build_forecasters(fold_df, which=["load_fc", "load_nm", "pv_fc", "w_sig"])
    tabs = {"fc": cb.error_tables_for(fold_df, "fc", "fc"), "nm": cb.error_tables_for(fold_df, "nomodel", "fc")}
    A = {}
    info = {}
    cfg_rt = {k: cfg[k] for k in ("scen", "H", "Q", "scen_hours", "spread", "v_end", "truncate_end", "dep", "bias_tail", "end_ts")}
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
        if v.dtype.kind == "f" and not np.isfinite(v).all() and not k.startswith(("sig_S0", "sig_CL", "tree_check_x", "tree_thr")):
            raise ValueError(f"{k}: non-finite values in the state")
    if verbose:
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


def harness_agent(fcs, tabs, config=None, load="fc"):
    """The integrator's agent built from the SAME fitted objects as the state (what make_agent builds)."""
    cfg = dict(CONFIG if config is None else config)
    cb = combined()
    tag = "fc" if load == "fc" else "nm"
    net = cb.NetScen(fcs["load_fc" if load == "fc" else "load_nm"], fcs["pv_fc"], cfg["scen"], tabs[tag][0], tabs[tag][1],
                     dep=cfg["dep"], bias_tail=cfg["bias_tail"])
    return cb.CombinedAgent(fcs["w_sig"], net, H=cfg["H"], Q=cfg["Q"], scen_hours=cfg["scen_hours"], spread=cfg["spread"],
                            v_end=cfg["v_end"], truncate_end=cfg["truncate_end"], end_ts=cfg["end_ts"])


def package(out_dir=ROOT / "submissions" / "h2", verbose=True):
    """Fit on every train row, regenerate the run-time copies and agent.py, build submissions/h2/."""
    bundle = _load("h2_bundle", HERE / "bundle.py")
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
