"""
experiments/02_stress/stress.py  --  red-team harness for any candidate agent.py

    .venv/Scripts/python.exe experiments/02_stress/stress.py agents/v1/agent.py
        [--fast]              scenarios on fold B only (3x quicker)
        [--calendar train]    keep the original train timestamps (default: relabel the folds to
                              the hidden test's calendar = last 7 days of each month)
        [--ref other.py]      reference agent for the scenario table (default: the candidate
                              with its guard disabled, if it has a _discharge_limits method)
        [--skip scenarios,evaluator,...]

Sections
  1 packaging   AST scan, imports, check_submission.py on the bare .py and on a zip
  2 fuzz        adversarial obs: never crash / NaN / inf / non-number / infraction
                (infractions counted by pushing the command through the OFFICIAL physics,
                 starter_kit/environment.py, from the same state of charge)
  3 sequences   timestamp flavours, DST-like days, state poisoning, on a full real run
  4 scenarios   data-shape scenarios: candidate vs reference (no guard) vs oracle
  5 timing      wall time of a 2,016-step run, slowest step
  6 official    full runs through starter_kit CampusBatteryEnv == src.sim.simulate
  7 evaluator   what if the hidden evaluator's physics differ slightly (informational)

Exit code 1 if any check FAILs. Details are written to experiments/02_stress/out/<tag>/.
"""
import argparse
import datetime as dt
import math
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import (CORE, OUT, ROOT, drop_weeks, folds, load_agent_module, make_noguard,  # noqa: E402
                 oracle_score, raw_score, run, suite)

sys.path.insert(0, str(ROOT / "starter_kit"))
import check_submission as cs  # noqa: E402
from environment import CampusBatteryEnv, run_agent_simulation  # noqa: E402

CAP, PMAX, ETA = 2000.0, 500.0, 0.9 ** 0.5
NAN, INF = float("nan"), float("inf")
DEL = object()
P, K, M = "preus_properes_24h", "co2_properes_24h", "noves_dades_5min"
ALLOWED_IMPORTS = {"numpy", "pandas", "sklearn", "joblib", "math", "datetime", "pathlib", "collections",
                   "itertools", "functools", "bisect", "heapq", "json", "pickle", "typing", "dataclasses",
                   "statistics", "random", "time", "calendar", "warnings", "copy", "operator", "re",
                   "__future__", "numbers", "decimal", "enum", "abc"}


# ==============================================================================================
# bookkeeping
# ==============================================================================================
class Report:
    def __init__(self):
        self.rows = []

    def add(self, section, name, status, detail=""):
        self.rows.append((section, name, status, detail))
        print("  [%-4s] %-58s %s" % (status, name[:58], detail), flush=True)

    def count(self, status):
        return sum(r[2] == status for r in self.rows)


def head(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100, flush=True)


# ==============================================================================================
# official physics for one step
# ==============================================================================================
_ENV = CampusBatteryEnv(pd.DataFrame({"load_kw": [1000.0, 1000.0], "pv_kw": [0.0, 0.0],
                                      "precio_eur_per_kwh": [0.1, 0.1], "co2_kg_per_kwh": [0.1, 0.1]}))


def official_infractions(soc, action):
    """Infractions the official environment would count for `action` taken at `soc`."""
    _ENV.reset()
    _ENV.soc = float(soc)
    _ENV.step(action)
    return int(_ENV.infraccions_per_step[-1])


def valid_number(a):
    if isinstance(a, bool) or not isinstance(a, (int, float, np.number)):
        return False
    return math.isfinite(float(a))


def safe_act(agent, obs):
    try:
        return agent.act(obs), None
    except BaseException as e:  # noqa: BLE001  (an agent must not even raise SystemExit)
        return None, "%s: %s" % (type(e).__name__, str(e)[:80])


# ==============================================================================================
# 1. packaging
# ==============================================================================================
def imports_of(path):
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def section_packaging(R, agent_path):
    head("1. PACKAGING")
    err = cs.check_ast(agent_path)
    R.add("packaging", "AST security scan (check_submission.check_ast)", "FAIL" if err else "PASS", err or "")
    mods = imports_of(agent_path)
    bad = sorted(mods & cs.FORBIDDEN_MODULES)
    odd = sorted(mods - ALLOWED_IMPORTS - cs.FORBIDDEN_MODULES)
    R.add("packaging", "imports", "FAIL" if bad else ("WARN" if odd else "PASS"),
          "%s%s" % (sorted(mods), ("  <- may not exist on the server: %s" % odd) if odd else ""))
    size = agent_path.stat().st_size
    aux = [f for f in agent_path.parent.iterdir()
           if f.is_file() and f.suffix.lower() in (".pkl", ".joblib", ".npy", ".npz", ".json", ".txt")]
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    if not Path(py).exists():
        py = sys.executable
    chk = str(ROOT / "starter_kit" / "check_submission.py")
    r = subprocess.run([py, chk, str(agent_path)], capture_output=True, text=True, cwd=str(ROOT),
                       encoding="utf-8", errors="replace")
    R.add("packaging", "check_submission.py on the bare agent.py", "PASS" if r.returncode == 0 else "FAIL",
          "exit %d%s" % (r.returncode, "" if r.returncode == 0 else " | " + (r.stdout + r.stderr)[-300:].replace("\n", " ")))
    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / "submission.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(agent_path, "agent.py")
            for f in aux:
                z.write(f, f.name)
        zsize = zp.stat().st_size
        r = subprocess.run([py, chk, str(zp)], capture_output=True, text=True, cwd=str(ROOT),
                           encoding="utf-8", errors="replace")
    R.add("packaging", "check_submission.py on a zip (agent.py at the root)", "PASS" if r.returncode == 0 else "FAIL",
          "exit %d, zip %.1f KB, aux files %s" % (r.returncode, zsize / 1024, [f.name for f in aux]))
    R.add("packaging", "size under the 25 MB limit", "PASS" if max(size, zsize) < 25e6 else "FAIL",
          "%.1f KB" % (size / 1024))


# ==============================================================================================
# 2. interface fuzz
# ==============================================================================================
def real_obs(df, t, soc, ts=None):
    h_end = min(t + 24, len(df))
    tl = t - 1 if t > 0 else t
    stamp = df.index[t] if ts is None else ts
    return {
        "timestamp": stamp, "hour": int(df.index[t].hour), "day_of_week": int(df.index[t].dayofweek),
        "soc_kwh": float(soc), "soc_pct": float(soc / CAP), "soc_percent": float(soc / CAP * 100),
        "preu_actual": float(df["precio_eur_per_kwh"].iloc[t]), "co2_actual": float(df["co2_kg_per_kwh"].iloc[t]),
        P: df["precio_eur_per_kwh"].iloc[t:h_end].tolist(), K: df["co2_kg_per_kwh"].iloc[t:h_end].tolist(),
        M: {"ultim_consum_kw": float(df["load_kw"].iloc[tl]), "ultima_solar_kw": float(df["pv_kw"].iloc[tl])},
    }


def resize(lst, n):
    return (list(lst) * (n // max(1, len(lst)) + 1))[:n]


def fuzz_cases():
    """(name, mutate(obs) -> None, true_socs or None). true_socs: states of charge the physics
    are checked from when the obs itself does not carry a usable one."""
    C = []

    def case(name, f, socs=None):
        C.append((name, f, socs))

    def setk(key, val):
        def f(o):
            if val is DEL:
                o.pop(key, None)
            else:
                o[key] = val(o[key]) if callable(val) else val
        return f

    def both(fp, fk):
        def f(o):
            fp(o)
            fk(o)
        return f

    # ---- lookahead length -------------------------------------------------------------------
    for n in (0, 1, 2, 23, 24, 30, 48, 200):
        case("lookahead length %d" % n, both(setk(P, lambda x, n=n: resize(x, n)), setk(K, lambda x, n=n: resize(x, n))))
    for a, b in ((24, 23), (23, 24), (24, 0), (0, 24), (1, 24), (24, 1), (30, 24), (24, 30), (0, 1)):
        case("unequal lengths price=%d co2=%d" % (a, b),
             both(setk(P, lambda x, a=a: resize(x, a)), setk(K, lambda x, b=b: resize(x, b))))
    # ---- container / element types -----------------------------------------------------------
    convs = {
        "np.ndarray": np.array, "tuple": tuple, "pd.Series": pd.Series,
        "np.float32 array": lambda x: np.array(x, np.float32),
        "list of np.float64": lambda x: [np.float64(v) for v in x],
        "list of int": lambda x: [1 + (i % 3) for i in range(len(x))],
        "list of numeric str": lambda x: ["%.5f" % v for v in x],
        "2-D array (1,n)": lambda x: np.array(x)[None, :], "2-D array (n,1)": lambda x: np.array(x)[:, None],
        "python float": lambda x: float(x[0]), "np.float64 scalar": lambda x: np.float64(x[0]),
        "int scalar": lambda x: 1, "None": lambda x: None, "str 'abc'": lambda x: "abc", "dict": lambda x: {},
        "list of list": lambda x: [[v] for v in x], "missing key": DEL,
    }
    for nm, cv in convs.items():
        case("lookahead as %s (both)" % nm, both(setk(P, cv), setk(K, cv)))
        case("price lookahead as %s" % nm, setk(P, cv))
        case("co2 lookahead as %s" % nm, setk(K, cv))
    # ---- bad values inside the lists ---------------------------------------------------------
    vals = {"NaN": NAN, "None": None, "inf": INF, "-inf": -INF, "negative": -0.5, "zero": 0.0,
            "1e12": 1e12, "1e300": 1e300, "1e-300": 1e-300, "str 'x'": "x"}
    for vn, v in vals.items():
        for where in ("first", "middle", "last", "all"):
            def put(x, v=v, where=where):
                x = list(x)
                idx = {"first": [0], "middle": [len(x) // 2], "last": [len(x) - 1], "all": range(len(x))}[where]
                for i in idx:
                    x[i] = v
                return x
            case("%s in price list (%s)" % (vn, where), setk(P, put))
            case("%s in co2 list (%s)" % (vn, where), setk(K, put))
            case("%s in both lists (%s)" % (vn, where), both(setk(P, put), setk(K, put)))
    # ---- current price / co2 -----------------------------------------------------------------
    for key in ("preu_actual", "co2_actual"):
        for vn, v in {"missing": DEL, "None": None, "NaN": NAN, "inf": INF, "negative": -1.0, "zero": 0.0,
                      "1e12": 1e12, "str": "0.1", "np.float64": np.float64(0.1), "int": 1}.items():
            case("%s %s" % (key, vn), setk(key, v))
            case("%s %s + empty lookahead" % (key, vn), both(setk(key, v), both(setk(P, []), setk(K, []))))
    # ---- last-hour readings ------------------------------------------------------------------
    for vn, v in {"missing": DEL, "None": None, "{}": {}, "[]": [], "str": "x", "float": 5.0,
                  "only consum": {"ultim_consum_kw": 900.0}, "only solar": {"ultima_solar_kw": 30.0},
                  "extra keys": {"ultim_consum_kw": 900.0, "ultima_solar_kw": 30.0, "foo": 1},
                  "pd.Series": pd.Series({"ultim_consum_kw": 900.0, "ultima_solar_kw": 30.0})}.items():
        case("noves_dades_5min %s" % vn, setk(M, v))
    rvals = {"None": None, "NaN": NAN, "inf": INF, "-inf": -INF, "-1100 (raw meter sign)": -1100.0, "zero": 0.0,
             "1e9": 1e9, "str '850.0'": "850.0", "np.float32": np.float32(850), "int": 850, "[850.0]": [850.0],
             "tiny 1e-9": 1e-9, "49.9": 49.9, "50.0": 50.0}
    for vn, v in rvals.items():
        case("ultim_consum_kw = %s" % vn, setk(M, lambda d, v=v: dict(d, ultim_consum_kw=v)))
        case("ultima_solar_kw = %s" % vn, setk(M, lambda d, v=v: dict(d, ultima_solar_kw=v)))
        case("both readings = %s" % vn, setk(M, lambda d, v=v: dict(d, ultim_consum_kw=v, ultima_solar_kw=v)))
    # ---- state of charge fields --------------------------------------------------------------
    case("soc_percent missing", setk("soc_percent", DEL))
    case("soc_pct missing", setk("soc_pct", DEL))
    case("soc_pct and soc_percent missing", both(setk("soc_pct", DEL), setk("soc_percent", DEL)))
    case("soc_pct inconsistent with soc_kwh", setk("soc_pct", 0.93))
    for vn, cv in {"int": lambda s: int(s), "np.float64": np.float64, "np.float32": np.float32,
                   "np.int64": lambda s: np.int64(s), "0-d array": np.array, "numeric str": lambda s: "%.1f" % s}.items():
        case("soc_kwh as %s" % vn, setk("soc_kwh", cv))
    for vn, v in {"missing": DEL, "None": None, "NaN": NAN, "str 'abc'": "abc", "inf": INF}.items():
        case("soc_kwh %s (pct fields still valid)" % vn, setk("soc_kwh", v), socs=[1000.0])
        case("soc_kwh %s and pct fields missing" % vn,
             both(setk("soc_kwh", v), both(setk("soc_pct", DEL), setk("soc_percent", DEL))), socs=[0.0, 1000.0, 2000.0])
    # ---- timestamp / calendar fields ---------------------------------------------------------
    for vn, v in {"pd.Timestamp": pd.Timestamp("2026-01-28 12:00"), "str 'Y-m-d H:M:S'": "2026-01-28 12:00:00",
                  "str ISO T": "2026-01-28T12:00:00", "str d/m/Y": "28/01/2026 12:00", "str garbage": "yesterday",
                  "datetime": dt.datetime(2026, 1, 28, 12), "date": dt.date(2026, 1, 28),
                  "np.datetime64": np.datetime64("2026-01-28T12:00"), "None": None, "missing": DEL,
                  "int epoch": 1769601600, "float": 1.7696e9, "tz-aware Timestamp": pd.Timestamp("2026-01-28 12:00", tz="Europe/Madrid"),
                  "pd.NaT": pd.NaT, "year 1900": pd.Timestamp("1900-01-01"), "year 2262": pd.Timestamp("2262-01-01")}.items():
        case("timestamp %s" % vn, setk("timestamp", v))
    for key in ("hour", "day_of_week"):
        for vn, v in {"missing": DEL, "None": None, "np.int64": np.int64(3), "str": "3", "out of range": 99, "NaN": NAN}.items():
            case("%s %s" % (key, vn), setk(key, v))
    case("extra unknown keys", lambda o: o.update(foo=1, bar=[1, 2], baz=None))
    case("only soc_kwh present", lambda o: [o.pop(k) for k in list(o) if k != "soc_kwh"])
    return C


def soc_edge_cases(df):
    """(name, obs, true_soc, reachable). Pressure to charge / discharge at the physical limits."""
    out = []
    u = PMAX / ETA / 10.0
    lattice = sorted({k * u for k in range(38)} | {CAP - k * u for k in range(38)})
    socs = [0.0, 1e-12, 1e-9, 1e-7, 1e-6, 1e-4, 1.1e-4, 1e-3, 0.05, 1.0, CAP - 1.0, CAP - 0.05, CAP - 1e-3, CAP - 1.1e-4,
            CAP - 1e-4, CAP - 1e-6, CAP - 1e-7, CAP - 1e-9, CAP - 1e-12, CAP, -1e-9, -1e-7, CAP + 1e-9, CAP + 1e-7]
    for g in lattice[1:-1:3]:
        socs += [g, g - 1e-9, g + 1e-9, g - 1e-12, g + 1e-12]
    rng = np.random.default_rng(3)
    socs += rng.uniform(0, CAP, 40).tolist()
    unreachable = [-1e-3, -1.0, CAP + 1e-3, CAP + 1.0]
    lo, hi = 0.02, 0.30
    windows = {"charge now": [lo] + [hi] * 23, "discharge now": [hi] + [lo] * 23, "flat": [0.1] * 24,
               "ramp up": list(np.linspace(lo, hi, 24)), "ramp down": list(np.linspace(hi, lo, 24)),
               "single hour, high": [hi], "single hour, low": [lo]}
    for j in range(6):
        t = int(rng.integers(0, len(df) - 24))
        windows["real slice %d" % j] = None if False else (df["co2_kg_per_kwh"].iloc[t:t + 24] + 0.257 * df["precio_eur_per_kwh"].iloc[t:t + 24]).tolist()
    readings = {"normal": (1000.0, 50.0), "outage": (0.0, 0.0), "low": (300.0, 20.0), "NaN": (NAN, NAN)}
    for s in socs + unreachable:
        for wn, w in windows.items():
            for rn, (ld, pv) in readings.items():
                o = real_obs(df, 400, s)
                o[P] = [x / 0.257 / 2 for x in w]
                o[K] = [x / 2 for x in w]
                o["preu_actual"], o["co2_actual"] = o[P][0], o[K][0]
                o[M] = {"ultim_consum_kw": ld, "ultima_solar_kw": pv}
                out.append(("soc=%r | %s | reading %s" % (s, wn, rn), o, s, s not in unreachable))
    return out


def section_fuzz(R, mod, df):
    head("2. INTERFACE FUZZ  (official physics, starter_kit/environment.py)")
    t0 = 500
    cases = fuzz_cases()
    stats = {"crash": [], "bad": [], "infr": [], "post": []}
    n_calls = 0
    for mode in ("warm", "cold"):
        for name, mutate, socs in cases:
            agent = mod.BatteryAgent()
            if mode == "warm":
                for k in (3, 2, 1):
                    safe_act(agent, real_obs(df, t0 - k, 1000.0))
            obs = real_obs(df, t0, 1000.0)
            try:
                mutate(obs)
            except Exception as e:  # harness bug, not agent bug
                raise RuntimeError("fuzz case %r could not be built: %s" % (name, e))
            a, err = safe_act(agent, obs)
            n_calls += 1
            tag = "%s [%s]" % (name, mode)
            if err:
                stats["crash"].append((tag, err))
            elif not valid_number(a):
                stats["bad"].append((tag, repr(a)))
            else:
                for s in (socs or [1000.0]):
                    k = official_infractions(s, a)
                    if k:
                        stats["infr"].append((tag, "soc=%g action=%r -> %d infraction(s)" % (s, a, k)))
            # the agent must keep working afterwards (state not poisoned)
            for k in (1, 2, 3):
                o2 = real_obs(df, t0 + k, 1000.0)
                a2, err2 = safe_act(agent, o2)
                if err2 or not valid_number(a2) or official_infractions(1000.0, a2):
                    stats["post"].append((tag, err2 or repr(a2)))
                    break
    for key, label in (("crash", "no exception on any adversarial obs"), ("bad", "return is always a finite int/float"),
                       ("infr", "no infraction on any adversarial obs"), ("post", "agent keeps working after a bad obs")):
        bad = stats[key]
        R.add("fuzz", "%s (%d calls)" % (label, n_calls), "FAIL" if bad else "PASS",
              "" if not bad else "%d failing, e.g. %s" % (len(bad), "; ".join("%s -> %s" % b for b in bad[:3])))
    # --- SOC edges ---
    edge = soc_edge_cases(df)
    fails, unre, n = [], [], 0
    worst_margin = None
    for name, obs, s, reachable in edge:
        agent = mod.BatteryAgent()
        a, err = safe_act(agent, obs)
        n += 1
        if err or not valid_number(a):
            (fails if reachable else unre).append((name, err or repr(a)))
            continue
        k = official_infractions(s, a)
        if k:
            (fails if reachable else unre).append((name, "action=%r -> %d infraction(s)" % (a, k)))
    R.add("fuzz", "SOC at / next to the limits and lattice points (%d cases)" % sum(e[3] for e in edge),
          "FAIL" if fails else "PASS", "" if not fails else "%d failing, e.g. %s" % (len(fails), "; ".join("%s -> %s" % b for b in fails[:3])))
    R.add("fuzz", "SOC outside the range by more than the env tolerance (1e-6)", "INFO",
          "%d of %d cases give an infraction (state unreachable in the official env)" % (len(unre), sum(not e[3] for e in edge)))
    return stats, fails


# ==============================================================================================
# 3. sequences
# ==============================================================================================
class TsRewrite:
    def __init__(self, agent, fn):
        self.agent, self.fn, self.t = agent, fn, 0

    def act(self, obs):
        obs = dict(obs)
        v = self.fn(obs["timestamp"], self.t)
        if v is DEL:
            obs.pop("timestamp")
        else:
            obs["timestamp"] = v
        self.t += 1
        return self.agent.act(obs)


class Poison:
    """Every `every` steps, first calls the agent with a garbage obs (output discarded)."""

    def __init__(self, agent, garbage, every=41):
        self.agent, self.garbage, self.every, self.t, self.errors = agent, garbage, every, 0, 0

    def act(self, obs):
        if self.t % self.every == self.every - 1:
            g = dict(obs)
            self.garbage[(self.t // self.every) % len(self.garbage)](g)
            try:
                self.agent.act(g)
            except BaseException:  # noqa: BLE001
                self.errors += 1
        self.t += 1
        return self.agent.act(obs)


def section_sequences(R, mod, df):
    head("3. SEQUENCES  (full 2,016-step run, fold B)")
    try:
        base = run(mod.BatteryAgent(), df)
    except Exception as e:
        R.add("sequences", "baseline run", "FAIL", "%s: %s" % (type(e).__name__, e))
        return
    flav = {
        "str 'Y-m-d H:M:S'": lambda ts, t: str(ts), "str ISO 'T'": lambda ts, t: ts.isoformat(),
        "datetime.datetime": lambda ts, t: ts.to_pydatetime(), "np.datetime64": lambda ts, t: ts.to_datetime64(),
        "tz-aware (UTC)": lambda ts, t: ts.tz_localize("UTC"), "None": lambda ts, t: None, "missing": lambda ts, t: DEL,
        "constant (repeating)": lambda ts, t: pd.Timestamp("2026-01-28 12:00"),
        "going backwards": lambda ts, t: pd.Timestamp("2026-08-31 23:00") - pd.Timedelta(hours=t),
        "int epoch seconds": lambda ts, t: int(ts.timestamp()),
        "random type each step": lambda ts, t: [ts, str(ts), None, ts.to_pydatetime(), ts.to_datetime64()][t % 5],
    }
    for name, fn in flav.items():
        try:
            r = run(TsRewrite(mod.BatteryAgent(), fn), df)
            d = r["raw"] - base["raw"]
            st = "FAIL" if r["infractions"] else ("WARN" if abs(d) > 0.005 else "PASS")
            R.add("sequences", "timestamp as %s" % name, st, "score %+.4f vs pd.Timestamp run, %d infr" % (d, r["infractions"]))
        except Exception as e:
            R.add("sequences", "timestamp as %s" % name, "FAIL", "%s: %s" % (type(e).__name__, str(e)[:90]))
    # DST-like days: a 23-hour day (02:00 missing) and a 25-hour day (02:00 twice)
    try:
        d2 = df.copy()
        i_spring = int(np.argmax((d2.index.month == 3) & (d2.index.day >= 27) & (d2.index.hour == 2)))
        d2 = d2.drop(d2.index[i_spring])
        i_fall = int(np.argmax((d2.index.month == 10) & (d2.index.day >= 27) & (d2.index.hour == 2)))
        d2 = pd.concat([d2.iloc[:i_fall + 1], d2.iloc[i_fall:]])
        r = run(mod.BatteryAgent(), d2)
        reg = oracle_score(d2) - r["raw"]
        R.add("sequences", "DST-like days (one hour missing, one duplicated)",
              "FAIL" if r["infractions"] else ("WARN" if reg > 0.05 else "PASS"),
              "regret vs oracle %.4f, %d infr" % (reg, r["infractions"]))
    except Exception as e:
        R.add("sequences", "DST-like days", "FAIL", "%s: %s" % (type(e).__name__, str(e)[:90]))
    # poison
    garbage = [lambda o: o.update({P: None, K: None}), lambda o: o.update(timestamp="garbage"),
               lambda o: o.update({M: {"ultim_consum_kw": NAN, "ultima_solar_kw": None}}),
               lambda o: o.update({M: {"ultim_consum_kw": 0.0, "ultima_solar_kw": 0.0}}),
               lambda o: o.pop("soc_kwh"), lambda o: o.update(timestamp=None),
               lambda o: o.update({P: [NAN] * 24}), lambda o: o.update(timestamp=pd.Timestamp("1999-01-01")),
               lambda o: o.update(soc_kwh=NAN), lambda o: o.update({P: [], K: []})]
    try:
        pz = Poison(mod.BatteryAgent(), garbage)
        r = run(pz, df)
        d = r["raw"] - base["raw"]
        st = "FAIL" if r["infractions"] else ("WARN" if (abs(d) > 0.02 or pz.errors) else "PASS")
        R.add("sequences", "49 garbage calls interleaved in a real run", st,
              "score %+.4f vs clean run, %d infr, %d garbage calls raised" % (d, r["infractions"], pz.errors))
    except Exception as e:
        R.add("sequences", "garbage calls interleaved", "FAIL", "%s: %s" % (type(e).__name__, str(e)[:90]))
    # determinism
    try:
        a = run(mod.BatteryAgent(), df, trace=True)["trace"]["p_kw"].to_numpy()
        b = run(mod.BatteryAgent(), df, trace=True)["trace"]["p_kw"].to_numpy()
        R.add("sequences", "deterministic (two runs, same actions)", "PASS" if np.array_equal(a, b) else "WARN", "")
    except Exception as e:
        R.add("sequences", "deterministic", "FAIL", str(e)[:90])


# ==============================================================================================
# 4. scenarios
# ==============================================================================================
def section_scenarios(R, mod, ref_cls, ref_name, calendar, fold_names, outdir):
    head("4. DATA-SHAPE SCENARIOS  (score in points, unclamped; regret = oracle - agent; mean of folds %s)" % ",".join(fold_names))
    S = suite(calendar, fold_names)
    rows = []
    crashed = 0
    for sc in S:
        row = {"family": sc["family"], "scenario": sc["name"], "fold": sc["fold"], "events": sc["events"]}
        row["oracle"] = oracle_score(sc["df"])
        try:
            r = run(mod.BatteryAgent(), sc["df"], sc["obs_load"], sc["obs_pv"])
            row["agent"], row["infr"] = r["raw"], r["infractions"]
        except Exception as e:
            crashed += 1
            row["agent"], row["infr"], row["error"] = np.nan, 0, "%s: %s" % (type(e).__name__, str(e)[:60])
        if ref_cls is not None:
            try:
                r = run(ref_cls(), sc["df"], sc["obs_load"], sc["obs_pv"])
                row["ref"], row["ref_infr"] = r["raw"], r["infractions"]
            except Exception:
                row["ref"], row["ref_infr"] = np.nan, 0
        rows.append(row)
    d = pd.DataFrame(rows)
    d["regret"] = d["oracle"] - d["agent"]
    if ref_cls is not None:
        d["ref_regret"] = d["oracle"] - d["ref"]
        d["gain"] = d["agent"] - d["ref"]
    clean = d[d.scenario == "clean"].set_index("fold")["regret"]
    d["extra_per_event"] = np.where(d.events > 0, (d.regret - d.fold.map(clean)) / d.events.clip(lower=1), np.nan)
    d.to_csv(outdir / "scenarios_by_fold.csv", index=False)
    agg = {"agent": "mean", "oracle": "mean", "regret": "mean", "infr": "sum", "extra_per_event": "mean"}
    if ref_cls is not None:
        agg.update({"ref": "mean", "ref_regret": "mean", "gain": "mean"})
    g = d.groupby(["family", "scenario"], sort=False).agg(agg)
    g["worst_fold_regret"] = d.groupby(["family", "scenario"], sort=False)["regret"].max()
    g.to_csv(outdir / "scenarios.csv")
    hdr = "%-42s %8s %8s %8s | %8s %8s %8s | %9s %5s" % ("scenario", "agent", ref_name[:8], "oracle", "regret", "ref.regr", "worst", "extra/evt", "infr")
    print(hdr)
    fam_prev = None
    for (fam, name), r in g.iterrows():
        if fam != fam_prev:
            print("-- %s" % fam)
            fam_prev = fam
        ref = "%8.4f" % r["ref"] if "ref" in r else "       -"
        rr = "%8.4f" % r["ref_regret"] if "ref_regret" in r else "       -"
        ev = "%9.4f" % r["extra_per_event"] if r["extra_per_event"] == r["extra_per_event"] else "        -"
        print("%-42s %8.4f %s %8.4f | %8.4f %s %8.4f | %s %5d" % (name[:42], r["agent"], ref, r["oracle"], r["regret"], rr, r["worst_fold_regret"], ev, r["infr"]))
    fam = d.groupby("family")
    print("\nfamily means:  " + "   ".join("%s %.4f" % (k[2:], v) for k, v in fam["regret"].mean().items()))
    n_inf = int(d["infr"].sum())
    R.add("scenarios", "no crash in any scenario (%d runs)" % len(d), "FAIL" if crashed else "PASS",
          "" if not crashed else "%d crashed: %s" % (crashed, d.dropna(subset=["error"]).iloc[0]["error"]))
    R.add("scenarios", "no infraction in any scenario", "FAIL" if n_inf else "PASS", "%d infractions" % n_inf)
    cr = float(d[d.family == "a.clean"]["regret"].mean())
    R.add("scenarios", "clean folds: regret vs oracle", "PASS" if cr < 0.02 else "WARN", "mean %.4f points" % cr)
    worst = d.loc[d["regret"].idxmax()]
    R.add("scenarios", "largest regret vs oracle", "WARN" if worst["regret"] > 0.15 else "PASS",
          "%.4f points: %s, fold %s" % (worst["regret"], worst["scenario"], worst["fold"]))
    if ref_cls is not None:
        lose = d[d["gain"] < -0.005]
        R.add("scenarios", "candidate never worse than %s by > 0.005" % ref_name, "WARN" if len(lose) else "PASS",
              "" if not len(lose) else "%d runs, worst %.4f: %s (fold %s)" % (
                  len(lose), lose["gain"].min(), lose.loc[lose["gain"].idxmin(), "scenario"], lose.loc[lose["gain"].idxmin(), "fold"]))
    return d


# ==============================================================================================
# 5. timing
# ==============================================================================================
class Timed:
    def __init__(self, agent):
        self.agent, self.times = agent, []

    def act(self, obs):
        t0 = time.perf_counter()
        a = self.agent.act(obs)
        self.times.append(time.perf_counter() - t0)
        return a


def section_timing(R, agent_path, df):
    head("5. TIMING")
    t0 = time.perf_counter()
    mod = load_agent_module(agent_path, "agent_timing")
    t_imp = time.perf_counter() - t0
    t0 = time.perf_counter()
    agent = mod.BatteryAgent()
    t_init = time.perf_counter() - t0
    best = None
    for _ in range(3):
        ta = Timed(mod.BatteryAgent())
        run(ta, df)
        tt = np.array(ta.times)
        if best is None or tt.sum() < best.sum():
            best = tt
    R.add("timing", "2,016-step run: time spent inside act()", "PASS" if best.sum() < 10 else ("WARN" if best.sum() < 40 else "FAIL"),
          "%.3f s total, %.3f ms/step mean, p99 %.3f ms, slowest step %.3f ms (limit: 60 s per run)" % (
              best.sum(), 1e3 * best.mean(), 1e3 * np.quantile(best, 0.99), 1e3 * best.max()))
    R.add("timing", "import + BatteryAgent()", "PASS" if t_imp + t_init < 5 else "WARN",
          "import %.1f ms (modules already warm), init %.2f ms" % (1e3 * t_imp, 1e3 * t_init))
    # cold start in a fresh interpreter: import numpy/pandas + agent + 2,016 steps of the smoke obs
    code = ("import time,sys;t0=time.perf_counter();import importlib.util as u;"
            "s=u.spec_from_file_location('a',sys.argv[1]);m=u.module_from_spec(s);s.loader.exec_module(m);"
            "a=m.BatteryAgent();o={'timestamp':'2026-01-15 12:00:00','hour':12,'day_of_week':3,'soc_kwh':1000.0,"
            "'soc_pct':0.5,'soc_percent':50.0,'preu_actual':0.12,'co2_actual':0.15,"
            "'preus_properes_24h':[0.1+0.01*(i%7) for i in range(24)],'co2_properes_24h':[0.1+0.01*(i%5) for i in range(24)],"
            "'noves_dades_5min':{'ultim_consum_kw':900.0,'ultima_solar_kw':20.0}};"
            "[a.act(o) for _ in range(2016)];print(time.perf_counter()-t0)")
    try:
        r = subprocess.run([sys.executable, "-c", code, str(agent_path)], capture_output=True, text=True, cwd=str(agent_path.parent))
        cold = float(r.stdout.strip().splitlines()[-1])
        R.add("timing", "cold start: fresh interpreter, import + 2,016 steps", "PASS" if cold < 15 else "WARN", "%.2f s" % cold)
    except Exception as e:
        R.add("timing", "cold start", "WARN", "could not measure: %s" % str(e)[:80])


# ==============================================================================================
# 6. official environment
# ==============================================================================================
def section_official(R, mod, calendar):
    head("6. OFFICIAL ENVIRONMENT  (starter_kit CampusBatteryEnv, full runs)")
    F = folds(calendar)
    tests = {"fold A": F["A"], "fold B": F["B"], "fold C (real 14 h outage)": F["C"],
             "fold A, Nov+Aug dropped": drop_weeks(F["A"], [2, 11])}
    for name, df in tests.items():
        try:
            off = run_agent_simulation(mod.BatteryAgent(), df[CORE])
            fast = run(mod.BatteryAgent(), df)
            same = (abs(off["score"] - round(max(0.0, fast["raw"]), 2)) < 0.0101
                    and abs(off["co2_estalviat_pct"] - fast["co2_pct"]) < 0.0051
                    and abs(off["cost_estalviat_pct"] - fast["cost_pct"]) < 0.0051
                    and off["infraccions"] == fast["infractions"])
            st = "FAIL" if off["infraccions"] else ("PASS" if same else "WARN")
            R.add("official", name, st, "official %.2f (co2 %.2f%%, cost %.2f%%, infr %d) | src.sim %.4f (%.3f%%, %.3f%%, infr %d)" % (
                off["score"], off["co2_estalviat_pct"], off["cost_estalviat_pct"], off["infraccions"],
                fast["raw"], fast["co2_pct"], fast["cost_pct"], fast["infractions"]))
        except Exception as e:
            R.add("official", name, "FAIL", "%s: %s" % (type(e).__name__, str(e)[:90]))


# ==============================================================================================
# 7. evaluator variants
# ==============================================================================================
def run_variant(agent, df, eta_c=ETA, eta_d=ETA, soc0=1000.0, cap=CAP, pmax=PMAX, tol=1e-6):
    """src.sim.simulate with adjustable physics (the agent still believes the official ones)."""
    load = df["load_kw"].to_numpy(float)
    pv = df["pv_kw"].to_numpy(float)
    price = df["precio_eur_per_kwh"].to_numpy(float)
    co2 = df["co2_kg_per_kwh"].to_numpy(float)
    pl, cl = price.tolist(), co2.tolist()
    n = len(df)
    soc, infr = soc0, 0
    grid = np.zeros(n)
    for t in range(n):
        tl = t - 1 if t > 0 else t
        obs = {"timestamp": df.index[t], "hour": int(df.index[t].hour), "day_of_week": int(df.index[t].dayofweek),
               "soc_kwh": float(soc), "soc_pct": float(soc / cap), "soc_percent": float(soc / cap * 100),
               "preu_actual": pl[t], "co2_actual": cl[t], P: pl[t:t + 24], K: cl[t:t + 24],
               M: {"ultim_consum_kw": float(load[tl]), "ultima_solar_kw": float(pv[tl])}}
        p_req = float(agent.act(obs))
        p = min(max(p_req, -pmax), pmax)
        infr += p != p_req
        if p > 0:
            p_act = min(p, max(0.0, (cap - soc) / eta_c))
            infr += p_act < p
            soc += p_act * eta_c
        elif p < 0:
            p_act = -min(-p, max(0.0, soc * eta_d))
            infr += p_act > p
            soc += p_act / eta_d
        else:
            p_act = 0.0
        infr += (soc < -tol or soc > cap + tol)
        soc = min(max(soc, 0.0), cap)
        grid[t] = max(load[t] - pv[t] + p_act, 0.0)
    base = np.maximum(load - pv, 0.0)
    sc = 70 * (1 - (grid * co2).sum() / (base * co2).sum()) + 30 * (1 - (grid * price).sum() / (base * price).sum())
    return sc, int(infr)


def section_evaluator(R, mod, df):
    head("7. WHAT IF THE HIDDEN EVALUATOR DIFFERS FROM environment.py?  (informational, fold B)")
    variants = {
        "official physics (control)": {},
        "one-way efficiency 0.9487 (4 decimals, as starter agent.py)": dict(eta_c=0.9487, eta_d=0.9487),
        "one-way efficiency 0.95": dict(eta_c=0.95, eta_d=0.95),
        "all the loss on charge (0.9 / 1.0)": dict(eta_c=0.9, eta_d=1.0),
        "all the loss on discharge (1.0 / 0.9)": dict(eta_c=1.0, eta_d=0.9),
        "no SOC tolerance (0 instead of 1e-6)": dict(tol=0.0),
        "run starts empty (SOC 0)": dict(soc0=0.0),
        "run starts full (SOC 2000)": dict(soc0=2000.0),
    }
    for name, kw in variants.items():
        try:
            sc, k = run_variant(mod.BatteryAgent(), df, **kw)
            st = "PASS" if k == 0 else ("FAIL" if not kw or "starts" in name or "tolerance" in name else "WARN")
            R.add("evaluator", name, st, "score %.4f before penalties, %d infractions (%.1f points)" % (sc, k, 0.2 * k))
        except Exception as e:
            R.add("evaluator", name, "WARN", "%s: %s" % (type(e).__name__, str(e)[:90]))


# ==============================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="path to the candidate agent.py")
    ap.add_argument("--fast", action="store_true", help="scenarios on fold B only")
    ap.add_argument("--calendar", choices=("test", "train"), default="test")
    ap.add_argument("--ref", default=None, help="reference agent.py for the scenario table")
    ap.add_argument("--skip", default="", help="comma list: packaging,fuzz,sequences,scenarios,timing,official,evaluator")
    a = ap.parse_args()
    agent_path = Path(a.agent).resolve()
    skip = set(filter(None, a.skip.split(",")))
    t_start = time.time()
    tag = agent_path.parent.name + "_" + agent_path.stem
    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    print("stress.py  candidate = %s   calendar = %s   %s" % (agent_path, a.calendar, time.strftime("%Y-%m-%d %H:%M")))

    R = Report()
    if "packaging" not in skip:
        section_packaging(R, agent_path)
    try:
        mod = load_agent_module(agent_path)
        mod.BatteryAgent()
    except BaseException as e:  # noqa: BLE001
        R.add("load", "import the agent and build BatteryAgent()", "FAIL", "%s: %s" % (type(e).__name__, e))
        traceback.print_exc()
        sys.exit(1)
    if a.ref:
        ref_mod = load_agent_module(a.ref, "agent_ref")
        ref_cls, ref_name = ref_mod.BatteryAgent, Path(a.ref).resolve().parent.name
    else:
        ref_cls, ref_name = make_noguard(mod), "no-guard"
    F = folds(a.calendar)
    if "fuzz" not in skip:
        section_fuzz(R, mod, F["B"])
    if "sequences" not in skip:
        section_sequences(R, mod, F["B"])
    if "scenarios" not in skip:
        section_scenarios(R, mod, ref_cls, ref_name, a.calendar, ("B",) if a.fast else ("A", "B", "C"), outdir)
    if "timing" not in skip:
        section_timing(R, agent_path, F["A"])
    if "official" not in skip:
        section_official(R, mod, a.calendar)
    if "evaluator" not in skip:
        section_evaluator(R, mod, F["B"])

    head("SUMMARY  %s" % agent_path)
    pd.DataFrame(R.rows, columns=["section", "check", "status", "detail"]).to_csv(outdir / "checks.csv", index=False)
    for sec in dict.fromkeys(r[0] for r in R.rows):
        rs = [r for r in R.rows if r[0] == sec]
        print("  %-10s PASS %3d  WARN %2d  FAIL %2d" % (sec, sum(r[2] == "PASS" for r in rs), sum(r[2] == "WARN" for r in rs), sum(r[2] == "FAIL" for r in rs)))
    for st in ("FAIL", "WARN"):
        for r in R.rows:
            if r[2] == st:
                print("  %s  %s / %s: %s" % (st, r[0], r[1], r[3][:160]))
    nf, nw = R.count("FAIL"), R.count("WARN")
    print("\nRESULT: %s   (%d FAIL, %d WARN, %d PASS)   %.0f s   details: %s" % (
        "FAIL" if nf else "PASS", nf, nw, R.count("PASS"), time.time() - t_start, outdir))
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
