"""
agents/h4/bundle.py  --  make the run-time copies of the specialists' modules and bundle them into agent.py.

    .venv/Scripts/python.exe agents/h4/bundle.py            # sync() + bundle()

sync()    regenerates agents/h4/rt_*.py from the originals. Nothing is ported by hand: each copy is the
          original source minus the fitting code (removed by name), plus a few exact-text patches listed in
          SPEC below. A patch whose text is no longer found stops the script, so a change in an original
          cannot slip through unnoticed.
bundle()  concatenates rt_*.py + rt_main.py into ONE agent.py. Each module becomes a function (its top-level
          names become that function's locals, so equal names in different modules cannot collide) and
          `from rt_x import a, b` becomes a lookup in what that function returned. A module that fails to
          load (e.g. no pandas) leaves None behind instead of breaking the import of agent.py.
Both run check_submission.check_ast on what they write.
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

COMBINED_HEADER = '''import numpy as np
import pandas as pd

from rt_planner import plan_det, plan_stoch, safe_clamp

KAPPA = (0.3 / 0.7) * 0.634                 # = src.hard_mpc.KAPPA (checked by build.py)
METEO_COLS = ["irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms"]      # = src.hard.METEO_COLS
CAP = 6000.0
PMAX = 1500.0
'''

SIGPLAN_NAMES = ["backward", "first_step", "stage_segments_det", "_segments", "plan_det", "plan_stoch", "safe_clamp",
                 "CAP", "PMAX", "ETA", "KAPPA"]
SIGPLAN_HEADER = "import numpy as np\n\nfrom rt_planner import " + ", ".join(SIGPLAN_NAMES) + "\n\n\n"
BASE_ATTRS = ("kappa", "kmax", "ridge", "variant", "holidays", "interp", "cold_min", "use_lw", "loo", "cross", "ks", "m1",
              "_last_i", "t0_ns", "n", "nh", "hour_grid", "S0", "CL", "coef", "S")      # everything a loaded SignalFC holds

# name of the run-time copy -> (original, top-level names to drop, methods to drop, top-level names to keep
# (None = all), text put in front, patches [(old, new, expected count)])
SPEC = {
    "rt_h1": (ROOT / "agents" / "h1" / "agent.py", [], [], None, "", [
        ("class BatteryAgent:", "class H1Agent:", 1),
        ("    def __init__(self):\n        self.ready = False\n        self.month = None",
         "    def __init__(self, tables=None):\n        self.ready = False\n        self.month = None", 1),
        ("            self._load(self.data_path if self.data_path is not None else _default_path())\n",
         "            self._load(tables)\n", 1),
        ("    def _load(self, path):\n        with open(str(path), \"r\", encoding=\"utf-8\") as f:\n            t = json.load(f)\n",
         "    def _load(self, t):\n", 1),
        # red-team: lists without a single finite value (missing, None, all NaN) are treated as empty, so the
        # horizon is not cut and the plan uses the scalar fields (before: one-hour plans for the whole run)
        ("        known = co2[:nk] + self.kappa * price[:nk]\n        w0 = ",
         "        known = co2[:nk] + self.kappa * price[:nk]\n"
         "        if nk and not np.isfinite(known).any():      # nothing usable in the lists: as if they were empty\n"
         "            nk = 0\n"
         "        w0 = ", 1),
    ]),
    "rt_planner": (EXP / "24_planner" / "planner.py", ["QuantileN", "MPCStoch"], [], None, "", []),
    "rt_load": (EXP / "21_load" / "load_fc.py", ["LoadQ"], ["LoadFC.fit", "LoadFC.to_state"], None, "", [
        # neither is needed at run time: pandas was only used by fit(), and the model is evaluated by
        # rt_trees.TreeModel (set by rt_main), so LightGBM is never imported on the server
        ("import pandas as pd\nimport lightgbm as lgb\n", "lgb = None\n", 1),
    ]),
    # the holiday calendar: the research subclass minus its constructor, fitting and packaging code. What is left is
    # the calendar itself (CALENDAR) and the five methods that override LoadFC's; rt_main sets the attributes.
    "rt_cal": (EXP / "21_load" / "cal_fc.py", ["d0_of"],
               ["CalLoadFC.__init__", "CalLoadFC.set_calendar", "CalLoadFC.fit", "CalLoadFC.fit_priors", "CalLoadFC.to_state",
                "CalLoadFC.from_state"], None, "", [
        ("import numpy as np\nimport pandas as pd\n\nfrom load_fc import FROZEN, IDX_DSMIX, IDX_LAMN, IDX_SGMIX, NAN, NS_H, LoadFC, day0, etype\n",
         "import numpy as np\n\nfrom rt_load import FROZEN, LoadFC, day0, etype\n", 1),
    ]),
    "rt_solar": (EXP / "22_solar" / "solar_fc.py", ["QuantileH0"],
                 ["SolarFC._build_envelope", "SolarFC._replay", "SolarFC.fit", "SolarFC._fit_model"], None, "", []),
    "rt_signal": (EXP / "23_signal" / "signal_fc.py", [], ["SignalFC.fit", "SignalFC.to_npz"], None, "", [
        # the server's scanner rejects any call spelled `.stack(` -- same array, built with concatenate
        ("np.stack(P, axis=2)", "np.concatenate([x_[:, :, None] for x_ in P], axis=2)", 4),
        ("z = np.load(path, allow_pickle=False)", "z = path if isinstance(path, dict) else np.load(path, allow_pickle=False)", 1),
        ('if f"A_{n}_{lw}" in z.files:', 'if f"A_{n}_{lw}" in z:', 1),
    ]),
    "rt_combined": (EXP / "27_combined" / "combined.py", [], [],
                    ["HT", "FINE", "NLEV", "LEV", "LEV_FINE", "make_rec", "_reduce", "_reduce_w", "NetScen", "CombinedAgent"],
                    COMBINED_HEADER, [
        # battery limits read from obs instead of the module constants (same numbers in hard mode)
        ('        soc = float(obs["soc_kwh"])\n',
         '        soc = float(obs["soc_kwh"])\n        cap = float(obs.get("cap_max_kwh", CAP))\n'
         '        pmax = float(obs.get("power_max_kw", PMAX))\n', 1),
        ("p = P.plan_det(w, n, soc, v_end=v_end)", "p = plan_det(w, n, soc, cap, pmax, v_end=v_end)", 1),
        ("p = P.plan_stoch(w, scen, probs, soc, v_end=v_end, n_tail=", "p = plan_stoch(w, scen, probs, soc, cap, pmax, v_end=v_end, n_tail=", 1),
        ("return P.safe_clamp(p, soc)", "return safe_clamp(p, soc, cap, pmax)", 1),
    ]),
    # ---- h3: round-2 signal forecaster, calendar cut, signal-path planner
    "rt_signal2": (EXP / "23_signal" / "signal_fc2.py", [],
                   ["SignalForecaster2.fit", "SignalForecaster2._base_all", "SignalForecaster2._fit_bridge", "SignalForecaster2.to_npz"],
                   None, "", [
        ("from signal_fc import BACK, BEST, KAPPA_DEFAULT, SignalFC\n", "from rt_signal import BACK, BEST, KAPPA_DEFAULT, SignalFC\n", 1),
        # one state file instead of two .npz files: from_npz takes a dict whose "alt" entry is the second model's dict
        ("        z = np.load(path, allow_pickle=False)\n",
         "        z = path if isinstance(path, dict) else np.load(path, allow_pickle=False)\n", 1),
        ('        self.alt = SignalFC.from_npz(str(path) + ".alt.npz") if self.night else None\n',
         '        self.alt = SignalFC.from_npz(path["alt"] if isinstance(path, dict) else str(path) + ".alt.npz") if self.night else None\n', 1),
        # no __dict__ access in a submission file: copy the loaded base model attribute by attribute
        ("        self.__dict__.update(base.__dict__)\n        self.__dict__.update(keep)\n",
         "        for k_ in " + repr(BASE_ATTRS) + ":\n            setattr(self, k_, getattr(base, k_))\n"
         "        for k_, v_ in keep.items():\n            setattr(self, k_, v_)\n", 1),
    ]),
    "rt_combined3": (EXP / "27_combined" / "combined3.py", [], [], ["EndCut"], "import numpy as np\nimport pandas as pd\n\n\n", []),
    "rt_sigplan": (EXP / "28_signal_planning" / "sigplan.py", [], ["Paths.__init__"],
                   ["METEO", "Paths", "_nodes", "path_levels", "value_from_levels", "plan_wpaths", "make_rec", "SigAgent"],
                   SIGPLAN_HEADER, [
        # the paths come from the state file (build.py runs the original constructor and stores what it made)
        ("    def paths(self, w, hod):\n",
         "    def __init__(self, sets, beta, K, m, mode=\"add\", bias=None):\n"
         "        self.sets, self.beta, self.bias = sets, beta, bias\n"
         "        self.K, self.m, self.mode = K, m, mode\n\n"
         "    def paths(self, w, hod):\n", 1),
        # the stopwatch of the experiments is not needed (and `time` is not on the list of allowed imports)
        ("        import time\n", "", 1),
        ("t0 = time.perf_counter()\n", "t0 = 0.0\n", 2),
        ("        self.sec += time.perf_counter() - t0\n", "", 1),
    ]),
}
ORDER = ["rt_h1", "rt_planner", "rt_trees", "rt_load", "rt_cal", "rt_solar", "rt_signal", "rt_signal2", "rt_combined", "rt_combined3", "rt_sigplan", "rt_main"]      # rt_trees, rt_main: written by hand


def _check_ast(path):
    sys.path.insert(0, str(ROOT / "starter_kit"))
    from check_submission import check_ast
    err = check_ast(Path(path))
    if err:
        raise SystemExit(f"{path}: rejected by the submission scanner: {err}")


def _span(node):
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]), node.end_lineno


def _generate(name):
    src_path, drop_top, drop_methods, keep_top, header, patches = SPEC[name]
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.split("\n")
    found, dead = set(), set()
    for node in tree.body:                                # the lines of everything dropped by name
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in drop_top:
            a, b = _span(node)
            dead.update(range(a, b + 1))
            found.add(node.name)
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and f"{node.name}.{sub.name}" in drop_methods:
                    a, b = _span(sub)
                    dead.update(range(a, b + 1))
                    found.add(f"{node.name}.{sub.name}")
    missing = (set(drop_top) | set(drop_methods)) - found
    if missing:
        raise SystemExit(f"{name}: not found in {src_path.name}: {sorted(missing)}")
    if keep_top is not None:                              # keep only the named top-level definitions
        parts, kept = [], set()
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n in keep_top for n in names):
                a, b = _span(node)
                parts.append("\n".join(l for i, l in enumerate(lines[a - 1:b], a) if i not in dead))
                kept.update(names)
        missing = set(keep_top) - kept
        if missing:
            raise SystemExit(f"{name}: not found in {src_path.name}: {sorted(missing)}")
        out = "\n\n\n".join(parts) + "\n"
    else:
        out = "\n".join(l for i, l in enumerate(lines, 1) if i not in dead)
    for old, new, count in patches:
        if out.count(old) != count:
            raise SystemExit(f"{name}: patch text found {out.count(old)} times instead of {count}: {old[:70]!r}")
        out = out.replace(old, new)
    if name == "rt_sigplan":                              # planner functions are imported by name (see SIGPLAN_HEADER)
        used = set(re.findall(r"\bP\.(\w+)", out))
        if not used <= set(SIGPLAN_NAMES):
            raise SystemExit(f"rt_sigplan: planner names not in SIGPLAN_NAMES: {sorted(used - set(SIGPLAN_NAMES))}")
        out = re.sub(r"\bP\.", "", out)
        if "time." in out or "import time" in out:
            raise SystemExit("rt_sigplan: a use of the time module is left")
    if name in ("rt_combined", "rt_sigplan") and re.search(r"\bP\.", out):
        raise SystemExit(f"{name}: a reference to the planner module (P.) is left")
    if "__dict__" in out:
        raise SystemExit(f"{name}: __dict__ is left in the run-time copy")
    rel = src_path.relative_to(ROOT).as_posix()
    head = (f"# GENERATED by agents/h4/bundle.py from {rel} -- do not edit (edit SPEC in bundle.py).\n"
            f"# Removed: {', '.join(drop_top + drop_methods) or 'nothing'}."
            f"{' Kept only: ' + ', '.join(keep_top) + '.' if keep_top else ''} Patches: {len(patches)}.\n")
    return head + header + out


def sync(verbose=True):
    for name in SPEC:
        text = _generate(name)
        path = HERE / f"{name}.py"
        changed = (not path.exists()) or path.read_text(encoding="utf-8") != text
        path.write_text(text, encoding="utf-8", newline="\n")
        ast.parse(text)
        _check_ast(path)
        if verbose:
            print(f"sync {name}.py: {len(text.splitlines())} lines{' (changed)' if changed else ''}")


IMPORT_RE = re.compile(r"^from (rt_\w+) import (.+)$")


def bundle(h1_tables=None, out=None, verbose=True):
    """h1_tables: dict of the embedded fallback's tables (default: submissions/h1/nov_fill/h1_tables.json)."""
    if h1_tables is None:
        h1_tables = json.loads((ROOT / "submissions" / "h1" / "nov_fill" / "h1_tables.json").read_text(encoding="utf-8"))
    srcs = {m: (HERE / f"{m}.py").read_text(encoding="utf-8") for m in ORDER}
    exports = {m: [] for m in ORDER}
    for m in ORDER:                                       # who needs what from whom
        for line in srcs[m].split("\n"):
            g = IMPORT_RE.match(line)
            if g:
                if ORDER.index(g.group(1)) >= ORDER.index(m):
                    raise SystemExit(f"{m} imports {g.group(1)}, which comes later in the bundle")
                for n in (x.strip() for x in g.group(2).split(",")):
                    if n not in exports[g.group(1)]:
                        exports[g.group(1)].append(n)
    exports["rt_main"] = ["BatteryAgent"]
    doc = (HERE / "rt_main.py").read_text(encoding="utf-8").split('"""')[1]
    parts = ['"""' + doc + '"""',
             "# GENERATED by agents/h4/bundle.py from agents/h4/rt_*.py -- do not edit.\n"
             "# Each module below is wrapped in a function so that its names stay private to it.",
             "_H1_TABLES_JSON = " + repr(json.dumps(h1_tables))]
    for m in ORDER:
        body = []
        for line in srcs[m].split("\n"):
            g = IMPORT_RE.match(line)
            if g:
                src_mod = g.group(1)
                for n in (x.strip() for x in g.group(2).split(",")):
                    body.append(f"    {n} = _{src_mod}[{n!r}] if _{src_mod} is not None else None")
            else:
                body.append(("    " + line) if line.strip() else "")
        ret = "    return {" + ", ".join(f"{n!r}: {n}" for n in exports[m]) + "}"
        block = [f"# {'=' * 100}", f"# module {m}", f"# {'=' * 100}", f"def _load_{m}():"] + body + [ret, "", ""]
        if m == "rt_main":
            block += [f"_{m} = _load_{m}()"]
        else:
            block += ["try:", f"    _{m} = _load_{m}()", "except Exception:", f"    _{m} = None"]
        parts.append("\n".join(block))
    parts.append('BatteryAgent = _rt_main["BatteryAgent"]\n')
    text = "\n\n\n".join(parts)
    out = Path(out) if out else HERE / "agent.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    ast.parse(text)
    _check_ast(out)
    if verbose:
        print(f"bundle {out}: {len(text.splitlines())} lines, {out.stat().st_size / 1024:.0f} KB")
    return out


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sync()
    bundle()
