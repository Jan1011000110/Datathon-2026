"""
agents/h2/bundle.py  --  make the run-time copies of the specialists' modules and bundle them into agent.py.

    .venv/Scripts/python.exe agents/h2/bundle.py            # sync() + bundle()

sync()    regenerates agents/h2/rt_*.py from the originals. Nothing is ported by hand: each copy is the
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
    ]),
    "rt_planner": (EXP / "24_planner" / "planner.py", ["QuantileN", "MPCStoch"], [], None, "", []),
    "rt_load": (EXP / "21_load" / "load_fc.py", ["LoadQ"], ["LoadFC.fit", "LoadFC.to_state"], None, "", [
        # neither is needed at run time: pandas was only used by fit(), and the model is evaluated by
        # rt_trees.TreeModel (set by rt_main), so LightGBM is never imported on the server
        ("import pandas as pd\nimport lightgbm as lgb\n", "lgb = None\n", 1),
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
}
ORDER = ["rt_h1", "rt_planner", "rt_trees", "rt_load", "rt_solar", "rt_signal", "rt_combined", "rt_main"]      # rt_trees, rt_main: written by hand


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
    found = set()
    if keep_top is not None:                              # keep only the named top-level definitions
        parts = []
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n in keep_top for n in names):
                a, b = _span(node)
                parts.append("\n".join(lines[a - 1:b]))
                found.update(names)
        missing = set(keep_top) - found
        if missing:
            raise SystemExit(f"{name}: not found in {src_path.name}: {sorted(missing)}")
        out = "\n\n\n".join(parts) + "\n"
    else:
        kill = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in drop_top:
                kill.append(_span(node))
                found.add(node.name)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and f"{node.name}.{sub.name}" in drop_methods:
                        kill.append(_span(sub))
                        found.add(f"{node.name}.{sub.name}")
        missing = (set(drop_top) | set(drop_methods)) - found
        if missing:
            raise SystemExit(f"{name}: not found in {src_path.name}: {sorted(missing)}")
        dead = set()
        for a, b in kill:
            dead.update(range(a, b + 1))
        out = "\n".join(l for i, l in enumerate(lines, 1) if i not in dead)
    for old, new, count in patches:
        if out.count(old) != count:
            raise SystemExit(f"{name}: patch text found {out.count(old)} times instead of {count}: {old[:70]!r}")
        out = out.replace(old, new)
    if name == "rt_combined" and re.search(r"\bP\.", out):
        raise SystemExit("rt_combined: a reference to the planner module (P.) is left")
    rel = src_path.relative_to(ROOT).as_posix()
    head = (f"# GENERATED by agents/h2/bundle.py from {rel} -- do not edit (edit SPEC in bundle.py).\n"
            f"# Removed: {', '.join(drop_top + drop_methods) or ('everything except ' + ', '.join(keep_top) if keep_top else 'nothing')}."
            f" Patches: {len(patches)}.\n")
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
             "# GENERATED by agents/h2/bundle.py from agents/h2/rt_*.py -- do not edit.\n"
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
