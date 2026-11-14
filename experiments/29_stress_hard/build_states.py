"""
experiments/29_stress_hard/build_states.py
Fold-out states for the packaged agents (fitted on the train rows OUTSIDE a fold), so that stress runs on a
fold are honest.   usage: build_states.py B [C ...]
    states/h1_<fold>.json     tables of h1 (November filled from neighbours, as the shipped nov_fill variant)
    states/h2_<fold>.npz      state of h2 (contains its own h1 tables fitted outside the fold)
"""
import importlib.util
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from src.hard import get_folds, train_complement  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    folds = get_folds()
    h1b = _load("h1_build", ROOT / "agents" / "h1" / "build.py")
    h2b = _load("h2_build", ROOT / "agents" / "h2" / "build.py")
    for name in sys.argv[1:]:
        t0 = time.time()
        p1 = HERE / "states" / f"h1_{name}.json"
        if not p1.exists():
            h1b.write_tables(train_complement(folds[name]), p1, fill_missing="neighbours")
        print(name, "h1 tables", round(time.time() - t0, 1), "s", flush=True)
        p2 = HERE / "states" / f"h2_{name}.npz"
        if not p2.exists():
            arrays, _, _ = h2b.build_state(folds[name])
            h2b.write_state(arrays, p2)
        print(name, "h2 state", round(time.time() - t0, 1), "s", flush=True)
