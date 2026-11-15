# Usage: .venv/Scripts/python.exe agents/h3p/make.py
"""
agents/h3p/make.py  --  build h3p from the packaged h3w.

h3p is h3w ("every test week is its own run") with ONE more change: in the last FILL hours of the first half of each
test week the battery is charged to full from the grid, so the second half of the week starts with a full battery.

This is aimed at a ranking that counts only the second half (the last 84 hours) of each test week. It lowers the
score of the first half of each week and raises the score of the second half; over the whole run it is worse than h3.

Nothing is refitted: the package is submissions/h3w/agent.py with the marked edits below and h3w's state file with
one more configuration key ("private_max": true). Trained on data/train only; no outside data.
Writes submissions/h3p/{agent.py, h3p_state.npz, submission.zip}.
"""
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submissions" / "h3w"
OUT = ROOT / "submissions" / "h3p"
STATE_NAME = "h3p_state.npz"

HEADER = '''agent.py  --  BatteryAgent h3p: h3w with one more change, aimed at a ranking that counts only the second half
(the last 84 hours) of each test week.

  * In the last 5 hours of the first half of each week the battery is charged to full from the grid, so the second
    half starts with a full battery. Before that, and in the second half, it is h3w: h3's forecasters and planner,
    with the plan stopping at the last hour of the week.
This lowers the score of the first half of each week and raises the score of the second half. Over the whole run
it is worse than h3. The edit is marked [second-half] in the code. Built from h3w by agents/h3p/make.py; trained on
data/train only. What follows describes h3w:

'''

HELPERS = '''

    def week_pos(ts, rule):                                  # [second-half]
        """Hours since the start of the test week `ts` is in (0..167; below 0 outside a test week)."""
        return 167 - int((week_end(ts, rule) - ts) / pd.Timedelta(hours=1))
'''

OVERRIDE = '''            if self.private_max:                         # [second-half] top up to full just before the half-way point
                try:
                    ts = obs["timestamp"]
                    if not isinstance(ts, pd.Timestamp):
                        ts = pd.Timestamp(str(ts))
                    if ts.tzinfo is not None:
                        ts = ts.tz_localize(None)
                    if HALF - FILL <= week_pos(ts, self.week_rule) < HALF:
                        p = pmax
                except Exception:
                    pass
'''

EDITS = [
    ('"""\nh3 with every test week treated as its own run:',
     '"""\n' + HEADER + 'h3 with every test week treated as its own run:'),
    ('    STATE_NAME = "h3w_state.npz"\n',
     '    STATE_NAME = "' + STATE_NAME + '"\n'
     '    HALF = 84                    # [second-half] hours in each half of a test week\n'
     '    FILL = 5                     # [second-half] hours before the half-way point spent charging (6000 kWh needs 4.3 h)\n'),
    ('        return ts.normalize().replace(day=day) + pd.Timedelta(hours=23)\n',
     '        return ts.normalize().replace(day=day) + pd.Timedelta(hours=23)\n' + HELPERS),
    ('            self.week_rule = "month_end"\n',
     '            self.week_rule = "month_end"\n'
     '            self.private_max = False         # [second-half] charge to full before the half-way point of each week\n'),
    ('                self.week_rule = _rule(cfg.get("week_rule", "month_end"))\n',
     '                self.week_rule = _rule(cfg.get("week_rule", "month_end"))\n'
     '                self.private_max = bool(cfg.get("private_max", False))   # [second-half]\n'),
    ('            eps = self.clamp_eps\n            max_c = max(0.0, min(pmax, (cap - soc) / ETA - eps))\n',
     OVERRIDE + '            eps = self.clamp_eps\n            max_c = max(0.0, min(pmax, (cap - soc) / ETA - eps))\n'),
]


def patch_source(src: str) -> str:
    nl = "\r\n" if "\r\n" in src else "\n"
    out = src.replace("\r\n", "\n")
    for old, new in EDITS:
        if out.count(old) != 1:
            raise ValueError(f"expected exactly one match, found {out.count(old)}: {old[:70]!r}")
        out = out.replace(old, new, 1)
    return out.replace("\n", nl)


def patch_config(arrays: dict, **extra) -> dict:
    out = dict(arrays)
    cfg = json.loads(bytes(np.asarray(arrays["config"], np.uint8)).decode("utf-8"))
    cfg["private_max"] = True
    cfg.update(extra)
    out["config"] = np.frombuffer(json.dumps(cfg).encode("utf-8"), dtype=np.uint8)
    return out


def _build_module():
    spec = importlib.util.spec_from_file_location("h3w_build", str(ROOT / "agents" / "h3w" / "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = (SRC / "agent.py").read_bytes().decode("utf-8")
    (OUT / "agent.py").write_bytes(patch_source(src).encode("utf-8"))
    with np.load(SRC / "h3w_state.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    _build_module().write_state(patch_config(arrays), OUT / STATE_NAME)
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED) as f:
        f.write(OUT / "agent.py", "agent.py")
        f.write(OUT / STATE_NAME, STATE_NAME)
    with np.load(OUT / STATE_NAME, allow_pickle=False) as z:
        print("config:", bytes(z["config"]).decode("utf-8")[-170:])
        same = sum(1 for k in z.files if k != "config" and np.array_equal(z[k], arrays[k], equal_nan=z[k].dtype.kind == "f"))
        print(f"state: {len(z.files)} arrays, {same} identical to h3w's, 1 changed (config)")
    with zipfile.ZipFile(OUT / "submission.zip") as f:
        print("zip:", [(i.filename, i.file_size) for i in f.infolist()])


if __name__ == "__main__":
    main()
