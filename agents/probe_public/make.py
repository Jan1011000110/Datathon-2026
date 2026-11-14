# Usage: .venv/Scripts/python.exe agents/probe_public/make.py
"""
agents/probe_public/make.py  --  build the "public-max" probe from the packaged h3w.

NOT a candidate for the jury selection. It shows how far the public leaderboard (the first 84 hours of each test
week) moves when energy is shifted between the two halves of a week, with no better forecasting or planning:

  * first 84 hours of each week: h3's forecasters and planner, with the plan stopping at hour 84 (battery emptied by then);
  * last 84 hours of each week:  the battery is refilled from the grid and held full for the next week
                                 (not in the last week of the run).

Nothing is refitted. The package is submissions/h3w/agent.py with the marked edits below, and h3w's state file with
one more configuration key ("public_max": true). `patch_source` and `patch_config` are importable for tests.
Writes submissions/probe_public/{agent.py, probe_public_state.npz, submission.zip}.
"""
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submissions" / "h3w"
OUT = ROOT / "submissions" / "probe_public"
STATE_NAME = "probe_public_state.npz"

HEADER = '''PROBE of the public/private split, "public-max" variant. NOT a candidate for the jury selection.

It shows how far the public leaderboard (the first 84 hours of each test week) can be moved by shifting energy
between the two halves of a week, with no better forecasting or planning at all:
  * first 84 hours of each week: h3's forecasters and planner, with the plan stopping at hour 84, so the battery
    is emptied by then;
  * last 84 hours of each week: the battery is refilled from the grid and held full for the next week (not in
    the last week of the run).
Over the whole run, and on the last half of each week, this is far worse than h3. The edits are marked
[public-max] in the code. Built from h3w by agents/probe_public/make.py; what follows describes h3w:

'''

HELPERS = '''

    def week_pos(ts, rule):                                  # [public-max]
        """Hours since the start of the test week `ts` is in (0..167; below 0 outside a test week)."""
        return 167 - int((week_end(ts, rule) - ts) / pd.Timedelta(hours=1))


    def plan_end(ts, rule, public_max=False):                # [public-max]
        """Last hour the plan may look at: the end of the week, or, in the public-max probe, hour 83 of the week
        while the week is in its first half."""
        end = week_end(ts, rule)
        if public_max and week_pos(ts, rule) < HALF:
            return end - pd.Timedelta(hours=HALF)
        return end
'''

OVERRIDE = '''            if self.public_max:                          # [public-max] second half of the week: refill and hold
                try:
                    ts = obs["timestamp"]
                    if not isinstance(ts, pd.Timestamp):
                        ts = pd.Timestamp(str(ts))
                    if ts.tzinfo is not None:
                        ts = ts.tz_localize(None)
                    ahead = [e for e in self.ends if e >= ts]
                    last_week = bool(ahead) and ahead[0] <= week_end(ts, self.week_rule)
                    if week_pos(ts, self.week_rule) >= HALF and not last_week:
                        p = pmax
                except Exception:
                    pass
'''

EDITS = [
    # (old, new) -- each `old` must occur exactly once in h3w's agent.py
    ('"""\nh3 with every test week treated as its own run:',
     '"""\n' + HEADER + 'h3 with every test week treated as its own run:'),
    ('    STATE_NAME = "h3w_state.npz"\n',
     '    STATE_NAME = "' + STATE_NAME + '"\n'
     '    HALF = 84                    # [public-max] the public leaderboard counts the first HALF hours of each test week\n'),
    ('        return ts.normalize().replace(day=day) + pd.Timedelta(hours=23)\n',
     '        return ts.normalize().replace(day=day) + pd.Timedelta(hours=23)\n' + HELPERS),
    ('            r = int((week_end(ts, self.rule) - ts) / pd.Timedelta(hours=1)) + 1      # rows left in the week, this one included\n',
     '            r = int((plan_end(ts, self.rule, getattr(self, "public_max", False)) - ts) / pd.Timedelta(hours=1)) + 1   # [public-max] rows left to plan for\n'),
    ('        w_fc = WeekRule(sig2, cfg.get("beyond_scale", 1.0), cfg.get("week_rule", "month_end"), cfg.get("week_cut", False))\n',
     '        w_fc = WeekRule(sig2, cfg.get("beyond_scale", 1.0), cfg.get("week_rule", "month_end"), cfg.get("week_cut", False))\n'
     '        w_fc.public_max = bool(cfg.get("public_max", False))     # [public-max]\n'),
    ('            self.week_rule = "month_end"\n',
     '            self.week_rule = "month_end"\n'
     '            self.public_max = False          # [public-max] plan to hour 84 of the week, refill in the second half\n'),
    ('                self.week_rule = _rule(cfg.get("week_rule", "month_end"))\n',
     '                self.week_rule = _rule(cfg.get("week_rule", "month_end"))\n'
     '                self.public_max = bool(cfg.get("public_max", False))     # [public-max]\n'),
    ('                wk = week_end(ts, self.week_rule)\n',
     '                wk = plan_end(ts, self.week_rule, self.public_max)       # [public-max]\n'),
    ('            eps = self.clamp_eps\n            max_c = max(0.0, min(pmax, (cap - soc) / ETA - eps))\n',
     OVERRIDE + '            eps = self.clamp_eps\n            max_c = max(0.0, min(pmax, (cap - soc) / ETA - eps))\n'),
]


def patch_source(src: str) -> str:
    """h3w's agent.py (text, any newline style) -> the public-max probe's agent.py."""
    nl = "\r\n" if "\r\n" in src else "\n"
    out = src.replace("\r\n", "\n")
    for old, new in EDITS:
        if out.count(old) != 1:
            raise ValueError(f"expected exactly one match, found {out.count(old)}: {old[:70]!r}")
        out = out.replace(old, new, 1)
    return out.replace("\n", nl)


def patch_config(arrays: dict, **extra) -> dict:
    """A copy of a state dict with "public_max": true (and any other key given) added to its configuration."""
    out = dict(arrays)
    cfg = json.loads(bytes(np.asarray(arrays["config"], np.uint8)).decode("utf-8"))
    cfg["public_max"] = True
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
        print("config:", bytes(z["config"]).decode("utf-8"))
        same = sum(1 for k in z.files if k != "config" and np.array_equal(z[k], arrays[k], equal_nan=z[k].dtype.kind == "f"))
        print(f"state: {len(z.files)} arrays, {same} identical to h3w's, 1 changed (config)")
    with zipfile.ZipFile(OUT / "submission.zip") as f:
        print("zip:", [(i.filename, i.file_size) for i in f.infolist()], f"{(OUT / 'submission.zip').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
