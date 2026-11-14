"""Clock-change days fall only in test weeks (26 Oct 2025, 29 Mar 2026). How do the shipped agents behave if the
timestamps there have a repeated or a missing 02:00?  Usage: dst_check.py <repo> <dir with extracted h3j/, h3w/>"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo, work = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
from src import sim as S  # noqa: E402
from src.hard import get_folds, simulate  # noqa: E402


def load(p, n):
    spec = importlib.util.spec_from_file_location(n, str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_like(df, last_fold_day=14):
    """Fold weeks relabelled to the last 7 days of each month, cut after July: the layout of the real test."""
    shifted = [(ts + pd.offsets.MonthEnd(0)).normalize() - pd.Timedelta(days=last_fold_day - ts.day) + pd.Timedelta(hours=ts.hour) for ts in df.index]
    d = df.copy()
    d.index = pd.DatetimeIndex(shifted, name=df.index.name)
    return d[d.index <= pd.Timestamp("2026-07-31 23:00")]


def variants(d):
    out = {"as is": d}
    idx = d.index
    oct_w = (idx >= "2025-10-25") & (idx <= "2025-10-31 23:00")
    mar_w = (idx >= "2026-03-25") & (idx <= "2026-03-31 23:00")
    # realistic for local-time data: one row repeated / one row missing, every other label untouched
    i = int(np.flatnonzero(idx == pd.Timestamp("2025-10-26 02:00"))[0])
    out["26 Oct: 02:00 twice (one extra row)"] = pd.concat([d.iloc[:i + 1], d.iloc[i:i + 1], d.iloc[i + 1:]])
    j = int(np.flatnonzero(idx == pd.Timestamp("2026-03-29 02:00"))[0])
    out["29 Mar: 02:00 missing (one row less)"] = pd.concat([d.iloc[:j], d.iloc[j + 1:]])
    both = pd.concat([d.iloc[:i + 1], d.iloc[i:i + 1], d.iloc[i + 1:j], d.iloc[j + 1:]])
    out["both of the above"] = both
    # stress: the labels slide by an hour for the rest of the week while the data stay put
    a = idx.to_series().reset_index(drop=True)
    sel = oct_w & (idx >= "2025-10-26 03:00")
    a[sel] = a[sel] - pd.Timedelta(hours=1)
    sel = mar_w & (idx >= "2026-03-29 02:00")
    a[sel] = a[sel] + pd.Timedelta(hours=1)
    s = d.copy()
    s.index = pd.DatetimeIndex(a.values, name=idx.name)
    out["stress: labels slide an hour to the week's end, 168 rows kept (March week ends 1 Apr 00:00)"] = s
    return out


print("=== HARD MODE (fold B relabelled to month ends)")
hard = variants(test_like(get_folds()["B"]))
mods = {"h3j": load(work / "h3j" / "agent.py", "h3j_zip"), "h3w": load(work / "h3w" / "agent.py", "h3w_zip")}
for name, m in mods.items():
    base = None
    for label, d in hard.items():
        try:
            a = m.BatteryAgent()
            r = simulate(a, d)
            base = r["score"] if base is None else base
            print(f"{name} | {label}: {len(d)} h, score {r['score']:.3f} ({r['score'] - base:+.3f}), infractions {r['infractions']}, answered {getattr(a, 'answered', None)}")
        except Exception as e:
            print(f"{name} | {label}: EXCEPTION {type(e).__name__}: {str(e)[:200]}")

print("=== STANDARD MODE (fold B of the filled table relabelled to month ends)")
std = variants(test_like(S.get_folds("filled")["B"]))
smods = {"v2": load(repo / "agents/v2/agent.py", "v2_ref"), "v2w": load(repo / "agents/v2w/agent.py", "v2w_ref")}
for name, m in smods.items():
    base = None
    for label, d in std.items():
        try:
            r = S.simulate(m.BatteryAgent(), d)
            base = r["score"] if base is None else base
            print(f"{name} | {label}: {len(d)} h, score {r['score']:.3f} ({r['score'] - base:+.3f}), infractions {r['infractions']}")
        except Exception as e:
            print(f"{name} | {label}: EXCEPTION {type(e).__name__}: {str(e)[:200]}")
