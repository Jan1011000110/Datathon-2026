# Usage: .venv/Scripts/python.exe experiments/25_integration/build_probe_h2.py
"""Builds submissions/probe_h2/submission.zip: the h2 agent, unchanged, inside a wrapper that lets it
play the site's free 48-hour quick validation and then stops on purpose with a report in the error
text. For "Validacio Rapida Mode Dificil" ONLY; never submit it to the ranking (it fails on purpose).

The report says (a) whether h2 built correctly on the server and how fast it runs there, and
(b) whether a few assumptions about the server's observations hold. It reports counts, yes/no
checks and timings; no price, CO2, load or solar value of the test."""
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submissions" / "h2"
OUT = ROOT / "submissions" / "probe_h2"

WRAPPER = r'''


# ==================================================================================================
# PROBE WRAPPER -- quick validation only. Everything above this line is the h2 agent, unchanged.
# ==================================================================================================
import time as _ptime

_ProbedAgent = BatteryAgent
_P_ETA = 0.9 ** 0.5


def _pnum(x):
    try:
        x = float(x)
        return x if (x == x and abs(x) != float("inf")) else None
    except Exception:
        return None


class BatteryAgent:
    def __init__(self):
        t0 = _ptime.perf_counter()
        self.a = _ProbedAgent()
        self.t_init = _ptime.perf_counter() - t0
        self.n = 0
        self.dt = []
        self.lens = []
        self.prev = None                 # (soc, command, hour) of the previous step
        self.soc_dev, self.soc_n = 0.0, 0
        self.ts_ok, self.prev_ts = True, None
        self.ratio = []                  # solar reading / irradiance reading, sunlit hours
        self.night_same, self.night_n = 0, 0
        self.wind_nonzero = 0
        self.panel_gt, self.mid_n = 0, 0
        self.bad_readings = 0
        self.prev_load = None
        self.cmd = [0, 0, 0]             # charge, discharge, idle
        self.notes = []

    def _checks(self, obs):
        n1 = len(obs.get("preus_properes_24h"))
        n2 = len(obs.get("co2_properes_24h"))
        self.lens.append(n1 if n1 == n2 else -1)
        soc = _pnum(obs.get("soc_kwh"))
        if self.prev is not None and soc is not None:
            s0, p = self.prev
            exp = s0 + (p * _P_ETA if p > 0 else p / _P_ETA)
            self.soc_dev = max(self.soc_dev, abs(soc - exp))
            self.soc_n += 1
        ts = str(obs.get("timestamp"))
        hour = int(ts[11:13])
        if self.prev_ts is not None:
            import pandas as _ppd
            if (_ppd.Timestamp(ts) - _ppd.Timestamp(self.prev_ts)) != _ppd.Timedelta(hours=1):
                self.ts_ok = False
        self.prev_ts = ts
        last = obs.get("noves_dades_5min")
        vals = {k: _pnum(last.get(k)) for k in ("ultim_consum_kw", "ultima_solar_kw", "irradiancia_wm2",
                                                 "temp_ambient_c", "temp_panel_c", "vent_ms")}
        self.bad_readings += sum(1 for v in vals.values() if v is None)
        load, pv, irr = vals["ultim_consum_kw"], vals["ultima_solar_kw"], vals["irradiancia_wm2"]
        if irr is not None and pv is not None and irr > 200.0:
            self.ratio.append(pv / irr)
        if 1 <= hour <= 6 and load is not None and self.prev_load is not None:
            self.night_n += 1
            self.night_same += int(abs(load - self.prev_load) < 1e-6)
        self.prev_load = load
        if vals["vent_ms"] not in (None, 0.0):
            self.wind_nonzero += 1
        if 12 <= hour <= 16 and vals["temp_panel_c"] is not None and vals["temp_ambient_c"] is not None:
            self.mid_n += 1
            self.panel_gt += int(vals["temp_panel_c"] > vals["temp_ambient_c"])
        return soc

    def _report(self):
        d = sorted(self.dt)
        runs, out = [], []
        for x in self.lens:
            if runs and runs[-1][0] == x:
                runs[-1][1] += 1
            else:
                runs.append([x, 1])
        for x, c in runs:
            out.append("%dx%d" % (x, c) if c > 1 else str(x))
        r = sorted(self.ratio)
        try:
            import numpy as _pnp
            import pandas as _ppd
            vers = "numpy=%s pandas=%s" % (_pnp.__version__, _ppd.__version__)
        except Exception as e:
            vers = "versions n/a (%s)" % repr(e)[:60]
        a = self.a
        parts = [
            "steps=%d" % self.n,
            "init_s=%.2f" % self.t_init,
            "ms_per_step mean=%.2f median=%.2f max=%.1f" % (1000 * sum(d) / len(d), 1000 * d[len(d) // 2], 1000 * d[-1]),
            vers,
            "build_error=%r" % (getattr(a, "build_error", "?"),),
            "load_mode=%r" % (getattr(a, "load_mode", "?"),),
            "fallbacks=%r" % (getattr(a, "fallbacks", "?"),),
            "last_error=%r" % (getattr(a, "last_error", "?"),),
            "list_len(value x steps)=" + ",".join(out),
            "soc_max_dev_kwh=%.2e (n=%d)" % (self.soc_dev, self.soc_n),
            "ts_consecutive=%s" % self.ts_ok,
            "solar/irradiance median=%s (n=%d)" % ("%.2f" % r[len(r) // 2] if r else "n/a", len(r)),
            "night_load_same_as_previous=%d/%d" % (self.night_same, self.night_n),
            "wind_nonzero=%d" % self.wind_nonzero,
            "panel_warmer_than_air_midday=%d/%d" % (self.panel_gt, self.mid_n),
            "unusable_readings=%d" % self.bad_readings,
            "commands charge/discharge/idle=%d/%d/%d" % tuple(self.cmd),
        ] + self.notes
        return "H2-PROBE >>> " + "; ".join(parts) + " <<<"

    def act(self, obs: dict) -> float:
        self.n += 1
        soc = None
        try:
            soc = self._checks(obs)
        except Exception as e:
            self.notes.append("check_error@%d=%s" % (self.n, repr(e)[:80]))
        t0 = _ptime.perf_counter()
        p = float(self.a.act(obs))
        self.dt.append(_ptime.perf_counter() - t0)
        self.cmd[0 if p > 0 else (1 if p < 0 else 2)] += 1
        self.prev = (soc, p) if soc is not None else None
        if self.n >= 48 or (self.lens and self.lens[-1] == 1):
            raise RuntimeError(self._report())
        return p
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = (SRC / "agent.py").read_text(encoding="utf-8")
    assert src.rstrip().endswith('BatteryAgent = _rt_main["BatteryAgent"]'), "h2 bundle layout changed"
    (OUT / "agent.py").write_text(src.rstrip() + WRAPPER, encoding="utf-8", newline="\n")
    shutil.copyfile(SRC / "h2_state.npz", OUT / "h2_state.npz")
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / "agent.py", "agent.py")
        z.write(OUT / "h2_state.npz", "h2_state.npz")
    sys.path.insert(0, str(ROOT / "starter_kit"))
    import check_submission as cs
    print("AST check:", cs.check_ast(OUT / "agent.py") or "clean")
    print("built", OUT / "submission.zip", (OUT / "submission.zip").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
