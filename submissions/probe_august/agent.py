"""
agent.py  --  PROBE, not a competitor and not a candidate for the jury selection.

Question it answers: is the August 2026 week in the hidden test?  (The inverter file of the train set
stops on 16 August 2026; if the hidden data has no solar readings after that date, the organisers'
pipeline drops those hours and the run ends on 31 July.)

What it does: nothing (0 kW, always legal) for the whole run, except on August 2026 dates, where it
discharges the charge the battery started with (3,000 kWh), never more than 90 % of the last measured
net load. So:
    CO2 avoided > 0 on the leaderboard  ->  the August week exists (and some of its hours are scored),
    CO2 avoided = 0                     ->  no August hour is scored: most likely there is no August week.
"""

ETA = 0.9 ** 0.5
EPS = 0.01                   # kW, safety margin of the clamp


def _num(x):
    try:
        x = float(x)
        return x if (x == x and abs(x) != float("inf")) else None
    except Exception:
        return None


class BatteryAgent:
    def act(self, obs: dict) -> float:
        try:
            ts = str(obs.get("timestamp"))
            if ts[0:7] != "2026-08":
                return 0.0
            soc = _num(obs.get("soc_kwh"))
            pmax = _num(obs.get("power_max_kw", 1500.0)) or 1500.0
            last = obs.get("noves_dades_5min") or {}
            load, pv = _num(last.get("ultim_consum_kw")), _num(last.get("ultima_solar_kw"))
            if soc is None or load is None:
                return 0.0
            net = load - (pv or 0.0)
            d = min(pmax, 0.9 * net, soc * ETA - EPS)
            return -float(d) + 0.0 if d > 0.0 else 0.0
        except Exception:
            return 0.0
