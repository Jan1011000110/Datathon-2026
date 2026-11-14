"""Regression checks for experiments/02_stress/candidate/agent.py (run after every edit of it):
  1. with the three new guard rules switched off it must reproduce agents/v1 action for action
     on every scenario of the stress suite (so the interface hardening changes nothing);
  2. with them on it must equal the lab agent (guard_lab.LabAgent) with the same parameters;
  3. how often does each guard rule fire on the official train table.

    .venv/Scripts/python.exe experiments/02_stress/check_candidate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *          # noqa: F401,F403
from guard_lab import LabAgent
from src.sim import get_folds


def main():
    v1 = load_agent_module(ROOT / "agents" / "v1" / "agent.py", "chk_v1")
    off = load_agent_module(HERE / "candidate" / "agent.py", "chk_off")
    off.DROP_RATIO, off.PERSIST_X, off.CALENDAR_EXIT = 0.0, 2, False
    on = load_agent_module(HERE / "candidate" / "agent.py", "chk_on")
    lab = dict(drop_ratio=on.DROP_RATIO, drop_skip6=True, pers_a=on.PERSIST_MIN, pers_b=on.PERSIST_X, cal_exit=on.CALENDAR_EXIT)
    S = suite("test", ("A", "B", "C"))
    w1 = w2 = 0.0
    for sc in S:
        a = run(v1.BatteryAgent(), sc["df"], sc["obs_load"], sc["obs_pv"], trace=True)["trace"].p_kw.to_numpy()
        b = run(off.BatteryAgent(), sc["df"], sc["obs_load"], sc["obs_pv"], trace=True)["trace"].p_kw.to_numpy()
        c = run(on.BatteryAgent(), sc["df"], sc["obs_load"], sc["obs_pv"], trace=True)["trace"].p_kw.to_numpy()
        d = run(LabAgent(**lab), sc["df"], sc["obs_load"], sc["obs_pv"], trace=True)["trace"].p_kw.to_numpy()
        w1 = max(w1, float(np.abs(a - b).max()))
        w2 = max(w2, float(np.abs(c - d).max()))
    print("1. candidate with new rules off vs v1  : max |action difference| over %d runs = %.3g kW" % (len(S), w1))
    print("2. candidate vs lab agent, same params : max |action difference| over %d runs = %.3g kW" % (len(S), w2))

    # 3. rule firings on the official train table (all train days, NaN rows dropped: 5,993 h)
    df = get_folds("official")["ALL"][CORE]
    ag = on.BatteryAgent()
    fired = []
    orig = ag._discharge_limits

    def spy(obs, horizon):
        prev = ag.prev_net
        cap, nb = orig(obs, horizon)
        last = obs["noves_dades_5min"]
        net = last["ultim_consum_kw"] - last["ultima_solar_kw"]
        if cap == 0.0 and nb > 0:
            fired.append((obs["timestamp"], "zero reading" if net < on.OUTAGE_KW else "drop %.0f -> %.0f kW" % (prev, net)))
        return cap, nb

    ag._discharge_limits = spy
    r1 = run(ag, df)
    r0 = run(v1.BatteryAgent(), df)
    drops = [f for f in fired if f[1].startswith("drop")]
    print("3. official train table, %d h: outage rule fired %d times (zero reading %d, sudden drop %d)" % (
        len(df), len(fired), len(fired) - len(drops), len(drops)))
    for f in drops:
        print("     ", f[0], f[1])
    print("   score v1 %.4f, candidate %.4f, oracle %.4f" % (r0["raw"], r1["raw"], oracle_score(df)))


if __name__ == "__main__":
    main()
