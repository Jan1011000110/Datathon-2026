"""
experiments/29_stress_hard/eps_margin.py
Safety margin of the final clamp (EPS, kW) against a server that sends a rounded state of charge.
    eps_margin.py <h1|h2>
The module global EPS of the packaged agent is overwritten in memory; nothing on disk changes.
"""
import sys

import pandas as pd

import lib as L

kind = sys.argv[1]
folds = L.clean_folds()
mod = L.agent_module(kind)
rows = []
for eps in (1e-4, 0.01, 0.05, 0.6):
    mod.EPS = eps
    for fold in ("A", "B", "C"):
        for name, fn in (("clean", None), ("soc rounded to 0.01 kWh", L.o_soc("round2")), ("soc rounded to 1 kWh", L.o_soc("round0")),
                         ("soc float noise outside [0, cap]", L.o_soc("out"))):
            if fold == "A" and name != "clean":
                continue
            r = L.run(L.load_agent(kind, L.state_path(kind, fold)), folds[fold], fn)
            rows.append(dict(agent=kind, eps=eps, fold=fold, scenario=name, score_raw=r["score_raw"], infractions=r["infractions"],
                             fallbacks=r["fallbacks"]))
            print(rows[-1], flush=True)
mod.EPS = 1e-4
d = pd.DataFrame(rows)
d.to_csv(L.HERE / "out" / f"eps_{kind}.csv", index=False)
print(d.pivot_table(index=["scenario", "fold"], columns="eps", values="infractions"))
print(d[d.scenario == "clean"].pivot_table(index="fold", columns="eps", values="score_raw").round(5))
