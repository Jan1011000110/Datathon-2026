"""Why is the public-leaderboard baseline ~23% below what the train set predicts?

Public half (1,008 h nominal), reconstructed by the project lead from the leaderboard rows:
    baseline CO2 71,797-71,832 kg, cost 118,733-118,750 EUR, best plain-24h score 6.43
    (CO2 -5.97 %, cost -7.5 %, 4,289.4 kg avoided, 8,907.87 EUR saved)

The battery's absolute savings do not depend on the load (as long as load > 500 kW), the
baseline does. So comparing both against train-month tables separates 'fewer scored hours'
from 'lower load'.
"""
from lib import *
from src.data import TEST_MONTHS

pd.set_option("display.width", 220)
PUB = {"co2_base": 71815.0, "cost_base": 118741.0, "co2_saved": 4289.4, "cost_saved": 8907.87}

folds = get_folds("filled")
df = folds["ALL"]
r = run(v1.BatteryAgent(), df)
base = np.maximum(df.load_kw - df.pv_kw, 0.0).to_numpy()
grid = r["trace"]["grid_kw"].to_numpy()
d = pd.DataFrame({"co2_base": base * df.co2_kg_per_kwh, "cost_base": base * df.precio_eur_per_kwh,
                  "co2_saved": (base - grid) * df.co2_kg_per_kwh, "cost_saved": (base - grid) * df.precio_eur_per_kwh,
                  "net": base}, index=df.index)
m = d.groupby([df.index.year, df.index.month]).mean()
m["net_p05"] = d["net"].groupby([df.index.year, df.index.month]).quantile(0.05)
m["h<526"] = (d["net"] < 526.4).groupby([df.index.year, df.index.month]).sum()
print("per-hour means by month (v1 on ALL, train days only):")
print(m.round(2).to_string())

cols = ["co2_base", "cost_base", "co2_saved", "cost_saved"]
print("\npredicted public-half totals vs leaderboard (ratio = leaderboard / prediction):")
hyp = {"12 weeks, 1,008 h": (list(m.index), 84),
       "11 weeks (no Nov 2025), 924 h": ([k for k in m.index if k != (2025, 11)], 84),
       "10 weeks (no Nov 2025, no Aug 2026), 840 h": ([k for k in m.index if k not in ((2025, 11), (2026, 8))], 84)}
for name, (keys, h) in hyp.items():
    pred = m.loc[keys, cols].sum() * h
    print("  %-45s" % name, "  ".join("%s %.0f (x%.3f)" % (c, pred[c], PUB[c] / pred[c]) for c in cols))
print("\nReading: 'saved' ratios ~ fewer scored hours or smaller spreads; 'base' ratio / 'saved' ratio ~ load level.")

# how low does the load get in holiday periods that ARE in train?
net = d["net"]
for lab, sl in (("Jan 1-6 2026 (Christmas holidays)", slice("2026-01-01", "2026-01-06")), ("Jan 8-21 2026", slice("2026-01-08", "2026-01-21")),
                ("Dec 1-21 2025", slice("2025-12-01", "2025-12-21")), ("Dec 22-24 2025", slice("2025-12-22", "2025-12-24")),
                ("Aug 1-21 2026", slice("2026-08-01", "2026-08-21")), ("Jul 1-21 2026", slice("2026-07-01", "2026-07-21")),
                ("Apr 1-6 2026 (Easter)", slice("2026-04-01", "2026-04-06")), ("Apr 7-21 2026", slice("2026-04-07", "2026-04-21"))):
    x = net.loc[sl]
    if len(x):
        print("  %-36s mean %6.0f  min %6.0f  p05 %6.0f  hours<526: %3d of %d" % (lab, x.mean(), x.min(), x.quantile(0.05), int((x < 526.4).sum()), len(x)))
print("\nlast train day of each month (day before the test week) -- mean net load of the last 3 train days:")
for (y, mo), g in net.groupby([net.index.year, net.index.month]):
    last3 = g[g.index.normalize() >= g.index.normalize().max() - pd.Timedelta(days=2)]
    print("  %d-%02d  last train day %s  mean %6.0f  min %6.0f" % (y, mo, g.index.max().date(), last3.mean(), last3.min()))
