# Usage: .venv/Scripts/python.exe experiments/26_hard_eda/extra_numbers.py
"""Numbers quoted in reports/15_hard_eda/report.md that no figure script prints.
Rows: folds A, B, C (perfect-foresight schedule of each fold, pooled) and the organisers' hourly table."""
import numpy as np
import pandas as pd

from common import ETA, CAP, PMAX, TOL, folds_abc, load_hard, run_oracle

parts = []
for k, f in folds_abc().items():
    tr = run_oracle(f)
    tr["fold"], tr["pv"], tr["load"] = k, f["pv_kw"], f["load_kw"]
    parts.append(tr)
tr = pd.concat(parts)
tr["surplus"] = np.maximum(-tr.net_kw, 0.0)
dis, ch = tr[tr.discharge > TOL], tr[tr.charge > TOL]
h = tr.index.hour

print("--- discharge and the sun")
print(f"discharge energy in hours with no solar (< 10 kW): {dis.discharge[dis.pv < 10].sum() / dis.discharge.sum():.1%}; "
      f"with solar below 10 % of the load: {dis.discharge[dis.pv < 0.1 * dis.load].sum() / dis.discharge.sum():.1%}")
print(f"discharge energy by window: 18-22 {dis.discharge[np.isin(dis.index.hour, range(18, 23))].sum() / dis.discharge.sum():.1%}, "
      f"23-05 {dis.discharge[np.isin(dis.index.hour, [23, 0, 1, 2, 3, 4, 5])].sum() / dis.discharge.sum():.1%}, "
      f"06-08 {dis.discharge[np.isin(dis.index.hour, [6, 7, 8])].sum() / dis.discharge.sum():.1%}, "
      f"09-17 {dis.discharge[np.isin(dis.index.hour, range(9, 18))].sum() / dis.discharge.sum():.1%}")
print(f"charge energy by window: 10-17 {ch.charge[np.isin(ch.index.hour, range(10, 18))].sum() / ch.charge.sum():.1%}, "
      f"23-05 {ch.charge[np.isin(ch.index.hour, [23, 0, 1, 2, 3, 4, 5])].sum() / ch.charge.sum():.1%}")
print(f"time to empty a full battery at the median discharge power ({dis.discharge.median():.0f} kW): "
      f"{CAP * ETA / dis.discharge.median():.1f} h; at 1,500 kW: {CAP * ETA / PMAX:.1f} h")
print(f"hours idle {((tr.charge <= TOL) & (tr.discharge <= TOL)).mean():.1%}, charging {(tr.charge > TOL).mean():.1%}, "
      f"discharging {(tr.discharge > TOL).mean():.1%}")

print("--- surplus days in the folds")
rows = []
for (fold, date), g in tr.groupby(["fold", tr.index.date]):
    if g.surplus.sum() <= 0:
        continue
    first = g.index[g.surplus > 0][0]
    soc_before = tr.loc[tr.fold == fold, "soc_kwh"].shift(1).loc[first]
    rows.append({"fold": fold, "date": date, "surplus_kwh": g.surplus.sum(), "soc_before": soc_before,
                 "room_needed": g.surplus.sum() * ETA, "captured": np.minimum(g.charge, g.surplus).sum(),
                 "soc_max": g.soc_kwh.max(), "weekend": pd.Timestamp(date).day_of_week >= 5})
sd = pd.DataFrame(rows)
print(f"surplus days {len(sd)} of {tr.groupby(['fold', tr.index.date]).ngroups} fold-days; weekend share {sd.weekend.mean():.1%}; "
      f"surplus captured {sd.captured.sum() / sd.surplus_kwh.sum():.2%}")
print(f"surplus per surplus day: median {sd.surplus_kwh.median():.0f} kWh, p90 {sd.surplus_kwh.quantile(.9):.0f}, max {sd.surplus_kwh.max():.0f}; "
      f"days above 3,000 kWh: {(sd.surplus_kwh > 3000).sum()}")
print(f"state of charge just before the first surplus hour: median {sd.soc_before.median():.0f} kWh, p90 {sd.soc_before.quantile(.9):.0f}; "
      f"at or below 1,500 kWh on {(sd.soc_before <= 1500).mean():.1%} of surplus days")
print(f"battery reaches 5,990+ kWh on {(sd.soc_max > 5990).mean():.1%} of surplus days")
alld = tr.groupby(["fold", tr.index.date]).soc_kwh.max()
print(f"battery reaches 5,990+ kWh on {(alld > 5990).mean():.1%} of all fold-days; touches 10 kWh or less on "
      f"{(tr.groupby(['fold', tr.index.date]).soc_kwh.min() < 10).mean():.1%}")

print("--- the 06:00 jump (organisers' table)")
df = load_hard()
ld = df["load_kw"]
contig = (df.index.to_series().diff() == pd.Timedelta("1h")).to_numpy()
d = (ld - ld.shift(1))[contig]
d6 = d[d.index.hour == 6]
we = d6.index.day_of_week >= 5
for nm, m in (("weekdays", ~we), ("weekends", we)):
    print(f"  06:00 minus the held night value, {nm}: n {m.sum()}, median {d6[m].median():+.0f} kW, "
          f"p10 {d6[m].quantile(.1):+.0f}, p90 {d6[m].quantile(.9):+.0f}, share lower than held {(d6[m] < 0).mean():.1%}")
print(f"  all days: share lower {(d6 < 0).mean():.1%}; when lower, median drop {d6[d6 < 0].median():.0f} kW")
night = ld[np.isin(ld.index.hour, [23, 0, 1, 2, 3, 4, 5])]
print(f"  held night load: median {night.median():.0f} kW, IQR {night.quantile(.25):.0f}-{night.quantile(.75):.0f}")
