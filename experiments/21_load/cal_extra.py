# Usage: .venv/Scripts/python.exe experiments/21_load/cal_extra.py
"""Supporting checks for the holiday-calendar study.
 1. Why week-start national holidays gain nothing and 11 Sep does: the signal w on those days (night
    against morning) and how much h3 charges from the grid at night, by month, on normal weekdays.
 2. What the calendar changes at 00:00 of the test-week days it names: forecasts of the final-fit
    forecaster (every train row) with and without the calendar, from a cold start as in the test.
 3. Translation of fold gains into test-run points (public baseline from the leaderboard)."""
import pickle

import numpy as np
import pandas as pd

from cal_lib import CALENDAR, HERE, CalLoadFC, cb, fold_cut, load_fit_table
from src.hard_mpc import KAPPA

pd.set_option("display.width", 250)
with open(HERE / "cal_p1.pkl", "rb") as fh:
    P1 = pickle.load(fh)

print("== 1a. signal w = co2 + kappa * price on low days: mean over 00-05, 06-09, 10-16, 18-21 (and a normal weekday of the same week) ==")
rows = []
for f, day, ref in (("A", "2026-01-01", "2026-01-07"), ("A", "2026-05-01", "2026-05-05"), ("B", "2025-12-08", "2025-12-10"),
                    ("B", "2025-09-11", "2025-09-10"), ("B", "2026-02-12", "2026-02-11"), ("A", "2026-04-02", "2026-04-07")):
    df = fold_cut(f)
    w = df["co2_kg_per_kwh"] + KAPPA * df["precio_eur_per_kwh"]
    for lab, d in (("low day", day), ("normal weekday", ref)):
        x = w[d]
        h = x.index.hour
        rows.append({"day": d, "what": lab, "night 00-05": x[h < 6].mean(), "morning 06-09": x[(h >= 6) & (h < 10)].mean(),
                     "midday 10-16": x[(h >= 10) & (h < 17)].mean(), "evening 18-21": x[(h >= 18) & (h < 22)].mean(),
                     "morning/night": x[(h >= 6) & (h < 10)].mean() / x[h < 6].mean(),
                     "price 08-13 / price 00-05": df["precio_eur_per_kwh"][d][(h >= 8) & (h < 14)].mean() / df["precio_eur_per_kwh"][d][h < 6].mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n== 1b. h3 on weekdays, by month: kWh charged from the grid 00-06 and discharged 06-10 (mean per day; folds A and B) ==")
rows = []
for f in "AB":
    df = fold_cut(f)
    tr = P1[(f, "h3")]["trace"]
    p = tr["p_kw"].to_numpy()
    h = df.index.hour.to_numpy()
    d = pd.DataFrame({"date": df.index.normalize(), "night_charge": np.where(h < 6, np.maximum(p, 0), 0.0),
                      "morning_discharge": np.where((h >= 6) & (h < 10), np.maximum(-p, 0), 0.0)}).groupby("date").sum()
    d = d[d.index.dayofweek < 5]
    d["month"] = d.index.strftime("%Y-%m")
    d["fold"] = f
    rows.append(d)
D = pd.concat(rows)
print(D.groupby("month").agg(days=("night_charge", "size"), night_charge=("night_charge", "mean"),
                             share_days_charging=("night_charge", lambda x: float((x > 300).mean())),
                             morning_discharge=("morning_discharge", "mean")).round(2).to_string())

print("\n== 2. final-fit forecaster at 00:00 of the test-week calendar days (cold start: the day before comes from the table) ==")
fit = load_fit_table()
plain = CalLoadFC(fit)
cal = CalLoadFC(fit, calendar=CALENDAR, cal_blend=1.0)
print("priors (every train row):", {k: round(v, 3) for k, v in cal.prior.items()}, "days per kind:", cal.prior_n)
rows = []
for day in ("2025-09-24", "2025-12-25", "2026-03-30", "2026-05-25", "2026-06-24"):
    ts = pd.Timestamp(day)
    obs = {"timestamp": ts, "noves_dades_5min": {"ultim_consum_kw": 900.0, "ultima_solar_kw": 0.0}}
    out = {}
    for name, fc in (("plain", plain), ("cal", cal)):
        fc.reset()
        p = fc.predict_load(obs, [{"ts": ts}], 36)
        out[name] = (p, fc._lam_now(fc._cur + 6)[0] if False else None)
    a, b = out["plain"][0], out["cal"][0]
    rows.append({"day": day, "dow": ts.day_name()[:3], "kind": CALENDAR[day], "prev hour in table": bool(plain.tab.get(int(ts.value // 3_600_000_000_000) - 1) is not None),
                 "now": a[0], "now cal": b[0], "09:00": a[9], "09:00 cal": b[9], "12:00": a[12], "12:00 cal": b[12], "18:00": a[18], "18:00 cal": b[18],
                 "next day 12:00": a[35] if False else a[min(35, 36 - 1)], "day kWh 06-22": a[6:23].sum(), "day kWh 06-22 cal": b[6:23].sum()})
print(pd.DataFrame(rows).round(0).to_string(index=False))

print("\n== 3. baselines: folds without August against the test (public half from the leaderboard: 60,270 kg, 94,960 EUR) ==")
for f in "AB":
    print(f, "fold baseline kg / EUR:", round(P1[(f, "h3")]["co2_base"]), round(P1[(f, "h3")]["cost_base"]))
PUB = (60270.0, 94960.0)


def pts(kg, eur, base):
    return 100.0 * (0.7 * kg / base[0] + 0.3 * eur / base[1])


S = pd.read_csv(HERE / "cal_p1_stretches.csv")
print("upper bounds (phase 1) as test points: whole run = 2 x public baseline; public half if the event is in it")
for f, st in (("B", "11-12 Sep (Thu hol, +Fri)"), ("A", "1-3 Jan (Thu hol, Fri closure, +Sat)"), ("A", "1-2 May (Fri hol, +Sat)"),
              ("B", "8-9 Dec (Mon hol, +Tue)"), ("A", "5-7 Jan (Mon closure, Tue hol, +Wed)"), ("A", "2-4 Apr (Thu Holy Week, Good Fri, +Sat)"),
              ("A", "6-7 Apr (Easter Mon, +Tue)")):
    for run in ("O1b", "O2b", "Pday"):
        x = S[(S.fold == f) & (S.stretch == st) & (S.run == run)].iloc[0]
        print(f"  {st:42s} {run:5s} fold pts {x.pts:+.4f}  kg {x.kg:+7.1f}  EUR {x.eur:+7.1f}  -> whole-run pts {pts(x.kg, x.eur, (2 * PUB[0], 2 * PUB[1])):+.4f}"
              f"  half pts {pts(x.kg, x.eur, PUB):+.4f}")
