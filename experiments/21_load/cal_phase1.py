# Usage: .venv/Scripts/python.exe experiments/21_load/cal_phase1.py run | show
"""Phase 1 of the holiday-calendar study: how much is there to gain in the h3 stack?
Folds A and B without August (fold C has no low weekday), run ending on the fold's last July day.
  h3            the packaged configuration (taped forecasts)
  O1            LoadFC given the TRUE holiday index of every low weekday of holidays.csv, at every horizon
  O1b           O1 + hour 0 pulled towards the blended reference on those days (cal_blend = 1)
  O2 / O2b      as O1 / O1b, plus the true index of the 3 weekdays after each low day
  Pday          perfect load forecast (all horizons, zero-width hour 0) for the target hours of those days
  Pall          perfect load everywhere (scale)
Writes cal_p1.pkl (traces), cal_p1_lam.csv (index the forecaster starts each day with)."""
import pickle
import sys
import time

import numpy as np
import pandas as pd

from cal_lib import (HERE, LOW_DATES, NS_H, CalLoadFC, base_tape, cb, d0_of, day0, deltas, etype, fit_table_for,
                     fold_cut, load_fit_table, masked_score, record_load, run_h3, simulate, true_index, week_pos,
                     window, with_load)

FOLDS = "AB"


class LamLog:
    """Pseudo-agent: runs the forecaster and logs the holiday index it would use from each hour on."""
    def __init__(self, fc):
        self.fc, self.hist, self.rows = fc, [], []

    def act(self, obs):
        self.hist.append({"ts": obs["timestamp"]})
        p = self.fc.predict_load(obs, self.hist, 36)
        i = self.fc._cur
        lam, dref = self.fc._lam_now(i)
        self.rows.append((obs["timestamp"], lam, self.fc._lam_prev(day0(i)), p[0], p[min(6, len(p) - 1)], p[min(12, len(p) - 1)]))
        return 0.0


def run():
    out = {}
    lamrows = []
    for f in FOLDS:
        t0 = time.perf_counter()
        df = fold_cut(f)
        tp = base_tape(f)
        n = len(df)
        fc = CalLoadFC(fit_table_for(f))
        F0, Q0 = record_load(fc, df)
        same = bool(np.array_equal(F0, tp["load_fc"][:n]) and np.array_equal(Q0, tp["load_fc_q"][:n]))
        print(f"fold {f}: {n} rows; CalLoadFC without calendar reproduces the cached tape bit for bit: {same}"
              f" (max abs diff {np.abs(F0 - tp['load_fc'][:n]).max():.3g} kW)", flush=True)
        lg = LamLog(fc)
        simulate(lg, df)
        lamrows += [(f,) + r for r in lg.rows]
        low = [d for d in LOW_DATES if d in df.index]
        o1, o2 = true_index(low, 0), true_index(low, 3)
        hrs = (df.index.values.astype("datetime64[ns]").astype(np.int64) // NS_H)
        full = load_fit_table()["load_kw"].dropna()
        truth = dict(zip((full.index.values.astype("datetime64[ns]").astype(np.int64) // NS_H).tolist(), full.to_numpy(float).tolist()))
        days = {d0_of(d) for d in low}
        tapes = {"h3": tp}
        for name, lam_day, blend in (("O1", o1, 0.0), ("O1b", o1, 1.0), ("O2", o2, 0.0), ("O2b", o2, 1.0)):
            fc.lam_day, fc.cal_blend = dict(lam_day), blend
            fc.reset()
            tapes[name] = with_load(tp, *record_load(fc, df))
        fc.lam_day, fc.cal_blend = {}, 0.0
        # perfect load on the low days (their meter day: 06:00 -> 05:59), and everywhere
        for name, pick in (("Pday", lambda j: day0(j) in days), ("Pall", lambda j: True)):
            F, Q = tp["load_fc"][:n].copy(), tp["load_fc_q"][:n].copy()
            for t in range(n):
                for k in range(F.shape[1]):
                    j = int(hrs[t]) + k
                    if pick(j) and j in truth:
                        F[t, k] = truth[j]
                if pick(int(hrs[t])):
                    Q[t] = df["load_kw"].iloc[t]
            tapes[name] = with_load(tp, F, Q)
        for name, t in tapes.items():
            r, tr = run_h3(f, df, t)
            out[(f, name)] = {"score": r["score"], "infr": int(r["infractions"]), "trace": tr,
                              "co2_base": r["co2_base_kg"], "cost_base": r["cost_base_eur"],
                              "F0": t["load_fc"][:n, 0].copy()}
            print(f"  {f} {name:5s} score {r['score']:.4f}  infractions {int(r['infractions'])}", flush=True)
        out[(f, "meta")] = {"o1": o1, "o2": o2, "low": low}
        print(f"  fold {f} done in {time.perf_counter() - t0:.0f} s", flush=True)
    with open(HERE / "cal_p1.pkl", "wb") as fh:
        pickle.dump(out, fh)
    pd.DataFrame(lamrows, columns=["fold", "ts", "lam_now", "lam_prev", "p0", "p6", "p12"]).to_csv(HERE / "cal_p1_lam.csv", index=False)


def show():
    with open(HERE / "cal_p1.pkl", "rb") as fh:
        out = pickle.load(fh)
    lam = pd.read_csv(HERE / "cal_p1_lam.csv", parse_dates=["ts"])
    pd.set_option("display.width", 250)
    names = ["O1", "O1b", "O2", "O2b", "Pday", "Pall"]
    print("== fold scores (no August, run ends on the fold's last July day) ==")
    rows = []
    for f in FOLDS:
        rows.append({"fold": f, **{k: out[(f, k)]["score"] for k in ["h3"] + names},
                     "co2_base_kg": out[(f, "h3")]["co2_base"], "cost_base_eur": out[(f, "h3")]["cost_base"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n== true holiday index of the low days (O1) and of the 3 weekdays after (O2) ==")
    for f in FOLDS:
        m = out[(f, "meta")]
        print(f, {str(pd.Timestamp((d0 - 6) * 3600, unit="s").date()): round(v, 2) for d0, v in sorted(m["o2"].items())})
    # ---- step 1: week starts that are holidays
    print("\n== step 1: week starts that are holidays, h3 against the bounds (kWh) ==")
    for f, day in (("A", "2026-01-01"), ("A", "2026-05-01"), ("B", "2025-12-08"), ("B", "2025-09-11"), ("B", "2026-02-12"), ("A", "2026-04-02"), ("A", "2026-01-05")):
        df = fold_cut(f)
        m = window(df, day)
        net = out[(f, "h3")]["trace"]["net_kw"].to_numpy()
        hod = df.index.hour.to_numpy()
        l = lam[(lam.fold == f) & (lam.ts >= day) & (lam.ts < pd.Timestamp(day) + pd.Timedelta(days=1))].set_index("ts")
        print(f"\n{day} ({pd.Timestamp(day).day_name()}), fold {f}: index used by LoadFC from 00:00 = {l['lam_now'].iloc[0]:.2f}, "
              f"06:00 = {l['lam_now'].iloc[6]:.2f}, 09:00 = {l['lam_now'].iloc[9]:.2f}, 12:00 = {l['lam_now'].iloc[12]:.2f}; "
              f"true index {out[(f, 'meta')]['o1'].get(int(pd.Timestamp(day).value // NS_H) + 6, float('nan')):.2f}")
        true = df["load_kw"].to_numpy()[m]
        print(f"   load forecast made at 00:00 for 06:00 / 12:00: {l['p6'].iloc[0]:.0f} / {l['p12'].iloc[0]:.0f} kW; "
              f"true {true[6]:.0f} / {true[12]:.0f} kW; true net load of the day {np.maximum(net[m], 0).sum():.0f} kWh, "
              f"solar surplus {np.maximum(-net[m], 0).sum():.0f} kWh")
        rr = []
        for k in ["h3", "O1", "O1b", "O2b", "Pday"]:
            tr = out[(f, k)]["trace"]
            p, soc, sp = tr["p_kw"].to_numpy(), tr["soc_kwh"].to_numpy(), tr["spill_kw"].to_numpy()
            i0 = int(np.flatnonzero(m)[0])
            night = m & (hod < 6)
            dsp = np.where(p < 0, np.maximum(-p - np.maximum(net, 0), 0), 0.0)
            rr.append({"run": k, "soc_at_00": soc[i0 - 1] if i0 > 0 else 3000.0,
                       "grid_charge_00_06": np.maximum(p[night], 0).sum(), "soc_at_06": soc[i0 + 5],
                       "discharge_used_06_24": (np.maximum(-p, 0) - dsp)[m & (hod >= 6)].sum(),
                       "discharge_thrown": dsp[m].sum(),
                       "solar_surplus_stored": np.minimum(np.maximum(p, 0), np.maximum(-net, 0))[m].sum(),
                       "solar_surplus_lost": np.where(net < 0, np.maximum(-(net + p), 0), 0.0)[m].sum(),
                       "grid_charge_06_24": (np.maximum(p, 0) - np.minimum(np.maximum(p, 0), np.maximum(-net, 0)))[m & (hod >= 6)].sum(),
                       "soc_at_24": soc[i0 + 23], "grid_import": tr["grid_kw"].to_numpy()[m].sum()})
        print(pd.DataFrame(rr).round(0).to_string(index=False))
    # ---- steps 2 and 3: paired by day
    print("\n== steps 2-3: gain over h3 by low day (D = the day 00:00-24:00, D+1 = the day after) ==")
    rows = []
    for f in FOLDS:
        df = fold_cut(f)
        g0 = out[(f, "h3")]["trace"]["grid_kw"].to_numpy()
        low = out[(f, "meta")]["low"]
        pos = week_pos(df, f)
        for k in names:
            pts, kg, eur = deltas(df, g0, out[(f, k)]["trace"]["grid_kw"].to_numpy())
            inwin = np.zeros(len(df), bool)
            for d in low:
                a, b = window(df, d), window(df, d + pd.Timedelta(days=1))
                first = (d - pd.Timedelta(days=1)) not in low and not (d.dayofweek == 0 and (d - pd.Timedelta(days=3)) in low)
                wk_start = int(pos[np.flatnonzero(a)[0]]) == 0
                rows.append({"fold": f, "run": k, "day": d.date(), "dow": d.day_name()[:3], "first": first, "week_start": wk_start,
                             "half": "public" if pos[np.flatnonzero(a)[0]] < 84 else "private",
                             "pts_D": pts[a].sum(), "pts_D1": pts[b].sum(), "kg_D": kg[a].sum(), "kg_D1": kg[b].sum(),
                             "eur_D": eur[a].sum(), "eur_D1": eur[b].sum()})
                inwin |= a | b
            rows.append({"fold": f, "run": k, "day": "elsewhere", "dow": "", "first": False, "week_start": False, "half": "",
                         "pts_D": pts[~inwin].sum(), "pts_D1": 0.0, "kg_D": kg[~inwin].sum(), "kg_D1": 0.0,
                         "eur_D": eur[~inwin].sum(), "eur_D1": 0.0})
    R = pd.DataFrame(rows)
    R["pts"] = R.pts_D + R.pts_D1
    R["kg"] = R.kg_D + R.kg_D1
    R["eur"] = R.eur_D + R.eur_D1
    R.to_csv(HERE / "cal_p1_days.csv", index=False)
    for k in names:
        x = R[R.run == k]
        print(f"\n--- {k}: fold-score points, kg CO2, EUR (day + day after; windows of consecutive days overlap by design? no: D and D+1 are summed per row, "
              f"so a stretch counts its inner days twice; see the stretch table)")
        print(x[["fold", "day", "dow", "first", "week_start", "half", "pts_D", "pts_D1", "kg_D", "kg_D1", "eur_D", "eur_D1"]].round(3).to_string(index=False))
    # stretches without double counting
    print("\n== by stretch (consecutive low days + the day after), no double counting ==")
    S = []
    stretches = {"A": [("1-3 Jan (Thu hol, Fri closure, +Sat)", "2026-01-01", 3), ("5-7 Jan (Mon closure, Tue hol, +Wed)", "2026-01-05", 3),
                       ("2-4 Apr (Thu Holy Week, Good Fri, +Sat)", "2026-04-02", 3), ("6-7 Apr (Easter Mon, +Tue)", "2026-04-06", 2),
                       ("1-2 May (Fri hol, +Sat)", "2026-05-01", 2)],
                 "B": [("11-12 Sep (Thu hol, +Fri)", "2025-09-11", 2), ("8-9 Dec (Mon hol, +Tue)", "2025-12-08", 2),
                       ("12-13 Feb (unexplained low Thu, +Fri)", "2026-02-12", 2)]}
    for f in FOLDS:
        df = fold_cut(f)
        g0 = out[(f, "h3")]["trace"]["grid_kw"].to_numpy()
        allw = np.zeros(len(df), bool)
        for lab, d, nd in stretches[f]:
            allw |= window(df, d, nd)
        for k in names:
            pts, kg, eur = deltas(df, g0, out[(f, k)]["trace"]["grid_kw"].to_numpy())
            for lab, d, nd in stretches[f]:
                m = window(df, d, nd)
                S.append({"fold": f, "stretch": lab, "run": k, "pts": pts[m].sum(), "kg": kg[m].sum(), "eur": eur[m].sum()})
            S.append({"fold": f, "stretch": "elsewhere", "run": k, "pts": pts[~allw].sum(), "kg": kg[~allw].sum(), "eur": eur[~allw].sum()})
            S.append({"fold": f, "stretch": "WHOLE FOLD", "run": k, "pts": pts.sum(), "kg": kg.sum(), "eur": eur.sum()})
    S = pd.DataFrame(S)
    S.to_csv(HERE / "cal_p1_stretches.csv", index=False)
    for v in ("pts", "kg", "eur"):
        print(f"\n{v}:")
        print(S.pivot_table(index=["fold", "stretch"], columns="run", values=v, sort=False)[names].round(3 if v == "pts" else 1).to_string())


if __name__ == "__main__":
    (run if (len(sys.argv) < 2 or sys.argv[1] == "run") else show)()
    if len(sys.argv) < 2 or sys.argv[1] == "run":
        show()
