# Usage: .venv/Scripts/python.exe experiments/21_load/cal_phase2.py run | show
"""Phase 2 of the holiday-calendar study: CalLoadFC (cal_fc.py) with the typed-in calendar in the h3 stack.
Folds A, B, C without August, run ending on the fold's last July day; priors fitted outside each fold.
  h3              packaged configuration (cached tape; CalLoadFC without calendar reproduces it)
  cal_b1          calendar, clean references, hour 0 pulled to the blended reference (cal_blend = 1)
  cal_b0, cal_b05 the same with cal_blend = 0 / 0.5
  cal_b1_noclean  calendar days stay in the weekday reference
  wrong entries   the calendar plus normal weekdays declared "local" holidays: every week start that is a
                  normal weekday (fake_ws) or one mid-week normal weekday per week (fake_mid), with
                  local_trust = 1 and 0.5 (prior = trust x the holiday prior)
Writes cal_p2.pkl."""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cal_lib import (CALENDAR, HERE, LOW_DATES, CalLoadFC, base_tape, cb, d0_of, deltas, fit_table_for, fold_cut,
                     masked_score, record_load, run_h3, week_pos, window, with_load)

FOLDS = "ABC"
CAL_DATES = {pd.Timestamp(d) for d in CALENDAR}


def normal_weekday(d):
    """Weekday that is neither in the calendar nor a low day, and whose neighbours (+-1 weekday) are not either."""
    if d.dayofweek >= 5:
        return False
    for k in (-3, -2, -1, 0, 1, 2, 3):
        x = d + pd.Timedelta(days=k)
        if x in CAL_DATES or x in LOW_DATES:
            return False
    return True


def fakes(f, df):
    pos = week_pos(df, f)
    ws, mid = [], []
    for wk, g in pd.Series(pos, index=df.index).groupby(df.index.strftime("%Y-%m")):
        d_first = g.index[0].normalize()
        if g.iloc[0] == 0 and normal_weekday(d_first):
            ws.append(d_first)
        for k in (3, 4, 2):
            d = d_first + pd.Timedelta(days=k)
            if normal_weekday(d) and d in df.index:
                mid.append(d)
                break
    return ws, mid


def run_fold(f):
    t0 = time.perf_counter()
    df = fold_cut(f)
    tp = base_tape(f)
    n = len(df)
    fit = fit_table_for(f)
    ws, mid = fakes(f, df)
    out = {"meta": {"fake_ws": ws, "fake_mid": mid}}
    r, tr = run_h3(f, df, tp)
    out["h3"] = {"score": r["score"], "infr": int(r["infractions"]), "grid": tr["grid_kw"].to_numpy(),
                 "co2_base": r["co2_base_kg"], "cost_base": r["cost_base_eur"]}
    fc = CalLoadFC(fit, calendar=CALENDAR, cal_blend=1.0)
    out["meta"]["prior"], out["meta"]["prior_n"], out["meta"]["dropped"] = dict(fc.prior), dict(fc.prior_n), list(fc.dropped)
    plain = CalLoadFC.__new__(CalLoadFC)                      # the same fitted object with the calendar switched off
    plain.__dict__.update(fc.__dict__)
    plain.cal, plain.lam_day = {}, {}
    plain.reset()
    F0, Q0 = record_load(plain, df)
    out["meta"]["identical_without_calendar"] = bool(np.array_equal(F0, tp["load_fc"][:n]) and np.array_equal(Q0, tp["load_fc_q"][:n]))

    def variant(name, calendar, blend, clean=True, trust=1.0, scale=1.0, forced=None, active=None):
        fc.cal_blend, fc.clean_ref, fc.local_trust = blend, clean, trust
        fc.active = None if active is None else {d0_of(d) for d in active}
        fc.prior_forced = dict(forced or {})
        fc.set_calendar(calendar)
        fc.fit_priors()
        if scale != 1.0:                                       # what an entry given part of the prior gains when it is right
            fc.prior = {k: scale * v for k, v in fc.prior.items()}
            fc.reset()
        F, Q = record_load(fc, df)
        r, tr = run_h3(f, df, with_load(tp, F, Q))
        out[name] = {"score": r["score"], "infr": int(r["infractions"]), "grid": tr["grid_kw"].to_numpy(),
                     "rows_changed": int((np.abs(F - tp["load_fc"][:n]).max(axis=1) > 0).sum()),
                     "days_changed": sorted({str(x.date()) for x in df.index[np.abs(F - tp["load_fc"][:n]).max(axis=1) > 0]}),
                     "prior": dict(fc.prior)}

    variant("cal_b1", CALENDAR, 1.0)
    variant("cal_b0", CALENDAR, 0.0)
    variant("cal_b05", CALENDAR, 0.5)
    variant("cal_b1_noclean", CALENDAR, 1.0, clean=False)
    variant("cal_half", CALENDAR, 1.0, scale=0.5)
    # Holy Week has no fit day outside fold A: give it half the fitted holiday prior (no leak) so the kind is exercised
    variant("cal_hw", CALENDAR, 1.0, forced={"holyweek": 0.5 * out["meta"]["prior"]["holiday"]})
    # sensitivity only (leaks the fold's own days): the priors of the final fit on every train row
    variant("cal_allprior", CALENDAR, 1.0, forced={"holiday": 0.935, "closure": 0.624, "holyweek": 0.456})
    # "blind": same calendar and clean references, but every calendar day starts with index 0 (as on a first
    # day that nothing announces). cal_b1 - cal_blind = what the prior alone is worth on each calendar day.
    variant("cal_blind", CALENDAR, 1.0, forced={"holiday": 0.0, "closure": 0.0, "holyweek": 0.0, "local": 0.0})
    # minimal calendars: priors from every entry, but only some dates act at run time
    variant("cal_min", CALENDAR, 1.0, active=["2025-09-11"])                     # the one regional holiday billed as a weekday
    variant("cal_first", CALENDAR, 1.0, active=["2025-09-11", "2025-12-08", "2026-01-01", "2026-05-01"])   # first days only
    variant("cal_holonly", {d: k for d, k in CALENDAR.items() if k in ("holiday", "local")}, 1.0)
    variant("cal_noclosure", {d: k for d, k in CALENDAR.items() if k != "closure"}, 1.0)
    for lab, days in (("ws", ws), ("mid", mid)):
        cal = dict(CALENDAR)
        cal.update({str(d.date()): "local" for d in days})
        for trust in (1.0, 0.5):
            variant(f"fake_{lab}_t{trust}", cal, 1.0, trust=trust)
            variant(f"fake_{lab}_b0_t{trust}", cal, 0.0, trust=trust)
    print(f"fold {f} done in {time.perf_counter() - t0:.0f} s", flush=True)
    return f, out


def run():
    res = Parallel(n_jobs=3)(delayed(run_fold)(f) for f in FOLDS)
    with open(HERE / "cal_p2.pkl", "wb") as fh:
        pickle.dump(dict(res), fh)


def show():
    with open(HERE / "cal_p2.pkl", "rb") as fh:
        R = pickle.load(fh)
    pd.set_option("display.width", 250)
    cals = ["cal_b1", "cal_b05", "cal_b0", "cal_b1_noclean", "cal_half", "cal_holonly", "cal_noclosure", "cal_hw", "cal_allprior", "cal_min", "cal_first"]
    print("== priors fitted outside each fold; bit-identity without calendar ==")
    for f in FOLDS:
        m = R[f]["meta"]
        print(f, "priors", {k: round(v, 3) for k, v in m["prior"].items()}, "days per kind", m["prior_n"], "dropped", m["dropped"],
              "| no calendar == cached h3 tape:", m["identical_without_calendar"])
    print("\n== whole run, public half (first 84 h of each week), private half (last 84 h): score and gain over h3 ==")
    rows = []
    for f in FOLDS:
        df = fold_cut(f)
        pos = week_pos(df, f)
        for k in ["h3"] + cals:
            g = R[f][k]["grid"]
            rows.append({"fold": f, "run": k, "whole": masked_score(df, g)["score"], "public": masked_score(df, g, pos < 84)["score"],
                         "private": masked_score(df, g, pos >= 84)["score"], "infr": R[f][k]["infr"],
                         "rows_changed": R[f][k].get("rows_changed", 0)})
    T = pd.DataFrame(rows)
    for col in ("whole", "public", "private"):
        p = T.pivot(index="run", columns="fold", values=col).loc[["h3"] + cals]
        p["mean"] = p.mean(axis=1)
        d = (p - p.loc["h3"]).add_prefix("d")
        print(f"\n{col}:")
        print(pd.concat([p, d], axis=1).round(4).to_string())
    print("\ninfractions:", T.groupby("run")["infr"].sum().to_dict(), "| wrong-entry runs:",
          sum(v["infr"] for f in FOLDS for k, v in R[f].items() if k.startswith("fake")))
    print("forecast rows that differ from LoadFC:", T.pivot(index="run", columns="fold", values="rows_changed").loc[cals].to_dict("index"))
    for f in FOLDS:
        print(f"  fold {f} days with any changed forecast (cal_b1):", R[f]["cal_b1"]["days_changed"])
    # ---- paired by week and by calendar day
    best = "cal_b1"
    for best in ("cal_b1", "cal_half", "cal_hw", "cal_min", "cal_first"):
        print(f"\n== {best}: gain over h3 by week (fold-score points) ==")
        wk_rows = []
        for f in FOLDS:
            df = fold_cut(f)
            pts, kg, eur = deltas(df, R[f]["h3"]["grid"], R[f][best]["grid"])
            pos = week_pos(df, f)
            s = pd.DataFrame({"wk": df.index.strftime("%Y-%m"), "pts": pts, "pub": np.where(pos < 84, pts, 0.0), "priv": np.where(pos >= 84, pts, 0.0),
                              "kg": kg, "eur": eur}).groupby("wk").sum()
            s.insert(0, "fold", f)
            wk_rows.append(s.reset_index())
        W = pd.concat(wk_rows)
        ch = W[W.pts.abs() > 1e-9]
        print(ch.round(4).to_string(index=False))
        print(f"weeks changed {len(ch)} of {len(W)}; better {int((ch.pts > 0).sum())}, worse {int((ch.pts < 0).sum())}; "
              f"sum {W.pts.sum():.4f} (public-half hours {W['pub'].sum():.4f}, private-half hours {W['priv'].sum():.4f})")
    print("\n== gain over h3 by calendar day in the folds (D = that day, D+1 = the day after; fold-score points, kg, EUR) ==")
    rows = []
    for f in FOLDS:
        df = fold_cut(f)
        pos = week_pos(df, f)
        for k in cals:
            pts, kg, eur = deltas(df, R[f]["h3"]["grid"], R[f][k]["grid"])
            for d in sorted(CAL_DATES):
                a = window(df, d)
                if not a.any() or d.dayofweek >= 5:
                    continue
                b = window(df, d + pd.Timedelta(days=1))
                rows.append({"fold": f, "run": k, "day": str(d.date()), "dow": d.day_name()[:3], "kind": CALENDAR[str(d.date())],
                             "half": "public" if pos[np.flatnonzero(a)[0]] < 84 else "private",
                             "pts_D": pts[a].sum(), "pts_D1": pts[b].sum(), "kg": kg[a | b].sum(), "eur": eur[a | b].sum()})
    D = pd.DataFrame(rows)
    D.to_csv(HERE / "cal_p2_days.csv", index=False)
    print(D.pivot_table(index=["fold", "day", "dow", "kind", "half"], columns="run", values=["pts_D", "pts_D1"], sort=False).round(4).to_string())
    print("== value of the prior alone: cal_b1 against cal_blind (same clean references, calendar days start at index 0), by week ==")
    rows = []
    for f in FOLDS:
        df = fold_cut(f)
        pts, kg, eur = deltas(df, R[f]["cal_blind"]["grid"], R[f]["cal_b1"]["grid"])
        s_ = pd.DataFrame({"wk": df.index.strftime("%Y-%m"), "pts": pts, "kg": kg, "eur": eur}).groupby("wk").sum()
        s_.insert(0, "fold", f)
        rows.append(s_[s_.pts.abs() > 1e-9].reset_index())
        for d in sorted(CAL_DATES):
            a_ = window(df, d)
            if a_.any() and d.dayofweek < 5:
                print(f"   {f} {d.date()} {d.day_name()[:3]} {CALENDAR[str(d.date())]:9s} D {pts[a_].sum():+.4f}  D+1 {pts[window(df, d + pd.Timedelta(days=1))].sum():+.4f}")
    print(pd.concat(rows).round(4).to_string(index=False))
    print("cal_blind against h3 (whole run):", {f: round(R[f]["cal_blind"]["score"] - R[f]["h3"]["score"], 4) for f in FOLDS})
    # ---- wrong entries
    print("\n== cost of a wrong entry: normal weekdays declared 'local' holidays (loss against cal of the same blend, D and D+1) ==")
    rows = []
    for f in FOLDS:
        df = fold_cut(f)
        for lab in ("ws", "mid"):
            for trust in (1.0, 0.5):
                for b, refname in (("", "cal_b1"), ("_b0", "cal_b0")):
                    k = f"fake_{lab}{b}_t{trust}"
                    pts, kg, eur = deltas(df, R[f][refname]["grid"], R[f][k]["grid"])
                    days = R[f]["meta"]["fake_" + lab]
                    inw = np.zeros(len(df), bool)
                    for d in days:
                        m = window(df, d, 2)
                        inw |= m
                        rows.append({"fold": f, "where": lab, "blend": 0 if b else 1, "trust": trust, "day": str(d.date()), "dow": d.day_name()[:3],
                                     "month": d.month, "pts": pts[m].sum(), "kg": kg[m].sum(), "eur": eur[m].sum(), "pts_D": pts[window(df, d)].sum()})
                    rows.append({"fold": f, "where": lab, "blend": 0 if b else 1, "trust": trust, "day": "elsewhere", "dow": "", "month": 0,
                                 "pts": pts[~inw].sum(), "kg": kg[~inw].sum(), "eur": eur[~inw].sum(), "pts_D": 0.0})
    Fk = pd.DataFrame(rows)
    Fk.to_csv(HERE / "cal_p2_fakes.csv", index=False)
    ev = Fk[Fk.day != "elsewhere"]
    print(ev.groupby(["where", "blend", "trust"]).agg(n=("pts", "size"), mean_pts=("pts", "mean"), median_pts=("pts", "median"), worst=("pts", "min"),
                                                      best=("pts", "max"), share_worse=("pts", lambda x: float((x < 0).mean())),
                                                      mean_kg=("kg", "mean"), mean_eur=("eur", "mean")).round(4).to_string())
    print("\nelsewhere (knock-on outside D, D+1), sum over folds:")
    print(Fk[Fk.day == "elsewhere"].groupby(["where", "blend", "trust"])["pts"].sum().round(4).to_string())
    print("\nper wrong entry, week starts, blend 1, trust 1 and 0.5:")
    x = ev[(ev["where"] == "ws") & (ev.blend == 1)].pivot_table(index=["fold", "day", "dow"], columns="trust", values="pts", sort=False)
    print(x.round(4).to_string())
    print("by season (week starts, blend 1):")
    ev2 = ev[(ev["where"] == "ws") & (ev.blend == 1)].assign(season=lambda d: np.where(d.month.isin([5, 6, 7, 9]), "May-Sep", "Oct-Apr"))
    print(ev2.groupby(["season", "trust"])["pts"].agg(["size", "mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "run":
        run()
    show()
