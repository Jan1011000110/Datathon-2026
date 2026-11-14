"""
experiments/29_stress_hard/suite.py
Scenario suite for the packaged hard-mode agents.

    suite.py <h1|h2> <fold> <families, comma separated | all> [shipped]
        families: obs, layout, lowload, faults
        "shipped" = the state inside the ZIP (fitted on every train row) instead of the fold-out state
Output: out/suite_<agent>_<fold>_<families>[_shipped].csv   (one row per scenario; the clean run is always first)
"""
import sys
import time

import numpy as np
import pandas as pd

import lib as L


def wk_of(df, month):
    wk = L.weeks(df)
    hit = np.flatnonzero(df.index.month == month)
    return int(wk[hit[0]]) if len(hit) else None


def scenarios(df, families):
    """List of (family, name, df, obs_fn, soc0, note)."""
    S = [("clean", "clean", df, None, L.SOC0, "")]
    nw = int(L.weeks(df).max()) + 1
    dec, jul, aug, oct_ = wk_of(df, 12), wk_of(df, 7), wk_of(df, 8), wk_of(df, 10)

    if "obs" in families:
        for k in ("str", "iso", "datetime", "np", "tz", "utc", "dmy", "epoch", "none", "missing", "const"):
            S.append(("obs", f"timestamp: {k}", df, L.o_ts(k), L.SOC0, ""))
        for k in ("24", "short_random", "short_weekend", "numpy", "series", "nan_random", "none_random",
                  "empty_random", "missing", "strings"):
            S.append(("obs", f"price/CO2 lists: {k}", df, L.o_lists(k), L.SOC0, ""))
        S.append(("obs", "cap_max_kwh / power_max_kw missing", df, L.o_drop("cap_max_kwh", "power_max_kw"), L.SOC0, ""))
        S.append(("obs", "cap / power as text", df, L.o_set(cap_max_kwh="6000.0", power_max_kw="1500.0"), L.SOC0, ""))
        S.append(("obs", "cap / power None", df, L.o_set(cap_max_kwh=None, power_max_kw=None), L.SOC0, ""))
        S.append(("obs", "hour / day_of_week missing", df, L.o_drop("hour", "day_of_week"), L.SOC0, ""))
        S.append(("obs", "preu_actual / co2_actual missing", df, L.o_drop("preu_actual", "co2_actual"), L.SOC0, ""))
        S.append(("obs", "soc_pct / soc_percent missing", df, L.o_drop("soc_pct", "soc_percent"), L.SOC0, ""))
        for k in ("out", "round2", "round0", "str", "pct_only", "none"):
            S.append(("obs", f"soc: {k}", df, L.o_soc(k), L.SOC0, ""))
        for s0 in (0.0, 1500.0, 6000.0):
            S.append(("obs", f"start charge {s0:.0f} kWh", df, None, s0, ""))
        for k in ("dict_none", "dict_missing", "dict_none_random", "all_nan_random", "load_nan_random", "pv_nan_random",
                  "numpy_scalars", "ints", "pv_not_x12", "current_hour",
                  "w_nan", "w_missing", "w_other_names", "w_const", "w_zero", "w_absurd", "w_inf", "w_none", "w_str",
                  "w_nan_random", "w_irr_kw", "w_irr_x2", "w_irr_half", "w_kelvin", "w_swapped"):
            S.append(("obs", f"readings: {k}", df, L.o_last(k), L.SOC0, ""))
        S.append(("obs", "string timestamps + numpy lists + no cap/power + soc out of range", df,
                  L.chain(L.o_ts("str"), L.o_lists("numpy"), L.o_drop("cap_max_kwh", "power_max_kw"), L.o_soc("out")),
                  L.SOC0, ""))

    if "layout" in families:
        S.append(("layout", "test calendar (last 7 days, weekdays shifted)", L.relabel(df, "last7"), None, L.SOC0, ""))
        S.append(("layout", "shifted by whole weeks (weekdays kept)", L.relabel(df, "plus7k"), None, L.SOC0, ""))
        d12 = df
        if wk_of(df, 11) is None and oct_ is not None:
            d12 = L.add_week(df, oct_, 28)                       # a November week: the October one, 4 weeks later
        S.append(("layout", "12 weeks (November = October week copied), real dates", d12, None, L.SOC0, ""))
        t12 = L.relabel(d12, "last7")
        S.append(("layout", "12 weeks on the test calendar", t12, None, L.SOC0, ""))
        S.append(("layout", "12 weeks on the test calendar + both clock changes", L.dst(t12), None, L.SOC0, ""))
        S.append(("layout", "12 weeks, test calendar, text timestamps", L.dst(t12), L.o_ts("str"), L.SOC0, ""))
        jan_dec = t12.iloc[np.argsort(t12.index.month * 10 ** 6 + np.arange(len(t12)), kind="stable")]
        S.append(("layout", "12 weeks, test calendar, weeks ordered January to December", jan_dec, None, L.SOC0, ""))
        S.append(("layout", "12 weeks, test calendar, 24 price/CO2 values", t12, L.o_lists("24"), L.SOC0, ""))
        S.append(("layout", "12 weeks, test calendar, lists shortened at every week end", t12, L.o_lists("short_weekend"), L.SOC0, ""))
        if aug is not None:
            S.append(("layout", "August week dropped", df[~L.week_mask(df, [aug])], None, L.SOC0, ""))
        n = len(df)
        S.append(("layout", "run ends 60 h early", df.iloc[:n - 60], None, L.SOC0, ""))
        S.append(("layout", "run ends after 5 weeks", df[L.weeks(df) < 5], None, L.SOC0, ""))
        S.append(("layout", "gaps of 2-10 h, one per week", L.gaps(df, 1), None, L.SOC0, f"{nw} gaps"))
        S.append(("layout", "gaps of 2-10 h starting 18:00, one per week", L.gaps(df, 2, hour=18), None, L.SOC0, f"{nw} gaps"))
        S.append(("layout", "gaps of 2-10 h, three per week", L.gaps(df, 3, per_week=3), None, L.SOC0, f"{3 * nw} gaps"))
        S.append(("layout", "12 duplicated hours", L.duplicate_rows(df, 4), None, L.SOC0, ""))

    if "lowload" in families:
        for mname, i in (("Dec", dec), ("Jul", jul)):
            if i is None:
                continue
            m = L.week_mask(df, [i])
            for f in (0.5, 0.6, 0.7):
                S.append(("lowload", f"{mname} week load x{f}", L.scale_load(df, f, m), None, L.SOC0, ""))
            S.append(("lowload", f"{mname} week: weekdays take the Sunday profile", L.sunday_week(df, [i]), None, L.SOC0, ""))
        S.append(("lowload", "every week: weekdays take the Sunday profile", L.sunday_week(df, range(nw)), None, L.SOC0, ""))
        S.append(("lowload", "whole run load x0.6", L.scale_load(df, 0.6), None, L.SOC0, ""))
        S.append(("lowload", "whole run load x1.3", L.scale_load(df, 1.3), None, L.SOC0, ""))

    if "faults" in families:
        S.append(("faults", "meter 0 for 3 h from 19:00, every week", L.meter_fault(df, L.event_mask(df, 3, 19)), None, L.SOC0, f"{nw} events"))
        S.append(("faults", "meter 0 for 14 h from 12:00, every week", L.meter_fault(df, L.event_mask(df, 14, 12)), None, L.SOC0, f"{nw} events"))
        S.append(("faults", "meter and solar 0 for 14 h from 12:00, every week", L.meter_fault(df, L.event_mask(df, 14, 12), pv_too=True), None, L.SOC0, f"{nw} events"))
        S.append(("faults", "meter 0 for 48 h from 06:00, every week", L.meter_fault(df, L.event_mask(df, 48, 6)), None, L.SOC0, f"{nw} events"))
        if oct_ is not None:
            m = L.week_mask(df, [oct_])
            S.append(("faults", "meter 0 for a whole week (Oct), solar normal", L.meter_fault(df, m, freeze=False), None, L.SOC0, "1 week"))
            S.append(("faults", "meter and solar 0 for a whole week (Oct)", L.meter_fault(df, m, pv_too=True, freeze=False), None, L.SOC0, "1 week"))
            S.append(("faults", "load held constant for a whole week (Oct)", L.const_week(df, [oct_]), None, L.SOC0, "1 week"))
        last = nw - 1 if aug is None else aug
        m = L.week_mask(df, [last])
        S.append(("faults", "solar 0 for the last week, irradiance normal", L.inverter_fault(df, m), None, L.SOC0, "1 week"))
        S.append(("faults", "solar and irradiance 0 for the last week", L.inverter_fault(df, m, irr_too=True), None, L.SOC0, "1 week"))
        S.append(("faults", "solar 0 for one day, every week", L.inverter_fault(df, L.event_mask(df, 24, 0)), None, L.SOC0, f"{nw} events"))
    return S


def main():
    kind, fold, fams = sys.argv[1], sys.argv[2], sys.argv[3]
    shipped = len(sys.argv) > 4 and sys.argv[4] == "shipped"
    families = ("obs", "layout", "lowload", "faults") if fams == "all" else tuple(fams.split(","))
    df = L.clean_folds()[fold]
    state = None if shipped else L.state_path(kind, fold)
    rows = []
    clean = None
    out = L.HERE / "out" / f"suite_{kind}_{fold}_{'-'.join(families)}{'_shipped' if shipped else ''}.csv"
    t00 = time.time()
    for fam, name, d, obs_fn, soc0, note in scenarios(df, families):
        agent = L.load_agent(kind, state)
        try:
            r = L.run(agent, d, obs_fn, soc0)
        except Exception as e:                       # the harness itself failed: keep going
            rows.append(dict(agent=kind, fold=fold, family=fam, scenario=name, harness_error=repr(e)[:200]))
            print("HARNESS ERROR", name, repr(e)[:200], flush=True)
            continue
        orc = L.oracle_score(d, soc0)
        row = dict(agent=kind, fold=fold, family=fam, scenario=name, note=note, n=r["n"], score=r["score"],
                   score_raw=r["score_raw"], oracle=orc, regret=orc - r["score"], infractions=r["infractions"],
                   exceptions=r["exceptions"], bad_returns=r["bad_returns"], fallbacks=r["fallbacks"],
                   load_mode=r["load_mode"], secs=r["secs"], max_step_ms=r["max_step_ms"], spill_kwh=r["spill_kwh"],
                   first_exc=r["first_exc"], last_error=r["last_error"])
        if clean is None:
            clean = row
        row["d_score"] = row["score"] - clean["score"]
        row["d_regret"] = row["regret"] - clean["regret"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{kind} {fold} {fam:8s} {name[:60]:60s} score {row['score']:7.3f} regret {row['regret']:6.3f} "
              f"d_reg {row['d_regret']:+7.3f} infr {row['infractions']} exc {row['exceptions']} bad {row['bad_returns']} "
              f"fb {row['fallbacks']} {row['secs']:.1f}s {row['first_exc']} {row['last_error']}", flush=True)
    print("done", round(time.time() - t00), "s ->", out, flush=True)


if __name__ == "__main__":
    main()
