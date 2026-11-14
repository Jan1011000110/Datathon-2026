"""What is the right RHO worth, and can an online estimate do better than a constant?

1. Fine sweep of a fixed RHO (0.50 .. 0.70, step 0.005) on A, B, C, ALL -> loss vs the fold's
   true RHO; fit loss ~ c * (rho - rho_true)^2.
2. Estimators (paired against v1 = fixed 0.60, per week):
     fixed 0.587 (train-wide), fixed true (cheat), calendar-table prediction (complement only),
     online blends of the agent's own observed baseline with a prior.
Output: out/rho_sweep.csv, out/rho_estimators.csv
"""
import importlib.util
from joblib import Parallel, delayed
from lib import *

_spec = importlib.util.spec_from_file_location("rho_tables", Path(__file__).parent / "04_rho_tables.py")
rho_tables = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rho_tables)
from src.data import TEST_MONTHS

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MEAN_COST_PER_H = 155.0          # EUR/h of baseline import, train average (only sets the prior's weight)
_F = {}


def fold(f):
    if not _F:
        _F.update(get_folds("filled"))
    return _F[f]


class OnlineRho:
    """Running estimate of the run's final CO2_base / Cost_base.

    mode 'fixed-mass' : (cum_co2 + rho0*M) / (cum_cost + M)                 M = prior weight in EUR
    mode 'remaining'  : (cum_co2 + rho0*R) / (cum_cost + R)   R = expected baseline cost still to come
    mode 'table'      : like 'remaining' but the part still to come is taken month by month from a
                        calendar table (mean hourly baseline CO2 / cost of that month in the train data)
    """

    def __init__(self, rho0=0.60, mode="fixed-mass", weeks=4.0, total_hours=2016, table=None, block_hours=168):
        self.rho0, self.mode = rho0, mode
        self.mass = weeks * 168 * MEAN_COST_PER_H
        self.total = total_hours * MEAN_COST_PER_H
        self.table, self.block_hours = table, block_hours
        self.cum_co2 = self.cum_cost = 0.0
        self.prev = None
        self.h_in_block = 0
        self.rem = (0.0, 0.0)

    def update(self, agent, obs):
        last = obs["noves_dades_5min"]
        net = max(0.0, float(last["ultim_consum_kw"]) - float(last["ultima_solar_kw"]))
        ts = obs["timestamp"]
        if self.prev is not None:                      # last hour's load x last hour's signals
            self.cum_co2 += net * self.prev[0]
            self.cum_cost += net * self.prev[1]
            jump = abs((ts - self.prev[2]).total_seconds() - 3600.0) > 1.0
            self.h_in_block = 0 if jump else self.h_in_block + 1
        self.prev = (float(obs["co2_actual"]), float(obs["preu_actual"]), ts)
        if self.mode == "table":
            key = (ts.year, ts.month)
            left = max(0, self.block_hours - self.h_in_block)
            c, e = self.table.get(key, (0.0, 0.0))
            rc, re = left * c, left * e
            for k in TEST_MONTHS[TEST_MONTHS.index(key) + 1:] if key in TEST_MONTHS else []:
                c, e = self.table.get(k, (0.0, 0.0))
                rc += self.block_hours * c
                re += self.block_hours * e
            self.rem = (rc, re)

    def value(self):
        if self.mode == "fixed-mass":
            return (self.cum_co2 + self.rho0 * self.mass) / (self.cum_cost + self.mass)
        if self.mode == "remaining":
            r = max(0.0, self.total - self.cum_cost)
            return (self.cum_co2 + self.rho0 * r) / (self.cum_cost + r) if self.cum_cost + r > 0 else self.rho0
        den = self.cum_cost + self.rem[1]
        return (self.cum_co2 + self.rem[0]) / den if den > 0 else self.rho0


def table_for(f):
    df = fold(f)
    t = rho_tables.month_table(train_complement(df, "full"))
    return {k: (float(r["co2"]), float(r["cost"])) for k, r in t.iterrows()}


def table_pred(f):
    t = table_for(f)
    return sum(t[k][0] for k in TEST_MONTHS) / sum(t[k][1] for k in TEST_MONTHS)


def estimators(f):
    df = fold(f)
    n = len(df)
    e = {"fixed 0.60 (v1)": lambda: 0.60,
         "fixed 0.587 (train-wide)": lambda: 0.587,
         "fixed 0.605 (public LB)": lambda: 0.605,
         "fixed true (cheat)": lambda: true_rho(df)}
    for wk in (1, 4, 12):
        e["online fixed-mass %dwk, prior 0.60" % wk] = lambda wk=wk: OnlineRho(0.60, "fixed-mass", weeks=wk)
    e["online remaining-mass, prior 0.60"] = lambda: OnlineRho(0.60, "remaining", total_hours=n)
    if f != "ALL":                                   # tables need a complement to be fitted on
        e["fixed calendar-table prediction"] = lambda: table_pred(f)
        e["online calendar-table remainder"] = lambda: OnlineRho(0.60, "table", table=table_for(f))
    return e


def sweep_one(f, rho):
    return f, rho, run(GapAgent(rho=rho), fold(f))["score"]


def est_one(f, name):
    df = fold(f)
    r = run(GapAgent(rho=estimators(f)[name]()), df)
    return f, name, r["score"], r["infractions"], {"contrib": r["contrib"], "trace": r["trace"][["soc_kwh"]]}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    rhos = np.round(np.arange(0.50, 0.7001, 0.005), 3)
    res = Parallel(n_jobs=4, batch_size=8)(delayed(sweep_one)(f, r) for f in FOLDS for r in rhos)
    sw = pd.DataFrame(res, columns=["fold", "rho", "score"]).pivot(index="rho", columns="fold", values="score")[list(FOLDS)]
    best = {f: run(GapAgent(rho=true_rho(fold(f))), fold(f))["score"] for f in FOLDS}
    loss = (pd.Series(best) - sw) * 1e4
    loss.to_csv(OUT / "rho_sweep.csv")
    print("loss vs the fold's true RHO (1e-4 points); true RHO:", {f: round(true_rho(fold(f)), 4) for f in FOLDS})
    print(loss.round(2).to_string())
    # curvature: loss ~ c * d^2 (least squares through the origin, |d| <= 0.06)
    for f in FOLDS:
        d = loss.index.values - true_rho(fold(f))
        m = np.abs(d) <= 0.06
        c = float(np.sum(loss[f].values[m] * d[m] ** 2) / np.sum(d[m] ** 4))
        print("  fold %-3s  loss ~ %.2f * (d_rho)^2 points   -> d=0.005: %.2fe-4   d=0.01: %.2fe-4   d=0.02: %.2fe-4   d=0.05: %.2fe-4" % (
            f, c * 1e-4, c * 0.005 ** 2, c * 0.01 ** 2, c * 0.02 ** 2, c * 0.05 ** 2))
    # empirical: mean / worst loss when the RHO used is off by a given amount (interpolated on the sweep)
    print("\nempirical loss for an error of +-d (linear interpolation of the sweep), 1e-4 points:")
    for d in (0.005, 0.01, 0.02, 0.03, 0.05):
        row = []
        for f in FOLDS:
            rt = true_rho(fold(f))
            row.append("%s: -d %.2f / +d %.2f" % (f, np.interp(rt - d, loss.index.values, loss[f].values), np.interp(rt + d, loss.index.values, loss[f].values)))
        print("  d=%.3f  " % d + "   ".join(row))

    tasks = [(f, n) for f in FOLDS for n in estimators(f)]
    res = Parallel(n_jobs=4, batch_size=2)(delayed(est_one)(f, n) for f, n in tasks)
    store = {(f, n): (s, k, c) for f, n, s, k, c in res}
    rows = []
    for f, n in tasks:
        s, k, c = store[(f, n)]
        ref = store[(f, "fixed 0.60 (v1)")]
        g = paired(fold(f), c, ref[2])
        st = summarize(g)
        rows.append({"fold": f, "estimator": n, "score": s, "gain_1e-4": (s - ref[0]) * 1e4, "weeks+": st["pos"], "weeks-": st["neg"],
                     "worst_week_1e-4": st["min"] * 1e4, "best_week_1e-4": st["max"] * 1e4, "infr": k})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "rho_estimators.csv", index=False)
    print("\nestimators, paired against fixed 0.60 (gain in 1e-4 points; weeks with gain / loss):")
    for f in FOLDS:
        print("fold", f, " table prediction: %s" % ("%.4f" % table_pred(f) if f != "ALL" else "n/a"))
        print(tab[tab.fold == f].drop(columns="fold").round(4).to_string(index=False))
