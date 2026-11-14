"""Same 45 test-like runs as 13_many_runs.py: (a) which fixed RHO is the most robust, (b) is the
candidate v1_tv ever worse than v1. Output: out/many_runs_choices.csv"""
import importlib.util
from joblib import Parallel, delayed
from lib import *

_s = importlib.util.spec_from_file_location("many", Path(__file__).parent / "13_many_runs.py")
many = importlib.util.module_from_spec(_s); _s.loader.exec_module(many)
OUT = Path(__file__).parent / "out"
RHOS = [0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63]


def one(name, starts):
    df = many.build(starts)
    cand = load_agent_module(Path(__file__).parent / "candidates" / "v1_tv" / "agent.py", "cand_v1_tv")
    rt = true_rho(df)
    row = {"run": name, "true_rho": rt, "best": simulate(GapAgent(rho=rt), df)["score"]}
    for r in RHOS:
        row["rho=%.2f" % r] = simulate(GapAgent(rho=r), df)["score"]
    a = run(v1.BatteryAgent(), df)
    b = run(cand.BatteryAgent(), df)
    row["v1"] = a["score"]
    row["v1_tv"] = b["score"]
    row["tv_hours_changed"] = int((np.abs(a["trace"].p_kw - b["trace"].p_kw) > 1e-6).sum())
    row["tv_infr"] = b["infractions"]
    row["tv_worst_week"] = float(paired(df, b, a).min())
    return row


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    tab = pd.DataFrame(Parallel(n_jobs=4, batch_size=1)(delayed(one)(n, s) for n, s in many.plans())).set_index("run")
    tab.to_csv(OUT / "many_runs_choices.csv")
    loss = pd.DataFrame({c: (tab["best"] - tab[c]) * 1e4 for c in tab.columns if c.startswith("rho=")})
    print("loss vs each run's true RHO (1e-4 points) over 45 runs; true RHO range %.3f-%.3f, mean %.3f, sd %.3f" % (
        tab.true_rho.min(), tab.true_rho.max(), tab.true_rho.mean(), tab.true_rho.std()))
    print(pd.DataFrame({"mean": loss.mean(), "median": loss.median(), "p90": loss.quantile(0.9), "max": loss.max(),
                        "runs>10 (0.001)": (loss > 10).sum()}).round(2).to_string())
    g = (tab["v1_tv"] - tab["v1"]) * 1e4
    print("\nv1_tv - v1 over 45 runs (1e-4 points): mean %.3f  min %.3f  max %.3f ; runs better %d, equal %d, worse %d ; hours changed max %d ; infractions %d ; worst single week %.3f" % (
        g.mean(), g.min(), g.max(), int((g > 1e-6).sum()), int((g.abs() <= 1e-6).sum()), int((g < -1e-6).sum()),
        tab.tv_hours_changed.max(), tab.tv_infr.sum(), tab.tv_worst_week.min() * 1e4))
    print(g.round(3).to_string())
