"""
experiments/04_whatif/common.py
Shared code for the what-if analyses (reports/04_whatif/): data, a parametrised
perfect-foresight battery LP, physical accounting, annual scaling and chart style.

Run every script from the repo root:
    .venv/Scripts/python.exe experiments/04_whatif/a1_annual.py        (a1 .. a7)
    .venv/Scripts/python.exe experiments/04_whatif/run_all.py          (everything)

Conventions
-----------
* Data: all train hours of Sep 2025 .. Aug 2026 (`get_folds("filled")["ALL"]`, 6,744 h =
  281 days, the 12 month-blocks concatenated; missing load/PV imputed, see src/data.py).
* The LP is the same model as src.sim.solve_lp, with capacity, power, efficiency, the weights
  and a few optional features (export credit, cap on grid import) as arguments.
* ENERGY-NEUTRAL by default: the battery must end at the state of charge it started with
  (50 %). The official simulator starts half full and lets you end empty, which hands every
  agent 1,000 kWh for free; that gift grows with capacity and would flatter the sizing sweep.
  `terminal="free"` reproduces src.sim.oracle exactly.
* A tiny throughput penalty (TIE) breaks ties towards not cycling when a trade is worth nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_hourly_filled  # noqa: E402
from src.sim import get_folds  # noqa: E402

OUT_DIR = ROOT / "reports" / "04_whatif"
FIG_DIR = OUT_DIR / "figs"
STATS_DIR = OUT_DIR / "stats"
for _d in (FIG_DIR, STATS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- design point
CAP = 2000.0      # kWh
PMAX = 500.0      # kW
RTE = 0.90        # round-trip efficiency
SOC0_FRAC = 0.5
ALPHA = 0.7       # weight on CO2 in the challenge score
TIE = 1e-6        # throughput penalty, as a fraction of the mean weight (tie-breaker only)

MONTH_ORDER = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]       # the competition year, Sep -> Aug
MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
              9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Autumn", "Winter", "Spring", "Summer"]


# ----------------------------------------------------------------------------- data
def load_all() -> pd.DataFrame:
    """The 6,744 train hours (281 days) of Sep 2025 .. Aug 2026, with `net` = load - pv."""
    df = get_folds("filled")["ALL"].copy()
    df["net"] = df["load_kw"] - df["pv_kw"]
    df["imputed"] = load_hourly_filled()["imputed"].reindex(df.index).to_numpy()
    return df


class Problem:
    """Arrays + baseline totals for one scenario (optionally with PV scaled by `pv_scale`)."""

    def __init__(self, df: pd.DataFrame | None = None, pv_scale: float = 1.0, export_price: float = 0.0):
        self.df = load_all() if df is None else df
        self.idx = self.df.index
        self.load = self.df["load_kw"].to_numpy(float)
        self.pv = self.df["pv_kw"].to_numpy(float) * pv_scale
        self.net = self.load - self.pv
        self.price = self.df["precio_eur_per_kwh"].to_numpy(float)
        self.co2 = self.df["co2_kg_per_kwh"].to_numpy(float)
        self.export_price = float(export_price)
        self.base = np.maximum(self.net, 0.0)                 # no-battery grid import (kWh per hour)
        self.base_export = np.maximum(-self.net, 0.0)
        self.co2_base = float(np.sum(self.base * self.co2))
        self.cost_base = float(np.sum(self.base * self.price) - export_price * np.sum(self.base_export))
        self.energy_base = float(np.sum(self.base))
        self.n = len(self.df)
        self.days = self.n / 24.0

    def weights(self, alpha: float = ALPHA, ref: "Problem | None" = None) -> np.ndarray:
        """Per-kWh weight of grid import: alpha*co2/CO2_base + (1-alpha)*price/Cost_base.
        With these weights `1 - sum(w*grid)` is the score as a fraction. `ref` = scenario whose
        baseline totals set the kg <-> EUR exchange rate (default: this one)."""
        r = self if ref is None else ref
        return alpha * self.co2 / r.co2_base + (1.0 - alpha) * self.price / r.cost_base

    def carbon_price_eur_per_t(self, alpha: float = ALPHA) -> float:
        """EUR per tonne of CO2 at which the score is indifferent (the weights' exchange rate)."""
        return 1000.0 * alpha / (1.0 - alpha) * self.cost_base / self.co2_base


# ----------------------------------------------------------------------------- the LP
def solve(prob: Problem, w: np.ndarray, cap: float = CAP, pmax: float = PMAX, rte: float = RTE,
          soc0_frac: float = SOC0_FRAC, terminal: str = "neutral", export_credit: float = 0.0,
          grid_cap: np.ndarray | float | None = None, tie: float = TIE, duals: bool = False,
          cyclic: np.ndarray | None = None) -> dict:
    """Perfect-foresight optimum.

        min  sum_t w_t g_t - export_credit * sum_t e_t  (+ tiny throughput penalty)
        s.t. s_t = s_{t-1} + eta*c_t - d_t/eta          (s_0 = soc0_frac*cap)
             g_t - e_t = net_t + c_t - d_t              (g = import, e = export, both >= 0)
             0 <= c_t, d_t <= pmax ;  0 <= s_t <= cap ;  g_t <= grid_cap_t (optional)
             s_T >= s_0  if terminal == "neutral"

    `w` and `export_credit` share units (score fraction per kWh). export_credit must stay below
    min(w), otherwise importing and exporting in the same hour would look profitable.
    `cyclic`: block id per hour (contiguous blocks, e.g. the calendar day). Each block is then
    solved on its own with a free but periodic state of charge (end = start), i.e. the best
    the battery can do on that block's signal alone; soc0_frac / terminal are ignored.
    Returns dict(c, d, s, g, e, p=c-d, obj) and, if duals=True, the bound sensitivities.
    """
    net = prob.net
    T = len(w)
    eta = rte ** 0.5
    w = np.asarray(w, float)
    if export_credit > 0 and export_credit >= w.min():
        raise ValueError("export_credit must be below min(w)")
    scale = 1.0 / float(np.mean(np.abs(w)))        # score weights are ~1e-7: rescale for HiGHS
    eps = tie                                       # in scaled units the mean weight is 1

    I = sparse.identity(T, format="csr")
    Z = sparse.csr_matrix((T, T))
    b_soc = np.zeros(T)
    if cyclic is None:
        L = sparse.diags([np.ones(T), -np.ones(T - 1)], [0, -1], format="csr")  # s_t - s_{t-1}
        b_soc[0] = soc0_frac * cap
    else:
        blk = np.asarray(cyclic)
        first = np.r_[True, blk[1:] != blk[:-1]]
        last_of_block = np.r_[np.flatnonzero(first)[1:] - 1, T - 1]             # per block
        prev = np.arange(T) - 1
        prev[first] = last_of_block                                             # wrap inside the block
        L = sparse.csr_matrix((np.r_[np.ones(T), -np.ones(T)], (np.r_[np.arange(T), np.arange(T)],
                                                                 np.r_[np.arange(T), prev])), shape=(T, T))
        terminal = "free"
    # x = [c, d, s, g, e]
    A_soc = sparse.hstack([-eta * I, I / eta, L, Z, Z], format="csr")
    A_bal = sparse.hstack([-I, I, Z, I, -I], format="csr")                      # g - e - c + d = net
    A_eq = sparse.vstack([A_soc, A_bal], format="csr")
    b_eq = np.concatenate([b_soc, net])

    cost = np.concatenate([np.full(T, eps), np.full(T, eps), np.zeros(T), w * scale,
                           np.full(T, -export_credit * scale)])
    lb = np.zeros(5 * T)
    ub = np.concatenate([np.full(T, pmax), np.full(T, pmax), np.full(T, cap),
                         np.full(T, np.inf), np.full(T, np.inf)])
    if terminal == "neutral":
        lb[3 * T - 1] = soc0_frac * cap
    elif terminal != "free":
        raise ValueError(terminal)
    if grid_cap is not None:
        ub[3 * T:4 * T] = np.broadcast_to(np.asarray(grid_cap, float), (T,))
    r = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=np.column_stack([lb, ub]), method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    x = r.x
    out = {"c": x[:T], "d": x[T:2 * T], "s": x[2 * T:3 * T], "g": x[3 * T:4 * T], "e": x[4 * T:],
           "obj": r.fun / scale, "cap": cap, "pmax": pmax, "rte": rte}
    out["p"] = out["c"] - out["d"]
    if duals:
        up = r.upper.marginals / scale            # d(obj)/d(upper bound) <= 0
        lo = r.lower.marginals / scale
        out["d_obj_d_cap"] = float(np.sum(up[2 * T:3 * T]))          # s <= cap only (soc0 held fixed)
        out["d_obj_d_pmax_c"] = float(np.sum(up[:T]))
        out["d_obj_d_pmax_d"] = float(np.sum(up[T:2 * T]))
        out["hours_at_cap"] = int(np.sum(up[2 * T:3 * T] < -1e-15))
        out["hours_at_empty"] = int(np.sum(lo[2 * T:3 * T] > 1e-15))
        out["hours_pmax_c"] = int(np.sum(up[:T] < -1e-15))
        out["hours_pmax_d"] = int(np.sum(up[T:2 * T] < -1e-15))
        out["soc_price"] = -r.eqlin.marginals[:T] / scale            # value of 1 kWh stored after hour t
    return out


def account(prob: Problem, sol: dict) -> dict:
    """Physical results of a schedule against the scenario's no-battery baseline."""
    g, e = sol["g"], sol["e"]
    co2_a = float(np.sum(g * prob.co2))
    cost_a = float(np.sum(g * prob.price) - prob.export_price * np.sum(e))
    d_co2 = prob.co2_base - co2_a
    d_cost = prob.cost_base - cost_a
    eta = sol["rte"] ** 0.5
    out_cells = float(np.sum(sol["d"]) / eta)                    # energy drawn from the cells
    return {
        "score": 100 * (ALPHA * d_co2 / prob.co2_base + (1 - ALPHA) * d_cost / prob.cost_base),
        "co2_pct": 100 * d_co2 / prob.co2_base,
        "cost_pct": 100 * d_cost / prob.cost_base,
        "co2_saved_kg": d_co2,
        "cost_saved_eur": d_cost,
        "charged_kwh": float(np.sum(sol["c"])),
        "discharged_kwh": float(np.sum(sol["d"])),
        "cycles_per_day": out_cells / sol["cap"] / prob.days,
        "exported_kwh": float(np.sum(e)),
        "peak_import_kw": float(np.max(g)),
    }


def run(prob: Problem, alpha: float = ALPHA, **kw) -> tuple[dict, dict]:
    """solve + account with the challenge weights (alpha on CO2)."""
    sol = solve(prob, prob.weights(alpha), **kw)
    return sol, account(prob, sol)


# ----------------------------------------------------------------------------- annual scaling
def annual_factors(idx: pd.DatetimeIndex) -> pd.Series:
    """Per-hour multiplier that turns a sum over train hours into a full-year figure.

    Assumption: the missing last week of each month looks like the ~3 weeks we have, so each
    month's train total is scaled by (days in the month) / (train days in the month).
    Sums of `value * factor` over all train hours are then Sep 2025 .. Aug 2026 totals (365 d).
    """
    ym = idx.to_period("M")
    train_days = pd.Series(1.0, index=idx).groupby(ym).transform("size") / 24.0
    return pd.Series(ym.days_in_month.to_numpy() / train_days.to_numpy(), index=idx)


def annualise(values: np.ndarray, idx: pd.DatetimeIndex) -> float:
    return float(np.sum(np.asarray(values, float) * annual_factors(idx).to_numpy()))


def annual_savings(prob: Problem, sol: dict) -> dict:
    """Full-year savings of a schedule: tonnes of CO2 and euros (month-by-month scaling)."""
    fac = annual_factors(prob.idx).to_numpy()
    dg = prob.base - sol["g"]
    de = sol["e"] - prob.base_export
    return {
        "co2_t_per_year": float(np.sum(dg * prob.co2 * fac) / 1000.0),
        "eur_per_year": float(np.sum((dg * prob.price + de * prob.export_price) * fac)),
    }


# ----------------------------------------------------------------------------- style
# Palette checked with the dataviz skill's validate_palette.py (light surface, --pairs all):
#   CO2 / cost / score  #2a78d6, #eb6834, #1baf7a  -> pass (aqua needs direct labels: 2.7:1)
#   weekday / weekend   #6250d6, #eda100           -> pass
# Colour follows the entity in every figure: CO2 = blue, euros = orange, score = aqua.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GRAY = "#b5b3ab"

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
C_CO2, C_COST, C_SCORE = BLUE, ORANGE, AQUA
C_WEEKDAY, C_WEEKEND = VIOLET, YELLOW

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_AQUA = LinearSegmentedColormap.from_list(
    "seq_aqua", ["#e9f8f1", "#c6eedc", "#93dfbf", "#55c99b", "#1baf7a", "#128c60", "#0c6a48", "#084c33"])
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fdf1ea", "#fbd9c8", "#f6b596", "#f09064", "#eb6834", "#c44f1f", "#983a14", "#6e290c"])
# diverging: blue (charge) <-> red (discharge), neutral gray midpoint
DIV_BLUE_RED = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a3", "#e34948", "#a32626"])


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 12,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.labelsize": 12, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 11,
        "figure.dpi": 100, "savefig.dpi": 150,
    })


def titled(fig, title: str, subtitle: str | None = None, y: float = 0.975, x: float = 0.035) -> None:
    """Slide-style header: bold takeaway title + one grey context line."""
    fig.text(x, y, title, fontsize=17, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(x, y - 0.052, subtitle, fontsize=11.5, color=INK2, ha="left", va="top")


def footnote(fig, text: str, y: float = 0.012, x: float = 0.035) -> None:
    fig.text(x, y, text, fontsize=9.5, color=MUTED, ha="left", va="bottom")


FOOT = ("Train data Sep 2025 – Aug 2026 (281 days), perfect-foresight LP, battery ends each run as full as it "
        "started. Upper bound; our real-time controller reaches it to within ~0.2 %.")


def save(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig] {path.relative_to(ROOT)}")
    return path


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period)):
        return str(o)
    return o


def write_stats(name: str, stats: dict) -> None:
    path = STATS_DIR / f"{name}.json"
    path.write_text(json.dumps(_clean(stats), indent=2), encoding="utf-8")
    print(f"[stats] {path.relative_to(ROOT)}")
