"""
experiments/eda/common.py
Shared helpers for the EDA scripts: paths, chart style, calendar helpers, the combined
score signal `w`, and a small JSON "stats" sink so every number quoted in
reports/01_eda/report.md can be traced back to a script.

Run every script from the repo root:  .venv/Scripts/python.exe experiments/eda/qN_xxx.py
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252, which cannot print "CO₂" or "€"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from src.data import (  # noqa: E402
    RAW_DIR,
    TEST_MONTHS,
    load_hourly_filled,
    load_hourly_full,
    load_hourly_official,
)

OUT_DIR = ROOT / "reports" / "01_eda"
FIG_DIR = OUT_DIR / "figs"
STATS_DIR = OUT_DIR / "stats"
for _d in (FIG_DIR, STATS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- style
# Palette: validated with the dataviz skill's validate_palette.py (light surface).
#   signals  (CO2, price, w)       #2a78d6, #eb6834, #1baf7a  -> passes --pairs all
#   day type (weekday, weekend)    #6250d6, #eda100           -> passes --pairs all
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
GRAY = "#b5b3ab"  # de-emphasis

C_CO2, C_PRICE, C_W = BLUE, ORANGE, AQUA  # colour follows the entity in every figure
C_WEEKDAY, C_WEEKEND = VIOLET, YELLOW
C_LOAD, C_PV = VIOLET, YELLOW

# status colours (coverage map only)
ST_GOOD, ST_WARN, ST_CRIT = "#0ca30c", "#fab219", "#d03b3b"

# sequential ramps: one hue each, light -> dark
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fdf1ea", "#fbd9c8", "#f6b596", "#f09064", "#eb6834", "#c44f1f", "#983a14", "#6e290c"])
SEQ_AQUA = LinearSegmentedColormap.from_list(
    "seq_aqua", ["#e9f8f1", "#c6eedc", "#93dfbf", "#55c99b", "#1baf7a", "#128c60", "#0c6a48", "#084c33"])
SEQ_VIOLET = LinearSegmentedColormap.from_list(
    "seq_violet", ["#f1effc", "#dcd7f7", "#bbb2ef", "#998ce6", "#6250d6", "#4a3aa7", "#372a80", "#261c5c"])
SEQ_YELLOW = LinearSegmentedColormap.from_list(
    "seq_yellow", ["#fdf6e3", "#fae7b3", "#f6d27a", "#f1bb3d", "#eda100", "#bd8000", "#8d5f00", "#614100"])
# diverging: blue <-> red with a neutral gray midpoint
DIV_BLUE_RED = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a3", "#e34948", "#a32626"])

MONTH_ORDER = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]  # the competition year, Sep -> Aug
MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
              9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Autumn", "Winter", "Spring", "Summer"]

RTE = 0.90                 # round-trip efficiency
BREAK_EVEN = 1.0 / RTE     # the dear hour must be >= 11.1 % worse than the cheap hour
P_MAX = 500.0              # kW
E_CAP = 2000.0             # kWh


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
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def write_stats(name: str, stats: dict) -> None:
    path = STATS_DIR / f"{name}.json"
    path.write_text(json.dumps(_clean(stats), indent=2), encoding="utf-8")
    print(f"[stats] {path.relative_to(ROOT)}")


def hour_axis(ax, step: int = 3, label: str = "Hour of day (local time)") -> None:
    ax.set_xticks(range(0, 24, step))
    ax.set_xlim(-0.5, 23.5)
    if label:
        ax.set_xlabel(label)


# ----------------------------------------------------------------------------- calendar
def test_days() -> pd.DatetimeIndex:
    """Hidden test set: the last 7 days of each calendar month, Sep 2025 .. Aug 2026."""
    out = []
    for y, m in TEST_MONTHS:
        end = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
        out.extend(pd.date_range(end - pd.Timedelta(days=6), end, freq="D"))
    return pd.DatetimeIndex(out)


def is_weekend(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(idx.day_of_week >= 5)


def day_matrix(s: pd.Series) -> pd.DataFrame:
    """Hourly series -> DataFrame (index = date, columns = hour 0..23). Days with a missing
    hour keep NaN in that cell."""
    df = pd.DataFrame({"v": s.values, "d": s.index.normalize(), "h": s.index.hour})
    return df.pivot(index="d", columns="h", values="v").sort_index()


def block_id(days: pd.DatetimeIndex) -> np.ndarray:
    """Id of the contiguous train block each day belongs to (a new block starts after any gap)."""
    gap = np.r_[True, np.diff(days.values).astype("timedelta64[D]").astype(int) > 1]
    return np.cumsum(gap) - 1


# ----------------------------------------------------------------------------- score signal
def baseline_totals(df: pd.DataFrame | None = None) -> dict:
    """No-battery baseline on the organisers' table (official pipeline, NaN rows dropped)."""
    if df is None:
        df = load_hourly_official()
    grid = (df["load_kw"] - df["pv_kw"]).clip(lower=0)
    e = float(grid.sum())
    co2 = float((grid * df["co2_kg_per_kwh"]).sum())
    cost = float((grid * df["precio_eur_per_kwh"]).sum())
    return {"hours": int(len(df)), "energy_kwh": e, "co2_kg": co2, "cost_eur": cost,
            "co2_per_kwh": co2 / e, "eur_per_kwh": cost / e, "kg_per_eur": co2 / cost}


def add_w(df: pd.DataFrame, base: dict | None = None) -> pd.DataFrame:
    """Add the score's per-kWh signal, normalised so that its import-weighted mean is 1:

        w_t = 0.7 * co2_t / c_bar + 0.3 * price_t / p_bar ,
        c_bar = CO2_base / E_base ,  p_bar = Cost_base / E_base      (train baseline totals)

    Moving 1 kWh of grid import out of hour t changes the score by 100 * w_t / E_base points,
    so w is the lead's  0.7*co2/CO2_base + 0.3*price/Cost_base  multiplied by the constant E_base.
    """
    if base is None:
        base = baseline_totals()
    out = df.copy()
    out["w_co2"] = 0.7 * out["co2_kg_per_kwh"] / base["co2_per_kwh"]
    out["w_price"] = 0.3 * out["precio_eur_per_kwh"] / base["eur_per_kwh"]
    out["w"] = out["w_co2"] + out["w_price"]
    return out


SIGNALS = {
    "co2": {"col": "co2_kg_per_kwh", "label": "CO₂ intensity", "unit": "g CO₂/kWh", "scale": 1000.0,
            "color": C_CO2, "cmap": SEQ_BLUE},
    "price": {"col": "precio_eur_per_kwh", "label": "PVPC price", "unit": "c€/kWh", "scale": 100.0,
              "color": C_PRICE, "cmap": SEQ_ORANGE},
    "w": {"col": "w", "label": "Combined signal w", "unit": "relative value, mean = 1", "scale": 1.0,
          "color": C_W, "cmap": SEQ_AQUA},
}


def signals_frame() -> pd.DataFrame:
    """All 6,768 train hours with price, CO2 and w (plus calendar columns)."""
    df = add_w(load_hourly_full())
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["weekend"] = is_weekend(df.index)
    return df
