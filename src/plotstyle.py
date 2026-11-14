"""
src/plotstyle.py
One look for every figure of the project (matplotlib). Usage:

    from src.plotstyle import apply, C, INK, save
    apply()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, y, color=C[0])
    save(fig, "reports/10_hard_mode/figs/name.png")

Rules of thumb: colour by job (one hue for magnitude, C[...] in fixed order for distinct series,
grey + one accent to emphasise), never two y-axes, a title that states the finding, direct labels
where there are few marks, solid hairline grid.
"""
import matplotlib
import matplotlib.pyplot as plt

# categorical slots, fixed order (colour-blind-safe on adjacent slots; keep to <= 4 series if you can)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # one hue, light -> dark
DIV = ("#2a78d6", "#f0efec", "#e34948")                                               # low, neutral, high
GREY = "#b9b7ae"                                                                        # de-emphasised marks
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}


def apply():
    matplotlib.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "text.color": INK["primary"],
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10, "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": INK["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 9,
        "axes.prop_cycle": matplotlib.cycler(color=C),
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
