import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.prop_cycle": matplotlib.cycler(color=SERIES), "lines.linewidth": 2,
        "legend.frameon": False, "savefig.bbox": "tight",
    })
