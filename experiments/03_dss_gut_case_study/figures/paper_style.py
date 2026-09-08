"""One visual system for every case-study figure, so the panels read as a set.

Import and call `use()` before creating any figure. Everything here is deliberately
restrained: thin spines, no top/right frame, one accent colour, and a fixed colour per
DSS day so a day means the same thing in every panel of every figure.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# fixed colour per DSS timepoint -- consistent across every figure in the case study
DAY_COLORS = {0.0: "#3b6ea5", 3.0: "#4c9a6a", 9.0: "#e08214", 21.0: "#b2182b"}
ACCENT = "#b2182b"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
# sequential map for spatial fields / expression; perceptually ordered, prints legibly
FIELD_CMAP = "magma"
DENSITY_CMAP = "magma_r"


MICE_ORDERED = [
    "062921_D0_m3a", "082421_D0_m6", "082421_D0_m7",                       # Day 0
    "092421_D3_m1", "092421_D3_m2", "092421_D3_m3", "092421_D3_m4",        # Day 3
    "062221_D9_m3", "062921_D9_m2a", "062921_D9_m5",
    "100221_D9_m2", "100221_D9_m3", "100221_D9_m5",                        # Day 9
    "082421_D21_m1", "082421_D21_m2",                                      # Day 21
]
MOUSE_NUMBER = {m: i + 1 for i, m in enumerate(MICE_ORDERED)}


def day_label(d):
    """'Day 9'. Never an abbreviation, never a phase word."""
    return f"Day {int(round(float(d)))}"


def mouse_label(mouse_id, day=None, slice_id=None, sep=", "):
    """'Mouse 9, Day 9, Slide 20' -- spelled out, no abbreviations, no cell counts."""
    parts = [f"Mouse {MOUSE_NUMBER[mouse_id]}"]
    if day is not None:
        parts.append(day_label(day))
    if slice_id is not None:
        parts.append(f"Slide {int(slice_id)}")
    return sep.join(parts)


_CAL = None


def calibration():
    global _CAL
    if _CAL is None:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "calibration.json")) as fh:
            _CAL = json.load(fh)
    return _CAL


def calibrate(pred, marker=None):
    """Put a prediction on the observed scale. marker=None -> aggregate 17-marker score.

    Monotone in `pred` (every displayed marker has a_f > 0), so any readout defined by a
    rate-matched threshold -- the binary inflamed-region map -- is invariant to it.
    """
    cal = calibration()
    e = cal["aggregate"] if marker is None else cal["per_marker"][marker]
    return e["observed_baseline"] + e["a"] * (np.asarray(pred) - e["pred_baseline"])


def day_color(d):
    """Colour for a DSS day, snapping to the nearest known timepoint.

    Days arrive as float32 (3.0 reads back as 3.000000238), so a dict lookup keyed on
    exact floats raises KeyError. Snap instead of rounding blindly, so an unexpected
    value is still drawn rather than crashing a figure at the last line.
    """
    return DAY_COLORS[min(DAY_COLORS, key=lambda k: abs(k - float(d)))]


def use():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#4d4d4d",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "text.color": INK,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })


def panel_label(ax, letter, dx=-0.11, dy=1.06):
    """Bold panel letter in figure-corner position, outside the axes box."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left", color=INK)


def bare(ax):
    """Strip an axis down to the data (for spatial scatter panels)."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
