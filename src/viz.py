"""
Shared figure styling.

The categorical hues are used in fixed slot order and were validated for
colour-vision deficiency separation before use, rather than chosen by eye.
Slots 1-2 (blue, orange) measure CVD dE 24.7 and normal-vision dE 33.6 against
each other on this surface, clearing the >=8 and >=15 floors respectively.
Identity is never carried by colour alone: every multi-series figure here also
carries a legend and direct labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "output" / "figures"

# Categorical slots, assigned in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_BLUE, SERIES_ORANGE = SERIES[0], SERIES[1]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    """Install the shared rcParams. Call once at the top of a script."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
    })


def despine(ax, keep=("left", "bottom")) -> None:
    """Drop non-load-bearing chart chrome."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def caption(fig, text: str) -> None:
    """Add a source/reading note beneath the figure."""
    fig.text(0.01, -0.02, text, ha="left", va="top",
             fontsize=8, color=INK_MUTED, wrap=True)


def save(fig, name: str) -> Path:
    """Write a figure to output/figures/ and return the path."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
