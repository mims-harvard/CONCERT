#!/usr/bin/env python3
"""Perturb-map benchmark panel (Fig. 4j), drawn in the style of plot_perturbmap_benchmark.py.

Identical layout, sizes, colours, error bars, legend and limits to plot_perturbmap_benchmark.make_figure
(lines 80-129). One change requested by the author: bar faces (and the matching legend swatches)
are drawn at alpha = 0.7; the white bar edges, error bars and text stay fully opaque.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

# plot_perturbmap_benchmark (the shared style) lives with the Perturb-map figure scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "02_perturb_map_benchmark" / "figures"))
from plot_perturbmap_benchmark import GRID, INK, METHODS, MUTED, PERTS  # noqa: E402

BAR_ALPHA = 0.7


def make_figure(stats: dict, shared: dict, out: Path, stem: str = "pmap_benchmark") -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED, "axes.linewidth": 0.9,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.labelcolor": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
    })

    x = np.arange(len(PERTS))
    w = 0.78 / len(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

    def panel(ax, rk, sk, ylabel):
        for j, (m, _label, colour) in enumerate(METHODS):
            xs = x + (j - (len(METHODS) - 1) / 2) * w
            vals = np.array([stats.get((p, m), {}).get(rk, np.nan) for p in PERTS])
            errs = np.array([stats.get((p, m), {}).get(sk, np.nan) for p in PERTS])
            ax.bar(xs, vals, w, facecolor=to_rgba(colour, BAR_ALPHA), edgecolor="white",
                   linewidth=0.7, zorder=1)
            ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=MUTED,
                        elinewidth=1.0, capsize=2.5, zorder=2)
        ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n(n = {len(shared[p])})" for p in PERTS], fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_xlim(-0.5, len(PERTS) - 0.5)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    panel(axes[0], "rho", "rho_sem", "Log-fold-change rank correlation")
    panel(axes[1], "rmse", "rmse_sem", "Log-fold-change RMSE")
    lo = min(v["rho"] - (v["rho_sem"] if np.isfinite(v["rho_sem"]) else 0)
             for v in stats.values())
    axes[0].set_ylim(min(-0.02, lo * 1.35), None)
    axes[1].set_ylim(0, None)

    handles = [Patch(facecolor=to_rgba(c, BAR_ALPHA), edgecolor="white", label=lab) for _m, lab, c in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=len(METHODS), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=13, handlelength=1.4,
               columnspacing=1.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
