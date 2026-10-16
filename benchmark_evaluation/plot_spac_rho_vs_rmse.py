#!/usr/bin/env python3
"""Both benchmark metrics on one plane: rank correlation against negative RMSE.

Up and to the right is better on both axes at once. Two versions per arm:

  overall   one dot per method, its mean over all scored knockouts
  per_pert  one dot per method and knockout, with the method mean drawn on top

Usage: plot_spac_rho_vs_rmse.py [--indir DIR] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import release_paths as rp  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_spac_benchmark_bars import COLORS  # noqa: E402
from plot_spac_overall_rank import load_regime  # noqa: E402

REGIMES = [("seen", "within-niche"), ("unseen", "cross-slide")]
XLAB = "Log-fold-change rank correlation"
YLAB = "Negative log-fold-change RMSE"
LABEL_FS, AXIS_FS, LEG_FS = 15, 17, 14


def _finish(ax, fig, order, out_pdf):
    ax.set_xlabel(XLAB, fontsize=AXIS_FS)
    ax.set_ylabel(YLAB, fontsize=AXIS_FS)
    ax.tick_params(labelsize=LABEL_FS)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                          markerfacecolor=COLORS[m], markeredgecolor="#2B2B2B",
                          markeredgewidth=0.6, label=m) for m in order]
    ax.legend(handles=handles, fontsize=LEG_FS, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=rp.SPACSEQ_RESULTS)
    ap.add_argument("--outdir", type=Path,
                    default=rp.WORKDIR / "spacseq_results" / "figures" / "rho_vs_rmse")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for regime, tag in REGIMES:
        per_gene = load_regime(args.indir, regime)
        rho = per_gene.pivot(index="gene", columns="method_label", values="lfc_spearman")
        rmse = per_gene.pivot(index="gene", columns="method_label", values="lfc_dist")
        order = (rho.mean() - rmse.mean()).sort_values(ascending=False).index.tolist()

        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        for m in order:
            ax.scatter(rho[m].mean(), -rmse[m].mean(), s=320, color=COLORS[m],
                       edgecolor="#2B2B2B", linewidth=0.9, zorder=3)
        _finish(ax, fig, order, args.outdir / f"spacseq_{regime}_rho_vs_rmse_overall.pdf")

        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        for m in order:
            ax.scatter(rho[m], -rmse[m], s=26, color=COLORS[m], alpha=0.40,
                       linewidths=0, zorder=2)
        for m in order:
            ax.scatter(rho[m].mean(), -rmse[m].mean(), s=300, color=COLORS[m],
                       edgecolor="#2B2B2B", linewidth=1.3, zorder=4)
        _finish(ax, fig, order, args.outdir / f"spacseq_{regime}_rho_vs_rmse_per_pert.pdf")

        stats[regime] = {m: {"mean_rho": round(float(rho[m].mean()), 4),
                             "mean_rmse": round(float(rmse[m].mean()), 4),
                             "n_knockouts": int(rho[m].notna().sum())} for m in order}
        _ = tag

    (args.outdir / "rho_vs_rmse_stats.json").write_text(json.dumps(stats, indent=2))
    print("wrote rho_vs_rmse_stats.json")


if __name__ == "__main__":
    main()
