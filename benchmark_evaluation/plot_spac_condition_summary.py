#!/usr/bin/env python3
"""Summary of CONCERT's advantage over every competing method, for one benchmark condition.

Four views per condition, each over ALL scored knockouts rather than a displayed subset:

  1 value    mean of the metric per method, with the standard error across knockouts
  2 rank     mean rank of a method inside a knockout, with every knockout's rank behind the bar
  3 heatmap  every knockout as a column, normalized inside the knockout so 1 is the best method
  4 delta    per-knockout margin of CONCERT over the competitor, signed so positive favours CONCERT

Conditions: within-niche and cross-slide, each by log-fold-change RMSE and by rank correlation.
Each view is written on its own and the four are also written as one 2x2 sheet.

Usage: plot_spac_condition_summary.py [--indir DIR] [--outroot DIR]
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

from plot_spac_benchmark_bars import BAR_EDGE, BAR_EDGE_LW, COLORS  # noqa: E402
from plot_spac_overall_rank import LOWER_IS_BETTER, load_regime, rank_table  # noqa: E402

CONDITIONS = [
    ("seen", "lfc_dist", "RMSE"),
    ("seen", "lfc_spearman", "rank correlation"),
    ("unseen", "lfc_dist", "RMSE"),
    ("unseen", "lfc_spearman", "rank correlation"),
]
OURS = "CONCERT"
LABEL_FS, AXIS_FS = 15, 17
DOT = {"s": 11, "color": "#2B2B2B", "alpha": 0.18, "linewidths": 0, "zorder": 3}


def _style(ax, xlabel, order):
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels(order, fontsize=LABEL_FS)
    ax.set_xlabel(xlabel, fontsize=AXIS_FS)
    ax.tick_params(axis="x", labelsize=LABEL_FS)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_value(ax, per_gene, metric, tag):
    wide = per_gene.pivot(index="gene", columns="method_label", values=metric)
    means = wide.mean()
    order = means.sort_values(ascending=LOWER_IS_BETTER[metric]).index.tolist()
    y = np.arange(len(order))[::-1]
    sems = [float(wide[m].std(ddof=1) / np.sqrt(wide[m].notna().sum())) for m in order]
    ax.barh(y, [means[m] for m in order], height=0.74, xerr=sems,
            color=[COLORS[m] for m in order], edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW,
            error_kw={"elinewidth": 1.1, "ecolor": "#2B2B2B"}, zorder=2)
    if metric == "lfc_spearman":
        ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    _style(ax, f"Log-fold-change {tag}, mean over knockouts", order)
    return {m: {"mean": round(float(means[m]), 4), "n_knockouts": int(wide[m].notna().sum())}
            for m in order}


def panel_rank(ax, per_gene, metric, tag):
    ranks = rank_table(per_gene, metric)
    order = ranks.mean().sort_values().index.tolist()
    y = np.arange(len(order))[::-1]
    means = [float(ranks[m].mean()) for m in order]
    sems = [float(ranks[m].std(ddof=1) / np.sqrt(ranks[m].notna().sum())) for m in order]
    ax.barh(y, means, height=0.74, xerr=sems,
            color=[COLORS[m] for m in order], edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW,
            error_kw={"elinewidth": 1.1, "ecolor": "#2B2B2B"}, zorder=2)
    rng = np.random.default_rng(0)
    for yi, m in zip(y, order):
        v = ranks[m].dropna().to_numpy()
        ax.scatter(v, yi + rng.uniform(-0.23, 0.23, size=len(v)), **DOT)
    ax.set_xlim(0.5, len(order) + 0.5)
    ax.set_xticks(range(1, len(order) + 1))
    _style(ax, f"Mean rank over all knockouts ({tag})", order)
    return {m: {"mean_rank": round(float(ranks[m].mean()), 3),
                "share_rank_1": round(float((ranks[m] == 1).sum() / ranks[m].notna().sum()), 3)}
            for m in order}


def _pairs(per_gene, metric):
    """Per competitor: the signed margin of CONCERT on every knockout both methods scored."""
    wide = per_gene.pivot(index="gene", columns="method_label", values=metric)
    out = {}
    for m in wide.columns:
        if m == OURS:
            continue
        both = wide[[OURS, m]].dropna()
        if both.empty:
            continue
        margin = both[m] - both[OURS] if LOWER_IS_BETTER[metric] else both[OURS] - both[m]
        out[m] = margin.to_numpy()
    return out


def normalized_scores(per_gene, metric):
    """Per knockout, rescale the methods that scored it to [0, 1], with 1 on the best method."""
    wide = per_gene.pivot(index="gene", columns="method_label", values=metric)
    lo = wide.min(axis=1)
    hi = wide.max(axis=1)
    span = (hi - lo).replace(0.0, np.nan)
    norm = (hi.values[:, None] - wide) if LOWER_IS_BETTER[metric] else (wide - lo.values[:, None])
    return norm.div(span, axis=0)


def panel_heatmap(ax, per_gene, metric, tag):
    norm = normalized_scores(per_gene, metric)
    # CONCERT holds the first row in every condition; the rest follow by mean normalized score.
    rest = [m for m in norm.mean().sort_values(ascending=False).index if m != OURS]
    order = [OURS] + rest
    cols = norm[OURS].sort_values(ascending=False).index.tolist()
    mat = np.ma.masked_invalid(norm.loc[cols, order].to_numpy().T)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#DDDDDD")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=LABEL_FS)
    ax.set_xticks([])
    ax.set_xlabel(f"Knockouts ordered by CONCERT ({tag})", fontsize=AXIS_FS)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cb.set_label("Normalized within knockout", fontsize=14)
    cb.ax.tick_params(labelsize=13)
    return {m: {"mean_normalized": round(float(norm[m].mean()), 3),
                "n_knockouts": int(norm[m].notna().sum()),
                "share_best": round(float((norm[m] == 1.0).sum() / norm[m].notna().sum()), 3)}
            for m in order}


def panel_delta(ax, per_gene, metric, tag):
    pairs = _pairs(per_gene, metric)
    order = sorted(pairs, key=lambda m: float(np.median(pairs[m])), reverse=True)
    y = np.arange(len(order))[::-1]
    rng = np.random.default_rng(0)
    for yi, m in zip(y, order):
        v = pairs[m]
        ax.boxplot(v, positions=[yi], vert=False, widths=0.62, showfliers=False,
                   patch_artist=True,
                   boxprops={"facecolor": COLORS[m], "edgecolor": "#2B2B2B", "linewidth": 0.8},
                   medianprops={"color": "#FFFFFF", "linewidth": 1.6},
                   whiskerprops={"color": "#2B2B2B", "linewidth": 0.8},
                   capprops={"color": "#2B2B2B", "linewidth": 0.8}, zorder=2)
        ax.scatter(v, yi + rng.uniform(-0.2, 0.2, size=len(v)), **DOT)
    ax.axvline(0.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.7)
    unit = "RMSE units" if metric == "lfc_dist" else "correlation units"
    _style(ax, f"Margin of CONCERT per knockout ({unit})", order)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=LABEL_FS)
    return {m: {"median_margin": round(float(np.median(v)), 4),
                "mean_margin": round(float(np.mean(v)), 4)} for m, v in pairs.items()}


# (name, function, row height in inches) - the heatmap rows are shorter than the bar rows.
PANELS = [("value", panel_value, 0.52), ("rank", panel_rank, 0.52),
          ("heatmap", panel_heatmap, 0.30), ("delta", panel_delta, 0.52)]
# 2x2 summary sheet, read row by row: value | heatmap / rank | delta.
SHEET_ORDER = ["value", "heatmap", "rank", "delta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=rp.SPACSEQ_RESULTS)
    ap.add_argument("--outroot", type=Path, default=rp.WORKDIR / "spacseq_results" / "figures")
    args = ap.parse_args()

    stats, cache = {}, {}
    for regime, metric, tag in CONDITIONS:
        if regime not in cache:
            cache[regime] = load_regime(args.indir, regime)
        per_gene = cache[regime]
        mtag = "rmse" if metric == "lfc_dist" else "rho"
        outdir = args.outroot / "condition_summary" / f"{regime}_{mtag}"
        outdir.mkdir(parents=True, exist_ok=True)
        cond = {}
        n_methods = per_gene.method_label.nunique()
        for name, fn, row_h in PANELS:
            fig, ax = plt.subplots(figsize=(8.8, row_h * n_methods + 1.1))
            cond[name] = fn(ax, per_gene, metric, tag)
            fig.tight_layout(pad=0.4)
            f = outdir / f"spacseq_{regime}_{mtag}_{name}.pdf"
            fig.savefig(f, bbox_inches="tight")
            fig.savefig(f.with_suffix(".png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {f}")
        fig, axes = plt.subplots(2, 2, figsize=(17.6, 1.04 * n_methods + 2.2))
        panel_fn = {name: fn for name, fn, _row_h in PANELS}
        for name, ax in zip(SHEET_ORDER, axes.ravel()):
            panel_fn[name](ax, per_gene, metric, tag)
        fig.tight_layout(pad=0.6)
        f = outdir / f"spacseq_{regime}_{mtag}_summary_sheet.pdf"
        fig.savefig(f, bbox_inches="tight")
        fig.savefig(f.with_suffix(".png"), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {f}")
        stats[f"{regime}_{mtag}"] = cond

    (args.outroot / "condition_summary" / "condition_summary_stats.json").write_text(
        json.dumps(stats, indent=2))
    print("wrote condition_summary_stats.json")


if __name__ == "__main__":
    main()
