#!/usr/bin/env python3
"""CONCERT vs SPAC-seq competing methods -- paired dots + Overall bars.

Panels: Overall first (bar +/- SD over ALL tested perts), then top-N gene dots
(mean only, no niche replicates). No figure title/subtitle.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_spacseq_bars import (
    COLORS,
    DEFAULT_OUT,
    EXTRA_COMPETITORS,
    METHOD_ORDER,
    METHOD_SOURCES,
    SPAC_SEEN_TOP_N,
    SPAC_ROOT,
    collect_lfc_stats,
    collect_lfc_stats_unseen,
    load_concert_by_gene,
    load_eval_units,
    load_lfc_dist_from_csv,
    merge_dist,
    select_top_genes,
    select_unseen_genes,
    _stat_valid,
)

COMPETITORS = [(label, COLORS[label]) for label, _, _ in METHOD_SOURCES] + [
    (lb, COLORS[lb]) for lb in EXTRA_COMPETITORS
]
DOT_GRID_COLS = 3
OVERALL_KEY = "__overall__"
JITTER_FRAC = 0.04
JITTER_MIN_SEP = 0.02
PANEL_FS = 15      # panel titles (gene / Overall)
AXIS_FS = 14       # x/y axis titles
TICK_FS = 12       # tick numbers
BAR_LABEL_FS = 12  # method names under Overall bars
LEGEND_FS = 14


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top-n", type=int, default=SPAC_SEEN_TOP_N)
    p.add_argument("--regime", choices=("seen", "unseen"), default="seen")
    # Match plot_spacseq_bars.py: selects the CONCERT result directory without touching the
    # module-level default. CONCERT's plotted metrics come from the rescored --outdir CSV;
    # this keeps gene selection consistent with the plotted CONCERT result.
    p.add_argument("--concert-dir", type=Path, default=None)
    return p.parse_args()


def _merge_extra_csvs(stats, outdir: Path, all_genes: list[str], regime: str) -> None:
    """Merge naive baselines + STATE (prefer our-rule STATE over collab ntc20)."""
    if regime == "seen":
        naive = outdir / "spac_seen_naive_baselines_our_rule.csv"
        state_our = outdir / "spac_seen_state_our_rule.csv"
        state_fb = outdir / "spac_seen_state_collab_metrics.csv"
    else:
        naive = outdir / "spac_unseen_naive_baselines_our_rule.csv"
        state_our = outdir / "spac_unseen_state_our_rule.csv"
        state_fb = outdir / "spac_unseen_state_collab_metrics.csv"
    extras = [naive, state_our if state_our.is_file() else state_fb]
    for path in extras:
        if path.is_file():
            merge_dist(stats, load_lfc_dist_from_csv(path, all_genes), all_genes)


def build_stats(args):
    if args.regime == "seen":
        concert = load_concert_by_gene(args.concert_dir, "seen")
        display_genes = select_top_genes(concert, SPAC_ROOT, args.top_n)
        dist_csv = args.outdir / "spac_seen_lfc_dist_per_record.csv"
        prefix, tag = "spacseq_seen", "seen"
        all_genes = _all_genes_from_csv(dist_csv, concert)
        stats = collect_lfc_stats(all_genes, concert)
    else:
        eval_units = load_eval_units()
        concert = load_concert_by_gene(args.concert_dir, "unseen", eval_units)
        display_genes = select_unseen_genes(concert, eval_units, SPAC_ROOT)[: args.top_n]
        dist_csv = args.outdir / "spac_unseen_lfc_dist_per_record.csv"
        prefix, tag = "spacseq_unseen", "unseen"
        all_genes = _all_genes_from_csv(dist_csv, concert)
        for g in select_unseen_genes(concert, eval_units, SPAC_ROOT):
            if g not in all_genes:
                all_genes.append(g)
        stats = collect_lfc_stats_unseen(all_genes, eval_units, concert)
    merge_dist(stats, load_lfc_dist_from_csv(dist_csv, all_genes), all_genes)
    _merge_extra_csvs(stats, args.outdir, all_genes, args.regime)
    return stats, display_genes, all_genes, prefix, tag


def _all_genes_from_csv(csv_path: Path, concert: dict) -> list[str]:
    if not csv_path.is_file():
        return sorted(g for g in concert if concert[g].get("can_predict"))
    genes = sorted({r["gene"] for r in csv.DictReader(csv_path.open())})
    return [g for g in genes if concert.get(g, {}).get("can_predict", True)]


def mean_pair(stats: dict, gene: str, comp_label: str, metric_key: str):
    if not _stat_valid(stats[gene].get("CONCERT", {}), metric_key):
        return None
    if not _stat_valid(stats[gene].get(comp_label, {}), metric_key):
        return None
    bx, _, _ = stats[gene][comp_label][metric_key]
    cy, _, _ = stats[gene]["CONCERT"][metric_key]
    return float(bx), float(cy)


def method_gene_means(stats: dict, genes: list[str], metric_key: str, label: str) -> list[float]:
    vals = []
    for gene in genes:
        if not _stat_valid(stats[gene].get(label, {}), metric_key):
            continue
        if metric_key == "lfc_dist" and not _stat_valid(stats[gene].get("CONCERT", {}), metric_key):
            continue
        m, _, _ = stats[gene][label][metric_key]
        if np.isfinite(m):
            vals.append(float(m))
    return vals


def _axis_limits(points, pad=0.12):
    if not points:
        return (-0.5, 1.0)
    vals = [v for xy in points for v in xy]
    lo, hi = float(min(vals)), float(max(vals))
    span = hi - lo or 1.0
    return lo - pad * span, hi + pad * span


def _jitter(xs, ys, lim, rng, scale=1.0):
    span = lim[1] - lim[0]
    j = max(span * JITTER_FRAC, JITTER_MIN_SEP) * scale
    n = len(xs)
    ox, oy = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        off = i - (n - 1) / 2.0
        ox.append(x + off * j * 0.55 + rng.uniform(-j * 0.12, j * 0.12))
        oy.append(y + off * j * 0.55 + rng.uniform(-j * 0.12, j * 0.12))
    return np.asarray(ox), np.asarray(oy)


def _set_panel_title(ax, text):
    ax.set_title(text, fontsize=PANEL_FS, fontweight="normal", pad=8)
    ax.title.set_fontweight("normal")


def _draw_overall_bars(fig, gs_cell, stats, all_genes, metric_key, overall_ylabel):
    """Overall bars with a dedicated label strip so 90 degree names never enter the axis."""
    inner = gs_cell.subgridspec(2, 1, height_ratios=[2.6, 2.0], hspace=0.08)
    ax = fig.add_subplot(inner[0])
    ax_lab = fig.add_subplot(inner[1], sharex=ax)

    labels, means, errs, colors = [], [], [], []
    for label in METHOD_ORDER:
        vals = method_gene_means(stats, all_genes, metric_key, label)
        if not vals:
            continue
        labels.append(label)
        means.append(float(np.mean(vals)))
        errs.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        colors.append(COLORS[label])
    if not means:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=PANEL_FS)
        _set_panel_title(ax, "Overall")
        ax.axis("off")
        ax_lab.axis("off")
        return
    higher_better = metric_key == "lfc_rho"
    order = sorted(range(len(means)), key=lambda i: means[i], reverse=higher_better)
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    errs = [errs[i] for i in order]
    colors = [colors[i] for i in order]
    xp = np.arange(len(means))
    bars = ax.bar(
        xp, means, yerr=errs, capsize=3.0, width=0.78, color=colors,
        edgecolor="#555555", linewidth=0.4,
        error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
    )
    for i, lb in enumerate(labels):
        if lb == "CONCERT":
            bars[i].set_edgecolor("#8B0000")
            bars[i].set_linewidth(1.2)
    if metric_key == "lfc_rho":
        ax.axhline(0.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.55)
    ax.set_xticks(xp)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(overall_ylabel, fontsize=AXIS_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    _set_panel_title(ax, "Overall")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(sp == "bottom")
    ax.spines["bottom"].set_linewidth(0.8)

    # Dedicated strip: 90 degree names, ha=center, fully below the bar axis
    ax_lab.set_xlim(ax.get_xlim())
    ax_lab.set_ylim(0, 1)
    ax_lab.set_xticks([])
    ax_lab.set_yticks([])
    for sp in ax_lab.spines.values():
        sp.set_visible(False)
    for x, lb in zip(xp, labels):
        # Start slightly below the bar axis so names do not enter the plot.
        # 45 degree tilt (not 90 degree): fits in the strip without being clipped by the
        # dot panel below, and stays readable.
        ax_lab.text(
            x, 0.98, lb, rotation=45, ha="right", va="top",
            rotation_mode="anchor", fontsize=BAR_LABEL_FS, clip_on=False,
        )


def plot_metric(stats, display_genes, all_genes, metric_key, xlabel, ylabel, overall_ylabel, out_path):
    rng = np.random.default_rng(0)
    gene_means: dict[str, list] = {g: [] for g in display_genes}
    for gene in display_genes:
        for comp_label, color in COMPETITORS:
            pair = mean_pair(stats, gene, comp_label, metric_key)
            if pair is None:
                continue
            gene_means[gene].append((*pair, comp_label, color))

    panel_keys = [OVERALL_KEY] + display_genes
    panel_titles = ["Overall"] + list(display_genes)

    all_pts = []
    for gene in display_genes:
        all_pts.extend((p[0], p[1]) for p in gene_means[gene])
    lim = _axis_limits(all_pts) if all_pts else (-0.5, 1.0)

    n_panels = len(panel_keys)
    ncols = DOT_GRID_COLS
    nrows = int(np.ceil(n_panels / ncols))
    # panel / spacer / panel / spacer / ... / panel / spacer / legend
    # Equal panel-row heights so dots beside Overall stay the same size.
    height_ratios = []
    for i in range(nrows):
        height_ratios.append(1.0)
        if i < nrows - 1:
            height_ratios.append(0.18)  # spacer between panel rows
        else:
            height_ratios.append(0.45)  # larger gap above legend
    height_ratios.append(0.32)  # legend
    n_gs_rows = len(height_ratios)

    fig = plt.figure(figsize=(12.2, 2.4 * sum(height_ratios)))
    gs = GridSpec(
        n_gs_rows, ncols, figure=fig,
        height_ratios=height_ratios,
        hspace=0.08, wspace=0.18, left=0.08, right=0.99, top=0.97, bottom=0.03,
    )

    def _gs_row(panel_row: int) -> int:
        return 2 * panel_row

    for idx, (key, panel_title) in enumerate(zip(panel_keys, panel_titles)):
        r, c = divmod(idx, ncols)
        if key == OVERALL_KEY:
            _draw_overall_bars(fig, gs[_gs_row(r), c], stats, all_genes, metric_key, overall_ylabel)
            continue

        ax = fig.add_subplot(gs[_gs_row(r), c])
        solid = gene_means[key]
        if not solid:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=PANEL_FS)
            _set_panel_title(ax, panel_title)
            ax.axis("off")
            continue

        ax.plot(lim, lim, color="#888888", linestyle="--", linewidth=1.1, zorder=0)
        xs = [p[0] for p in solid]
        ys = [p[1] for p in solid]
        colors = [p[3] for p in solid]
        jx, jy = _jitter(xs, ys, lim, rng, scale=1.0)
        ax.scatter(
            jx, jy, s=90, c=colors, alpha=0.95,
            edgecolors="white", linewidths=0.8, zorder=3,
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_box_aspect(1)  # square panel; x/y share lim
        ax.grid(alpha=0.28, linewidth=0.5, color="#cccccc")
        _set_panel_title(ax, panel_title)
        # Only bottom row gets x-label to avoid colliding with the next row title
        if r == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=AXIS_FS, labelpad=4)
        ax.tick_params(labelsize=TICK_FS)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=AXIS_FS, labelpad=4)

    for idx in range(len(panel_keys), nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.add_subplot(gs[_gs_row(r), c]).axis("off")

    leg_ax = fig.add_subplot(gs[n_gs_rows - 1, :])
    leg_ax.axis("off")
    handles = [
        mlines.Line2D([], [], color=COLORS[lb], marker="o", linestyle="None",
                      markersize=9, label=lb)
        for lb in METHOD_ORDER if lb in COLORS
    ]
    leg_ax.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 6),
        frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, columnspacing=1.1,
        borderaxespad=0.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    pdf_path = out_path.with_suffix('.pdf')
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {pdf_path}")


def main():
    args = parse_args()
    stats, display_genes, all_genes, prefix, tag = build_stats(args)
    if not display_genes:
        print(f"No genes for {args.regime}; skip dots")
        return
    print(f"{tag}: display={len(display_genes)} overall_genes={len(all_genes)}", flush=True)

    plot_metric(
        stats, display_genes, all_genes, "lfc_rho",
        "Competing methods\nLFC Spearman rho", "CONCERT LFC rho", "LFC Spearman rho",
        args.outdir / f"{prefix}_concert_vs_competitor_lfc_rho_dots.png",
    )
    plot_metric(
        stats, display_genes, all_genes, "lfc_dist",
        "Competing methods\nLFC RMSE", "CONCERT LFC RMSE", "LFC RMSE",
        args.outdir / f"{prefix}_concert_vs_competitor_lfc_rmse_dots.png",
    )


if __name__ == "__main__":
    main()
