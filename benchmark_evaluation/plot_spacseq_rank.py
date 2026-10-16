#!/usr/bin/env python3
"""Overall standing of every method across ALL scored knockouts, one figure per benchmark condition.

The per-knockout bar panels show eight knockouts chosen for coverage across methods, so they say
nothing about how often a method leads over the whole benchmark. This script ranks the methods
inside every knockout and summarises those ranks: the bar is a method's mean rank over all the
knockouts it scored, and the dots behind it are its rank on each individual knockout.

Four conditions: within-niche and cross-slide, each by log-fold-change RMSE and by rank correlation.

Usage: plot_spacseq_rank.py [--indir DIR] [--outdir DIR]
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
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_spacseq_bars import BAR_EDGE, BAR_EDGE_LW, COLORS  # noqa: E402

CONDITIONS = [
    ("seen", "lfc_dist", "Log-fold-change RMSE", "RMSE"),
    ("seen", "lfc_spearman", "Log-fold-change rank correlation", "rank correlation"),
    ("unseen", "lfc_dist", "Log-fold-change RMSE", "RMSE"),
    ("unseen", "lfc_spearman", "Log-fold-change rank correlation", "rank correlation"),
]
LOWER_IS_BETTER = {"lfc_dist": True, "lfc_spearman": False}
REGIME_TAG = {"seen": "within-niche", "unseen": "cross-slide"}


def load_regime(indir: Path, regime: str) -> pd.DataFrame:
    parts = [pd.read_csv(indir / f"spac_{regime}_lfc_dist_per_record.csv"),
             pd.read_csv(indir / f"spac_{regime}_naive_baselines_our_rule.csv")]
    state = indir / f"spac_{regime}_state_our_rule.csv"
    if state.is_file():
        parts.append(pd.read_csv(state))
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["method_label", "gene"])[["lfc_spearman", "lfc_dist"]].mean().reset_index()


def rank_table(per_gene: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank the methods inside each knockout, over the methods that scored that knockout."""
    wide = per_gene.pivot(index="gene", columns="method_label", values=metric)
    ranks = wide.rank(axis=1, ascending=LOWER_IS_BETTER[metric], method="min")
    return ranks


def draw(ranks: pd.DataFrame, axis_tag: str, out_pdf: Path) -> dict:
    order = ranks.mean().sort_values().index.tolist()
    y = np.arange(len(order))[::-1]
    means = [float(ranks[m].mean()) for m in order]
    sems = [float(ranks[m].std(ddof=1) / np.sqrt(ranks[m].notna().sum())) for m in order]

    fig, ax = plt.subplots(figsize=(8.8, 0.52 * len(order) + 1.1))
    ax.barh(y, means, height=0.74, xerr=sems,
            color=[COLORS[m] for m in order], edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW,
            error_kw={"elinewidth": 1.1, "ecolor": "#2B2B2B"}, zorder=2)
    rng = np.random.default_rng(0)
    for yi, m in zip(y, order):
        v = ranks[m].dropna().to_numpy()
        ax.scatter(v, yi + rng.uniform(-0.23, 0.23, size=len(v)), s=11,
                   color="#2B2B2B", alpha=0.18, linewidths=0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=15)
    ax.set_xlabel(f"Mean rank over all knockouts ({axis_tag})", fontsize=17)
    ax.set_xlim(0.5, len(order) + 0.5)
    ax.set_xticks(range(1, len(order) + 1))
    ax.tick_params(axis="x", labelsize=15)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {m: {"n_knockouts": int(ranks[m].notna().sum()),
                "mean_rank": round(float(ranks[m].mean()), 3),
                "sem_rank": round(float(ranks[m].std(ddof=1) / np.sqrt(ranks[m].notna().sum())), 3),
                "share_rank_1": round(float((ranks[m] == 1).sum() / ranks[m].notna().sum()), 3)}
            for m in order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=rp.SPACSEQ_RESULTS)
    ap.add_argument("--outdir", type=Path, default=rp.WORKDIR / "spacseq_results" / "figures" / "overall_rank")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    stats, caption = {}, []
    cache = {}
    for regime, metric, metric_name, axis_tag in CONDITIONS:
        if regime not in cache:
            cache[regime] = load_regime(args.indir, regime)
        ranks = rank_table(cache[regime], metric)
        tag = "rmse" if metric == "lfc_dist" else "rho"
        out_pdf = args.outdir / f"spacseq_{regime}_overall_rank_{tag}.pdf"
        s = draw(ranks, axis_tag, out_pdf)
        stats[f"{regime}_{tag}"] = {"n_knockouts_total": int(len(ranks)), "methods": s}
        best = list(s)[0]
        caption.append(
            f"- **{REGIME_TAG[regime]}, {metric_name.lower()}** ({out_pdf.name}): ranks over "
            f"{len(ranks)} knockouts. Best mean rank {s[best]['mean_rank']} ({best}); "
            f"CONCERT mean rank {s['CONCERT']['mean_rank']} on {s['CONCERT']['n_knockouts']} knockouts, "
            f"top of {int(round(s['CONCERT']['share_rank_1'] * 100))}% of them."
        )
        print(f"wrote {out_pdf}")

    (args.outdir / "overall_rank_figure_stats.json").write_text(json.dumps(stats, indent=2))
    (args.outdir / "overall_rank_captions.md").write_text(
        "# Overall rank figures\n\n"
        "Every method is ranked inside each knockout over the methods that scored it (rank 1 is best).\n"
        "The bar is the mean of those ranks with the standard error across knockouts, and each dot is\n"
        "one knockout.\n\n" + "\n".join(caption) + "\n")
    print(f"wrote {args.outdir / 'overall_rank_figure_stats.json'}")


if __name__ == "__main__":
    main()
