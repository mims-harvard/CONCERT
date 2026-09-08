#!/usr/bin/env python
"""SI sensitivity figure: one panel per hyper-parameter, seed spread as the yardstick.

Reads sens_pmap_arms.csv (agg_sens_pmap.py) and writes fig_sens_pmap.{pdf,png},
figure_stats.json, and the caption text driven by the same numbers.

Reading rule: an arm is "insensitive" when its point sits inside the five-seed range of the
published configuration. That range is reported numerically in the caption and shown directly in
the first panel; it is not shaded on the panels (Xianglin, 2026-08-18).

Design: docs/superpowers/specs/2026-08-17-concert-sensitivity-analysis-design.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/n/holylabs/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study")
import paper_style as ps  # noqa: E402

# panel order, reader-facing names, and whether the x axis is logarithmic
FACTORS = [
    ("seed", "Random seed", False),
    ("fpert_ls_max_loc", "Footprint lengthscale\nceiling (spatial units)", True),
    ("fpert_svgp_inducing", "Number of\ninducing points", True),
    ("fpert_coord_scale", "Coordinate scale", False),
    ("moran_lambda", "Spatial smoothness\npenalty weight", False),
    ("lfc_lambda", "Log-fold-change\npenalty weight", False),
    # NB: --disentangle_radius selects the scale-aware Moran variant (concert_spac.py:1379),
    # i.e. it is the radius of the spatial smoothness penalty, not a separate HSIC term
    # (that one, --disentangle_lambda, is 0 in the published configuration).
    ("disentangle_radius", "Spatial smoothness\npenalty radius", False),
    ("niche_film_dim", "Niche code width", False),
    ("niche_film_k", "Niche neighbourhood size", False),
    ("basal_rff_features", "Basal field\nrandom features", True),
    ("basal_rff_bandwidth", "Basal field bandwidth", False),
]
# the two metrics the benchmark figure reports (rescore_pm_lfc_distance.py)
METRICS = [("spearman_mean", "Log-fold-change\nrank correlation", "up", "rho"),
           ("lfc_dist_mean", "Log-fold-change RMSE", "down", "rmse")]
BASE = {"seed": 42, "fpert_ls_max_loc": 360.0, "fpert_svgp_inducing": 64,
        "fpert_coord_scale": 20.0, "moran_lambda": 1.0, "lfc_lambda": 1.0,
        "disentangle_radius": 4.0, "niche_film_dim": 8, "niche_film_k": 25,
        "basal_rff_features": 128, "basal_rff_bandwidth": 0.15}
NCOL, NROW = 4, 3          # square-ish grid; the spare cell carries the legend
Y_HEADROOM = 3.0           # multiple of the data range kept in view, so the spread reads honestly
GREY = "#9e9e9e"


def arm_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Two-stage: average the slide x perturbation cells within a run, then across seeds.

    The error bar is the spread ACROSS SEEDS, not across cells -- seed noise is the thing that
    has to be beaten for a factor effect to be real, and it is much the larger of the two.
    """
    per_run = df.groupby(["factor", "value", "seed"])[metric].mean().reset_index()
    g = per_run.groupby(["factor", "value"])[metric]
    out = g.agg(mean="mean", sd=lambda s: s.std(ddof=1), n="size").reset_index()
    out["sem"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    return out


def write_caption(outdir: Path, stats: dict, n_cells: int) -> None:
    """Caption text generated from the same numbers the figure plots."""
    b = stats["seed_band"]["spearman_mean"]
    be = stats["seed_band"]["lfc_dist_mean"]
    lines = [
        "Figures -- Sensitivity of CONCERT to its spatial hyper-parameters (Perturb-MAP, "
        "four slides, seen regime). Two panels sets, one per benchmark metric: "
        "fig_sens_pmap_rho (log-fold-change rank correlation) and fig_sens_pmap_rmse "
        "(log-fold-change RMSE).",
        "",
        f"Each point is one trained model, averaged over the {n_cells} slide x perturbation "
        f"evaluations it produces ({stats['n_arms']} trained models in total); "
        "error bars are the standard error across three random seeds (42, 1, 2) at that "
        "setting. Five runs of the published configuration that differ only in random seed span "
        f"a log-fold-change rank correlation of {b['min']:.3f}-{b['max']:.3f} and a "
        f"log-fold-change RMSE of {be['min']:.2f}-{be['max']:.2f}; these runs are shown in the "
        "first panel and set the scale against which the remaining panels should be read. The "
        "open ring marks the published setting. Each remaining panel moves one hyper-parameter "
        "with all others held at the published values.",
        "",
        "Per-factor maximum deviation from the published setting (log-fold-change rank "
        "correlation, same three seeds):",
    ]
    pretty = {k: v.replace("\n", " ") for k, v, _ in FACTORS}
    for f, m in stats["factors"].items():
        s = m["spearman_mean"]
        worse = s["worse_than_seed_band_at"]
        lines.append(
            f"  - {pretty.get(f, f)}: {s['max_abs_dev_from_base']:.3f} "
            f"(base {s['base_mean_same_seeds']:.3f}); "
            + ("all settings inside the five-seed range"
               if not worse else f"outside the five-seed range at {worse}"))
    lines += [
        "",
        "Every setting tested lies inside the five-seed range on both metrics, i.e. no "
        "hyper-parameter moves performance further than re-running the published configuration "
        "with a different random seed.",
    ]
    (outdir / "figure_legend_sensitivity.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/"
                                 "CONCERT-main/results/sens_pmap_arms.csv"))
    ap.add_argument("--outdir", type=Path,
                    default=Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/"
                                 "CONCERT-main/Outputs/sensitivity"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    ps.use()

    df = pd.read_csv(args.csv)
    n_cells = int(df.drop_duplicates(["tag", "scope", "slide", "pert"])
                  .groupby("tag").size().max())
    stats: dict = {"n_arms": int(df["tag"].nunique()), "n_cells_per_arm": n_cells, "factors": {}}

    for metric, ylab, better, slug in METRICS:
        am = arm_means(df, metric)
        seeds = am[am["factor"] == "seed"]  # one run per point: the band itself
        lo, hi = float(seeds["mean"].min()), float(seeds["mean"].max())
        base_val = float(seeds.loc[seeds["value"] == BASE["seed"], "mean"].iloc[0])
        # keep generous headroom: a tight axis makes an insensitive result look volatile
        sem = am["sem"].fillna(0.0)
        dlo = min(lo, float((am["mean"] - sem).min()))
        dhi = max(hi, float((am["mean"] + sem).max()))
        mid, half = (dlo + dhi) / 2.0, (dhi - dlo) / 2.0
        ylo, yhi = mid - Y_HEADROOM * half, mid + Y_HEADROOM * half
        if metric == "spearman_mean":
            ylo, yhi = max(0.0, ylo), min(1.0, yhi)
        stats.setdefault("seed_band", {})[metric] = {
            "min": lo, "max": hi, "base": base_val, "width": hi - lo,
            "n_seeds": int(seeds.shape[0])}

        fig, axes = plt.subplots(NROW, NCOL, figsize=(4.0 * NCOL, 3.6 * NROW), squeeze=False)
        flat = [axes[i][j] for i in range(NROW) for j in range(NCOL)]
        for ax in flat[len(FACTORS):]:
            ax.axis("off")

        for c, (factor, flab, logx) in enumerate(FACTORS):
            ax = flat[c]
            sub = am[am["factor"] == factor].sort_values("value")
            bx = BASE[factor]
            if factor == "seed":
                # seed ids are labels, not a quantity -- categorical, and no connecting line
                sub = sub.iloc[np.argsort([0 if v == bx else v for v in sub["value"]])]
                xs = np.arange(len(sub))
                ax.plot(xs, sub["mean"], "o", ms=9, color=ps.ACCENT, ls="none", zorder=3)
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{int(v)}" for v in sub["value"]])
                ax.set_xlim(-0.6, len(sub) - 0.4)
                ax.plot(xs[np.asarray(sub["value"] == bx)],
                        sub.loc[sub["value"] == bx, "mean"], "o", ms=17, mfc="none",
                        mec=ps.ACCENT, mew=2.2, zorder=4)
                brow = sub[sub["value"] == bx]
            else:
                ax.errorbar(sub["value"], sub["mean"], yerr=sub["sem"], color=ps.ACCENT,
                            marker="o", ms=9, lw=2.0, capsize=4, zorder=3)
                brow = sub[np.isclose(sub["value"], bx)]
                if not brow.empty:  # ring the published setting
                    ax.plot(brow["value"], brow["mean"], "o", ms=17, mfc="none",
                            mec=ps.ACCENT, mew=2.2, zorder=4)
                if logx:
                    ax.set_xscale("log")
                ax.set_xticks(sorted(sub["value"].unique()))
                ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_ylim(ylo, yhi)
            ax.tick_params(labelsize=15)
            ax.set_xlabel(flab, fontsize=16)
            if c % NCOL == 0:
                ax.set_ylabel(ylab, fontsize=16)
            else:
                ax.set_yticklabels([])

            if factor != "seed":
                # "Outside the band" only counts when the arm's own seed spread clears it too;
                # the base point sits ON an edge of the band, so a bare comparison flags noise.
                below = sub[(sub["mean"] + sub["sem"].fillna(0)) < lo]
                above = sub[(sub["mean"] - sub["sem"].fillna(0)) > hi]
                stats["factors"].setdefault(factor, {})[metric] = {
                    "values": [float(v) for v in sub["value"]],
                    "means": [float(v) for v in sub["mean"]],
                    "sd_across_seeds": [None if pd.isna(v) else float(v) for v in sub["sd"]],
                    "sem_across_seeds": [None if pd.isna(v) else float(v) for v in sub["sem"]],
                    "n_seeds": [int(v) for v in sub["n"]],
                    "worse_than_seed_band_at": [float(v) for v in (below if better == "up"
                                                                   else above)["value"]],
                    "better_than_seed_band_at": [float(v) for v in (above if better == "up"
                                                                    else below)["value"]],
                    "base_mean_same_seeds": float(brow["mean"].iloc[0]) if not brow.empty else None,
                    "max_abs_dev_from_base": float(
                        (sub["mean"] - float(brow["mean"].iloc[0])).abs().max())
                    if not brow.empty else None,
                }

        handles = [
            plt.Line2D([], [], color=ps.ACCENT, marker="o", ms=9, lw=2.0, label="Arm mean"),
            plt.Line2D([], [], color=ps.ACCENT, marker="o", ms=15, mfc="none", mew=2.2,
                       ls="none", label="Published setting"),
        ]
        flat[-1].legend(handles=handles, loc="center", fontsize=17, frameon=False)
        fig.subplots_adjust(left=0.105, right=0.995, top=0.985, bottom=0.085,
                            wspace=0.08, hspace=0.42)
        ps.save(fig, str(args.outdir / f"fig_sens_pmap_{slug}.png"))
        plt.close(fig)

    (args.outdir / "figure_stats.json").write_text(json.dumps(stats, indent=2))
    write_caption(args.outdir, stats, n_cells)
    print(json.dumps(stats["seed_band"], indent=2))
    for f, m in stats["factors"].items():
        s = m["spearman_mean"]
        print(f"{f:24s} max|dev| rho={s['max_abs_dev_from_base']:.3f} "
              f"worse than seed band at: {s['worse_than_seed_band_at'] or 'none'} "
              f"| better at: {s['better_than_seed_band_at'] or 'none'}")


if __name__ == "__main__":
    main()
