#!/usr/bin/env python3
"""Perturb-map benchmark panel: CONCERT vs published methods, per perturbation.

Reads the per-record CSV written by rescore_pm_lfc_distance.py and scores every
method on the SAME set of niche-patch records (the intersection across methods) --
without that restriction CONCERT is scored on 22/3/2 records where the competitors
have 23/6/2, which is not a like-for-like comparison.

Emits <out>/pmap_benchmark.{png,pdf}, figure_stats.json and figure_legend.txt.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PERTS = ["Tgfbr2", "Ifngr2", "Jak2"]

# Method order and colour by FAMILY, matching the SPAC-seq benchmark bars:
# accent red = CONCERT, blues = non-spatial perturbation models, greens = spatial models.
METHODS = [
    ("concert",   "CONCERT",      "#6E4F7D"),
    ("scgen",     "scGEN",        "#4E5D9D"),
    ("biolord",   "Biolord",      "#2F6FA3"),
    ("gears",     "GEARS",        "#9CC7D8"),
    ("steamboat", "Steamboat",    "#356B5B"),
    ("celcomen",  "CelComen",     "#77B89C"),
]

INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"


def load_records(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def restrict_to_shared_records(rows: list[dict], methods: list[str]) -> dict:
    """Per perturbation, keep only the records every plotted method scored."""
    by_method = defaultdict(set)
    for r in rows:
        by_method[(r["gene"], r["method"])].add((r["slide"], r["niche_key"]))
    shared = {}
    for gene in PERTS:
        sets = [by_method[(gene, m)] for m in methods if by_method[(gene, m)]]
        shared[gene] = set.intersection(*sets) if sets else set()
    return shared


def aggregate(rows: list[dict], shared: dict) -> dict:
    vals = defaultdict(list)
    for r in rows:
        key = (r["slide"], r["niche_key"])
        if key not in shared.get(r["gene"], ()):
            continue
        rho, dist = float(r["lfc_spearman"]), float(r["lfc_dist"])
        if np.isfinite(rho) and np.isfinite(dist):
            vals[(r["gene"], r["method"])].append((rho, dist))
    out = {}
    for key, pairs in vals.items():
        arr = np.asarray(pairs, dtype=float)
        n = len(arr)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.array([np.nan, np.nan])
        out[key] = {
            "n_records": n,
            "rho": float(arr[:, 0].mean()), "rho_sem": float(sem[0]),
            "rmse": float(arr[:, 1].mean()), "rmse_sem": float(sem[1]),
        }
    return out


def make_figure(stats: dict, shared: dict, out: Path) -> None:
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
            # Bars stay BELOW their own error bars (zorder 2) -- raising a bar above
            # its error bar paints over the lower half and renders a one-sided cap.
            ax.bar(xs, vals, w, color=colour, edgecolor="white", linewidth=0.7, zorder=1)
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
    # Tight limits: the default margins leave a band of empty axes under the bars.
    lo = min(v["rho"] - (v["rho_sem"] if np.isfinite(v["rho_sem"]) else 0)
             for v in stats.values())
    axes[0].set_ylim(min(-0.02, lo * 1.35), None)
    axes[1].set_ylim(0, None)

    handles = [Patch(facecolor=c, edgecolor="white", label=lab) for _m, lab, c in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=len(METHODS), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=13, handlelength=1.4,
               columnspacing=1.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pmap_benchmark.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(stats: dict, shared: dict, out: Path) -> None:
    payload = {
        "perturbations": PERTS,
        "n_shared_records": {p: len(shared[p]) for p in PERTS},
        "methods": [lab for _m, lab, _c in METHODS],
        "values": {
            p: {lab: stats.get((p, m), {}) for m, lab, _c in METHODS} for p in PERTS
        },
    }
    (out / "figure_stats.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "Perturb-map benchmark panel (pmap_benchmark.pdf).",
        "",
        "CONCERT and five published perturbation-response methods predict the response "
        "of each of the three Perturb-map knockouts, scored against the measured "
        "perturbed spots. Left, agreement between predicted and observed log-fold-change "
        "vectors measured by Spearman rank correlation (higher is better). Right, root "
        "mean squared error of the same vectors (lower is better). Bars are means over "
        "niche-patch records pooled across the four slides, error bars are the standard "
        "error of the mean across those records; n per perturbation is given on the axis. "
        "All methods are scored on the identical set of records.",
        "",
        "Values (mean +/- s.e.m.):",
    ]
    for p in PERTS:
        lines.append(f"  {p} (n = {len(shared[p])} records)")
        for m, lab, _c in METHODS:
            s = stats.get((p, m))
            if not s:
                lines.append(f"    {lab}: not scored")
                continue
            lines.append(
                f"    {lab}: rank correlation {s['rho']:.3f} +/- {s['rho_sem']:.3f}, "
                f"RMSE {s['rmse']:.3f} +/- {s['rmse_sem']:.3f}"
            )
    lines += [
        "",
        "CellAgentChat is omitted: its Perturb-map run wrote no per-niche log-fold-change "
        "vectors, so it cannot be scored on this metric.",
    ]
    (out / "figure_legend.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-record", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_records(args.per_record)
    methods = [m for m, _l, _c in METHODS]
    shared = restrict_to_shared_records(rows, methods)
    stats = aggregate(rows, shared)
    make_figure(stats, shared, args.out)
    write_outputs(stats, shared, args.out)

    for p in PERTS:
        print(f"{p} (n={len(shared[p])})")
        for m, lab, _c in METHODS:
            s = stats.get((p, m))
            if s:
                print(f"  {lab:12s} rho {s['rho']:+.3f}+/-{s['rho_sem']:.3f}"
                      f"   RMSE {s['rmse']:.3f}+/-{s['rmse_sem']:.3f}")
    print("wrote", args.out / "pmap_benchmark.pdf")


if __name__ == "__main__":
    main()
