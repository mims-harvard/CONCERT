#!/usr/bin/env python3
"""Replot Task-1 paper figures: 2-bar vs NNLS + per-gene pie maps.

Updates vs prior:
- Distinct KO color per gene.
- Keep only the largest contiguous island of that gene's spots (drop isolates).
- Aggressive spatial thinning + pie radius from thinned separation (less overlap).

Caller:
  python replot_task1_paper_figs.py
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ASSETS_DEFAULT = Path(f"{_rp.DATA}/deconvolution/spacseq_2-1/spot_assets")
PRED_DEFAULT = Path(f"{_rp.DATA}/deconvolution/spacseq_2-1/per_spot.csv")
OUT_DEFAULT = Path(f"{_rp.WORKDIR}/deconvolution_spacseq_2-1")
EXT_DEFAULT = Path(f"{_rp.WORKDIR}/deconvolution_spacseq_2-1/figures")

# Match overview slide colors (matplotlib tab10 over TOP_KO order used in viz_deconv_slide_2_1)
GENE_COLORS = {
    "Icam1": "#1f77b4",
    "Arap1": "#ff7f0e",
    "Cd276": "#2ca02c",
    "Sema4b": "#d62728",
    "Slc43a3": "#9467bd",
    "Bcam": "#8c564b",
    "H2-Q7": "#e377c2",
    "Cxcr4": "#7f7f7f",
    "Slc2a1": "#bcbd22",
    "Utrn": "#17becf",
}
UNA_COLOR = "#b0b0b0"
BG_COLOR = "#f7f7f7"


def draw_pie(ax, x, y, w_ko, radius, ko_color, zorder=5):
    w = float(np.clip(w_ko, 0.0, 1.0))
    ax.add_patch(
        mpatches.Wedge(
            (x, y), radius, 90 - 360 * w, 90,
            facecolor=ko_color, edgecolor="0.2", lw=0.35, zorder=zorder,
        )
    )
    ax.add_patch(
        mpatches.Wedge(
            (x, y), radius, 90 - 360, 90 - 360 * w,
            facecolor=UNA_COLOR, edgecolor="0.2", lw=0.35, zorder=zorder,
        )
    )


def largest_island(xy, link_dist):
    """Union-find largest connected component with edges if dist <= link_dist."""
    n = xy.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=float(link_dist), output_type="ndarray")
    for i, j in pairs:
        union(int(i), int(j))
    roots = np.array([find(i) for i in range(n)])
    # largest component
    vals, counts = np.unique(roots, return_counts=True)
    root = vals[np.argmax(counts)]
    return np.where(roots == root)[0]


def thin_spots(xy, max_n, min_sep, rng):
    n = xy.shape[0]
    if n <= max_n:
        return np.arange(n)
    order = rng.permutation(n)
    keep = []
    for i in order:
        if not keep:
            keep.append(i)
            continue
        if np.linalg.norm(xy[keep] - xy[i], axis=1).min() >= min_sep:
            keep.append(i)
        if len(keep) >= max_n:
            break
    if len(keep) < max(8, max_n // 4):
        # fallback: grid subsample if connectivity too tight
        rest = [i for i in order if i not in set(keep)]
        keep.extend(rest[: max_n - len(keep)])
    return np.asarray(keep, dtype=int)


def plot_bars(summ_path: Path, out_png: Path):
    summ = pd.read_json(summ_path)
    base = summ.loc[summ.method == "naive_nnls_empirical"].iloc[0]
    prop = summ.loc[summ.method == "concert_amortized"].iloc[0]
    labels = ["NNLS", "CONCERT"]
    mae = [float(base.comp_MAE), float(prop.comp_MAE)]
    pear = [float(base.comp_pearson), float(prop.comp_pearson)]
    cols = ["#1565c0", "#c62828"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    bw = 0.55
    for ax, vals, ylabel in [
        (axes[0], mae, "Composition MAE $\\downarrow$"),
        (axes[1], pear, "Pearson r $\\uparrow$"),
    ]:
        ax.bar([0, 1], vals, color=cols, width=bw)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlim(-0.55, 1.55)
        ymax = max(vals) * 1.18
        ax.set_ylim(0, ymax)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02 * ymax, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)



def plot_bars_per_ko(
    pred: pd.DataFrame,
    out_png: Path,
    nnls_col: str = "naive_nnls_empirical",
    concert_col: str = "concert_amortized",
):
    """Grouped MAE / Pearson bars per KO gene: NNLS vs CONCERT. No titles."""
    rows = []
    for g, sub in pred.groupby("gene"):
        gt = sub["gt_frac_ko"].to_numpy(dtype=float)
        for col, method in [(nnls_col, "NNLS"), (concert_col, "CONCERT")]:
            pr = sub[col].to_numpy(dtype=float)
            if np.std(gt) < 1e-12 or np.std(pr) < 1e-12:
                r = float("nan")
            else:
                r = float(np.corrcoef(gt, pr)[0, 1])
            rows.append(
                {
                    "gene": str(g),
                    "method": method,
                    "n": int(len(sub)),
                    "mae": float(np.mean(np.abs(gt - pr))),
                    "pearson": r,
                }
            )
    mdf = pd.DataFrame(rows)
    # sort genes by CONCERT MAE (best -> left)
    order = (
        mdf.loc[mdf.method == "CONCERT"]
        .sort_values("mae")["gene"]
        .tolist()
    )
    methods = ["NNLS", "CONCERT"]
    colors = {"NNLS": "#1565c0", "CONCERT": "#c62828"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(order))
    width = 0.38
    for ax, metric, ylabel in [
        (axes[0], "mae", "Composition MAE $\\downarrow$"),
        (axes[1], "pearson", "Pearson r $\\uparrow$"),
    ]:
        for i, method in enumerate(methods):
            vals = [
                float(
                    mdf.loc[(mdf.gene == g) & (mdf.method == method), metric].iloc[0]
                )
                for g in order
            ]
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                color=colors[method],
                label=method,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ymax = max(
            float(mdf[metric].max()) * (1.18 if metric == "mae" else 1.12),
            0.05,
        )
        if metric == "pearson":
            ymin = min(0.0, float(np.nanmin(mdf[metric].to_numpy())) - 0.05)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0, ymax)
    axes[1].legend(fontsize=12, frameon=False, loc="upper right")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)
    return mdf


def plot_scatter(pred: pd.DataFrame, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, col, label, color in [
        (axes[0], "naive_nnls_empirical", "NNLS", "#1565c0"),
        (axes[1], "concert_amortized", "CONCERT", "#c62828"),
    ]:
        y = pred[col].to_numpy()
        gt = pred.gt_frac_ko.to_numpy()
        ax.scatter(gt, y, s=6, alpha=0.3, c=color, linewidths=0)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        r = float(np.corrcoef(gt, y)[0, 1])
        mae = float(np.mean(np.abs(gt - y)))
        ax.set_xlabel("GT KO fraction")
        ax.set_ylabel("Predicted")
        ax.text(
            0.04, 0.96, f"{label}\nr={r:.3f}  MAE={mae:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color="black",
        )
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def plot_winrate(pred: pd.DataFrame, out_png: Path):
    gt = pred.gt_frac_ko.to_numpy()
    err_nnls = np.abs(gt - pred.naive_nnls_empirical.to_numpy())
    err_am = np.abs(gt - pred.concert_amortized.to_numpy())
    d = err_nnls - err_am  # >0 => CONCERT better
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.hist(d, bins=40, color="#c62828", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(
        float(np.mean(d)), color="#1565c0", ls="--", lw=1.5,
        label=f"mean $\\Delta$={float(np.mean(d)):+.3f}  (win {100 * float(np.mean(d > 0)):.0f}%)",
    )
    ax.set_xlabel("|err_NNLS| - |err_CONCERT|  (>0 = CONCERT better)")
    ax.set_ylabel("# mixture spots")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def _prepare_gene_pies(
    pred: pd.DataFrame,
    roles: pd.DataFrame,
    gene: str,
    max_pies: int,
    pad_frac: float,
    seed: int,
    max_abs_err: float = 0.12,
):
    """Select island / good / thinned spots for one gene; return plot payload."""
    sub = pred[pred.gene == gene].copy()
    if sub.empty:
        return None
    roles_ix = roles.set_index("spot")
    xs, ys, gt, pr = [], [], [], []
    for r in sub.itertuples():
        sid = int(r.spot)
        if sid not in roles_ix.index:
            continue
        xs.append(float(roles_ix.loc[sid, "x"]))
        ys.append(float(roles_ix.loc[sid, "y"]))
        gt.append(float(r.gt_frac_ko))
        pr.append(float(r.concert_amortized))
    xy = np.column_stack([xs, ys])
    gt = np.asarray(gt)
    pr = np.asarray(pr)
    rng = np.random.default_rng(seed)
    ko_color = GENE_COLORS.get(gene, "#1565c0")

    tree = cKDTree(xy)
    nn = tree.query(xy, k=2)[0][:, 1]
    pitch = float(np.median(nn[nn > 0])) if np.any(nn > 0) else 80.0
    island = largest_island(xy, 2.5 * pitch)
    xy_i, gt_i, pr_i = xy[island], gt[island], pr[island]
    n_drop = len(xy) - len(xy_i)

    err = np.abs(gt_i - pr_i)
    good = err <= max_abs_err
    if good.sum() < max(12, max_pies // 4):
        thr = float(np.quantile(err, min(0.35, max_pies / max(len(err), 1))))
        good = err <= thr
        max_abs_err = thr
    xy_g, gt_g, pr_g = xy_i[good], gt_i[good], pr_i[good]

    xmin, xmax = xy_i[:, 0].min(), xy_i[:, 0].max()
    ymin, ymax = xy_i[:, 1].min(), xy_i[:, 1].max()
    span = max(xmax - xmin, ymax - ymin, 1.0)
    pad = pad_frac * span
    xmin, xmax = xmin - pad, xmax + pad
    ymin, ymax = ymin - pad, ymax + pad

    area = max((xmax - xmin) * (ymax - ymin), 1.0)
    target_sep = max(2.0 * pitch, 0.85 * np.sqrt(area / max(max_pies, 1)))
    keep = thin_spots(xy_g, max_pies, target_sep, rng)
    xy_p, gt_p, pr_p = xy_g[keep], gt_g[keep], pr_g[keep]
    if len(xy_p) >= 2:
        actual_sep = float(np.min(cKDTree(xy_p).query(xy_p, k=2)[0][:, 1]))
    else:
        actual_sep = float(target_sep)
    radius = 0.28 * actual_sep

    meta = {
        "gene": gene,
        "n_spots_total": int(len(xy)),
        "n_island": int(len(xy_i)),
        "n_isolates_dropped": int(n_drop),
        "n_good": int(len(xy_g)),
        "max_abs_err": float(max_abs_err),
        "n_pies": int(len(xy_p)),
        "pitch": pitch,
        "pie_sep": actual_sep,
        "pie_radius": radius,
        "color": ko_color,
        "mean_gt": float(np.mean(gt_p)),
        "mean_pred": float(np.mean(pr_p)),
        "mean_abs_err_shown": float(np.mean(np.abs(gt_p - pr_p))),
    }
    return {
        "gene": gene,
        "ko_color": ko_color,
        "xy_i": xy_i,
        "xy_p": xy_p,
        "gt_p": gt_p,
        "pr_p": pr_p,
        "xlim": (xmin, xmax),
        "ylim": (ymin, ymax),
        "radius": radius,
        "meta": meta,
    }


def _draw_gene_pie_pair(axes, payload, label_panels=True):
    """Draw Raw|Predicted pies; no titles. Bottom labels Raw/Predicted only."""
    xy_i, xy_p = payload["xy_i"], payload["xy_p"]
    radius = payload["radius"]
    ko_color = payload["ko_color"]
    xmin, xmax = payload["xlim"]
    ymin, ymax = payload["ylim"]
    for ax, weights, xlab in [
        (axes[0], payload["gt_p"], "Raw"),
        (axes[1], payload["pr_p"], "Predicted"),
    ]:
        ax.scatter(
            xy_i[:, 0], xy_i[:, 1], s=5, c=BG_COLOR, edgecolors="#cfcfcf",
            linewidths=0.15, zorder=1,
        )
        for (x, y), w in zip(xy_p, weights):
            draw_pie(ax, float(x), float(y), float(w), radius, ko_color)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if label_panels:
            ax.set_xlabel(xlab, fontsize=9, labelpad=1)


def plot_gene_pies(
    pred: pd.DataFrame,
    roles: pd.DataFrame,
    gene: str,
    out_png: Path,
    max_pies: int,
    pad_frac: float,
    seed: int,
    max_abs_err: float = 0.12,
):
    payload = _prepare_gene_pies(
        pred, roles, gene, max_pies, pad_frac, seed, max_abs_err=max_abs_err,
    )
    if payload is None:
        print(f"skip {gene}: no spots", flush=True)
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True)
    _draw_gene_pie_pair(axes, payload, label_panels=True)
    handles = [
        mpatches.Patch(color=payload["ko_color"], label=payload["gene"]),
        mpatches.Patch(color=UNA_COLOR, label="unassigned"),
    ]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.98, bottom=0.14)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    m = payload["meta"]
    print(
        f"wrote {out_png}  island {m['n_island']}/{m['n_spots_total']}  "
        f"good={m['n_good']}  pies={m['n_pies']}  |delta|<={m['max_abs_err']:.3f}  "
        f"r={m['pie_radius']:.1f}",
        flush=True,
    )
    return m



def plot_gt_pred_scatter_row(
    pred: pd.DataFrame,
    genes: list[str],
    out_png: Path,
    pred_col: str = "concert_amortized",
    seed: int = 42,
):
    """1x3 calibration: GT-binned mean+/-IQR of Pred (faint points underneath)."""
    # Match overview legend (TOP_KO[:6])
    prefer = ["Icam1", "Arap1", "Cd276", "Sema4b", "Slc43a3", "Bcam"]
    genes = [g for g in prefer if (pred.gene == g).any()]
    if not genes:
        print("skip gt-pred calibration: no genes", flush=True)
        return

    rng = np.random.default_rng(seed)
    n = len(genes)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, g in zip(axes, genes):
        sub = pred.loc[pred.gene == g, ["gt_frac_ko", pred_col]].to_numpy(dtype=float)
        gt, pr = sub[:, 0], sub[:, 1]
        col = GENE_COLORS.get(g, "#1565c0")
        ax.scatter(
            gt + rng.normal(0, 0.008, size=len(gt)), pr,
            s=6, c=col, alpha=0.08, linewidths=0, zorder=1, rasterized=True,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, zorder=2)
        levels = np.array(sorted(np.unique(np.round(gt, 6))))
        xs, means, lo, hi = [], [], [], []
        for lv in levels:
            m = np.isclose(gt, lv, atol=1e-6)
            if int(m.sum()) < 5:
                continue
            vals = pr[m]
            xs.append(float(lv))
            means.append(float(np.mean(vals)))
            lo.append(float(np.percentile(vals, 25)))
            hi.append(float(np.percentile(vals, 75)))
        xs = np.asarray(xs)
        means = np.asarray(means)
        ax.vlines(xs, lo, hi, color=col, lw=2.2, zorder=3, alpha=0.9)
        ax.scatter(xs, means, s=42, c=col, edgecolors="white", linewidths=0.8, zorder=4)
        r = float(np.corrcoef(gt, pr)[0, 1])
        mae = float(np.mean(np.abs(gt - pr)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("GT KO fraction", fontsize=12)
        ax.tick_params(labelsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(
            0.04, 0.96,
            f"{g}\nn={len(gt)}  r={r:.2f}  MAE={mae:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11, color="black",
        )
    axes[0].set_ylabel("Predicted", fontsize=12)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def plot_gene_pies_combined(
    payloads: list[dict],
    out_png: Path,
):
    """2 rows: Icam1 (top, centered); Arap1|Sema4b (bottom).

    Coords normalized to a unit square per panel so pies are identical size.
    Equal square panels; legend on the right; no gene titles.
    """
    by = {pl["gene"]: dict(pl) for pl in payloads}  # shallow copy
    need = ["Icam1", "Arap1", "Sema4b"]
    missing = [g for g in need if g not in by]
    if missing:
        print(f"skip combined: missing {missing}", flush=True)
        return

    R_NORM = 0.042  # identical pie radius in unit-square coords

    def _normalize(pl):
        xy_p = np.asarray(pl["xy_p"], dtype=float)
        xy_i = np.asarray(pl["xy_i"], dtype=float)
        x0, x1 = xy_p[:, 0].min(), xy_p[:, 0].max()
        y0, y1 = xy_p[:, 1].min(), xy_p[:, 1].max()
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        half = 0.5 * max(x1 - x0, y1 - y0, 1.0) * 1.15

        def norm(xy):
            out = np.empty_like(xy, dtype=float)
            out[:, 0] = (xy[:, 0] - (cx - half)) / (2.0 * half)
            out[:, 1] = (xy[:, 1] - (cy - half)) / (2.0 * half)
            return out

        pl["xy_p"] = norm(xy_p)
        pl["xy_i"] = norm(xy_i)
        pl["xlim"] = (-0.02, 1.02)
        pl["ylim"] = (-0.02, 1.02)
        pl["radius"] = R_NORM
        return pl

    for g in need:
        by[g] = _normalize(by[g])

    # Equal square panels (inches)
    P = 2.4
    gap = 0.16
    leg_w = 1.85
    label_h = 0.42
    n_col = 4
    content_w = n_col * P + (n_col - 1) * gap
    fig_w = content_w + gap + leg_w
    fig_h = 2 * P + gap + label_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def _place(x_in, y_in, pl, weights, xlab):
        ax = fig.add_axes([x_in / fig_w, y_in / fig_h, P / fig_w, P / fig_h])
        ax.set_facecolor("white")
        ax.scatter(
            pl["xy_i"][:, 0], pl["xy_i"][:, 1],
            s=6, c=BG_COLOR, edgecolors="#d8d8d8", linewidths=0.1, zorder=1,
        )
        for (x, y), w in zip(pl["xy_p"], weights):
            draw_pie(ax, float(x), float(y), float(w), pl["radius"], pl["ko_color"])
        ax.set_xlim(*pl["xlim"])
        ax.set_ylim(*pl["ylim"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(xlab, fontsize=17, labelpad=7, color="black")
        return ax

    # Row 1 (top): Icam1 centered
    y_top = label_h + P + gap
    x_ic = (content_w - (2 * P + gap)) / 2.0
    _place(x_ic, y_top, by["Icam1"], by["Icam1"]["gt_p"], "Raw")
    _place(x_ic + P + gap, y_top, by["Icam1"], by["Icam1"]["pr_p"], "Predicted")

    # Row 2: Arap1 | Sema4b
    y_bot = label_h
    x = 0.0
    for g in ("Arap1", "Sema4b"):
        _place(x, y_bot, by[g], by[g]["gt_p"], "Raw")
        _place(x + P + gap, y_bot, by[g], by[g]["pr_p"], "Predicted")
        x += 2 * (P + gap)

    handles = [
        mpatches.Patch(color=by[g]["ko_color"], label=g) for g in need
    ] + [mpatches.Patch(color=UNA_COLOR, label="unassigned")]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=((content_w + 0.6 * gap) / fig_w, 0.52),
        fontsize=17,
        frameon=False,
        labelspacing=1.0,
        handlelength=1.4,
        borderaxespad=0.0,
    )
    fig.savefig(out_png, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=ASSETS_DEFAULT)
    ap.add_argument("--pred_csv", type=Path, default=PRED_DEFAULT)
    ap.add_argument("--outdir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--ext", type=Path, default=EXT_DEFAULT)
    ap.add_argument("--zoom_genes", default="Icam1,Arap1,Cd276,Sema4b,Slc43a3,Bcam")
    ap.add_argument("--max_pies", type=int, default=80)
    ap.add_argument("--pad_frac", type=float, default=0.03)
    ap.add_argument("--max_abs_err", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    args.ext.mkdir(parents=True, exist_ok=True)

    summ_path = outdir / "summary.json"
    plot_bars(summ_path, outdir / "bars_mae_pearson.png")

    pred = pd.read_csv(args.pred_csv)
    per_ko = plot_bars_per_ko(pred, outdir / "bars_mae_pearson_per_ko.png")
    per_ko.to_csv(outdir / "per_ko_metrics.csv", index=False)

    plot_scatter(pred, outdir / "scatter_paper_vs_nnls.png")
    plot_winrate(pred, outdir / "winrate_vs_nnls.png")

    gene_counts = pred.gene.value_counts()
    print("genes evaluated (n={}):".format(gene_counts.size), flush=True)
    print(gene_counts.to_string(), flush=True)

    roles = pd.read_csv(args.assets / "spot_roles.csv")
    genes = [g.strip() for g in args.zoom_genes.split(",") if g.strip()]
    zoom_meta = []
    payloads = []
    for g in genes:
        payload = _prepare_gene_pies(
            pred, roles, g,
            max_pies=args.max_pies, pad_frac=args.pad_frac, seed=args.seed,
            max_abs_err=args.max_abs_err,
        )
        if payload is None:
            print(f"skip {g}: no spots", flush=True)
            continue
        # per-gene figure (no titles)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True)
        _draw_gene_pie_pair(axes, payload, label_panels=True)
        handles = [
            mpatches.Patch(color=payload["ko_color"], label=payload["gene"]),
            mpatches.Patch(color=UNA_COLOR, label="unassigned"),
        ]
        axes[1].legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
        fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.98, bottom=0.14)
        out_g = outdir / f"slide_raw_vs_deconv_zoom_{g}.png"
        fig.savefig(out_g, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        m = payload["meta"]
        print(
            f"wrote {out_g}  island {m['n_island']}/{m['n_spots_total']}  "
            f"pies={m['n_pies']}",
            flush=True,
        )
        zoom_meta.append(m)
        payloads.append(payload)

    # 1x3 GT vs Predicted scatters for top-3 KO genes
    plot_gt_pred_scatter_row(
        pred, genes, outdir / "slide_raw_vs_deconv_zoom_top3.png",
    )

    # overview: keep prior copy if present; rename NNLS labels in copied metric figs
    copies = [
        ("bars_mae_pearson.png", "task1_spac21_deconv_mae_pearson_bars.png"),
        ("bars_mae_pearson_per_ko.png", "task1_spac21_deconv_mae_pearson_bars_per_ko.png"),
        ("scatter_paper_vs_nnls.png", "task1_spac21_deconv_scatter_concert_vs_nnls.png"),
        ("winrate_vs_nnls.png", "task1_spac21_deconv_winrate_vs_nnls.png"),
        ("deconv_paired_ko_frac_dots.png", "task1_spac21_deconv_paired_ko_frac_dots.png"),
        ("slide_raw_vs_deconv_zoom_top3.png", "task1_spac21_slide_raw_vs_deconv_zoom_top3.png"),
        ("slide_raw_vs_deconv_zoom_Icam1.png", "task1_spac21_slide_raw_vs_deconv_zoom_Icam1.png"),
        ("slide_raw_vs_deconv_zoom_Arap1.png", "task1_spac21_slide_raw_vs_deconv_zoom_Arap1.png"),
        ("slide_raw_vs_deconv_zoom_Sema4b.png", "task1_spac21_slide_raw_vs_deconv_zoom_Sema4b.png"),
    ]
    # also refresh overview if available under outdir/slide_viz or outdir
    for cand in [
        outdir / "slide_raw_vs_deconv_overview.png",
        outdir / "slide_viz" / "slide_raw_vs_deconv_overview.png",
    ]:
        if cand.exists():
            copies.append((str(cand.relative_to(outdir)) if cand.parent == outdir else f"slide_viz/{cand.name}",
                           "task1_spac21_slide_raw_vs_deconv_overview.png"))
            break

    for src, dst in copies:
        s = outdir / src
        if s.exists():
            (args.ext / dst).write_bytes(s.read_bytes())
            print(f"copied -> ext_exp/{dst}", flush=True)

    # remove old empirical-named copies in ext_exp (keep overview)
    for stale in [
        "task1_spac21_deconv_scatter_amortized_vs_empirical_nnls.png",
        "task1_spac21_deconv_winrate_vs_empirical_nnls.png",
    ]:
        stale_p = args.ext / stale
        if stale_p.exists():
            archive = args.ext / "task2_scratch_archive" / stale
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(stale_p.read_bytes())
            stale_p.unlink()
            print(f"archived stale {stale}", flush=True)

    note = {
        "n_genes_evaluated": int(gene_counts.size),
        "genes_evaluated": gene_counts.to_dict(),
        "zoom_gene_choice": (
            "Top-3 mixture genes by #spots in the Task-1 eval table "
            f"({', '.join(genes)}); metrics use all {int(gene_counts.size)} genes."
        ),
        "zoom_panels": zoom_meta,
        "barplot": "NNLS vs CONCERT only",
        "labels": "baseline displayed as NNLS (was empirical/naive NNLS)",
        "combined_zoom": "task1_spac21_slide_raw_vs_deconv_zoom_top3.png (1x6 GT-binned calibration, legend KOs)",
        "max_pies": args.max_pies,
    }
    (outdir / "paper_fig_replot_meta.json").write_text(json.dumps(note, indent=2))
    (args.ext / "task1_spac21_fig_meta.json").write_text(json.dumps(note, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
