#!/usr/bin/env python3
"""Four-arm rebuild of SI figure resp_deconv_spacseq (SPAC-seq slide 2-1 deconvolution).

Copy of replot_task1_paper_figs.py restricted to what the SI figure needs.
Only panel b changes: it now shows four arms so the learned unmixing head and
the signature source are separated:
  NNLS + empirical signatures        (naive_nnls_empirical)
  NNLS + CONCERT signatures          (naive_nnls_concert)
  Learned head + empirical signatures (empirical_amortized)
  Learned head + CONCERT signatures   (concert_amortized)

Panels a, c and d are taken pixel-for-pixel from the previous composite
(resp_deconv_spacseq.png.bak_4arm), so layout and panel letters are unchanged.
Panel c and d are also re-rendered standalone with the original, unmodified
functions (plot_winrate, plot_gt_pred_scatter_row) into the work directory as a
provenance check; their numbers go into the stats JSON.

Caller:
  python replot_task1_paper_figs_4arm.py
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
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

DECONV = Path(f"{_rp.DATA}/deconvolution/spacseq_2-1")
FIG = Path(f"{_rp.WORKDIR}/deconvolution_spacseq_2-1/four_arm")
WORK = FIG / "work"

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

# Arms in display order. Colours: blue family = NNLS, red family = learned head;
# the two original arms keep their exact colours, the added arms are lighter tints.
ARMS = [
    dict(method="naive_nnls_empirical", group="NNLS", sig="Empirical",
         label="NNLS, empirical signatures", color="#1565c0"),
    dict(method="naive_nnls_concert", group="NNLS", sig="CONCERT",
         label="NNLS, CONCERT signatures", color="#7fb0e6"),
    dict(method="empirical_amortized", group="Learned head", sig="Empirical",
         label="Learned head, empirical signatures", color="#e88a86"),
    dict(method="concert_amortized", group="Learned head", sig="CONCERT",
         label="Learned head, CONCERT signatures", color="#c62828"),
]

# Geometry of the previous composite (1890 x 768 px), measured from its pixels
# (probe_layout.py): panel b axes boxes and the band that panel b owns.
IMG_W, IMG_H = 1890, 768
B_CLEAR = dict(x0=660, x1=1399, y0=42, y1=448)  # rows/cols wiped before redraw; keeps letter b
B_AX_LEFT = (745, 1026)   # x-range (px) of the MAE axes spines
B_AX_RIGHT = (1110, 1391)  # x-range (px) of the Pearson axes spines
B_AX_Y = (54, 391)        # y-range (px, from top) of both axes
# The old composite shrank the dpi-160 bar figure by 0.647, an effective 103.5 dpi.
# Rendering at 100 dpi, point sizes are scaled by 1.035 to keep the same pixel size.
DPI = 100
FS = 103.5 / 100.0


def plot_winrate(pred: pd.DataFrame, out_png: Path):
    """Unmodified from replot_task1_paper_figs.py (panel c)."""
    gt = pred.gt_frac_ko.to_numpy()
    err_nnls = np.abs(gt - pred.naive_nnls_empirical.to_numpy())
    err_am = np.abs(gt - pred.concert_amortized.to_numpy())
    d = err_nnls - err_am  # >0 means the CONCERT-signature head is better
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
    return d


def plot_gt_pred_scatter_row(
    pred: pd.DataFrame,
    out_png: Path,
    pred_col: str = "concert_amortized",
    seed: int = 42,
):
    """Unmodified from replot_task1_paper_figs.py (panel d); bars are the 25th-75th percentile."""
    prefer = ["Icam1", "Arap1", "Cd276", "Sema4b", "Slc43a3", "Bcam"]
    genes = [g for g in prefer if (pred.gene == g).any()]
    rng = np.random.default_rng(seed)
    n = len(genes)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharex=True, sharey=True)
    per_gene = []
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
        per_gene.append(dict(gene=g, n=int(len(gt)), pearson=r, MAE=mae))
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
    return per_gene


def _px_axes(fig, xr, yr):
    """Add axes from pixel ranges (x from left, y from top) of an IMG_W x IMG_H figure."""
    x0, x1 = xr
    y0, y1 = yr
    return fig.add_axes([x0 / IMG_W, 1.0 - y1 / IMG_H, (x1 - x0) / IMG_W, (y1 - y0) / IMG_H])


def draw_bars_4arm(fig, arm_vals):
    """Panel b: two axes (MAE, Pearson r), four bars each, one shared 2 x 2 legend below.

    Four full two-part labels do not fit as tick labels under 4 bars in a 281 px
    wide axes, so the bars are identified by a legend whose column-first order
    matches the left-to-right bar order.
    """
    import matplotlib.patches as mpatches

    xpos = np.arange(len(ARMS), dtype=float)
    bw = 0.62
    cols = [a["color"] for a in ARMS]
    axes = [_px_axes(fig, B_AX_LEFT, B_AX_Y), _px_axes(fig, B_AX_RIGHT, B_AX_Y)]
    specs = [
        (axes[0], [v["MAE"] for v in arm_vals], "Composition MAE \u2193"),
        (axes[1], [v["pearson"] for v in arm_vals], "Pearson r \u2191"),
    ]
    for ax, vals, ylabel in specs:
        ax.bar(xpos, vals, color=cols, width=bw)
        ax.set_xticks([])
        ax.set_ylabel(ylabel, fontsize=12 * FS)
        ax.tick_params(axis="y", labelsize=11 * FS)
        ax.set_xlim(-0.55, len(ARMS) - 0.45)
        # 1.28 x headroom reproduces the y-range of the previous composite (0.38 / 0.64)
        ymax = max(vals) * 1.28
        ax.set_ylim(0, ymax)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02 * ymax, f"{v:.3f}", ha="center", va="bottom", fontsize=11 * FS)
    handles = [mpatches.Patch(color=a["color"], label=a["label"]) for a in ARMS]
    x_mid = 0.5 * (B_AX_LEFT[0] + B_AX_RIGHT[1]) / IMG_W
    y_top = 1.0 - (B_AX_Y[1] + 13) / IMG_H  # clears the half-line of the y=0 tick labels
    leg = fig.legend(
        handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(x_mid, y_top),
        frameon=False, fontsize=11 * FS, handlelength=1.2, handleheight=0.9,
        handletextpad=0.4, columnspacing=1.2, labelspacing=0.25, borderpad=0.0,
        borderaxespad=0.0,
    )
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    # display coords have origin bottom-left; convert to image rows from top
    lx0, lx1 = bb.x0, bb.x1
    ly_top, ly_bot = IMG_H - bb.y1, IMG_H - bb.y0
    print(f"legend px box x=[{lx0:.0f},{lx1:.0f}] y=[{ly_top:.0f},{ly_bot:.0f}]", flush=True)
    assert B_CLEAR["x0"] <= lx0 and lx1 <= B_CLEAR["x1"], "legend overflows panel b horizontally"
    assert ly_bot <= B_CLEAR["y1"], "legend overflows panel b vertically"
    return axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DECONV / "summary.json")
    ap.add_argument("--pred_csv", type=Path, default=DECONV / "per_spot.csv")
    ap.add_argument("--base_png", type=Path, default=DECONV / "resp_deconv_spacseq_previous_composite.png")
    ap.add_argument("--out_png", type=Path, default=FIG / "resp_deconv_spacseq.png")
    ap.add_argument("--out_pdf", type=Path, default=FIG / "resp_deconv_spacseq.pdf")
    ap.add_argument("--stats", type=Path, default=FIG / "resp_deconv_spacseq_stats.json")
    ap.add_argument("--work", type=Path, default=WORK)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    summ = pd.read_json(args.summary)
    pred = pd.read_csv(args.pred_csv)
    gt = pred.gt_frac_ko.to_numpy(dtype=float)

    arm_vals = []
    for a in ARMS:
        row = summ.loc[summ.method == a["method"]].iloc[0]
        p = pred[a["method"]].to_numpy(dtype=float)
        mae_re = float(np.mean(np.abs(gt - p)))
        r_re = float(np.corrcoef(gt, p)[0, 1])
        # summary.json is the plotted source; per_spot.csv must reproduce it
        assert abs(mae_re - float(row.comp_MAE)) < 1e-6, (a["method"], mae_re, row.comp_MAE)
        assert abs(r_re - float(row.comp_pearson)) < 1e-6, (a["method"], r_re, row.comp_pearson)
        assert int(row.n) == len(pred)
        arm_vals.append(dict(method=a["method"], label=a["label"], color=a["color"],
                             n=int(row.n), MAE=float(row.comp_MAE),
                             pearson=float(row.comp_pearson)))
        print(f"{a['method']:26s} n={int(row.n)} MAE={float(row.comp_MAE):.4f} "
              f"r={float(row.comp_pearson):.4f}", flush=True)

    # Standalone re-renders of the unchanged panels c and d (provenance only).
    d = plot_winrate(pred, args.work / "panel_c_winrate_standalone.png")
    per_gene = plot_gt_pred_scatter_row(pred, args.work / "panel_d_calibration_standalone.png")

    # Base image: previous composite flattened on white, panel b band wiped.
    base = np.asarray(Image.open(args.base_png).convert("RGBA")).astype(np.float64)
    assert base.shape[:2] == (IMG_H, IMG_W), base.shape
    alpha = base[..., 3:4] / 255.0
    rgb = np.clip(np.rint(base[..., :3] * alpha + 255.0 * (1.0 - alpha)), 0, 255).astype(np.uint8)
    flat_old = rgb.copy()
    rgb[B_CLEAR["y0"]:B_CLEAR["y1"], B_CLEAR["x0"]:B_CLEAR["x1"]] = 255

    # +0.001 px guards against float truncation (18.9 * 100 = 1889.999...)
    fig = plt.figure(figsize=((IMG_W + 1e-3) / DPI, (IMG_H + 1e-3) / DPI), dpi=DPI, facecolor="white")
    # figimage places the pixels 1:1 (imshow resampled and shifted edges by a pixel)
    fig.figimage(rgb, xo=0, yo=0, origin="upper", resize=False, zorder=-10)  # below the panel b axes
    draw_bars_4arm(fig, arm_vals)

    fig.savefig(args.out_png, dpi=DPI, facecolor="white")
    fig.savefig(args.out_pdf, dpi=DPI, facecolor="white")  # figimage is sized by dpi
    plt.close(fig)
    print(f"wrote {args.out_png}\nwrote {args.out_pdf}", flush=True)

    # Check: size unchanged, and every pixel outside the panel b band identical.
    new = np.asarray(Image.open(args.out_png).convert("RGB")).astype(int)
    assert new.shape[:2] == (IMG_H, IMG_W), new.shape
    outside = np.ones((IMG_H, IMG_W), bool)
    outside[B_CLEAR["y0"]:B_CLEAR["y1"], B_CLEAR["x0"]:B_CLEAR["x1"]] = False
    diff = np.abs(new - flat_old.astype(int)).max(-1)
    n_changed_outside = int((diff[outside] > 0).sum())
    print(f"pixels changed outside panel b band: {n_changed_outside} "
          f"(max diff {int(diff[outside].max())})", flush=True)
    Image.fromarray(new[:460, 640:1420].astype(np.uint8)).save(args.work / "panel_b_crop.png")

    stats = {
        "figure": str(args.out_png),
        "source_summary": str(args.summary),
        "source_per_spot": str(args.pred_csv),
        "script": str(Path(__file__).resolve()),
        "panel_b_arms": arm_vals,
        "panel_c": {
            "quantity": "per-spot |GT - naive_nnls_empirical| - |GT - concert_amortized|",
            "n_spots": int(len(d)),
            "mean_delta": float(np.mean(d)),
            "win_fraction": float(np.mean(d > 0)),
        },
        "panel_d": {
            "pred_col": "concert_amortized",
            "bars": "25th-75th percentile of predictions per GT level (levels with >=5 spots); dot = mean",
            "per_gene": per_gene,
        },
        "layout_check": {"pixels_changed_outside_panel_b": n_changed_outside},
    }
    args.stats.write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.stats}", flush=True)


if __name__ == "__main__":
    main()
