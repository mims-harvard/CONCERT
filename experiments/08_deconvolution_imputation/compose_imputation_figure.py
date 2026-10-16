#!/usr/bin/env python3
"""Rebuild resp_imputation_patch.png with CONCERT scored by the shared LFC rule.

Panel a: pixels reused unchanged from the backed-up composite.
Panel b: both bar groups redrawn from task2_spac22_impute_rescored_our_rule.csv
         with EVERY method scored by the shared rule (per-cell Spearman column
         `scc_percell_hvg`, LFC column `lfc_scc`). Error bars = s.e.m. across
         the 5 KOs (ddof=1).
Panel c: five true-vs-pred LFC scatters; pred LFC uses the same observed
         nearby-unassigned reference as the true LFC (points written by
         plot_imputation_shared_reference.py).

The style constants below were read from the vector PDFs of the original panel b
components (task2_spac22_impute_{scc,lfc_scc}_bars_our_rule.pdf) and from the
pixel geometry of the original composite. Before drawing the new panels, the
script redraws panels b and c with the OLD numbers and reports the pixel
difference to the original composite, and uses the best integer pixel shift
found there to place the new panels.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import spearmanr

FIG_DIR = Path(
    f"{_rp.WORKDIR}/paper/manuscript/MAIN/FIG/supp"
)
ORIG = FIG_DIR / "resp_imputation_patch.png.bak_sharedref"
OUT_PNG = FIG_DIR / "resp_imputation_patch.png"
OUT_PDF = FIG_DIR / "resp_imputation_patch.pdf"
OUT_JSON = FIG_DIR / "resp_imputation_patch_stats.json"
EXT = Path(f"{_rp.WORKDIR}/paper/Outputs/ext_exp")
CSV = EXT / "task2_spac22_impute_rescored_our_rule.csv"
POINTS = EXT / "task2_spac22_eval5_lfc10_lfc_scatter_sharedref_points.npz"
CHECK_DIR = Path(
    f"{_rp.WORKDIR}/outputs/resp_imputation_sharedref"
)

METHODS = ["CONCERT", "SpaIM", "Tangram", "SpatialScope"]
METHOD_COLORS = {  # from the component PDFs (seaborn deep)
    "CONCERT": (0.2980392157, 0.4470588235, 0.6901960784),
    "SpaIM": (0.768627451, 0.3058823529, 0.3215686275),
    "Tangram": (0.8666666667, 0.5176470588, 0.3215686275),
    "SpatialScope": (0.3333333333, 0.6588235294, 0.4078431373),
}
KOS = ["H2-Ea", "Iqgap1", "Cttn", "Ccn1", "Phgr1"]
KO_MARKERS = {"H2-Ea": "o", "Iqgap1": "s", "Cttn": "^", "Ccn1": "D", "Phgr1": "v"}
# Per (method, KO) horizontal marker offsets in category units, read from the PDF.
KO_XOFF = {
    "CONCERT": {"H2-Ea": 0.0383, "Iqgap1": -0.0641, "Cttn": -0.1283, "Ccn1": -0.1352, "Phgr1": 0.0879},
    "SpaIM": {"H2-Ea": 0.1156, "Iqgap1": 0.0299, "Cttn": 0.0643, "Ccn1": 0.0122, "Phgr1": 0.1216},
    "Tangram": {"H2-Ea": 0.0885, "Iqgap1": -0.1392, "Cttn": 0.1001, "Ccn1": -0.1305, "Phgr1": 0.0643},
    "SpatialScope": {"H2-Ea": -0.0908, "Iqgap1": 0.1017, "Cttn": 0.0116, "Ccn1": -0.0561, "Phgr1": -0.0217},
}
# Scatter colours of panel c (same rule as plot_imputation.py)
_KO_ORDER_MAP = ["Iqgap1", "Ccn1", "Cttn", "Phgr1", "H2-Ea"]
KO_COLORS = {
    g: to_hex(c)
    for g, c in zip(_KO_ORDER_MAP, plt.cm.tab10(np.linspace(0, 1, len(_KO_ORDER_MAP))))
}

# Geometry of the original composite (pixels, origin top-left)
B_BOX = (935, 0, 1728, 440)  # x0, y0, x1, y1
C_BOX = (1, 461, 1714, 834)
PC_AX_LEFT, PC_AX_BOTTOM, PC_DPI = 1035.0, 344.5, 139.45
LFC_AX_LEFT, LFC_AX_BOTTOM, LFC_DPI = 1361.5, 347.5, 135.9
AX_W_PT, AX_H_PT = 124.416, 219.024
C_DPI = 160 * (C_BOX[2] - C_BOX[0]) / 2695.0
C_AX_LEFT = [75.5, 414.5, 752.5, 1091.5, 1430.5]
C_AX_TOP, C_AX_BOTTOM, C_AX_W = 503.5, 779.5, 274.0
LETTER_C = (12, 455, 32, 475)


def sem(v):
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


def _render(fig, W, H):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    out = np.full((H, W, 3), 255, np.uint8)
    h, w = min(H, buf.shape[0]), min(W, buf.shape[1])
    out[:h, :w] = buf[:h, :w]
    return out


def bar_panel(vals, ylim, yticks, ylabel, ax_left, ax_bottom, dpi, ax_h_pt,
              label_off_pt, legend):
    """vals: {method: [v per KO in KOS]}. Returns an RGB array the size of B_BOX."""
    x0, y0, x1, y1 = B_BOX
    W, H = x1 - x0, y1 - y0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    ax_w_px, ax_h_px = AX_W_PT * dpi / 72, ax_h_pt * dpi / 72
    ax = fig.add_axes([(ax_left - x0) / W, (y1 - ax_bottom) / H, ax_w_px / W, ax_h_px / H])
    x = np.arange(len(METHODS))
    means = [float(np.mean(vals[m])) for m in METHODS]
    sems = [sem(vals[m]) for m in METHODS]
    ax.axhline(0, color="0.55", lw=0.5, zorder=0)
    ax.bar(x, means, width=0.62, color=[METHOD_COLORS[m] for m in METHODS],
           edgecolor="white", linewidth=0.5, zorder=1)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="0.25", elinewidth=0.9,
                capsize=3, capthick=0.9, zorder=2)
    for i, m in enumerate(METHODS):
        for k, g in enumerate(KOS):
            ax.scatter(i + KO_XOFF[m][g], vals[m][k], s=26, marker=KO_MARKERS[g],
                       facecolor="white", edgecolor="0.2", linewidths=0.65, zorder=3)
        ax.annotate(f"{means[i]:.2f}", (i, max(vals[m])), xytext=(0, label_off_pt),
                    textcoords="offset points", ha="center", va="baseline",
                    fontsize=9.5, color="0.1")
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([(f"{t:.2f}" if ylim[1] < 0.5 else f"{t:.1f}").replace("-", "\u2212")
                        for t in yticks],
                       fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        handles = [
            Line2D([0], [0], marker=KO_MARKERS[g], ls="", mfc="white", mec="0.2",
                   mew=0.65, ms=np.sqrt(26), label=g)
            for g in KOS
        ]
        # legend anchored in points above the axes bottom so it stays where it
        # was when the axes height changes (x anchor read from the PDF)
        leg = ax.legend(handles=handles, title="KO", loc="center left",
                        bbox_to_anchor=(1.0769, (AX_H_PT / 2 + LEG_DY_PT) / ax_h_pt),
                        frameon=False, fontsize=8, title_fontsize=9, handlelength=1.2)
        leg.set_zorder(5)
    return _render(fig, W, H), means, sems


LEG_DY_PT = -2.9


def facet_panel(true_by, pred_by):
    x0, y0, x1, y1 = C_BOX
    W, H = x1 - x0, y1 - y0
    dpi = C_DPI
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    rhos = {}
    for j, g in enumerate(KOS):
        ax_h = C_AX_BOTTOM - C_AX_TOP
        ax = fig.add_axes([(C_AX_LEFT[j] - x0) / W, (y1 - C_AX_BOTTOM) / H, C_AX_W / W, ax_h / H])
        o_h, m_h = true_by[g], pred_by[g]
        rs = float(spearmanr(o_h, m_h).correlation)
        rhos[g] = rs
        ax.scatter(o_h, m_h, s=12, alpha=0.55, c=KO_COLORS[g], edgecolors="none",
                   rasterized=True)
        lim = float(max(np.percentile(np.abs(o_h), 99), np.percentile(np.abs(m_h), 99), 0.05))
        lim *= 1.15
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.55)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(0, color="0.7", lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{g}\n$\\rho$={rs:.2f}", fontsize=11, color="black")
        ax.set_xlabel("true LFC", fontsize=10, color="black")
        ax.tick_params(colors="black")
        for sp in ax.spines.values():
            sp.set_color("black")
        if j == 0:
            ax.set_ylabel("pred LFC", fontsize=10, color="black")
    return _render(fig, W, H), rhos


def best_shift(render, target_rgb, r=4):
    tg = target_rgb.astype(float).mean(-1)
    rg = render.astype(float).mean(-1)
    best = (1e9, 0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = np.abs(np.roll(np.roll(rg, dy, 0), dx, 1) - tg).mean()
            if d < best[0]:
                best = (d, dy, dx)
    return best


def shift(img, dy, dx):
    """Integer translation with white fill (np.roll would wrap edge pixels)."""
    out = np.full_like(img, 255)
    H, W = img.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = img[ys, xs]
    return out


def main():
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    orig = np.array(Image.open(ORIG).convert("RGBA"))
    # composite over white for comparisons
    a = orig[..., 3:4].astype(float) / 255
    orig_flat = (orig[..., :3] * a + 255 * (1 - a)).astype(np.uint8)

    df = pd.read_csv(CSV)
    def per_ko(col):
        return {m: [float(df[(df.method == m) & (df.group == g)][col].iloc[0]) for g in KOS]
                for m in METHODS}
    pc_vals = per_ko("scc_percell_hvg")
    lfc_vals = per_ko("lfc_scc")
    lfc_old = per_ko("lfc_scc")
    lfc_old["CONCERT"] = [float(df[(df.method == "CONCERT") & (df.group == g)]["lfc_scc_vs_basal"].iloc[0]) for g in KOS]

    bx0, by0, bx1, by1 = B_BOX
    cx0, cy0, cx1, cy1 = C_BOX
    tgt_b = orig_flat[by0:by1, bx0:bx1]
    tgt_c = orig_flat[cy0:cy1, cx0:cx1]
    report = {}

    # reproduction check with the OLD numbers
    pc_old, _, _ = bar_panel(pc_vals, (0, 0.42), np.arange(0, 0.401, 0.05), "Per-cell Spearman",
                             PC_AX_LEFT, PC_AX_BOTTOM, PC_DPI, AX_H_PT, 11.84, legend=False)
    lf_old, _, _ = bar_panel(lfc_old, (0, 1.05), np.arange(0, 1.001, 0.2), "LFC Spearman",
                             LFC_AX_LEFT, LFC_AX_BOTTOM, LFC_DPI, AX_H_PT, 10.72, legend=True)
    mask_pc = np.zeros(tgt_b.shape[:2], bool); mask_pc[:, : 1290 - bx0] = True
    s_pc = best_shift(np.where(mask_pc[..., None], pc_old, 255), np.where(mask_pc[..., None], tgt_b, 255))
    s_lf = best_shift(np.where(~mask_pc[..., None], lf_old, 255), np.where(~mask_pc[..., None], tgt_b, 255))
    b_old = np.minimum(shift(pc_old, s_pc[1], s_pc[2]), shift(lf_old, s_lf[1], s_lf[2]))
    report["repro_b"] = {"percell_shift_dy_dx": s_pc[1:], "lfc_shift_dy_dx": s_lf[1:],
                         "mean_abs_gray_diff": float(np.abs(b_old.astype(float).mean(-1) - tgt_b.astype(float).mean(-1)).mean())}

    pts = np.load(POINTS, allow_pickle=True)
    true_by = {g: pts[f"true_{g}"] for g in KOS}
    pred_new = {g: pts[f"pred_{g}"] for g in KOS}
    pred_old = {g: pts[f"pred_vsbasal_{g}"] for g in KOS}
    c_old, rho_old = facet_panel(true_by, pred_old)
    s_c = best_shift(c_old, tgt_c)
    c_old_s = shift(c_old, s_c[1], s_c[2])
    report["repro_c"] = {"shift_dy_dx": s_c[1:], "mean_abs_gray_diff": float(s_c[0]),
                         "rho_old_rule": rho_old}
    Image.fromarray(np.concatenate([tgt_b, b_old], 1)).save(CHECK_DIR / "repro_check_b.png")
    Image.fromarray(np.concatenate([tgt_c, c_old_s], 0)).save(CHECK_DIR / "repro_check_c.png")

    # new panels
    pc_new, pc_means, pc_sems = bar_panel(pc_vals, (0, 0.42), np.arange(0, 0.401, 0.05),
                                          "Per-cell Spearman", PC_AX_LEFT, PC_AX_BOTTOM,
                                          PC_DPI, AX_H_PT, 11.84, legend=False)
    # LFC: a negative per-KO value forces the axis below 0; the bottom spine and
    # the visible top (canvas row 0) stay where they were.
    lfc_ylim = (-0.45, 0.87)
    lfc_ax_h_pt = (LFC_AX_BOTTOM - 0.0) * 72 / LFC_DPI
    lf_new, lf_means, lf_sems = bar_panel(lfc_vals, lfc_ylim, np.round(np.arange(-0.4, 0.801, 0.2), 1),
                                          "LFC Spearman", LFC_AX_LEFT, LFC_AX_BOTTOM,
                                          LFC_DPI, lfc_ax_h_pt, 10.72, legend=True)
    b_new = np.minimum(shift(pc_new, s_pc[1], s_pc[2]), shift(lf_new, s_lf[1], s_lf[2]))
    c_new, rho_new = facet_panel(true_by, pred_new)
    c_new = shift(c_new, s_c[1], s_c[2])

    out = orig.copy()
    out[by0:by1, bx0:bx1, :3] = b_new
    out[by0:by1, bx0:bx1, 3] = 255
    letter = orig[LETTER_C[1]:LETTER_C[3], LETTER_C[0]:LETTER_C[2]].copy()
    out[cy0:cy1, cx0:cx1, :3] = c_new
    out[cy0:cy1, cx0:cx1, 3] = 255
    lx0, ly0, lx1, ly1 = LETTER_C
    region = out[ly0:ly1, lx0:lx1]
    dark = (letter[..., 3] > 0) & (letter[..., :3].max(-1) < 200)
    region[dark] = letter[dark]
    Image.fromarray(out).save(OUT_PNG, dpi=(144, 144))
    a2 = out[..., 3:4].astype(float) / 255
    flat = (out[..., :3] * a2 + 255 * (1 - a2)).astype(np.uint8)
    Image.fromarray(flat).save(OUT_PDF, resolution=144.0)

    lfc_ref = {g: float(df[(df.method == "CONCERT") & (df.group == g)]["lfc_scc"].iloc[0]) for g in KOS}
    stats = {
        "figure": str(OUT_PNG),
        "backup_of_previous_version": str(ORIG),
        "scoring_rule": (
            "All methods scored with one rule. Per-cell Spearman = column scc_percell_hvg "
            "(mean per-cell Spearman on top-200 HVGs after CP10K->log1p). LFC Spearman = column "
            "lfc_scc: LFC = mean(log1p CP10K imputed KO cells) - mean(log1p CP10K nearby observed "
            "kept-unassigned cells, k=50 NN pool), Spearman vs the observed LFC (same reference) on "
            "top-200 HVGs. Previous version plotted CONCERT lfc_scc_vs_basal (reference = CONCERT imputed basal)."
        ),
        "source_table": str(CSV),
        "error_bars": "s.e.m. across the 5 KOs, std(ddof=1)/sqrt(5)",
        "panel_b": {
            "per_cell_spearman": {
                m: {"per_ko": dict(zip(KOS, pc_vals[m])), "mean": pc_means[i], "sem": pc_sems[i]}
                for i, m in enumerate(METHODS)
            },
            "lfc_spearman": {
                m: {"per_ko": dict(zip(KOS, lfc_vals[m])), "mean": lf_means[i], "sem": lf_sems[i]}
                for i, m in enumerate(METHODS)
            },
            "lfc_axis_ylim": list(lfc_ylim),
            "concert_lfc_previous_vs_basal": {
                "per_ko": dict(zip(KOS, lfc_old["CONCERT"])),
                "mean": float(np.mean(lfc_old["CONCERT"])), "sem": sem(lfc_old["CONCERT"]),
            },
        },
        "panel_c": {
            "genes": "top-200 HVG by variance of CP10K->log1p GT across held-out cells",
            "points": str(POINTS),
            "rho_shared_reference": rho_new,
            "rho_table_lfc_scc": lfc_ref,
            "abs_diff_vs_table": {g: abs(rho_new[g] - lfc_ref[g]) for g in KOS},
            "rho_previous_vs_basal": rho_old,
            "panel_order": KOS,
        },
        "layout_check": report,
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("Wrote", OUT_PNG, OUT_PDF, OUT_JSON)


if __name__ == "__main__":
    main()
