#!/usr/bin/env python
"""Per-mouse virtual course as a dot matrix: rows = mice, columns = virtual day.

The companion of plot_q4_gt_vs_pred_bubbles.py, one level up in aggregation. There the
rows were single markers for a few mice; here each row is a MOUSE and the quantity is the
AGGREGATED DSS marker level (mean log1p over the 17 core markers), so the whole cohort's
imputed injury course fits in one panel.

Same GT-anchored motif as every other figure in the set:

        left half-disc  = Observed      right half-disc = CONCERT
        empty dashed half-disc          = that mouse was never measured at that day

Each animal is sectioned once, so exactly ONE column per row carries an observed half --
the diagonal. Everything off that diagonal is the counterfactual re-decode, which is the
entire point: a terminal snapshot turned into a virtual longitudinal course.

Colour = the aggregate level; there is NO size channel here (every dot is the same size).
The per-cell "% expressing" that sized the marker dots is only available for the 8 mice in
percell_dump, and an encoding that exists for half the rows is worse than none. White in
the colour ramp sits at the observed healthy (Day 0) cohort level, so blue = below healthy
tissue, red = above.

Readout calibration is the ONE shared rule (paper_style.calibrate, aggregate entry):
calibrated(t) = observed(Day 0) + a * (pred(t) - pred(Day 0)), a = 0.2051. Nothing is
re-fitted here.

Figures
    fig_q4_mouse_time_dotplot.png            the four featured mice (one per day)
    fig_q4_mouse_time_dotplot_allmice.png    all 15 mice, ordered by sampled day

Input:  pathway_dump.npz (fixedt all-15-mice heldout_eval) -- per-mouse per-gene means,
        predicted at each virtual day and observed on the diagonal.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
import paper_style as ps
ps.use()

DUMP = ("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
        "dss_timekernel_markersign_timefield_fixedt/heldout_eval/pathway_dump.npz")
HERE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study"
DAYS = [0.0, 3.0, 9.0, 21.0]
# the same featured animals as the slide and marker figures (best structural completeness)
FEATURED = ["082421_D0_m6", "092421_D3_m1", "100221_D9_m5", "082421_D21_m1"]
MARKERS = ["Isg15", "Ifit1", "Tnf", "Cxcl1", "Nos2", "Il11", "Il1b", "Mmp3", "Mmp10",
           "Mmp13", "Plau", "Igfbp5", "Col18a1", "Mmp2", "Osm", "Cxcl10", "Cxcl5"]
BASE_CMAP = "coolwarm"
ROW_FS, DAY_FS, CBAR_FS, LEG_FS = 17, 18, 16, 16
# Geometry is fixed in INCHES, not left to the figure size: one cell is the same physical
# square in both figures, so a dot means the same thing in the 4-mouse and the 15-mouse
# panel and neither ends up with dots floating in whitespace.
CELL_W, CELL_H = 1.02, 0.82      # inches per column / per row; columns are the wider of
DOT_FRAC = 0.92                  # the two only so the day headers do not collide

MARGIN = dict(left=2.55, right=2.35, top=0.60, bottom=0.85)     # inches around the axes
CBAR_LABEL = "Aggregated marker level"


def pivot_cmap(name, pivot, n=512):
    """coolwarm with its white point moved to `pivot` of the colorbar LENGTH.

    The norm stays LINEAR, so position on the bar is proportional to value and the ticks
    come out evenly spaced; only WHICH colour sits where changes. Here white is placed at
    the observed healthy cohort level, which makes the ramp mean something: blue is below
    healthy tissue, red above it. Never buy contrast from the norm (a TwoSlopeNorm makes
    the bar piecewise, so tick spacing reports colour rather than value)."""
    base = plt.get_cmap(name)
    k = int(np.clip(round(n * pivot), 2, n - 2))
    return LinearSegmentedColormap.from_list(
        f"{name}_p{pivot:g}",
        np.vstack([base(np.linspace(0.0, 0.5, k)), base(np.linspace(0.5, 1.0, n - k))]))


# ── data: per-mouse per-gene means, all 15 mice ────────────────────────────────────
d = np.load(DUMP, allow_pickle=True)
genes = [str(g) for g in d["gene_names"]]
midx = [genes.index(g) for g in MARKERS]              # all 17 markers are on the panel
mice = [str(m) for m in d["mouse_order"]]
mday = np.asarray(d["mouse_day"], float)
# aggregate = mean over the 17 markers of the per-gene cell-mean log1p (averaging order
# commutes, so this is exactly the per-cell obs_score/pred score used everywhere else)
obs = d["gmean_obs"][:, midx].mean(1)                 # (15,)   diagonal only
pred = ps.calibrate(d["gmean_pred"][:, midx, :].mean(1))   # (15, 4) on the observed scale

order = sorted(range(len(mice)), key=lambda i: ps.MOUSE_NUMBER[mice[i]])

print("[q4mouse] aggregated DSS marker level (calibrated), one row per mouse")
print(f"{'mouse':>10s} {'sampled':>8s} {'observed':>9s} | " +
      " ".join(f"{ps.day_label(t):>7s}" for t in DAYS))
for i in order:
    print(f"{'Mouse ' + str(ps.MOUSE_NUMBER[mice[i]]):>10s} {ps.day_label(mday[i]):>8s} "
          f"{obs[i]:9.4f} | " + " ".join(f"{v:7.4f}" for v in pred[i]))


def _half_disc(side):
    """Unit half-disc path sized like matplotlib's 'o' (radius 0.5), flat edge vertical."""
    a0, a1 = (np.pi / 2, 3 * np.pi / 2) if side == "left" else (-np.pi / 2, np.pi / 2)
    th = np.linspace(a0, a1, 80)
    pts = np.column_stack([np.cos(th), np.sin(th)]) * 0.5
    return Path(np.vstack([pts, pts[:1]]), closed=True)


LEFT, RIGHT = _half_disc("left"), _half_disc("right")


def figure(rows, out_name):
    """rows = indices into the dump, already in display order (top row drawn first)."""
    n = len(rows)
    vmax = float(max(pred[rows].max(), obs[rows].max())) * 1.02
    norm = Normalize(vmin=0.0, vmax=vmax)
    # white sits at the observed healthy cohort level, not at half the bar
    cmap = pivot_cmap(BASE_CMAP, ps.calibration()["aggregate"]["observed_baseline"] / vmax)

    aw, ah = CELL_W * len(DAYS), CELL_H * n
    fig = plt.figure(figsize=(aw + MARGIN["left"] + MARGIN["right"],
                              ah + MARGIN["top"] + MARGIN["bottom"]))
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([MARGIN["left"] / fw, MARGIN["bottom"] / fh, aw / fw, ah / fh])
    dot = (DOT_FRAC * min(CELL_W, CELL_H) * 72.0) ** 2   # marker area in points^2

    for r, i in enumerate(rows):
        for c, t in enumerate(DAYS):
            if abs(mday[i] - t) < 1e-6:
                ax.scatter([c], [r], s=dot, c=[obs[i]], marker=LEFT,
                           cmap=cmap, norm=norm, edgecolors="0.20", linewidths=0.8)
            else:
                ax.scatter([c], [r], s=dot, marker=LEFT, facecolors="none",
                           edgecolors=ps.MUTED, linewidths=1.4, linestyle=(0, (2, 1.6)))
            ax.scatter([c], [r], s=dot, c=[pred[i, c]], marker=RIGHT,
                       cmap=cmap, norm=norm, edgecolors="0.20", linewidths=0.8)

    ax.set_xticks(range(len(DAYS)))
    ax.set_xticklabels([ps.day_label(t) for t in DAYS], fontsize=DAY_FS)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n))
    ax.set_yticklabels([ps.mouse_label(mice[i], mday[i]) for i in rows], fontsize=ROW_FS)
    ax.set_xlim(-0.5, len(DAYS) - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis(); ax.grid(False)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)

    cb_h = min(ah * 0.72, 3.6)
    cax = fig.add_axes([(MARGIN["left"] + aw + 0.40) / fw,
                        (MARGIN["bottom"] + 0.5 * (ah - cb_h)) / fh, 0.22 / fw, cb_h / fh])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    ticks = list(np.linspace(0.0, vmax, 4))            # evenly spaced BY VALUE
    cb.set_ticks(ticks); cb.set_ticklabels(["%.3f" % v for v in ticks])
    cb.set_label(CBAR_LABEL, fontsize=CBAR_FS)
    cb.ax.tick_params(labelsize=CBAR_FS - 3)

    keys = [Line2D([0], [0], marker=LEFT, ls='', mfc='0.55', mec='0.20', ms=17,
                   label="Observed"),
            Line2D([0], [0], marker=RIGHT, ls='', mfc='0.55', mec='0.20', ms=17,
                   label="CONCERT"),
            Line2D([0], [0], marker=LEFT, ls='', mfc='none', mec=ps.MUTED, mew=1.4,
                   ms=17, label="Never measured")]
    fig.legend(handles=keys, loc="upper center",
               bbox_to_anchor=((MARGIN["left"] + 0.5 * aw) / fw, (MARGIN["bottom"] - 0.12) / fh),
               ncol=3, frameon=False, fontsize=LEG_FS, handletextpad=0.5,
               columnspacing=2.0)

    out = f"{HERE}/{out_name}"
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


figure([mice.index(m) for m in FEATURED], "fig_q4_mouse_time_dotplot.png")
figure(order, "fig_q4_mouse_time_dotplot_allmice.png")
