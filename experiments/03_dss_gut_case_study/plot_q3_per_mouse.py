#!/usr/bin/env python
"""Q3 (GT-anchored): combined slide figures showing Observed BESIDE prediction.

Design principle: never display a prediction alone where an observed value exists.
Terminal sampling means each mouse is measured at exactly ONE day, so the observed
panel exists on the diagonal only -- the figure marks the other slots with an empty
circle instead of quietly showing prediction alone.

Layout (both figures)
    rows    = 4 mice, one per sampled day, ONE slide each
    columns = 4 virtual days, each SPLIT into two sub-columns: Observed | CONCERT

Two figures
    fig_q3_gt_vs_pred_slides.png   continuous DSS-marker inflammation score
    fig_q3_gt_vs_pred_region.png   binary inflamed region (score > healthy cutoff)

Colour scaling differs between them, on purpose:
  * continuous -- values in HEALTHY-TISSUE z-units (see below), scaled PER ROW. The
    displayed slides sit at very different absolute levels, so one global scale lets the
    hottest row flatten every other one. Each row has its own range and its own colorbar;
    compare Observed against CONCERT within a row, never across rows.
  * region -- a cutoff per modality, each set so that it flags the SAME fraction of
    healthy (Day-0) tissue as the other. See CUT_OBS/CUT_PRED below.

Neither figure uses the shared readout calibration (paper_style.calibrate), and both say
why. In short: that calibration is a between-day LEVEL correction, while these panels are
per-cell displays whose readability depends on within-panel DISPERSION. The region map is
additionally invariant to it -- it is monotone, so it cannot move a cell across a
rate-matched cutoff.

Input:  percell_dump_sizefactor.npz (fixedt all-mice heldout_eval) + calibration.json
"""
import json  # noqa: F401  (kept: mouse_id_map traceability block)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.legend_handler as mpl_handler
from matplotlib.lines import Line2D
import paper_style as ps
ps.use()

DUMP = ("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
        "dss_timekernel_markersign_timefield_fixedt/heldout_eval/percell_dump_sizefactor.npz")
HERE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study"

# one mouse per sampled day. Chosen for STRUCTURAL COMPLETENESS of the largest slide
# (fraction of the tissue bounding box occupied): 0.72 / 0.64 / 0.82 / 0.55. The
# previously featured 092421_D3_m4 (0.36) and 082421_D21_m1 (0.40) are thin partial
# rings and were dropped.
import os
MICE = [m for m in os.environ.get("DSS_MICE", "").split(",") if m] or \
       ["082421_D0_m6", "092421_D3_m1", "100221_D9_m5", "082421_D21_m1"]
ALL_SLIDES = os.environ.get("DSS_ALL_SLIDES", "") == "1"
PREFIX = os.environ.get("DSS_OUT_PREFIX", "fig_q3_gt_vs_pred")
TIMES = [0, 3, 9, 21]

DAY_FS, SUB_FS, ROW_FS, CBAR_FS = 25, 20, 18, 17
DOT = 0.45          # cells are dense enough that anything larger merges into a blob

d = np.load(DUMP, allow_pickle=True)
mouse_all, sid_all, coords = d["mouse"], d["sample_id"], d["coords"]

# ── rows: largest slide of each featured mouse ─────────────────────────────────────
rows = []
for M in MICE:
    m0 = mouse_all == M
    if not m0.any():
        raise SystemExit(f"{M} not in dump")
    s, c = np.unique(sid_all[m0], return_counts=True)
    # supp mode shows EVERY slide of the mouse; the main figure shows only its largest
    slides = [int(x) for x in s[np.argsort(-c)]] if ALL_SLIDES else [int(s[np.argmax(c)])]
    for slide in slides:
        m = m0 & (sid_all == slide)
        rows.append({"mouse": M, "slide": slide, "day": float(np.unique(d["src_day"][m])[0]),
                     "xy": coords[m], "n": int(m.sum()), "obs_raw": d["obs_score"][m],
                     "pred_raw": {t: d[f"pred_t{t}"][m] for t in TIMES}})

healthy = np.isclose(d["src_day"], 0.0, atol=1e-3)

# ── continuous panels: both modalities in HEALTHY-TISSUE z-units ────────────────────
# NOT the shared readout calibration. That calibration is a BETWEEN-DAY amplitude
# (a=0.205, fitted on population day-means); applied per cell it multiplies WITHIN-PANEL
# spread by the same a and flattens the predicted maps. Measured: observed/calibrated
# per-cell SD ran 3.8-6.0x across the four rows, while the RAW predicted spread already
# matches observed to 0.82-1.28x. Level and dispersion are different quantities, and this
# figure displays per-cell values, so it is scaled by the per-cell dispersion of healthy
# tissue rather than by the day-mean amplitude -- the same anchor-to-measured-healthy-
# tissue idea as calibration.json, applied at the level actually being shown.
#
# Each modality is referenced to ITS OWN healthy distribution, so on both sides 0 means
# "at the healthy mean" and 1 means "one healthy-tissue SD above it" -- which is what makes
# the two halves of a row directly comparable in colour. Rows are still scaled separately
# (see build): the displayed slides differ several-fold in level, and one slide of an
# animal can differ 10x from another slide of the SAME animal, so a shared ramp would
# flatten most of the figure.
def _z(v, ref):
    return (v - ref.mean()) / ref.std()


OBS_HEALTHY, PRED_HEALTHY = d["obs_score"][healthy], d["pred_t0"]
for r in rows:
    r["obs"] = _z(r["obs_raw"], OBS_HEALTHY)
    r["pred"] = {t: _z(v, PRED_HEALTHY) for t, v in r["pred_raw"].items()}

# ── region panels: one cutoff per modality, matched on EFFECTIVE healthy rate ────────
# A shared numeric cutoff is not comparable here. Observed is a sparse per-cell count
# score with a long tail; CONCERT is a smooth conditional mean with a far shorter one, so
# the same number flags wildly different fractions of the same healthy tissue. Worse, a
# shared NOMINAL percentile does not fix it either: ties at zero mean the observed 95th
# percentile actually flags only a few percent of healthy cells, while the predicted 95th
# percentile flags exactly 5.0%. So the observed cutoff is taken at the 95th percentile of
# healthy tissue, the rate it ACHIEVES there is measured, and CONCERT's cutoff is then set
# to whatever value reproduces that same rate on the Day-0 counterfactual decode. The two
# printed baseline rates below must agree -- that is the check that makes the word
# "inflamed" mean the same thing on both sides of the figure.
CUT_OBS = float(np.percentile(d["obs_score"][healthy], 95))
RATE_OBS = float((d["obs_score"][healthy] > CUT_OBS).mean())
CUT_PRED = float(np.percentile(d["pred_t0"], 100.0 * (1.0 - RATE_OBS)))
RATE_PRED = float((d["pred_t0"] > CUT_PRED).mean())
REGION_CMAP = ListedColormap(["#d9d9d9", ps.ACCENT])
print(f"[q3gt] region cutoffs   Observed {CUT_OBS:.4f} (flags {100*RATE_OBS:.2f}% of healthy)"
      f"   CONCERT {CUT_PRED:.4f} (flags {100*RATE_PRED:.2f}% of healthy)")
print("[q3gt] healthy-z panel means (0 = healthy mean, 1 = one healthy SD)")
for r in rows:
    print(f"  {ps.mouse_label(r['mouse'], r['day'], r['slide']):34s} {r['n']:,} cells   "
          f"observed {r['obs'].mean():+.2f} (sd {r['obs'].std():.2f})   CONCERT "
          + "/".join("%+.2f" % r["pred"][t].mean() for t in TIMES)
          + f"  (sd {r['pred'][int(r['day'])].std():.2f})")


class _CircleKey(mpl_handler.HandlerBase):
    """Legend key drawing the actual 'never measured' glyph (an empty dashed circle)."""

    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        rr = 0.42 * min(width, height)
        return [mpatches.Circle((-xd + width / 2, -yd + height / 2), rr, fill=False,
                                ec=ps.MUTED, lw=1.6, ls=(0, (3, 2)), transform=trans)]


def build(kind, out_name):
    """kind: 'continuous' (per-row scale, magma) or 'region' (global cutoff, binary)."""
    NR, GAP = len(rows), 0.30
    widths, colmap = [], []
    for i, t in enumerate(TIMES):
        if i:
            widths.append(GAP); colmap.append(None)
        widths += [1.0, 1.0]; colmap += [(t, "obs"), (t, "pred")]
    NC = len(widths)

    right = 0.88 if kind == "continuous" else 0.97
    fig = plt.figure(figsize=(1.95 * 8 + 2.6, 2.05 * NR + 1.6))
    gs = gridspec.GridSpec(NR, NC, figure=fig, width_ratios=widths, wspace=0.02,
                           hspace=0.02, left=0.135, right=right, top=0.845, bottom=0.055)
    # anchor every label to the GRIDSPEC CELL: set_aspect("equal") shrinks each scatter
    # axes inside its cell by a different amount per row, so axes-anchored text drifts
    cell = {(ri, ci): gs[ri, ci].get_position(fig) for ri in range(NR) for ci in range(NC)}

    for ri, r in enumerate(rows):
        if kind == "continuous":
            # PER ROW. The displayed slide of each mouse sits at a very different absolute
            # level (whole-animal healthy-z +4.0 to +8.6, and individual slides of ONE
            # animal span -0.0 to +16.3), so a single scale lets the hottest row flatten
            # every other one. Each row therefore gets its own range and its own colorbar:
            # compare Observed against CONCERT WITHIN a row, never across rows.
            # Robust cap -- a handful of extreme cells otherwise flattens the whole row.
            pool = np.concatenate([r["obs"]] + [r["pred"][t] for t in TIMES])
            vmin, vmax = np.percentile(pool, 1), np.percentile(pool, 97)
        sc = None
        for ci, spec in enumerate(colmap):
            if spec is None:
                continue
            t, what = spec
            ax = fig.add_subplot(gs[ri, ci])
            has_gt = what == "obs" and np.isclose(t, r["day"], atol=1e-3)
            xy = r["xy"]

            if what == "pred" or has_gt:
                if kind == "continuous":
                    v = r["obs"] if has_gt else r["pred"][t]
                    sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=DOT, cmap=ps.FIELD_CMAP,
                                    vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
                else:
                    inflamed = (r["obs_raw"] > CUT_OBS if has_gt
                                else r["pred_raw"][t] > CUT_PRED)
                    ax.scatter(xy[:, 0], xy[:, 1], c=inflamed.astype(int), s=DOT,
                               cmap=REGION_CMAP, vmin=0, vmax=1, linewidths=0,
                               rasterized=True)
                    ax.text(0.02, 0.02, f"r = {100*inflamed.mean():.0f}%", fontsize=19,
                            transform=ax.transAxes, ha="left", va="bottom", color=ps.INK)
                ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
                ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            else:
                # never measured: an empty circle the size of the slide, no text
                cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
                span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
                ax.add_patch(mpatches.Circle((cx, cy), 0.40 * span, fill=False,
                                             ec=ps.GRID, lw=2.2, ls=(0, (5, 4))))
                ax.set_xlim(cx - span / 2, cx + span / 2)
                ax.set_ylim(cy - span / 2, cy + span / 2)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); ax.grid(False)

        if kind == "continuous":                  # one colorbar per row (its own range)
            p0, p1 = cell[(ri, 0)], cell[(ri, NC - 1)]
            cax = fig.add_axes([p1.x1 + 0.012, p0.y0 + 0.10 * (p0.y1 - p0.y0),
                                0.008, 0.80 * (p0.y1 - p0.y0)])
            cb = fig.colorbar(sc, cax=cax)
            cb.ax.tick_params(labelsize=CBAR_FS - 3)
            if ri == NR - 1:
                # ONE label for the whole column of per-row bars, centred over all of them.
                # Attaching it to a single bar overflows the figure: the rotated string is
                # taller than one row.
                fig.text(cax.get_position().x1 + 0.046, 0.5,
                         "DSS-marker score (healthy-tissue SD)", rotation=90, va="center",
                         ha="left", fontsize=CBAR_FS, color=ps.INK)

    # headers
    row0_top = cell[(0, 0)].y1
    for t in TIMES:
        ci_o, ci_p = colmap.index((t, "obs")), colmap.index((t, "pred"))
        for ci, lbl in ((ci_o, "GT"), (ci_p, "CONCERT")):
            p = cell[(0, ci)]
            fig.text(0.5 * (p.x0 + p.x1), row0_top + 0.014, lbl, ha="center",
                     va="bottom", fontsize=SUB_FS, color=ps.INK)
        fig.text(0.5 * (cell[(0, ci_o)].x0 + cell[(0, ci_p)].x1), row0_top + 0.056,
                 ps.day_label(t), ha="center", va="bottom", fontsize=DAY_FS,
                 fontweight="bold", color=ps.INK)

    # row labels, horizontal in the left margin (rotated text collides at this row height)
    for ri, r in enumerate(rows):
        p = cell[(ri, 0)]
        fig.text(p.x0 - 0.010, 0.5 * (p.y0 + p.y1),
                 ps.mouse_label(r["mouse"]),
                 ha="right", va="center", fontsize=ROW_FS, color=ps.INK, linespacing=1.5)

    handles = [mpatches.Circle((0, 0), 1)]
    labels = ["No GT"]
    hmap = {handles[0]: _CircleKey()}
    if kind == "region":
        # dots, not squares -- the key should look like the mark it stands for (each cell
        # is drawn as a dot), and it then matches the circular "never measured" glyph
        handles = [Line2D([0], [0], marker="o", ls="", mfc=ps.ACCENT, mec=ps.ACCENT, ms=15),
                   Line2D([0], [0], marker="o", ls="", mfc="#d9d9d9", mec="#d9d9d9",
                          ms=15)] + handles
        labels = ["Inflamed", "Not inflamed"] + labels
    fig.legend(handles=handles, labels=labels, handler_map=hmap, loc="lower center",
               ncol=len(handles), frameon=False, fontsize=19, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.2, handleheight=2.2, handletextpad=0.7, columnspacing=2.2)

    out = f"{HERE}/{out_name}"
    ps.save(fig, out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


build("continuous", f"{PREFIX}_slides.png")
build("region", f"{PREFIX}_region.png")

# traceability: figure label -> raw internal id
with open(f"{HERE}/mouse_id_map.csv", "w") as fh:
    fh.write("figure_label,mouse_id,day\n")
    for m in ps.MICE_ORDERED:
        day = int(m.split("_")[1][1:])
        fh.write(f"Mouse {ps.MOUSE_NUMBER[m]},{m},{day}\n")
print(f"wrote {HERE}/mouse_id_map.csv")
