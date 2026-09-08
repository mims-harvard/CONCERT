#!/usr/bin/env python
"""Q3: Spatial time-lapse (imputation) of good predicted mice.

Framing = IMPUTATION, not holdout. Each mouse in the DSS course was measured at a
SINGLE timepoint. The all-mice fixedt CONCERT model imputes that same tissue's
inflammation field at the other (never-measured) timepoints. "Inflammatory region"
= per-cell DSS-marker gene score (mean log1p over the 20 DSS markers).

Two figures per featured mouse (observed panel REMOVED per reviewer request):
  fig_q3_spatial_timelapse_<mouse>.png   continuous marker score at t0/t3/t9/t21
  fig_q3_inflammatory_region_<mouse>.png  BINARY inflamed region (score > cutoff)

Styling (see the plotting-figures skill): NO title, large readable text, large
legend, rows packed tight with minimal whitespace.

Input:  percell_dump_sizefactor.npz from the fixedt all-mice heldout_eval.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import paper_style
paper_style.use()

DUMP = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/dss_timekernel_markersign_timefield_fixedt/heldout_eval/percell_dump_sizefactor.npz"
OUTDIR = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study"
FEATURED = ["092421_D3_m4", "092421_D3_m1", "062221_D9_m3"]

# text sizes -- large + readable
TITLE_FS, SLICE_FS, PCT_FS, LEG_FS, CBAR_FS = 18, 16, 16, 17, 16

d = np.load(DUMP, allow_pickle=True)
mouse_all = d["mouse"]
TIMES = [0, 3, 9, 21]
# column title = two lines: "Day X" over a short DSS-course phase word.
# The mouse's OWN observed day (star column) overrides the phase with "measured".
PHASE = {0: "healthy", 3: "onset", 9: "peak", 21: "resolution"}

# inflamed if imputed marker score > 95th pctile of the HEALTHY (t=0) imputed distribution
_t0_pool = np.concatenate([d["pred_t0"][mouse_all == M] for M in FEATURED if (mouse_all == M).any()])
CUTOFF = float(np.percentile(_t0_pool, 95))
print(f"[q3] inflamed cutoff = 95th pctile of healthy(t0) imputed score = {CUTOFF:.4f}")
_REGION_CMAP = ListedColormap(["#d9d9d9", paper_style.ACCENT])


def _slices_for(MOUSE):
    m0 = mouse_all == MOUSE
    sid = d["sample_id"]
    slices, counts = np.unique(sid[m0], return_counts=True)
    slices = slices[np.argsort(-counts)]
    src_day = float(np.unique(d["src_day"][m0])[0])
    return m0, sid, slices, src_day


def _grid(nrow, ncol):
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow),
                             squeeze=False)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.01,
                        wspace=0.01, hspace=0.01)   # top leaves room for 2-line titles
    return fig, axes


def _finish(ax, r, c, t, star, slices, s, m):
    if r == 0:
        sub = "measured" if star else PHASE[t]
        ax.set_title(f"Day {t}\n{sub}", fontsize=TITLE_FS,
                     color=paper_style.INK, pad=4)
    if c == 0:
        ax.text(-0.03, 0.5, f"slice {int(s)}\n{int(m.sum()):,} cells",
                transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=SLICE_FS)
    ax.set_aspect("equal"); ax.axis("off"); ax.invert_yaxis()


def continuous(MOUSE):
    m0, sid, slices, src_day = _slices_for(MOUSE)
    if not m0.any():
        print(f"  [skip] {MOUSE} not in dump"); return
    allvals = np.concatenate([d[f"pred_t{t}"][m0] for t in TIMES])
    vmin, vmax = np.percentile(allvals, 1), np.percentile(allvals, 99)
    fig, axes = _grid(len(slices), len(TIMES))
    sc = None
    for r, s in enumerate(slices):
        m = m0 & (sid == s); xy = d["coords"][m]
        for c, t in enumerate(TIMES):
            ax = axes[r][c]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=d[f"pred_t{t}"][m], s=2.0,
                            cmap=paper_style.FIELD_CMAP, vmin=vmin, vmax=vmax,
                            linewidths=0, rasterized=True)
            _finish(ax, r, c, t, np.isclose(t, src_day, atol=1e-3), slices, s, m)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.55, pad=0.008)
    cbar.set_label("DSS-marker inflammation score", fontsize=CBAR_FS)
    cbar.ax.tick_params(labelsize=CBAR_FS - 3)
    out = f"{OUTDIR}/fig_q3_spatial_timelapse_{MOUSE}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  wrote", out)


def region(MOUSE):
    m0, sid, slices, src_day = _slices_for(MOUSE)
    if not m0.any():
        return
    fig, axes = _grid(len(slices), len(TIMES))
    for r, s in enumerate(slices):
        m = m0 & (sid == s); xy = d["coords"][m]
        for c, t in enumerate(TIMES):
            ax = axes[r][c]
            inflamed = (d[f"pred_t{t}"][m] > CUTOFF).astype(int)
            ax.scatter(xy[:, 0], xy[:, 1], c=inflamed, s=2.0, cmap=_REGION_CMAP,
                       vmin=0, vmax=1, linewidths=0, rasterized=True)
            ax.text(0.02, 0.02, f"{100*inflamed.mean():.0f}% inflamed",
                    transform=ax.transAxes, fontsize=PCT_FS, color=paper_style.INK,
                    ha="left", va="bottom")
            _finish(ax, r, c, t, np.isclose(t, src_day, atol=1e-3), slices, s, m)
    handles = [plt.Line2D([0], [0], marker='o', ls='', mfc=paper_style.ACCENT, mec='none',
                          ms=15, label="inflamed"),
               plt.Line2D([0], [0], marker='o', ls='', mfc="#d9d9d9", mec='none',
                          ms=15, label="not inflamed")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=LEG_FS, bbox_to_anchor=(0.5, -0.055))
    out = f"{OUTDIR}/fig_q3_inflammatory_region_{MOUSE}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  wrote", out)


for M in FEATURED:
    print(M)
    continuous(M)
    region(M)
