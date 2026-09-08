#!/usr/bin/env python
"""Counterfactual inflammation-pathway heatmaps over virtual time (formal Reactome sets).

Two value modes are drawn, sharing layout, ordering and styling:
  * fig_reactome_pathway_heatmap.png      -> log2 fold-over-Day0 of the pathway score
  * fig_reactome_pathway_heatmap_nes.png  -> ssGSEA NES (self-contained per sample; no baseline)

Layout: rows = the 15 mice (label = DayX_mouse), columns = timepoint blocks
    [ Day 0 : 13 pathways ] gap [ Day 3 ] gap [ Day 9 ] gap [ Day 21 ]
Pathway order is shared across blocks, sorted by the population peak day so the panel reads
as a temporal cascade. Row colour strip = each mouse's actually-sampled day.

Scores come from the per-mouse per-gene decode means in pathway_dump.npz (gmean_pred (M,G,4),
gmean_obs (M,G)); gene sets = reactome_inflammation_gene_sets.json (MSigDB Reactome v2025.1,
mouse-mapped). Pure CPU/numpy (ssGSEA implemented inline, no gseapy).
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches
import matplotlib.legend_handler as mpl_handler

import paper_style as ps
ps.use()

DUMP = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
            "dss_timekernel_markersign_timefield_fixedt/heldout_eval/pathway_dump.npz")
SETS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/"
            "CONCERT-spac-niche-temporal/reactome_inflammation_gene_sets.json")
HERE = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study")
DAYS = [0, 3, 9, 21]

# --- load ---
d = np.load(DUMP, allow_pickle=True)
genes = [str(g) for g in d["gene_names"]]
gidx = {g: i for i, g in enumerate(genes)}
G = len(genes)
gmean_pred = d["gmean_pred"]          # (M, G, 4)  log1p count
gmean_obs = d["gmean_obs"]            # (M, G)     observed diagonal
mice = [str(m) for m in d["mouse_order"]]
mday = np.array(d["mouse_day"], float)
ncell = np.array(d["mouse_ncell"], float)
M = len(mice)
sets = json.loads(SETS.read_text())
pw_names = list(sets.keys())
pw_cols = {p: [gidx[g] for g in gs if g in gidx] for p, gs in sets.items()}
pw_mask = {p: np.zeros(G, bool) for p in pw_names}
for p, cols in pw_cols.items():
    pw_mask[p][cols] = True

# Row label = "Mouse 9, Day 9" from the shared cohort numbering in paper_style. The old
# abbreviated form (D9_m3) was also ambiguous: two DIFFERENT animals collapsed onto the
# same string (062221_D9_m3 / 100221_D9_m3, and likewise D9_m5), so two rows read as one.
def short(m):
    return ps.mouse_label(m, day=mday[mice.index(m)])

# --- mean-of-set pathway score (for fold view + column ordering + validation) ---
pred = {p: gmean_pred[:, cols, :].mean(1) for p, cols in pw_cols.items()}       # (M,4)
obs = {p: gmean_obs[:, cols].mean(1) for p, cols in pw_cols.items()}            # (M,)
fold = {p: np.log2(pred[p] / pred[p][:, [0]] + 1e-9) for p in pw_names}          # (M,4)
pop = {p: np.average(pred[p], axis=0, weights=ncell) for p in pw_names}          # (4,)
order = sorted(pw_names, key=lambda p: (DAYS[int(np.argmax(pop[p]))], -pop[p].max()))

# --- standardized pathway-activity score (the principled non-fold readout) ---
# A GSEA NES is NOT appropriate here: this is a 942-gene TARGETED inflammation panel,
# so the panel itself is the (biased) enrichment background -- rank-based ES (absolute
# ssGSEA, delta ssGSEA, pre-ranked KS) do not recover the course (peaks scatter to
# Day 0/3/21, or tie-artefact at Day 0). The right readout for a targeted panel is the
# gene-set ACTIVITY = mean of the set genes' decoded expression, standardized (z-score)
# across all mouse x virtual-day samples so it needs no Day-0 baseline. Positive z =
# pathway above its own cross-course average; the Day-9 peak / Day-21 resolution survive.
# Standardization is PER ROW (within each mouse, across its 4 virtual days), not over all
# 60 mouse x day samples. Measured on this cohort the between-mouse spread of pathway
# activity (SD 0.021) is LARGER than the within-mouse across-day spread (SD 0.018), so a
# global z-score mostly reports which animal a row is rather than which day a column is --
# a high-baseline mouse then reads red at every day, Day 0 included. Row-wise z removes
# that offset and leaves each row showing that animal's own temporal profile; the Day-9
# peak then appears in essentially every row.
act = {p: gmean_pred[:, cols, :].mean(1) for p, cols in pw_cols.items()}         # (M,4)
zact = {p: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
        for p, a in act.items()}

# --- observed values, placed on the SAME scale so they can sit beside the prediction ---
# Each mouse is measured once, so an observed value exists ONLY at that mouse's own day
# (the diagonal). The counterfactual decode carries a level offset (~1.8x on these gene
# means), so predictions are first put on the observed scale with ONE multiplicative
# readout factor per pathway, fit by least squares on the 15 diagonal pairs. Because the
# factor is a pure scale it cancels out of the z-transform -- zact above is unchanged --
# but it is what lets the observed value be expressed in the prediction's z units.
DIAG = [DAYS.index(int(round(mday[k]))) for k in range(M)]
cal_a, zobs, fold_obs = {}, {}, {}
obs_pop_d0 = {p: np.average(obs[p][mday == 0], weights=ncell[mday == 0]) for p in pw_names}
for p in pw_names:
    pdg = np.array([pred[p][k, DIAG[k]] for k in range(M)])
    cal_a[p] = float((pdg * obs[p]).sum() / (pdg ** 2).sum())
    cp = cal_a[p] * act[p]                                    # calibrated pred, obs units
    # observed placed on that mouse's OWN row scale, matching the row-wise z above. Here
    # the calibration genuinely matters: unlike the prediction's z (where a pure scale
    # cancels), the observed value is compared against the calibrated row mean and SD.
    zobs[p] = (obs[p] - cp.mean(1)) / (cp.std(1) + 1e-9)       # (M,)
    # fold view: the prediction folds each mouse over ITS OWN Day-0 decode; observed has
    # no own Day 0, so its only available baseline is the observed Day-0 POPULATION mean.
    # The two folds are therefore not identical quantities -- say so in the caption.
    fold_obs[p] = np.log2(obs[p] / (obs_pop_d0[p] + 1e-9) + 1e-9)



# --- cross-mouse diagonal validation r (predicted at own day vs observed) ---
def diag_r(p):
    pd_ = np.array([pred[p][k, DAYS.index(int(round(mday[k])))] for k in range(M)])
    od_ = obs[p]
    return float(np.corrcoef(pd_, od_)[0, 1]) if pd_.std() > 1e-9 and od_.std() > 1e-9 else np.nan

row_ord = np.argsort(mday, kind="stable")


class _SplitKey(mpl_handler.HandlerBase):
    """Legend key that draws the actual cell glyph: a square split by its anti-diagonal
    (upper-left = prediction, lower-right = observed), or a whole square. A plain Patch
    handle would be rendered by matplotlib as a rectangle, hiding the very thing the
    key exists to explain."""

    def __init__(self, split):
        self.split = split
        super().__init__()

    def create_artists(self, legend, orig_handle, xd, yd, width, height, fontsize, trans):
        s = min(width, height)
        x0, y0 = -xd, -yd
        if not self.split:
            return [mpatches.Rectangle((x0, y0), s, s, fc="0.72", ec="0.3", lw=0.7,
                                       transform=trans)]
        ul = mpatches.Polygon([(x0, y0 + s), (x0 + s, y0 + s), (x0, y0)], closed=True,
                              fc="0.45", ec="0.3", lw=0.7, transform=trans)
        lr = mpatches.Polygon([(x0 + s, y0), (x0 + s, y0 + s), (x0, y0)], closed=True,
                              fc="0.88", ec="0.3", lw=0.7, transform=trans)
        return [ul, lr]

# --- shared heatmap drawer ---
def draw(valmap, cbar_label, out_name, obsmap=None):
    """Heatmap of the counterfactual `valmap`; where an OBSERVED value exists, the cell is
    split by its anti-diagonal: upper-left triangle = CONCERT, lower-right = observed.

    Terminal sampling means each mouse is measured once, so observed values exist only in
    the day-block matching that mouse's own sampled day. Every other cell stays a whole
    square = prediction only, and the key says so.
    """
    nP = len(order); GAP = 1
    W = 4 * nP + 3 * GAP
    Mtx = np.full((M, W), np.nan, np.float32)
    col_pw = []
    c = 0; block_starts = []
    for ti in range(4):
        block_starts.append(c)
        for p in order:
            Mtx[:, c] = valmap[p][:, ti][row_ord]; col_pw.append(p); c += 1
        if ti < 3:
            for _ in range(GAP):
                col_pw.append(None); c += 1

    pool = Mtx[~np.isnan(Mtx)].ravel()
    vmax = float(np.percentile(np.abs(pool), 97))
    n_sat = int((np.abs(pool) > vmax).sum())
    msg = (f"  [{out_name}] colour capped at +/-{vmax:.2f} (from predictions); "
           f"{n_sat}/{pool.size} predicted ({100*n_sat/pool.size:.1f}%) saturate")
    if obsmap:
        ov = np.concatenate([obsmap[p] for p in order])
        msg += f"; {int((np.abs(ov) > vmax).sum())}/{ov.size} observed saturate"
    print(msg)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("white")

    # smaller cells (narrower), larger fonts
    fig, ax = plt.subplots(figsize=(0.17 * W + 3.4, 0.34 * M + 2.6))
    ax.imshow(Mtx, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # observed half-cells: lower-right triangle over the mouse's OWN measured day-block
    if obsmap:
        n_gt = 0
        for y, k in enumerate(row_ord):
            bs = block_starts[DIAG[k]]
            for j, p in enumerate(order):
                x = bs + j
                ax.add_patch(mpatches.Polygon(
                    [(x - 0.5, y + 0.5), (x + 0.5, y + 0.5), (x + 0.5, y - 0.5)],
                    closed=True, facecolor=cmap(norm(obsmap[p][k])), edgecolor="none",
                    zorder=2))
                ax.plot([x - 0.5, x + 0.5], [y + 0.5, y - 0.5], color="white", lw=0.6,
                        zorder=3, solid_capstyle="butt")
                n_gt += 1
        print(f"  [{out_name}] {n_gt} split cells carry an observed half "
              f"({n_gt / (M * 4 * nP) * 100:.0f}% of the panel)")

    # thin white gridlines delineate every cell (kills the "one name / two cells" read)
    for x in range(W + 1):
        ax.axvline(x - 0.5, color="white", lw=0.35, zorder=4)
    for y in range(M + 1):
        ax.axhline(y - 0.5, color="white", lw=0.35, zorder=4)

    # rows: short labels + sampled-day colour strip
    ax.set_yticks(range(M))
    ax.set_yticklabels([short(mice[k]) for k in row_ord], fontsize=13.5)
    for y, k in enumerate(row_ord):
        ax.add_patch(mpatches.Rectangle((-1.4, y - 0.5), 0.8, 1.0, clip_on=False,
                                        color=ps.day_color(mday[k]), lw=0))
    ax.set_xlim(-1.5, W - 0.5)

    # pathway labels under every block + bold Day header above each block
    for bi, bs in enumerate(block_starts):
        for j, p in enumerate(order):
            ax.text(bs + j, M - 0.35, p, rotation=90, va="top", ha="center",
                    fontsize=11, color=ps.INK)
        ax.text(bs + (nP - 1) / 2, -1.1, ps.day_label(DAYS[bi]),
                ha="center", va="bottom", fontsize=21, fontweight="bold")
    ax.set_xticks([]); ax.tick_params(axis="both", length=0)

    # thick separators between the four day-blocks
    for bs in block_starts[1:]:
        ax.axvline(bs - 1.0, color="white", lw=4)

    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.020, pad=0.012)
    cbar.set_label(cbar_label, fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    # day legend, pushed well BELOW the (long, vertical) pathway names
    handles = [mpatches.Patch(color=ps.day_color(dd), label=ps.day_label(dd)) for dd in DAYS]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.66),
                    ncol=4, fontsize=14, frameon=False, handlelength=1.3,
                    columnspacing=1.6, title="Sampled day", title_fontsize=14)
    if obsmap:
        ax.add_artist(leg)
        split, whole = mpatches.Patch(), mpatches.Patch()
        ax.legend(handles=[split, whole],
                  labels=["Upper-left CONCERT  /  lower-right Observed",
                          "CONCERT only (never measured)"],
                  handler_map={split: _SplitKey(True), whole: _SplitKey(False)},
                  loc="upper center", bbox_to_anchor=(0.5, -0.86), ncol=1, fontsize=13,
                  frameon=False, handlelength=1.6, handletextpad=0.8, labelspacing=0.6)
    fig.subplots_adjust(left=0.15, right=0.93, top=0.9, bottom=0.44)
    fig.savefig(HERE / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_name}")

print("[reactome] per-pathway readout factor a (pred -> observed scale, diagonal fit):")
for p in order:
    print(f"   {p:26s} a={cal_a[p]:.3f}   diag_r={diag_r(p):+.2f}")

draw(fold, "Log2 fold over Day 0", "fig_reactome_pathway_heatmap.png", obsmap=fold_obs)
# CAVEAT on the observed half of this view (kept, but read the fold view alongside it).
# Measured on this cohort, the
# observed whole-mouse Reactome set-mean does not move with time at all (Day-9/Day-0 ratio
# 0.68-1.12 across all 13 sets, 11 of them <= 1.0) because a broad set averaged over ALL
# cells dilutes an inflammation signal that lives in a minority of cells -- whereas the
# counterfactual decode applies the programme to every cell, so the prediction does move.
# On top of that, placing a single observed value on a row-wise z scale divides by that
# mouse's predicted across-day SD and amplified calibration residuals to z up to +9,
# which can render healthy Day-0 animals as apparently "inflamed". The fold view states the
# same thing more safely, against a like-for-like observed Day-0 baseline.
draw(zact, "Pathway activity", "fig_reactome_pathway_heatmap_activity.png", obsmap=zobs)

# --- curve plots of the activity-z values ---
import matplotlib.colors as mcolors
VDAYS = np.arange(4, dtype=float)            # evenly-spaced categorical x (Day 0/3/9/21)
dpos = {d: i for i, d in enumerate(DAYS)}
pop_z = {p: np.average(zact[p], axis=0, weights=ncell) for p in order}

# soft Tableau-20 palette (nicer than turbo; muted, distinguishable)
SOFT = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948",
        "#ff9da7", "#9c755f", "#8cd17d", "#d37295", "#a0cbe8", "#b6992d"]

def lighten(c, f=0.55):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

# population cascade: one activity-z line per pathway, wider, legend at BOTTOM
fig, ax = plt.subplots(figsize=(11.0, 6.8))
for i, p in enumerate(order):
    ax.plot(VDAYS, pop_z[p], "-o", lw=2.6, ms=6.5, color=SOFT[i % len(SOFT)], label=p, zorder=3)
ax.axhline(0, color=ps.MUTED, lw=0.8, ls="--", zorder=1)
ax.set_xlim(-0.15, 3.15)
ax.set_xticks(VDAYS); ax.set_xticklabels([f"Day {d}" for d in DAYS], fontsize=16)
ax.set_ylabel("pathway activity (z-score)", fontsize=17)
ax.tick_params(axis="y", labelsize=14)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=5, fontsize=12.5,
          frameon=False, handlelength=1.5, columnspacing=1.5, labelspacing=0.4)
fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.28)
fig.savefig(HERE / "fig_reactome_pathway_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("[fig] fig_reactome_pathway_curves.png")

# per-mouse small multiples: 3x3 of 9 representative pathways (peak-ordered), 15 soft
# per-mouse activity-z trajectories coloured by sampled day + bold population mean.
SEL9 = ["IL-17 signaling", "IFN-alpha/beta", "IFN-gamma", "IL-6 family", "TNF / NF-kB",
        "Neutrophil degranulation", "TLR cascades", "IL-1 family", "Chemokine signaling"]
sel = [p for p in order if p in SEL9]
fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0), sharex=True)
axes = axes.ravel()
for j, p in enumerate(sel):
    ax = axes[j]
    for k in range(M):
        ax.plot(VDAYS, zact[p][k], "-", lw=1.4, alpha=0.85,
                color=lighten(ps.day_color(mday[k]), 0.45), zorder=2)
    ax.plot(VDAYS, pop_z[p], "-", lw=3.0, color=ps.INK, zorder=5)
    ax.axhline(0, color=ps.GRID, lw=0.7, zorder=1)
    ax.set_title(p, fontsize=14, loc="left")
    ax.set_xticks(VDAYS); ax.set_xticklabels([f"D{d}" for d in DAYS], fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
handles = [plt.Line2D([], [], color=lighten(ps.day_color(dd), 0.45), lw=2.6,
                      label=ps.day_label(dd)) for dd in DAYS]
handles += [plt.Line2D([], [], color=ps.INK, lw=3.0, label="Population mean")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=13, frameon=False,
           bbox_to_anchor=(0.5, -0.015), title="Sampled day", title_fontsize=13)
fig.supylabel("pathway activity (z-score)", fontsize=15)
fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.09, wspace=0.20, hspace=0.28)
fig.savefig(HERE / "fig_reactome_pathway_curves_permouse.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("[fig] fig_reactome_pathway_curves_permouse.png")

# --- record table (adds population activity-z per timepoint) ---
with open(HERE / "reactome_pathway_trajectories.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    pop_z = {p: np.average(zact[p], axis=0, weights=ncell) for p in order}
    w.writerow(["pathway", "n_genes", "peak_day", "diag_r"] +
               [f"pop_pred_t{dd}" for dd in DAYS] + [f"pop_log2fc_t{dd}" for dd in DAYS] +
               [f"pop_activityz_t{dd}" for dd in DAYS])
    for p in order:
        pv = pop[p]; fc = np.log2(pv / pv[0] + 1e-9)
        w.writerow([p, len(pw_cols[p]), DAYS[int(np.argmax(pv))], f"{diag_r(p):.3f}"] +
                   [f"{v:.4f}" for v in pv] + [f"{v:.3f}" for v in fc] +
                   [f"{v:.3f}" for v in pop_z[p]])
print("[csv] reactome_pathway_trajectories.csv")
for p in order:
    pz = np.average(zact[p], axis=0, weights=ncell)
    print(f"  {p:22s} peak Day{DAYS[int(np.argmax(pop[p]))]:<2d}  diag_r={diag_r(p):.2f}  "
          f"activity-z(0/3/9/21)={'/'.join(f'{v:+.2f}' for v in pz)}")
