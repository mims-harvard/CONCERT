#!/usr/bin/env python
"""Per-mouse counterfactual PATHWAY trajectories over virtual time.

Two figures, house style (paper_style.py):
  fig_pathway_cascade.png     population-mean fold-over-t0 for every pathway on one axis --
                              the temporal cascade (inflammation peaks ~d9, repair/remodeling
                              rises later). This is the headline.
  fig_pathway_permouse.png    small-multiple grid, one panel per pathway: 15 per-mouse
                              CALIBRATED trajectories (a*pred+b, observed scale) coloured by
                              each mouse's SAMPLED day, with the observed-diagonal anchor (o)
                              at that day and the bold population mean.

Reads pathway_trajectories.csv + pathway_calibration.json (head node, no torch).
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import paper_style as ps
ps.use()

HERE = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study")
DAYS = [0, 3, 9, 21]

rows = list(csv.DictReader(open(HERE / "pathway_trajectories.csv")))
calib = json.load(open(HERE / "pathway_calibration.json"))
pathways = list(dict.fromkeys(r["pathway"] for r in rows))
mice = sorted(set(r["mouse"] for r in rows))

# inflammation vs repair/remodeling grouping (for cascade colouring + panel order)
INFLAM = ["Inflammatory_response", "TNFA_NFKB_signaling", "IL6_JAK_STAT3",
          "IFN_gamma_response", "IFN_alpha_response", "Neutrophil_granulocyte"]
REPAIR = ["Epithelial_repair_regeneration", "Angiogenesis", "MMP_ECM_remodeling", "EMT_fibrosis"]
order = [p for p in INFLAM + REPAIR if p in pathways] + \
        [p for p in pathways if p not in INFLAM + REPAIR]

def pretty(pw):
    return pw.replace("_", " ").replace("signaling", "sig.").replace("response", "resp.")

def per_mouse(pw):
    recs = {r["mouse"]: r for r in rows if r["pathway"] == pw}
    return recs

# a pathway's counterfactual is "per-mouse tracked" if the predicted score at each mouse's
# OWN day correlates with that mouse's observed score across the 15 mice (r >= threshold).
TRACK_THR = 0.35

# --- Figure A: cascade (population-mean fold-over-t0) ---
fig, ax = plt.subplots(figsize=(7.2, 5.0))
warm = colormaps["YlOrRd"](np.linspace(0.45, 0.95, len(INFLAM)))
cool = colormaps["GnBu"](np.linspace(0.55, 0.95, len(REPAIR)))
cmap_by = {}
for i, p in enumerate([p for p in INFLAM if p in pathways]): cmap_by[p] = warm[i]
for i, p in enumerate([p for p in REPAIR if p in pathways]): cmap_by[p] = cool[i]
for p in pathways: cmap_by.setdefault(p, "#888888")

for pw in order:
    recs = per_mouse(pw)
    ncell = np.array([float(recs[m]["n_cells"]) for m in mice])
    pop = np.array([np.average([float(recs[m][f"pred_t{d}"]) for m in mice], weights=ncell)
                    for d in DAYS])
    fold = pop / pop[0]                      # fold over virtual t0
    ls = "-" if pw in INFLAM else "--"
    ax.plot(DAYS, fold, ls, lw=2.4, color=cmap_by[pw], label=pretty(pw),
            marker="o", ms=5, zorder=3)
ax.axhline(1.0, color=ps.MUTED, lw=0.8, ls=":", zorder=1)
ax.set_xticks(DAYS); ax.set_xlabel("virtual day (counterfactual)")
ax.set_ylabel("pathway score, fold over day 0")
ax.margins(x=0.02)
ax.legend(ncol=1, fontsize=8.5, loc="upper right", handlelength=1.8,
          title="-- inflammation   -- repair/remodel", title_fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "fig_pathway_cascade.png"); plt.close(fig)
print("[fig] fig_pathway_cascade.png")

# --- Figure B: per-mouse small multiples (fold-over-t0 of RAW prediction) ---
# Fold-over-baseline avoids the per-pathway affine inversion (negative a flips the d9
# peak into a trough) and cancels each pathway's absolute-level differences, so the
# panels compare SHAPE/heterogeneity honestly. Each line = one mouse's counterfactual
# trajectory, coloured by the day it was actually sampled; bold black = population mean.
ncol = 5
nrow = int(np.ceil(len(order) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow), squeeze=False)
for k, pw in enumerate(order):
    ax = axes[k // ncol][k % ncol]
    recs = per_mouse(pw)
    for m in mice:
        r = recs[m]; day = float(r["day"])
        raw = np.array([float(r[f"pred_t{d}"]) for d in DAYS])
        ax.plot(DAYS, raw / raw[0], "-", lw=1.0, color=ps.day_color(day), alpha=0.7, zorder=2)
    ncell = np.array([float(recs[m]["n_cells"]) for m in mice])
    pop = np.array([np.average([float(recs[m][f"pred_t{d}"]) for m in mice], weights=ncell)
                    for d in DAYS])
    ax.plot(DAYS, pop / pop[0], "-", lw=2.6, color=ps.INK, zorder=5)
    ax.axhline(1.0, color=ps.MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_xticks(DAYS)
    c = calib[pw]; dr = c["diag_pearson_across_mice"]; tracked = dr >= TRACK_THR
    tag = f"tracks obs (r={dr:.2f})" if tracked else f"not per-mouse tracked (r={dr:.2f})"
    tcol = ps.INK if tracked else ps.MUTED
    ax.set_title(pretty(pw), fontsize=10, loc="left", fontweight="bold", pad=14)
    ax.text(0.0, 1.015, tag, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=7.5, color=tcol)
for k in range(len(order), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
fig.supxlabel("virtual day (counterfactual)", fontsize=11)
fig.supylabel("predicted pathway score, fold over day 0", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "fig_pathway_permouse.png"); plt.close(fig)
print("[fig] fig_pathway_permouse.png")

# --- Figure C: cross-mouse diagonal validation ---
# For each pathway, across the 15 mice, does the predicted score at each mouse's OWN
# (seen) day track that mouse's observed score? High r = the counterfactual is anchored
# to real per-mouse biology, not a population template.
val = [(pw, calib[pw]["diag_pearson_across_mice"], calib[pw]["diag_pearson_across_mice"] >= TRACK_THR)
       for pw in order]
val.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -1))
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ys = np.arange(len(val))
bars = ax.barh(ys, [v[1] for v in val],
               color=[ps.ACCENT if v[2] else "#b8b8b8" for v in val], height=0.7)
ax.set_yticks(ys); ax.set_yticklabels([pretty(v[0]) for v in val], fontsize=9)
ax.axvline(TRACK_THR, color=ps.MUTED, ls="--", lw=0.8)
ax.set_xlabel("diagonal Pearson r  (predicted vs observed, across 15 mice)")
ax.set_xlim(min(0, min(v[1] for v in val) - 0.05), 1.0)
for y, v in zip(ys, val):
    ax.text(v[1] + 0.01 if v[1] >= 0 else v[1] - 0.01, y, f"{v[1]:.2f}",
            va="center", ha="left" if v[1] >= 0 else "right", fontsize=8, color=ps.INK)
fig.tight_layout()
fig.savefig(HERE / "fig_pathway_validation.png"); plt.close(fig)
print("[fig] fig_pathway_validation.png")
