#!/usr/bin/env python3
"""Final Perturb-map case-study panels, one file each (PNG + PDF), for assembly."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from pmap_pathways_powered import niche_tumour_content, RAW  # noqa: E402

# Reads the two result directories the evaluation step wrote; must match the values used
# there. OUT is where the assembled panels are collected.
_BASE = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/figures"
SYS_D = Path(os.environ.get("PMAP_SYSTEMATIC_OUT", f"{_BASE}/pmap_systematic"))
INT_D = Path(os.environ.get("PMAP_INTERPRET_OUT", f"{_BASE}/pmap_interpret"))
OUT = Path(os.environ.get(
    "PMAP_PANEL_OUT",
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study_pm"))

PERTS = ["Tgfbr2", "Ifngr2", "Jak2"]
COL = {"Tgfbr2": "#3366BB", "Ifngr2": "#CC6677", "Jak2": "#228833"}
INK, MUTED, GRID, ACCENT = "#1a1a1a", "#5a5a5a", "#d9d9d9", "#b2182b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.linewidth": 0.9,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def tidy(ax, ylabel=None, xlabel=None, axis="y"):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(axis=axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_gain_vs_variability(interp, syst):
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    for p in PERTS:
        x = interp["per_perturbation"][p]["niche_gain"]
        y = syst["per_perturbation"][p]["across_niche_sd_over_effect"]
        ax.scatter(x, y, s=260, color=COL[p], edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(0, 20),
                    ha="center", fontsize=14, color=COL[p])
    ax.set_xscale("log")
    ax.set_xlim(0.01, 5)
    ax.set_ylim(0, 0.68)
    tidy(ax, ylabel="Variability of the predicted response\nacross microenvironments",
         xlabel="Learned niche sensitivity", axis="both")
    fig.tight_layout(pad=0.3)
    save(fig, "panel_gain_vs_variability")


def panel_response_spread(d):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    data, labels = [], []
    for p in PERTS:
        L = d[f"lfc_{p}"]
        C = np.corrcoef(L)
        iu = np.triu_indices(len(L), k=1)
        data.append(C[iu])
        labels.append(p)
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, p in zip(parts["bodies"], PERTS):
        body.set_facecolor(COL[p])
        body.set_alpha(0.75)
        body.set_edgecolor("white")
    for i, vals in enumerate(data, start=1):
        ax.plot([i], [np.median(vals)], marker="_", ms=26, color="white", mew=2.5, zorder=4)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0, 1.02)
    tidy(ax, ylabel="Similarity of the predicted response\nbetween two microenvironments")
    fig.tight_layout(pad=0.3)
    save(fig, "panel_response_spread")


def panel_decomposition(interp):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x = np.arange(len(PERTS))
    spatial = [interp["per_perturbation"][p]["spatial_fraction_of_effect"] for p in PERTS]
    niche = [interp["per_perturbation"][p]["niche_fraction_of_effect"] for p in PERTS]
    w = 0.36
    ax.bar(x - w / 2, spatial, w, color="#4393C3", edgecolor="white", linewidth=0.8,
           label="Varies with position")
    ax.bar(x + w / 2, niche, w, color=ACCENT, edgecolor="white", linewidth=0.8,
           label="Set by the surrounding niche")
    ax.set_xticks(x)
    ax.set_xticklabels(PERTS, fontsize=14)
    ax.set_ylim(0, 0.85)
    tidy(ax, ylabel="Fraction of the predicted effect")
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    save(fig, "panel_effect_decomposition")


def panel_propagation(interp):
    probe = interp.get("propagation_vs_overlap")
    if not probe:
        print("  propagation panel skipped (no overlap stats)")
        return
    bins = probe["bins"]
    labels = list(bins.keys())
    vals = [bins[k]["shift_change"] for k in labels]
    ns = [bins[k]["n_cells"] for k in labels]
    pretty = [k.replace("_edited_neighbours", "").replace("-", "-") for k in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = np.arange(len(labels))
    cols = [MUTED] + [ACCENT] * (len(labels) - 1)
    ax.bar(xs, vals, 0.64, color=cols, edgecolor="white", linewidth=0.8)
    for x, v, n in zip(xs, vals, ns):
        ax.text(x, v + max(vals) * 0.03, f"n = {n}", ha="center", fontsize=11, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty, fontsize=13)
    tidy(ax, ylabel="Change in an unedited cell's\npredicted response",
         xlabel="Perturbed cells among its neighbours")
    fig.tight_layout(pad=0.3)
    save(fig, "panel_propagation_dose_response")


def panel_amplitude(d, tum):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for p in PERTS:
        L = d[f"lfc_{p}"]
        amp = np.abs(L).mean(axis=1)
        ax.scatter(tum, amp, s=55, color=COL[p], alpha=0.75, edgecolor="white",
                   linewidth=0.5, label=p)
    tidy(ax, ylabel="Size of the predicted\nknockout effect",
         xlabel="Tumour content of the surrounding niche", axis="both")
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, "panel_effect_size_vs_niche")


def panel_pathways(csv_path):
    if not csv_path.is_file():
        print("  pathway panel skipped (no table)")
        return
    t = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    sets = ["Epithelial Mesenchymal Transition", "Hypoxia", "Glycolysis"]
    sets = [s for s in sets if s in t.index]
    # Only three bars per row and a shared axis origin -- the extra width bought nothing
    # but longer bars, so the panel is sized to the labels rather than to the bar length.
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    y = np.arange(len(sets))
    h = 0.26
    for j, p in enumerate(PERTS):
        if (p, "NES") not in t.columns:
            continue
        nes = [t.loc[s, (p, "NES")] for s in sets]
        q = [t.loc[s, (p, "q")] for s in sets]
        pos = y + (j - 1) * h
        ax.barh(pos, nes, h, color=COL[p], alpha=0.75, edgecolor="white",
                linewidth=0.7, label=p)
        for yy, v, qq in zip(pos, nes, q):
            if qq < 0.05:
                # Marker sits just beyond the bar tip, not on top of it.
                ax.text(v - 0.07, yy, "*", va="center", ha="right",
                        fontsize=18, color=INK)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("Epithelial Mesenchymal Transition",
                                  "Epithelial-mesenchymal\ntransition") for s in sets],
                       fontsize=13)
    ax.set_xlabel("Dependence of the response\non the microenvironment", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.06)
    fig.legend(fontsize=12, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04), handlelength=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3)
    save(fig, "panel_pathways_niche_dependence")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    interp = json.loads((INT_D / "interpret_stats.json").read_text())
    syst = json.loads((SYS_D / "systematic_stats.json").read_text())
    d = np.load(SYS_D / "systematic_transplant.npz", allow_pickle=True)
    patches = [np.asarray(p, dtype=int).ravel() for p in d["patches"]]
    with h5py.File(RAW, "r") as f:
        pos = np.array(f["pos"]).T.astype(float)
        pert_raw = np.array(f["perturbation"]).astype(str)
    tum = niche_tumour_content(pos, patches, pert_raw)

    print("building final panels ->", OUT)
    panel_gain_vs_variability(interp, syst)
    panel_response_spread(d)
    # DISABLED: this built the two-bar version whose fractions had different
    # denominators and could not be stacked. The correct three-way split is
    # Outputs/fig5_panels/plot_fig5_decomposition.py. Leaving this call in would
    # silently overwrite the corrected panel with the broken one.
    panel_propagation(interp)
    panel_amplitude(d, tum)
    panel_pathways(SYS_D / "hallmark_niche_correlation.csv")
    print("done")


if __name__ == "__main__":
    main()
