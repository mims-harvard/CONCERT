#!/usr/bin/env python
"""The PASTE2-aligned section stack, per mouse.

Each row is one mouse. The first four columns are its sections drawn in the common frame that
PASTE2 produced; the last column overlays all four, which is where the quality of the alignment is
actually visible -- a good registration puts the four outlines on top of one another.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import sys
from pathlib import Path
import anndata as ad, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style as ps
ps.use()

SL = Path(f"{_rp.WORKDIR}/stroke3d/slides")
OUT = Path(f"{_rp.WORKDIR}/paper/manuscript/MAIN/FIG/supp")
LABEL_FS, TICK_FS = 20, 15
SEC_C = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]

rows = []
for stack in ("PT", "Sham"):
    secs = []
    for i in range(1, 5):
        p = SL / f"stroke_paste2_{stack}{i}.h5ad"
        A = ad.read_h5ad(p)
        secs.append(np.asarray(A.obsm["array_coords"], np.float32))
    rows.append((stack, secs))

# one frame per mouse so the overlay is honest: sections are drawn in the SAME axes limits
fig, ax = plt.subplots(2, 5, figsize=(4.0 * 5 + 1.0, 4.0 * 2 + 0.8), squeeze=False)
for ri, (stack, secs) in enumerate(rows):
    allxy = np.vstack(secs)
    lo, hi = allxy.min(0), allxy.max(0)
    pad = 0.03 * (hi - lo)
    for ci in range(4):
        xy = secs[ci]
        ax[ri, ci].scatter(xy[:, 0], xy[:, 1], s=2.0, lw=0, color=SEC_C[ci], rasterized=True)
        ax[ri, ci].set_xlabel(f"Section {ci + 1}", fontsize=LABEL_FS)
    for ci in range(4):
        xy = secs[ci]
        ax[ri, 4].scatter(xy[:, 0], xy[:, 1], s=1.6, lw=0, color=SEC_C[ci], alpha=0.55,
                          rasterized=True, label=f"Section {ci + 1}")
    ax[ri, 4].set_xlabel("All sections", fontsize=LABEL_FS)
    for ci in range(5):
        a = ax[ri, ci]
        a.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); a.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal"); a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(False)
    ax[ri, 0].set_ylabel({"PT": "Stroke (PT)", "Sham": "Sham"}[stack], fontsize=LABEL_FS)
ax[0, 4].legend(fontsize=TICK_FS, frameon=False, markerscale=6, loc="upper right")
fig.subplots_adjust(wspace=0.03, hspace=0.10, left=0.055, right=0.995, top=0.995, bottom=0.065)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"supp_figS26.{ext}", dpi=300)
print("wrote", OUT / "supp_figS26.pdf")
for stack, secs in rows:
    print(f"  {stack}: " + ", ".join(f"S{i+1} n={len(s)}" for i, s in enumerate(secs)))
