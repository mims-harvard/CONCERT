#!/usr/bin/env python
"""Cross-section neighbourhood structure in the PASTE2-aligned 3D frame.

For each spot, take its k nearest neighbours in the aligned three-dimensional coordinate system
and ask how many come from a different section. If the sections were not registered onto a common
frame, a spot's neighbours would stay within its own section at every k; a shared frame is only
usable if cross-section neighbours appear as k grows.
"""
from __future__ import annotations
import sys
from pathlib import Path
import anndata as ad, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style as ps
ps.use()

SL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/slides")
OUT = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/manuscript/MAIN/FIG/supp")
KS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
Z_GAIN = 0.25
LABEL_FS, TICK_FS = 19, 15

fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.6))
for pi, stack in enumerate(("PT", "Sham")):
    xy, sec = [], []
    for i in range(1, 5):
        A = ad.read_h5ad(SL / f"stroke_paste2_{stack}{i}.h5ad")
        c = np.asarray(A.obsm["array_coords"], np.float32)
        xy.append(c); sec.append(np.full(len(c), i - 1))
    xy = np.vstack(xy); sec = np.concatenate(sec)
    # same normalisation the model uses: xy scaled to the stack frame, z the ordinal index
    lo, hi = xy.min(0), xy.max(0)
    n = (xy - lo) / np.maximum(hi - lo, 1e-8)
    pts = np.column_stack([n, sec * Z_GAIN]).astype(np.float32)
    tree = cKDTree(pts)
    frac, same = [], []
    for k in KS:
        idx = tree.query(pts, k=k + 1)[1][:, 1:]
        cross = (sec[idx] != sec[:, None]).mean()
        frac.append(100 * cross); same.append(100 * (1 - cross))
    a = ax[pi]
    a.plot(KS, same, "-o", ms=5, lw=2.0, color="#4d4d4d", label="same section")
    a.plot(KS, frac, "-o", ms=5, lw=2.0, color=ps.ACCENT, label="other sections")
    a.set_xlabel("Neighbourhood size $k$", fontsize=LABEL_FS)
    if pi == 0:
        a.set_ylabel("Neighbours (%)", fontsize=LABEL_FS)
    a.set_title({"PT": "Stroke (PT)", "Sham": "Sham"}[stack], fontsize=LABEL_FS)
    a.set_ylim(0, 100); a.tick_params(labelsize=TICK_FS); a.grid(False)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.legend(fontsize=TICK_FS - 1, frameon=False)
    print(f"  {stack}: cross-section neighbours " +
          ", ".join(f"k={k}:{f:.1f}%" for k, f in zip(KS, frac)))
fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.17, wspace=0.16)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"supp_figS27.{ext}", dpi=300)
print("wrote", OUT / "supp_figS27.pdf")
