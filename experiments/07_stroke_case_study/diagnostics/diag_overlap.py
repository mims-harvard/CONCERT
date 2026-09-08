#!/usr/bin/env python
"""Crux: are the 4 PT lesions at the SAME coordinates in the shared PASTE2 frame?

If they are, a coordinate-only basal field can predict "injured" without ever seeing the
perturbation label -- which is what leaves f_pert with nothing to learn.
"""
from pathlib import Path
import numpy as np, h5py
SL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/slides")
def cat(f, c):
    o = f["obs"][c]
    k = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in o["categories"][:]])
    return k[o["codes"][:]]
cen, pts = {}, {}
for s in ["PT1", "PT2", "PT3", "PT4"]:
    with h5py.File(SL / f"stroke_paste2_{s}.h5ad", "r") as f:
        p = cat(f, "perturbation"); xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)
    m = p == "stroke"; pts[s] = xy[m]; cen[s] = xy[m].mean(0)
    print(f"  {s}: n={m.sum():4d} centroid=({cen[s][0]:.2f},{cen[s][1]:.2f}) "
          f"x[{xy[m,0].min():.2f},{xy[m,0].max():.2f}] y[{xy[m,1].min():.2f},{xy[m,1].max():.2f}]")
ks = list(cen)
print("\n  pairwise centroid distance (slide span is ~4.5 x 6.6):")
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"    {ks[i]}-{ks[j]}: {np.linalg.norm(cen[ks[i]]-cen[ks[j]]):.2f}")
# how much of each lesion sits within 0.5 of another slide's lesion
from scipy.spatial import cKDTree
print("\n  fraction of each lesion within 0.5 of another PT lesion spot:")
for i in ks:
    o = np.vstack([pts[j] for j in ks if j != i])
    print(f"    {i}: {(cKDTree(o).query(pts[i])[0] <= 0.5).mean():.1%}")
