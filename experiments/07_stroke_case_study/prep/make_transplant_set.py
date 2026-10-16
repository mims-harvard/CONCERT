#!/usr/bin/env python
"""Transplant the REAL PT1 lesion footprint onto Sham3: nearest Sham3 spot to each PT1 lesion spot.

Same shape, same coordinates, healthy brain. If f_pert is a genuine perturbation effect the
response here should look like the response at the real lesion; if it is position-locked or
sign-flipped by niche, it will not.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
from pathlib import Path
import numpy as np, h5py
from scipy.spatial import cKDTree
SL = Path(f"{_rp.WORKDIR}/stroke3d/slides")
OUT = Path(f"{_rp.WORKDIR}/outputs/stroke_2dpara/pert_sets")
ORDER = ["PT1", "PT2", "PT3", "PT4", "Sham1", "Sham2", "Sham3", "Sham4"]
def cat(f, c):
    o = f["obs"][c]
    k = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in o["categories"][:]])
    return k[o["codes"][:]]
off, pt_xy, s3 = 0, None, None
for s in ORDER:
    with h5py.File(SL / f"stroke_paste2_{s}.h5ad", "r") as f:
        n = f["obs"]["_index"].shape[0]
        xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)
        if s == "PT1":
            pt_xy = xy[cat(f, "perturbation") == "stroke"]
        if s == "Sham3":
            s3 = (xy, off, n)
    off += n
xy3, off3, n3 = s3
j = np.unique(cKDTree(xy3).query(pt_xy)[1])                 # nearest Sham3 spot per lesion spot
idx = np.sort(j) + off3 + 1                                  # 1-based global
np.savetxt(OUT / "transplant_PT1.txt", idx, fmt="%d")
print(f"PT1 lesion {len(pt_xy)} spots -> {len(idx)} unique Sham3 spots "
      f"(centroid PT1 {pt_xy.mean(0).round(2)} vs Sham3 {xy3[j].mean(0).round(2)})")
