#!/usr/bin/env python
"""How far apart are the two cortical-depth patches, really?"""
import json, sys
from pathlib import Path
import h5py, numpy as np

SETS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/stroke_2dpara/pert_sets")
man = json.loads((SETS / "manifest_depth.json").read_text())
off, n_slide = man["offset"], man["n_slide"]
f = [p for p in man["files"] if man["target_slide"] in Path(p).stem][0]
with h5py.File(f, "r") as h:
    xy = np.asarray(h["obsm"]["array_coords"][:], np.float32)
    t = h["obs"]["tissue"]
    cats = np.array([c.decode() for c in t["categories"][:]]); tis = cats[t["codes"][:]]
A = np.loadtxt(SETS / "ctxdepth_superficial.txt", dtype=np.int64) - 1 - off
B = np.loadtxt(SETS / "ctxdepth_deep.txt", dtype=np.int64) - 1 - off
ctx = (tis == "CTX").sum()
inter = len(np.intersect1d(A, B))
ca, cb = xy[A].mean(0), xy[B].mean(0)
sep = float(np.linalg.norm(ca - cb))
ra = float(np.linalg.norm(xy[A] - ca, axis=1).mean())
rb = float(np.linalg.norm(xy[B] - cb, axis=1).mean())
from scipy.spatial import cKDTree
gap = float(cKDTree(xy[B]).query(xy[A])[0].min())
print(f"cortex spots on {man['target_slide']}: {ctx}")
print(f"patch sizes: superficial {len(A)}, deep {len(B)}  ({200*2/ctx:.0%} of cortex)")
print(f"overlapping spots: {inter}")
print(f"centroid separation {sep:.3f}; mean patch radius {ra:.3f} / {rb:.3f}")
print(f"ratio separation/(ra+rb) = {sep/(ra+rb):.2f}   (>1 = disjoint, <1 = interpenetrating)")
print(f"closest approach between the two patches: {gap:.3f} (spot spacing ~0.089)")
