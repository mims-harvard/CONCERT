#!/usr/bin/env python
"""Decisive: WHICH cells change, vs which cells were edited.

size_0800.txt is also the CONTROL's __basal__ set (BIG = ls -S | head -1), so for size 800 the
two passes edit the SAME 800 cells and every other cell must be bit-identical. Any unmasked cell
with delta != 0 is therefore either (a) genuine model spillover or (b) an index-mapping error.
"""
import json
from pathlib import Path
import numpy as np, anndata as ad, h5py

OUT = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/stroke_2dpara")
man = json.loads((OUT / "pert_sets" / "manifest.json").read_text())
off, n_slide = man["offset"], man["n_slide"]
rows = slice(off, off + n_slide)
genes = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
gi = int(np.where(genes == "Scgb1a1")[0][0])
tgt = next(p for p in man["files"] if man["target_slide"] in Path(p).stem)
with h5py.File(tgt, "r") as f:
    xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)

raw = lambda s: np.asarray(ad.read_h5ad(sorted(OUT.glob(f"{s}_stroke_*_perturbed_counts.h5ad"))[-1]).X[rows], np.float32)
Xc = raw("demo_CONTROL")
big = np.loadtxt(OUT / "pert_sets" / "size_0800.txt", dtype=np.int64) - 1 - off
mbig = np.zeros(n_slide, bool); mbig[big] = True
print("manifest offset", off, "n_slide", n_slide, "xy", xy.shape)
print("BIG (control __basal__ set) n =", mbig.sum())

for s in (25, 100, 800):
    Xp = raw(f"demo_size_{s:04d}")
    idx = np.loadtxt(OUT / "pert_sets" / f"size_{s:04d}.txt", dtype=np.int64) - 1 - off
    m = np.zeros(n_slide, bool); m[idx] = True
    D = np.abs(Xp - Xc)                        # full gene matrix, RAW mu
    changed = D.max(1) > 0                     # any gene changed at all
    print(f"\n=== size {s}: edited(mask)={m.sum()}  changed(any gene)={changed.sum()}")
    print(f"    changed & mask       = {(changed & m).sum():5d}   (of {m.sum()} edited)")
    print(f"    changed & BIG & !mask= {(changed & mbig & ~m).sum():5d}")
    print(f"    changed & !BIG &!mask= {(changed & ~mbig & ~m).sum():5d}  <-- must be 0 if no spillover")
    print(f"    mask centroid {xy[m].mean(0)}   changed centroid {xy[changed].mean(0)}")
    if (changed & ~mbig & ~m).sum():
        e = changed & ~mbig & ~m
        r = np.linalg.norm(xy - xy[m].mean(0), axis=1)
        print(f"    outside-any-edit cells: n={e.sum()}  r range [{r[e].min():.2f},{r[e].max():.2f}] "
              f"vs mask R={r[m].max():.2f}")
        print(f"    their max|d mu| = {D[e].max():.4g}   mean|d Scgb1a1| = {np.abs(Xp-Xc)[e, gi].mean():.4g}")
