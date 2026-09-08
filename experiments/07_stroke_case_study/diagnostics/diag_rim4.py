#!/usr/bin/env python
"""Compare the rim WITH paracrine (published run) vs WITHOUT (rerun), same model, same sets.

Also checks whether the 8 PASTE2-aligned slides share one coordinate frame, which would let
emitters on the PT (real-lesion) slides spill onto the Sham3 slide through the same r_max=5 graph.
"""
import json
from pathlib import Path
import numpy as np, anndata as ad, h5py

WITH = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/stroke_2dpara")
WOUT = Path("/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_rimtest/stroke_2dpara_nopara")
man = json.loads((WITH / "pert_sets" / "manifest.json").read_text())
off, n_slide = man["offset"], man["n_slide"]
rows = slice(off, off + n_slide)
genes = np.loadtxt(WITH / "hvg_gene_names.txt", dtype=str)
gi = int(np.where(genes == "Scgb1a1")[0][0])

print("=== slide coordinate frames (cross-slide paracrine check, r_max=5.0) ===")
for p in man["files"]:
    with h5py.File(p, "r") as f:
        c = np.asarray(f["obsm"]["array_coords"][:], np.float32)
    print(f"  {Path(p).stem:26s} n={c.shape[0]:6d} x[{c[:,0].min():6.2f},{c[:,0].max():6.2f}] "
          f"y[{c[:,1].min():6.2f},{c[:,1].max():6.2f}]")

tgt = next(p for p in man["files"] if man["target_slide"] in Path(p).stem)
with h5py.File(tgt, "r") as f:
    xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)


def prof(root, tag):
    g = lambda s: sorted(root.glob(f"{s}_stroke_*_perturbed_counts.h5ad"))
    if not g("demo_CONTROL"):
        print(f"  [{tag}] no control found in {root}"); return
    Xc = np.asarray(ad.read_h5ad(g("demo_CONTROL")[-1]).X[rows], np.float32)
    CS = np.where(Xc.sum(1, keepdims=True) == 0, 1.0, Xc.sum(1, keepdims=True))
    cpm = lambda X: np.log1p(X / CS * 1e4)
    base = cpm(Xc)[:, gi]
    for s in (25, 100, 800):
        hits = g(f"demo_size_{s:04d}")
        if not hits: continue
        Xp = np.asarray(ad.read_h5ad(hits[-1]).X[rows], np.float32)
        e = cpm(Xp)[:, gi]
        idx = np.loadtxt(WITH / "pert_sets" / f"size_{s:04d}.txt", dtype=np.int64) - 1 - off
        m = np.zeros(n_slide, bool); m[idx] = True
        cen = xy[m].mean(0); r = np.linalg.norm(xy - cen, axis=1); R = r[m].max(); rr = r / R
        d = e - base
        seg = []
        for lo, hi, mk in [(0, .5, 1), (.5, 1.0, 1), (1.0, 1.3, 0), (1.3, 1.8, 0), (1.8, 3.0, 0), (3.0, 99, 0)]:
            sel = (m if mk else ~m) & (rr >= lo) & (rr < hi)
            if sel.sum() >= 3:
                seg.append(f"{'in' if mk else 'out'}[{lo:.1f},{hi:.1f})={d[sel].mean():+.3f}")
        nz = int((d[~m] != 0).sum())
        print(f"  [{tag}] size {s:4d}  R={R:.2f}  " + " ".join(seg) + f"   unmasked_nonzero={nz}/{int((~m).sum())}")


print("\n=== delta profile: WITH paracrine (published figure) ===")
prof(WITH, "with")
print("\n=== delta profile: WITHOUT paracrine (rerun) ===")
prof(WOUT, "wout")
