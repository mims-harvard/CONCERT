#!/usr/bin/env python
"""Acceptance gate table for the retrained 2dpara arms (and anything else under gate_*)."""
import sys
from pathlib import Path
import numpy as np, anndata as ad
B = Path("/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_rimtest")
NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
SETS = NS / "outputs/stroke_2dpara/pert_sets"
GENES = ["Gm42418", "Scgb1a1", "Lcn2", "Vim", "Gfap", "Timp1"]
KEY = ["Lcn2", "Vim", "Timp1"]
gn = np.loadtxt(NS / "outputs/stroke_2dpara/hvg_gene_names.txt", dtype=str)
gi = {g: int(np.where(gn == g)[0][0]) for g in GENES if (gn == g).any()}
print(f"{'arm/mode':16s} {'placement':17s} {'n':>5} " + " ".join(f"{g:>9}" for g in gi) + "   verdict")
for d in sorted(B.glob("gate_fix*")):
    for s in ["real_lesion_PT1", "transplant_PT1", "region_TH"]:
        on = sorted(d.glob(f"g_{s}_ON_stroke_*_perturbed_counts.h5ad"))
        off = sorted(d.glob(f"g_{s}_OFF_stroke_*_perturbed_counts.h5ad"))
        if not on or not off:
            continue
        A = np.asarray(ad.read_h5ad(on[-1]).X, np.float32)
        C = np.asarray(ad.read_h5ad(off[-1]).X, np.float32)
        S = np.where(A.sum(1, keepdims=True) == 0, 1.0, A.sum(1, keepdims=True))
        cpm = lambda X: np.log1p(X / S * 1e4)
        idx = np.loadtxt(SETS / f"{s}.txt", dtype=np.int64) - 1
        D = (cpm(A) - cpm(C))[idx]
        v = {g: float(D[:, gi[g]].mean()) for g in gi}
        ok = all(v[g] > 0 for g in KEY if g in v)
        print(f"{d.name.replace('gate_',''):16s} {s:17s} {len(idx):5d} " +
              " ".join(f"{v[g]:+9.3f}" for g in gi) + f"   {'PASS' if ok else 'FAIL'}")
print("\nreference — ORIGINAL broken 2dpara, f_pert only (paracrine off), real lesion:")
print("   Gm42418 +1.369  Scgb1a1 -0.122  Lcn2 -0.574  Vim -0.460  Gfap -1.248  Timp1 -0.209  FAIL")
