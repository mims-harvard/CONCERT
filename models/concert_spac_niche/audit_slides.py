#!/usr/bin/env python3
"""Quick audit of SPAC-seq slides: perturbation overlap + NTC-vs-unassigned LFC consistency."""
import sys, h5py, numpy as np
from pathlib import Path
from collections import Counter

DATA = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq")
SLIDES = [s for s in sorted(DATA.glob("raw_output_1-*.h5ad"))]

UNASSIGNED = {"unassigned","Unassigned","No_guide","no_guide","unknown"}
CONTROL    = {"control","Control","NTC","ntc","non-targeting"}
EXCLUDED   = {"multiple","Multiple"}

def load_slide(path):
    with h5py.File(path, "r") as f:
        cats  = [x.decode() if isinstance(x,bytes) else x for x in f["obs/perturbation/categories"][:]]
        codes = f["obs/perturbation/codes"][:]
        perts = np.array([cats[c] for c in codes])
        n_cells = len(perts)
        keep = np.array([p not in EXCLUDED for p in perts])
        perts = perts[keep]
    una  = np.array([p in UNASSIGNED for p in perts])
    ntc  = np.array([p in CONTROL    for p in perts])
    ko   = np.array([p not in UNASSIGNED and p not in CONTROL for p in perts])
    gene_kos = set(perts[ko].tolist()) - EXCLUDED
    return dict(n_total=n_cells, n_pass=keep.sum(),
                n_una=una.sum(), n_ntc=ntc.sum(), n_ko=ko.sum(),
                n_perts=len(gene_kos), perts=gene_kos)

results = {}
for sl in SLIDES:
    name = sl.stem
    print(f"Loading {name}...", flush=True)
    results[name] = load_slide(sl)

print("\n=== Per-slide summary ===")
print(f"{'slide':<20} {'cells':>8} {'unassigned':>11} {'NTC':>7} {'KO':>8} {'n_perts':>9}")
for name, r in results.items():
    print(f"{name:<20} {r['n_pass']:>8,} {r['n_una']:>11,} {r['n_ntc']:>7,} {r['n_ko']:>8,} {r['n_perts']:>9,}")

names = list(results.keys())
all_perts = [results[n]["perts"] for n in names]
union = set.union(*all_perts)
inter = set.intersection(*all_perts)
print(f"\n=== Perturbation overlap ===")
print(f"Union:        {len(union):,}")
print(f"Intersection: {len(inter):,}  (in ALL slides)")
for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        shared = results[n1]["perts"] & results[n2]["perts"]
        print(f"  {n1} ∩ {n2}: {len(shared):,}")
