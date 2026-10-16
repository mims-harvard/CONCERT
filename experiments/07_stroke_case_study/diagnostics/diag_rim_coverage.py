#!/usr/bin/env python
"""Separate rim COVERAGE from rim AMPLITUDE.

A per-ring mean over ALL spots in the ring conflates two things: how many of that ring's spots
carry the PIA label at all (coverage), and how strongly a labelled spot responds (amplitude).
Unlabelled spots contribute exactly 0.000 (is_real_KO gate), so a patchy mask reads as a weak
response even when every labelled spot responds fully.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd
from scipy.spatial import cKDTree

NS = Path(f"{_rp.WORKDIR}")
SL = NS / "stroke3d/slides_cr"
SETS = NS / "outputs/thick_rim_sets"
DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")


def load_stack(stack):
    ps = sorted(SL.glob(f"stroke_paste2_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in ps], label="sf", keys=[p.stem for p in ps],
                  index_unique=None)
    return A, np.asarray(A.obsm["array_coords"], np.float32)


def log1p_cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


man = json.loads((SETS / "thick_manifest.json").read_text())
A_sh, xy = load_stack("Sham")
core = np.isin(np.arange(len(xy)), np.loadtxt(SETS / "thick_core_local.txt", dtype=np.int64))
sp = man["spacing"]
d = cKDTree(xy[core]).query(xy)[0] / sp
edges = [0.001, 1, 2, 3, 4, 6, 8, 12]

OUT = NS / "outputs/stroke_2d_cr"
side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o + c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
rd = lambda pat: log1p_cpm(np.asarray(ad.read_h5ad(next(OUT.glob(pat))).X[rows], np.float32))
L_off = rd("stroke_2d_cr_off_*_perturbed_counts.h5ad")
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
gi = int(np.where(gp == "Lcn2")[0][0])

print("Lcn2, 2D arm. 'cover' = fraction of the ring's spots carrying the PIA label.")
print("'all' = mean over every spot in the ring; 'labelled' = mean over labelled spots only.\n")
for tag in ("obs", "t3", "t5"):
    rim = np.isin(np.arange(len(xy)), np.loadtxt(SETS / f"thick_rim_{tag}_local.txt", dtype=np.int64)) & ~core
    L_rim = rd(f"stroke_2d_cr_onrim_{tag}_*_perturbed_counts.h5ad")
    delta = (L_rim - L_off)[:, gi]
    print(f"--- rim {tag}  ({rim.sum()} spots)")
    print(f"{'ring':>8s}{'n_ring':>8s}{'cover':>8s}{'all':>9s}{'labelled':>10s}")
    for i in range(len(edges) - 1):
        band = (d >= edges[i]) & (d < edges[i + 1])
        if not band.sum():
            continue
        lab = band & rim
        cov = lab.sum() / band.sum()
        allm = np.where(lab, delta, 0.0)[band].mean()
        labm = delta[lab].mean() if lab.sum() else float("nan")
        print(f"{edges[i]:>4.0f}-{edges[i+1]:<3.0f}{band.sum():>8d}{cov:>8.2f}{allm:>9.3f}{labm:>10.3f}")
    print()
