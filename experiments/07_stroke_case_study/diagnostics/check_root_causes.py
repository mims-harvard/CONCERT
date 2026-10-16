#!/usr/bin/env python
"""Root-cause evidence for: inverted marker direction, ring around the lesion, negative inductions.

Four hypotheses, one run. No fixes here -- evidence only.

H1 LIBRARY/CPM. Perturbation raises total counts inside the lesion, so a gene that does not
   respond is pushed DOWN once each spot is renormalised to a fixed library.
H2 SIGN INVERSION. f_pert learned the effect with the wrong sign (there is a known open
   f_pert-inversion bug in this codebase). Decisive test: correlate PREDICTED per-gene LFC against
   OBSERVED per-gene LFC over all genes. A negative correlation means inversion.
H3 RING. Response peaks just OUTSIDE the lesion boundary. Test: predicted expression binned on
   signed distance to the boundary (negative = inside).
H4 COORDINATE MISMATCH. 2dpara trains/evals with tile_by_stack=False, so coords are NOT folded to
   a per-slide local frame. If raw_coords/coord_scale lands far outside the inducing grid Z, every
   RBF feature saturates and the field is meaningless. Test: compare the coordinate range actually
   fed to the RBF against the span of Z.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd, torch
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

NS = Path(f"{_rp.WORKDIR}")
DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")
COORD_SCALE, LOC_RANGE = 20.0, 20.0


def cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


print("=" * 78)
print("H4  COORDINATE RANGE vs INDUCING GRID")
print("=" * 78)
for arm, tiled in (("stroke_3d", True), ("stroke_2d", True), ("stroke_2dpara", False)):
    p = NS / f"outputs/{arm}/model.pt"
    if not p.exists():
        continue
    sd = torch.load(p, map_location="cpu"); sd = sd.get("state_dict", sd.get("model_state_dict", sd))
    Z = sd["fpert_field.Z"].float().numpy()
    sl = NS / ("stroke3d/slides" if arm != "stroke_2dpara" else "stroke3d/slides")
    A = ad.read_h5ad(sl / "stroke_paste2_Sham3.h5ad")
    xy = np.asarray(A.obsm["array_coords"], np.float32)
    fed = (xy - LOC_RANGE * np.floor(xy / LOC_RANGE)) / COORD_SCALE if tiled else xy / COORD_SCALE
    print(f"{arm:15s} tile={tiled}  coords fed to RBF: x[{fed[:,0].min():.2f},{fed[:,0].max():.2f}] "
          f"y[{fed[:,1].min():.2f},{fed[:,1].max():.2f}]")
    print(f"{'':15s} inducing Z span:   x[{Z[:,0].min():.2f},{Z[:,0].max():.2f}] "
          f"y[{Z[:,1].min():.2f},{Z[:,1].max():.2f}]")
    off = max(0.0, float(max(fed[:, 0].max() - Z[:, 0].max(), fed[:, 1].max() - Z[:, 1].max())))
    print(f"{'':15s} -> data extends {off:.2f} BEYOND the grid " +
          ("*** OUT OF RANGE ***" if off > 0.5 else "(ok)"))

print()
print("=" * 78)
print("H2  PREDICTED vs OBSERVED per-gene LFC  (negative r = sign inversion)")
print("=" * 78)
A_pt = ad.concat([ad.read_h5ad(p) for p in sorted((NS/"stroke3d/slides").glob("stroke_paste2_PT*.h5ad"))],
                 index_unique=None)
Lpt = cpm(np.asarray(A_pt.X.todense(), np.float32)); gs = np.asarray(A_pt.var_names)
with h5py.File(DEL/"benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]]); rg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
rp = pd.Series(rg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
o_ica, o_ctx = rp == "ICA", rp == "CTX"
obs_lfc_all = Lpt[o_ica].mean(0) - Lpt[o_ctx].mean(0)          # observed ICA vs healthy cortex

for arm in ("stroke_3d", "stroke_2d"):
    OUT = NS / f"outputs/{arm}"
    side = json.loads((OUT/"sham_cells.json").read_text())
    rows = np.concatenate([np.arange(o, o+c) for o, c, t in zip(
        np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
    on = cpm(np.asarray(ad.read_h5ad(next(OUT.glob(f"{arm}_on_*_perturbed_counts.h5ad"))).X[rows], np.float32))
    off_ = cpm(np.asarray(ad.read_h5ad(next(OUT.glob(f"{arm}_off_*_perturbed_counts.h5ad"))).X[rows], np.float32))
    gp = np.loadtxt(OUT/"hvg_gene_names.txt", dtype=str)
    pred_lfc = (on - off_).mean(0)
    common = [g for g in gp if g in set(gs.tolist())]
    oi = np.array([int(np.where(gs == g)[0][0]) for g in common])
    pi = np.array([int(np.where(gp == g)[0][0]) for g in common])
    r = pearsonr(obs_lfc_all[oi], pred_lfc[pi])[0]
    rs = spearmanr(obs_lfc_all[oi], pred_lfc[pi])[0]
    print(f"{arm:15s} n_genes={len(common)}  Pearson={r:+.3f}  Spearman={rs:+.3f}  "
          + ("*** INVERTED ***" if r < -0.1 else "(not inverted)" if r > 0.1 else "(no relationship)"))
    for g in ("Gm42418", "Scgb1a1", "Vim", "Timp1"):
        if g in set(common):
            print(f"{'':17s}{g:9s} observed_LFC={obs_lfc_all[int(np.where(gs==g)[0][0])]:+.3f}  "
                  f"predicted={pred_lfc[int(np.where(gp==g)[0][0])]:+.3f}")

print()
print("=" * 78)
print("H1/H3  LIBRARY SIZE and RING PROFILE  (2dpara, size_0800, Scgb1a1)")
print("=" * 78)
OUT = NS/"outputs/stroke_2dpara"
man = json.loads((OUT/"pert_sets"/"manifest.json").read_text())
o2, n2 = man["offset"], man["n_slide"]; rws = slice(o2, o2+n2)
g2 = np.loadtxt(OUT/"hvg_gene_names.txt", dtype=str)
raw = lambda s: np.asarray(ad.read_h5ad(sorted(OUT.glob(f"{s}_stroke_*_perturbed_counts.h5ad"))[-1]).X[rws], np.float32)
Xb, Xe = raw("demo_CONTROL"), raw("demo_size_0800")
idx = np.loadtxt(OUT/"pert_sets"/"size_0800.txt", dtype=np.int64)-1-o2
m = np.zeros(n2, bool); m[idx] = True
print(f"  library inside : {Xb[m].sum(1).mean():.0f} -> {Xe[m].sum(1).mean():.0f} "
      f"({100*(Xe[m].sum(1).mean()/Xb[m].sum(1).mean()-1):+.1f}%)")
print(f"  library outside: {Xb[~m].sum(1).mean():.0f} -> {Xe[~m].sum(1).mean():.0f} "
      f"({100*(Xe[~m].sum(1).mean()/Xb[~m].sum(1).mean()-1):+.1f}%)")
Lb, Le = cpm(Xb), cpm(Xe)
A3 = ad.read_h5ad(NS/"stroke3d/slides/stroke_paste2_Sham3.h5ad")
xy3 = np.asarray(A3.obsm["array_coords"], np.float32)
d_out = cKDTree(xy3[m]).query(xy3)[0]
d_in = cKDTree(xy3[~m]).query(xy3)[0]
signed = np.where(m, -d_in, d_out)                     # negative inside, positive outside
for gname in ("Scgb1a1", "Gm42418"):
    i2 = int(np.where(g2 == gname)[0][0])
    dl = (Le - Lb)[:, i2]; dr = (np.log1p(Xe) - np.log1p(Xb))[:, i2]
    print(f"  --- {gname} ---")
    edges = [-1e9, -3, -1, 0.001, 1, 2, 3, 5, 1e9]
    lab = ["deep in", "mid in", "edge in", "0-1 out", "1-2 out", "2-3 out", "3-5 out", ">5 out"]
    for k in range(len(edges)-1):
        sel = (signed >= edges[k]) & (signed < edges[k+1])
        if sel.sum():
            print(f"    {lab[k]:9s} n={sel.sum():5d}  dCPM={dl[sel].mean():+.3f}  dRAW={dr[sel].mean():+.3f}")
