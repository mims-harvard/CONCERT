"""Convert the preprocessed perturb-FISH h5ad into the layout run_concert_spac.py reads.

The preprocessed file carries the perturbation assignment in obs and the MERFISH
centroids in obs/x and obs/y. The loader instead expects obsm/array_coords, a single
obs/perturbation column naming the targeted gene, and an obs/filter_pass QC flag, so
this script moves those over without touching the counts.

Coordinates are divided by a bin width so that one unit is one cell spacing.
That is the unit the downstream spatial descriptors assume: the array coordinates of the
SPAC-seq slides step by one per neighbouring spot, and the DBSCAN radius, density radius and
occupancy grid of 01_extract_kernel_params.py are all quoted in those units.
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in_h5ad", required=True)
ap.add_argument("--out_h5ad", required=True)
ap.add_argument("--pert_col", default="perturbation_gene",
                help="obs column naming the targeted gene per cell")
ap.add_argument("--bin_um", type=float, default=0.0,
                help="coordinate bin width; 0 = derive from the median nearest-neighbour distance")
ap.add_argument("--drop_labels", default="unmatched,none,nan",
                help="perturbation labels whose cells are marked filter_pass=0")
args = ap.parse_args()

adata = ad.read_h5ad(args.in_h5ad)
print("[in] %d cells x %d genes" % adata.shape)
print("[in] obs columns:", list(adata.obs.columns))

# ---------------------------------------------------------------- coordinates
xy = np.stack([adata.obs["x"].to_numpy(float), adata.obs["y"].to_numpy(float)], axis=1)
bin_um = args.bin_um
if bin_um <= 0:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    bin_um = float(np.median(d[:, 1]))
    print("[coords] median nearest-neighbour distance %.2f -> bin width %.3f"
          % (np.median(d[:, 1]), bin_um))
coords = (xy - xy.min(axis=0)) / bin_um
adata.obsm["array_coords"] = coords.astype(np.float32)
print("[coords] array_coords span: x %.1f, y %.1f bins"
      % (coords[:, 0].max(), coords[:, 1].max()))

# ---------------------------------------------------------------- perturbation
pert = adata.obs[args.pert_col].astype(str).to_numpy()
drop = {s.strip().lower() for s in args.drop_labels.split(",") if s.strip()}
keep = np.array([p.strip().lower() not in drop for p in pert])
adata.obs["perturbation"] = pd.Categorical(pert)
adata.obs["filter_pass"] = np.where(keep, "1", "0")
print("[pert] %d assigned cells kept, %d dropped" % (keep.sum(), (~keep).sum()))
counts = adata.obs.loc[keep, "perturbation"].value_counts()
print("[pert] %d labels among kept cells" % len(counts[counts > 0]))
print(counts[counts > 0].to_string())

# ---------------------------------------------------------------- gene ranking
# The panel is a few hundred targeted genes, so every gene is kept; the rank column
# only exists because the loader looks for one before falling back to all genes.
adata.var["hvg_rank"] = np.arange(1, adata.n_vars + 1, dtype=np.int32)

adata.write_h5ad(args.out_h5ad)
print("[out] wrote", args.out_h5ad)
