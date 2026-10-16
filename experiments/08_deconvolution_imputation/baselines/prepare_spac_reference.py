"""
Phase 1: prepare the SPAC-seq 2-2 scRNA-seq reference and the imputation inputs
1. Load the data, select HVGs, Leiden clustering
2. Build the scRNA-seq reference (sampled from non-masked cells)
3. Build sp_adata.h5ad (imputation input)
"""
import numpy as np
import anndata
import scanpy as sc
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# Configuration
SPAC_PATH = '/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/processed/SPAC/raw_output_2-2.h5ad'
MASK_PATH = '/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/imputation/patch_removal_2-2_top5_cellindex.npz'
OUT_DIR = '/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines/SpatialScope/SpatialScope/Ckpts_scRefs/SPAC_2-2'
IMPUTATION_DIR = '/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines/SpatialScope/test_output/spac_2-2'
N_HVGS = 2000
N_REF_SAMPLE = 30000
N_CELL_TYPES_TARGET = 15  # the Leiden resolution is tuned towards this

# Step 1: load the data
log("[1/6] Loading SPAC-seq 2-2 data...")
adata = anndata.read_h5ad(SPAC_PATH)
adata.var_names_make_unique()
log(f"    raw data: {adata.shape}")

# Step 2: preprocessing + HVG selection
log("[2/6] Preprocessing + HVG selection...")
# Keep raw counts
adata.layers['counts'] = adata.X.copy()

# Normalize + log1p
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVGS)
hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
log(f"    number of HVGs: {len(hvg_genes)}")

# Step 3: Leiden clustering (on all cells, before masking)
log("[3/6] PCA + Neighbors + Leiden clustering...")
adata_hvg = adata[:, hvg_genes].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50)
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=50)

# Try several resolutions and keep the one closest to N_CELL_TYPES_TARGET
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    sc.tl.leiden(adata_hvg, resolution=res, key_added=f'leiden_{res}')
    n_clusters = adata_hvg.obs[f'leiden_{res}'].nunique()
    log(f"    resolution={res}: {n_clusters} clusters")

# Pick the resolution closest to the target
best_res = None
best_diff = 999
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    n = adata_hvg.obs[f'leiden_{res}'].nunique()
    diff = abs(n - N_CELL_TYPES_TARGET)
    if diff < best_diff:
        best_diff = diff
        best_res = res

cell_type_col = f'leiden_{best_res}'
n_ct = adata_hvg.obs[cell_type_col].nunique()
log(f"    selected resolution={best_res}: {n_ct} cell types")

# Write the cell type labels back to the original adata
adata.obs['cell_type'] = adata_hvg.obs[cell_type_col].values

# Step 4: load the mask information
log("[4/6] Loading the mask information...")
mask_data = np.load(MASK_PATH, allow_pickle=True)
mask_cell_index = mask_data['cell_index']
mask_role = mask_data['role']
mask_group = mask_data['group']
mask_obs_name = mask_data['obs_name']

masked_indices = mask_cell_index[mask_role == 1]  # 5030 cells to impute
log(f"    experiment cells: {len(mask_cell_index)}")
log(f"    to impute (role=1): {len(masked_indices)}")
log(f"    kept (role=0): {(mask_role == 0).sum()}")
log(f"    dispersed (role=2): {(mask_role == 2).sum()}")

# Step 5: build the scRNA-seq reference
log("[5/6] Building the scRNA-seq reference...")
os.makedirs(OUT_DIR, exist_ok=True)

# Non-masked cells form the reference (the 5030 role=1 cells are excluded)
all_indices = np.arange(adata.shape[0])
non_masked_indices = np.setdiff1d(all_indices, masked_indices)
log(f"    non-masked cells: {len(non_masked_indices)}")

# Stratified sampling
adata_non_masked = adata[non_masked_indices].copy()
ct_counts = adata_non_masked.obs['cell_type'].value_counts()

# Proportional sampling, at least 100 cells per cell type
sample_indices = []
for ct in ct_counts.index:
    ct_idx = np.where(adata_non_masked.obs['cell_type'].values == ct)[0]
    n_sample = max(100, int(N_REF_SAMPLE * len(ct_idx) / len(non_masked_indices)))
    n_sample = min(n_sample, len(ct_idx))
    sampled = np.random.choice(ct_idx, size=n_sample, replace=False)
    sample_indices.extend(sampled.tolist())

sample_indices = np.array(sample_indices)
np.random.shuffle(sample_indices)
log(f"    sampled reference cells: {len(sample_indices)}")

# HVG subset, raw counts
ref_adata = adata_non_masked[sample_indices, hvg_genes].copy()
# Raw counts (Train_scRef.py applies log1p itself)
ref_adata.X = adata_non_masked[sample_indices].layers['counts'][:, adata.var['highly_variable'].values].copy()
try:
    ref_adata.X = ref_adata.X.toarray()
except:
    pass

# Marker column
ref_adata.var['Marker'] = True

# Cell type information
ref_adata.obs = ref_adata.obs[['cell_type']].copy()
ref_adata.obs.columns = ['cell_type']

ct_dist = ref_adata.obs['cell_type'].value_counts()
log(f"    Reference cell type distribution:")
for ct, count in ct_dist.items():
    log(f"      {ct}: {count}")

ref_path = os.path.join(OUT_DIR, 'Ref_SPAC_2-2.h5ad')
ref_adata.write(ref_path)
log(f"    saved to: {ref_path}")
log(f"    Reference shape: {ref_adata.shape}")
log(f"    X range: [{ref_adata.X.min():.1f}, {ref_adata.X.max():.1f}]")

# Step 6: Build sp_adata.h5ad (imputation input)
log("[6/6] Building sp_adata.h5ad...")
os.makedirs(IMPUTATION_DIR, exist_ok=True)

# The 10,690 experiment cells
exp_adata = adata[mask_cell_index, hvg_genes].copy()
# Raw counts
exp_adata.X = adata[mask_cell_index].layers['counts'][:, adata.var['highly_variable'].values].copy()
try:
    exp_adata.X = exp_adata.X.toarray()
except:
    pass

# Save the ground truth
gt_path = os.path.join(IMPUTATION_DIR, 'ground_truth.h5ad')
exp_adata_gt = exp_adata.copy()
exp_adata_gt.write(gt_path)
log(f"    Ground truth saved to: {gt_path}")

# Mask role=1 cells
# mask_role aligns with mask_cell_index
for i in range(len(mask_role)):
    if mask_role[i] == 1:
        exp_adata.X[i] = 0.0

n_zero = (exp_adata.X.sum(axis=1) == 0).sum()
log(f"    Masked (zero-expression) spots: {n_zero}")

# Set estimated_be = 1.0 (no batch effect)
exp_adata.var['estimated_be'] = 1.0

# Build the cell_locations DataFrame
# SpatialScope format: spot_index, cell_index, discrete_label_ct
cell_locations = pd.DataFrame()
cell_locations['spot_index'] = [f'spot_{i}' for i in range(exp_adata.shape[0])]
cell_locations['cell_index'] = [f'cell_{i}' for i in range(exp_adata.shape[0])]
cell_locations['discrete_label_ct'] = exp_adata.obs['cell_type'].values
# One cell per spot
cell_locations['cell_nums'] = 1

exp_adata.uns['cell_locations'] = cell_locations
exp_adata.obs_names = [f'spot_{i}' for i in range(exp_adata.shape[0])]

# Save the mask metadata
exp_adata.obs['role'] = mask_role
exp_adata.obs['patch_group'] = mask_group

sp_path = os.path.join(IMPUTATION_DIR, 'sp_adata.h5ad')
exp_adata.write(sp_path)
log(f"    sp_adata saved to: {sp_path}")
log(f"    sp_adata shape: {exp_adata.shape}")

# Save the mask index mapping for the evaluation
np.savez(os.path.join(IMPUTATION_DIR, 'mask_info.npz'),
         original_cell_index=mask_cell_index,
         role=mask_role,
         group=mask_group,
         obs_name=mask_obs_name)

log("\n" + "=" * 60)
log("Phase 1 done.")
log(f"  Reference: {ref_path} ({ref_adata.shape})")
log(f"  SP data:   {sp_path} ({exp_adata.shape})")
log(f"  Ground truth: {gt_path}")
log(f"  Cell types: {n_ct}")
log(f"  HVG genes: {len(hvg_genes)}")
log(f"  Masked cells: {len(masked_indices)}")
log("=" * 60)
