#!/usr/bin/env python3
"""
SpaIM baseline for contiguous-patch spatial imputation on SPAC-seq data.

SpaIM is a style transfer model designed for gene imputation (predicting
spatial expression of genes not in the ST panel using scRNA-seq reference).
For our cell imputation task, we adapt it as follows:

  1. Train SpaIM on shared genes (80% train / 20% val split)
     - scRNA reference aggregated by cell type -> (n_ct, n_genes)
     - Spatial non-masked cells -> (n_spots, n_genes)
     - Model learns: sc_profile(n_ct) -> st_profile(n_spots) per gene
  2. Run inference on ALL genes -> predicted spatial expression for non-masked spots
  3. Impute masked cells via spatial K-NN from the honest neighbor pool:
     - 5,660 non-masked experiment cells (SpaIM predictions)
     - ~324k retained context cells (actual expression): full slide minus the
       masked patches and unused perturbations (see build_context_pool.py).
     Masked cells are never in the pool -> no ground-truth leakage, and the
     original spatial state is preserved.
  4. Evaluate against ground truth (PCC, SCC, RMSE)

Usage:
  conda activate scprint
  python run_spaim_imputation.py --device cuda:0 --epochs 300
"""
import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Add SpaIM model imports
SPAIM_SRC = Path(__file__).parent / "SpaIM" / "src"
sys.path.insert(0, str(SPAIM_SRC))
from model import Imputation, ImputeModule


# Paths
SPAC_DIR = Path(
    "/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines"
    "/SpatialScope/test_output/spac_2-2"
)
REF_PATH = Path(
    "/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines"
    "/SpatialScope/SpatialScope/Ckpts_scRefs/SPAC_2-2/Ref_SPAC_2-2.h5ad"
)
OUT_DIR = Path(
    "/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines"
    "/spaim_output/spac_2-2"
)
SPAC_RAW = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data"
    "/processed/SPAC/raw_output_2-2.h5ad"
)
CONTEXT_POOL_PATH = SPAC_DIR / "context_pool.h5ad"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpaIM imputation baseline")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (default: 300, SpaIM paper default)")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Batch size (default: 500, SpaIM default)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--train_fraction", type=float, default=0.8,
                        help="Fraction of genes for training (default: 0.8)")
    parser.add_argument("--k_neighbors", type=int, default=20,
                        help="K for spatial KNN imputation of masked cells (default: 20)")
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="Random seed (default: 1234, SpaIM default)")
    parser.add_argument("--style_dim", type=int, default=1,
                        help="Style dimension (default: 1)")
    parser.add_argument("--model_layers", type=str, default="256,512",
                        help="Hidden layer dimensions (default: 256,512)")
    parser.add_argument("--aug_times", type=int, default=4,
                        help="Data augmentation multiplier (default: 4)")
    parser.add_argument("--aug_zero_frac", type=float, default=0.5,
                        help="Fraction of zeros in augmentation (default: 0.5)")
    parser.add_argument("--n_context", type=int, default=0,
                        help="Context cells for K-NN pool (0=ALL ~324k retained "
                             "cells, >0=subsample for quick tests)")
    return parser.parse_args()


# Data Loading
def load_data():
    """Load Phase 1 outputs: sp_adata, ground_truth, mask_info, reference."""
    log("Loading Phase 1 data...")

    sp_adata = anndata.read_h5ad(SPAC_DIR / "sp_adata.h5ad")
    log(f"  sp_adata: {sp_adata.shape}")

    ground_truth = anndata.read_h5ad(SPAC_DIR / "ground_truth.h5ad")
    log(f"  ground_truth: {ground_truth.shape}")

    mask_info = np.load(SPAC_DIR / "mask_info.npz", allow_pickle=True)
    role = mask_info["role"]
    group = mask_info["group"]
    log(f"  role=0 (kept): {(role == 0).sum()}, "
        f"role=1 (masked): {(role == 1).sum()}, "
        f"role=2 (dispersed): {(role == 2).sum()}")

    ref_adata = anndata.read_h5ad(REF_PATH)
    log(f"  reference: {ref_adata.shape}")
    log(f"  cell types: {ref_adata.obs['cell_type'].nunique()}")

    return sp_adata, ground_truth, role, group, ref_adata


def load_context_pool(hvg_names: list, n_context: int = 0, seed: int = 1234):
    """Load the honest spatial-context pool (built by build_context_pool.py).

    Retained context = full slide cells with perturbation in
    {unassigned, 5 tested KOs}, MINUS all 10,690 experiment patch cells
    (masked cells -> no leakage; non-masked experiment cells -> added
    separately with SpaIM predictions). These ~324k observed cells preserve
    the ORIGINAL spatial state for K-NN imputation.

    n_context=0 keeps ALL retained cells (default); n_context>0 subsamples.

    Returns:
        ctx_expr: (n, n_genes) raw count expression aligned to hvg_names
        ctx_coords: (n, 2) spatial coordinates
    """
    if not CONTEXT_POOL_PATH.exists():
        raise FileNotFoundError(
            f"Context pool not found: {CONTEXT_POOL_PATH}\n"
            f"Run `python build_context_pool.py` first."
        )
    log(f"Loading context pool: {CONTEXT_POOL_PATH.name}")
    ctx = anndata.read_h5ad(CONTEXT_POOL_PATH)

    if list(ctx.var_names) != list(hvg_names):
        ctx = ctx[:, list(hvg_names)].copy()

    if n_context > 0 and ctx.n_obs > n_context:
        rng = np.random.RandomState(seed)
        sel = rng.choice(ctx.n_obs, n_context, replace=False)
        ctx = ctx[sel].copy()

    ctx_expr = ctx.X.toarray() if issparse(ctx.X) else np.asarray(ctx.X)
    ctx_expr = ctx_expr.astype(np.float32)
    ctx_coords = np.asarray(ctx.obsm["spatial"], dtype=np.float32)

    log(f"  Context pool: {ctx_expr.shape} (actual raw counts)")
    log(f"  Expression range: [{ctx_expr.min():.0f}, {ctx_expr.max():.0f}]")
    return ctx_expr, ctx_coords


# SpaIM Data Preparation
def aggregate_by_cell_type(adata, cell_type_col="cell_type"):
    """Aggregate expression by cell type (mean per cell type per gene).

    Returns:
        aggregated: (n_cell_types, n_genes) numpy array
        cell_type_codes: integer codes for each cell
        n_cell_types: number of unique cell types
    """
    cell_types = adata.obs[cell_type_col].astype("category")
    codes = cell_types.cat.codes.values
    n_ct = len(cell_types.cat.categories)

    X = adata.X
    if issparse(X):
        X = X.toarray()

    aggregated = np.zeros((n_ct, X.shape[1]), dtype=np.float32)
    for i in range(n_ct):
        mask = codes == i
        if mask.sum() > 0:
            aggregated[i] = X[mask].mean(axis=0)

    return aggregated, codes, n_ct


def prepare_spaim_data(sp_adata, ref_adata, role, args):
    """Prepare scRNA reference (aggregated) and spatial data for SpaIM.

    Returns:
        sc_agg: (n_cell_types, n_genes) aggregated scRNA in log1p space
        st_data: (n_non_masked, n_genes) spatial data in log1p space
        train_genes_idx: indices of training genes
        val_genes_idx: indices of validation genes
        all_gene_names: list of gene names
        non_masked_idx: indices of non-masked cells in sp_adata
        n_cell_types: number of cell types
    """
    log("Preparing SpaIM data...")

    # Spatial: non-masked cells, convert to log1p
    non_masked_idx = np.where(role != 1)[0]
    st_X = sp_adata.X
    if issparse(st_X):
        st_X = st_X.toarray()
    st_data = np.log1p(st_X[non_masked_idx]).astype(np.float32)
    log(f"  Spatial (non-masked, log1p): {st_data.shape}")

    # Reference: log1p then aggregate by cell type
    ref_X = ref_adata.X
    if issparse(ref_X):
        ref_X = ref_X.toarray()
    ref_log1p = np.log1p(ref_X).astype(np.float32)

    # Create a temporary AnnData for aggregation
    ref_temp = anndata.AnnData(
        X=ref_log1p,
        obs=ref_adata.obs[["cell_type"]].copy(),
    )
    sc_agg, _, n_ct = aggregate_by_cell_type(ref_temp, "cell_type")
    log(f"  scRNA aggregated (log1p): ({n_ct}, {sc_agg.shape[1]})")

    # Gene split: random train/val
    n_genes = sp_adata.shape[1]
    all_gene_names = list(sp_adata.var_names)
    gene_indices = np.arange(n_genes)
    np.random.shuffle(gene_indices)

    n_train = int(n_genes * args.train_fraction)
    train_genes_idx = np.sort(gene_indices[:n_train])
    val_genes_idx = np.sort(gene_indices[n_train:])
    log(f"  Gene split: {len(train_genes_idx)} train, {len(val_genes_idx)} val")

    return sc_agg, st_data, train_genes_idx, val_genes_idx, all_gene_names, non_masked_idx, n_ct


# SpaIM Training Dataset
class SpaIMGeneDataset(Dataset):
    """Dataset where each sample is a gene.

    For each gene:
      - seq_x: scRNA cell-type profile (n_cell_types,) - aggregated
      - spa_x: spatial spot profile (n_spots,) - individual spots
    """

    def __init__(self, sc_agg, st_data, gene_indices, style_dim=1):
        """
        Args:
            sc_agg: (n_cell_types, n_genes) aggregated scRNA
            st_data: (n_spots, n_genes) spatial spot data
            gene_indices: which genes to include
            style_dim: style dimension
        """
        # Transpose: each row = one gene
        self.sc_genes = sc_agg[:, gene_indices].T.astype(np.float32)  # (n_train_genes, n_ct)
        self.st_genes = st_data[:, gene_indices].T.astype(np.float32)  # (n_train_genes, n_spots)
        self.style_dim = style_dim

    def __len__(self):
        return len(self.sc_genes)

    def __getitem__(self, idx):
        seq_x = torch.FloatTensor(self.sc_genes[idx])
        spa_x = torch.FloatTensor(self.st_genes[idx])
        st_style = torch.ones(self.style_dim)
        sc_style = torch.zeros(self.style_dim)
        return seq_x, spa_x, sc_style, st_style


class SpaIMInferenceDataset(Dataset):
    """Dataset for inference on all genes."""

    def __init__(self, sc_agg, gene_indices, style_dim=1):
        self.sc_genes = sc_agg[:, gene_indices].T.astype(np.float32)
        self.style_dim = style_dim

    def __len__(self):
        return len(self.sc_genes)

    def __getitem__(self, idx):
        seq_x = torch.FloatTensor(self.sc_genes[idx])
        st_style = torch.ones(self.style_dim)
        return seq_x, st_style


# Data Augmentation (from SpaIM)
def data_augmentation(data1, data2, data3, times=4, zero_fraction=0.5):
    """SpaIM's data augmentation: replicate with random zero masking."""
    all_data = [data1]
    for _ in range(times - 1):
        device = data1.device
        mask = torch.rand(data1.shape, device=device) < zero_fraction
        augmented = torch.where(mask, torch.zeros_like(data1), data1)
        all_data.append(augmented)

    return (
        torch.cat(all_data, dim=0),
        torch.cat([data2] * times, dim=0),
        torch.cat([data3] * times, dim=0),
    )


# Training
def train_spaim(sc_agg, st_data, train_genes_idx, n_cell_types, args):
    """Train SpaIM model.

    Returns trained model.
    """
    log("Training SpaIM model...")

    # Create dataset and dataloader
    train_dataset = SpaIMGeneDataset(
        sc_agg, st_data, train_genes_idx, args.style_dim,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    # Build model options (mimicking SpaIM's opt)
    model_layers = [int(x.strip()) for x in args.model_layers.split(",")]

    class Opt:
        pass

    opt = Opt()
    opt.sc_dim = n_cell_types
    opt.st_dim = st_data.shape[0]  # number of non-masked spots
    opt.style_dim = args.style_dim
    opt.model_layers = model_layers
    opt.lr = args.lr
    opt.beta1 = 0.9
    opt.beta2 = 0.999
    opt.parallel = 0
    opt.gpu = int(args.device.split(":")[-1]) if "cuda" in args.device else 0
    opt.device = torch.device(args.device)

    log(f"  Model config: sc_dim={opt.sc_dim}, st_dim={opt.st_dim}, "
        f"layers={model_layers}, style_dim={opt.style_dim}")

    model = ImputeModule(opt)
    model = model.to(args.device)

    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for seq, spa, sc_style, st_style in train_loader:
            # Data augmentation (SpaIM default: 4x with 50% zero masking)
            seq, spa, st_style = data_augmentation(
                seq, spa, st_style,
                times=args.aug_times,
                zero_fraction=args.aug_zero_frac,
            )

            # sc_style is not used in forward() but set_input needs it
            # Replicate to match augmented batch size
            sc_style_aug = torch.zeros(seq.shape[0], args.style_dim)

            inputs = {
                "scx": seq,
                "stx": spa,
                "sc_style": sc_style_aug,
                "st_style": st_style,
            }
            model.set_input(inputs, istrain=1)
            model.update_parameters()
            loss_stat = model.get_current_loss()
            epoch_loss += loss_stat["loss"]
            n_batches += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            log(f"  Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f} "
                f"({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    log(f"  Training complete in {elapsed:.1f}s")

    return model, opt


# Inference
def run_inference(model, sc_agg, all_gene_indices, args):
    """Run SpaIM inference on specified genes.

    Returns:
        predictions: (n_spots, n_genes) predicted spatial expression in log1p space
    """
    log("Running SpaIM inference...")

    infer_dataset = SpaIMInferenceDataset(
        sc_agg, all_gene_indices, args.style_dim,
    )
    infer_loader = DataLoader(
        infer_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    all_preds = []
    with torch.no_grad():
        for seq, st_style in infer_loader:
            inputs = {
                "scx": seq,
                "st_style": st_style,
                "sc_cls": None,
            }
            model.set_input(inputs, istrain=0)
            out = model.inference()
            pred = out["st_fake"].detach().cpu().numpy()
            all_preds.append(pred)

    # Shape: (n_genes, n_spots) -> transpose to (n_spots, n_genes)
    predictions = np.concatenate(all_preds, axis=0)
    predictions = predictions.T
    predictions[predictions < 0] = 0  # clip negatives (SpaIM convention)

    log(f"  Predictions shape: {predictions.shape}")
    log(f"  Value range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    return predictions


# K-NN Spatial Interpolation (with background cells)
def impute_masked_cells_knn(
    predictions, sp_adata, role, k=20,
    bg_expr=None, bg_coords=None,
):
    """Impute masked cells via spatial K-NN from an expanded neighbor pool.

    The neighbor pool includes:
      - Non-masked experiment cells with SpaIM-predicted expression
      - Unassigned background cells with actual expression (if provided)

    All interpolation is done in log1p space, then converted to counts.

    Args:
        predictions: (n_non_masked, n_genes) SpaIM predicted expression (log1p space)
        sp_adata: full experiment data with spatial coordinates (10,690 cells)
        role: role array
        k: number of nearest neighbors
        bg_expr: (n_bg, n_genes) background cell expression (RAW COUNTS)
        bg_coords: (n_bg, 2) background cell spatial coordinates

    Returns:
        imputed: (n_total, n_genes) imputed expression in COUNT space
    """
    log(f"Imputing masked cells via spatial K-NN (K={k})...")

    exp_coords = sp_adata.obsm["spatial"]
    masked_idx = np.where(role == 1)[0]
    non_masked_idx = np.where(role != 1)[0]

    n_total = len(role)
    n_genes = predictions.shape[1]

    # Build expanded neighbor pool: experiment non-masked + background
    pool_coords = exp_coords[non_masked_idx].copy()
    pool_expr = predictions.copy()  # already in log1p space

    if bg_expr is not None and bg_coords is not None:
        bg_expr_log = np.log1p(bg_expr)  # convert raw counts to log1p
        pool_coords = np.vstack([pool_coords, bg_coords])
        pool_expr = np.vstack([pool_expr, bg_expr_log])
        log(f"  Neighbor pool: {len(non_masked_idx)} experiment + "
            f"{len(bg_coords)} background = {len(pool_coords)} total")
    else:
        log(f"  Neighbor pool: {len(non_masked_idx)} experiment cells (no background)")

    # Build KD-tree from expanded pool
    tree = cKDTree(pool_coords)

    # Initialize output with predictions for non-masked experiment cells (log1p)
    imputed_log = np.zeros((n_total, n_genes), dtype=np.float32)
    imputed_log[non_masked_idx] = predictions

    # For each masked cell, find K nearest neighbors from expanded pool
    masked_coords = exp_coords[masked_idx]
    distances, indices = tree.query(masked_coords, k=k)

    # Distance-weighted average in log1p space
    weights = 1.0 / np.maximum(distances, 1e-10)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_normalized = weights / weights_sum

    for i in range(len(masked_idx)):
        neighbor_expr = pool_expr[indices[i]]  # (K, n_genes)
        imputed_log[masked_idx[i]] = (
            weights_normalized[i, :, np.newaxis] * neighbor_expr
        ).sum(axis=0)

    log(f"  Imputed {len(masked_idx)} masked cells (log1p space)")

    # Diagnostic: what fraction of neighbors come from the context tissue?
    if bg_expr is not None:
        n_exp = len(non_masked_idx)
        is_bg = indices >= n_exp  # indices into pool; >= n_exp means context
        bg_frac = is_bg.mean()
        log(f"  Context-tissue neighbor fraction: {bg_frac:.1%}")

    # Convert from log1p space to count space
    imputed_counts = np.expm1(imputed_log)
    imputed_counts[imputed_counts < 0] = 0

    log(f"  Count space range: [{imputed_counts[masked_idx].min():.2f}, "
        f"{imputed_counts[masked_idx].max():.2f}]")

    return imputed_counts


# Evaluation
def evaluate_imputation(imputed, ground_truth, role, group):
    """Evaluate imputed expression against ground truth for masked cells.

    Metrics: PCC, SCC, RMSE (same as SpatialScope and Tangram baselines).
    """
    log("Evaluating imputation...")

    gt_X = ground_truth.X
    if issparse(gt_X):
        gt_X = gt_X.toarray()

    masked_idx = np.where(role == 1)[0]

    gt_masked = gt_X[masked_idx].astype(np.float64)
    imp_masked = imputed[masked_idx].astype(np.float64)
    imp_masked = np.clip(imp_masked, 0, None)

    results = []

    def compute_metrics(gt_vals, imp_vals, label):
        gt_flat = gt_vals.flatten()
        imp_flat = imp_vals.flatten()

        nonzero_mask = (gt_flat != 0) | (imp_flat != 0)
        if nonzero_mask.sum() == 0:
            return {"group": label, "n_cells": len(gt_vals),
                    "pcc": 0.0, "scc": 0.0, "rmse": 0.0}

        gt_nz = gt_flat[nonzero_mask]
        imp_nz = imp_flat[nonzero_mask]

        pcc, _ = pearsonr(gt_nz, imp_nz)
        scc, _ = spearmanr(gt_nz, imp_nz)
        rmse = np.sqrt(np.mean((gt_nz - imp_nz) ** 2))

        return {
            "group": label,
            "n_cells": len(gt_vals),
            "pcc": round(pcc, 4),
            "scc": round(scc, 4),
            "rmse": round(rmse, 4),
        }

    # Overall
    overall = compute_metrics(gt_masked, imp_masked, "OVERALL")
    results.append(overall)
    log(f"  OVERALL: PCC={overall['pcc']:.4f}, SCC={overall['scc']:.4f}, "
        f"RMSE={overall['rmse']:.4f} ({overall['n_cells']} cells)")

    # Per-group
    masked_groups = group[masked_idx]
    for grp in np.unique(masked_groups):
        grp_mask = masked_groups == grp
        grp_result = compute_metrics(
            gt_masked[grp_mask], imp_masked[grp_mask], str(grp),
        )
        results.append(grp_result)
        log(f"  {grp}: PCC={grp_result['pcc']:.4f}, "
            f"SCC={grp_result['scc']:.4f}, RMSE={grp_result['rmse']:.4f} "
            f"({grp_result['n_cells']} cells)")

    return pd.DataFrame(results)


# Main
def main():
    args = parse_args()
    log("=" * 60)
    log("SpaIM Imputation Baseline (full spatial context)")
    log(f"  device={args.device}, epochs={args.epochs}, "
        f"lr={args.lr}, K={args.k_neighbors}")
    log(f"  train_fraction={args.train_fraction}, "
        f"layers={args.model_layers}, style_dim={args.style_dim}")
    log(f"  n_context={args.n_context if args.n_context > 0 else 'ALL'}")
    log("=" * 60)

    seed_everything(args.random_seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Phase 1: Load data
    sp_adata, ground_truth, role, group, ref_adata = load_data()

    # Phase 2: Prepare SpaIM data
    (
        sc_agg, st_data, train_genes_idx, val_genes_idx,
        all_gene_names, non_masked_idx, n_cell_types,
    ) = prepare_spaim_data(sp_adata, ref_adata, role, args)

    # Phase 3: Train SpaIM
    model, opt = train_spaim(
        sc_agg, st_data, train_genes_idx, n_cell_types, args,
    )

    # Save trained model
    model_path = OUT_DIR / "spaim_model.pth"
    model.save(str(model_path))
    log(f"Model saved to {model_path}")

    # Phase 4: Inference on ALL genes
    all_genes_idx = np.arange(len(all_gene_names))
    predictions = run_inference(model, sc_agg, all_genes_idx, args)

    # Phase 4.5: Load honest spatial-context pool (~324k observed cells)
    bg_expr, bg_coords = load_context_pool(
        all_gene_names, n_context=args.n_context, seed=args.random_seed,
    )

    # Phase 5: K-NN imputation over experiment + full context pool
    imputed_counts = impute_masked_cells_knn(
        predictions, sp_adata, role, k=args.k_neighbors,
        bg_expr=bg_expr, bg_coords=bg_coords,
    )

    # Phase 6: Evaluate
    results_df = evaluate_imputation(imputed_counts, ground_truth, role, group)

    # Save results
    results_path = OUT_DIR / "imputation_results.csv"
    results_df.to_csv(results_path, index=False)
    log(f"Results saved to {results_path}")

    # Save imputed expression for masked cells
    masked_idx = np.where(role == 1)[0]
    imputed_adata = anndata.AnnData(
        X=imputed_counts[masked_idx].astype(np.float32),
        obs=sp_adata[masked_idx].obs.copy(),
        var=ground_truth.var.copy(),
    )
    imputed_path = OUT_DIR / "imputed_masked.h5ad"
    imputed_adata.write(imputed_path)
    log(f"Imputed expression saved to {imputed_path}")

    # Save gene split info
    gene_split = {
        "train_genes": np.array([all_gene_names[i] for i in train_genes_idx]),
        "val_genes": np.array([all_gene_names[i] for i in val_genes_idx]),
    }
    np.savez(OUT_DIR / "gene_split.npz", **gene_split)

    log("=" * 60)
    log("SpaIM imputation complete!")
    log("=" * 60)


if __name__ == "__main__":
    main()
