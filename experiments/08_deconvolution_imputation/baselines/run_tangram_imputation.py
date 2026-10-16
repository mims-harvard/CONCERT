#!/usr/bin/env python3
"""
Tangram baseline for contiguous-patch spatial imputation on SPAC-seq data.

Strategy:
  1. Load Phase 1 data (sp_adata, ground_truth, mask_info, reference)
  2. Include unassigned background cells from the full slide
     so K-NN neighbors reflect real tissue context
  3. Train Tangram: map reference cells -> non-masked spots (exp + background)
  4. Project gene expression
  5. Impute masked cells via spatial K-NN from projected expression
  6. Evaluate against ground truth (PCC, SCC, RMSE)

Usage:
  conda activate scprint
  python run_tangram_imputation.py --use_ref --device cuda:0
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# Add Tangram to path
TANGRAM_DIR = Path(__file__).parent / "Tangram"
sys.path.insert(0, str(TANGRAM_DIR))
import tangram as tg


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tangram imputation baseline")
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1000,
        help="Training epochs (default: 1000)",
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=20,
        help="K for spatial KNN imputation of masked cells (default: 20)",
    )
    parser.add_argument(
        "--n_marker_genes", type=int, default=100,
        help="Top marker genes per cell type for training (default: 100)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use_ref", action="store_true", default=False,
        help="Use 30k reference (Ref_SPAC_2-2.h5ad) instead of non-masked experiment cells",
    )
    parser.add_argument(
        "--n_context", type=int, default=0,
        help="Context cells for K-NN pool (0=ALL ~324k retained cells, "
             ">0=subsample for quick tests)",
    )
    return parser.parse_args()


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
    "/tangram_output/spac_2-2"
)
SPAC_RAW_PATH = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data"
    "/processed/SPAC/raw_output_2-2.h5ad"
)
CONTEXT_POOL_PATH = SPAC_DIR / "context_pool.h5ad"


def load_data() -> tuple:
    """Load Phase 1 outputs: sp_adata, ground_truth, mask_info."""
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

    return sp_adata, ground_truth, role, group


def load_context_pool(
    gene_names: list[str],
    n_context: int = 0,
    seed: int = 42,
) -> anndata.AnnData:
    """
    Load the honest spatial-context pool (built by build_context_pool.py).

    Retained context = full slide cells with perturbation in
    {unassigned, 5 tested KOs}, MINUS all 10,690 experiment patch cells
    (masked cells -> no leakage; non-masked experiment cells -> added
    separately with Tangram-projected expression). These ~324k observed
    cells preserve the ORIGINAL spatial state for K-NN imputation.

    n_context=0 keeps ALL retained cells (default, faithful to
    "keep everything else"); n_context>0 subsamples for quick tests.
    """
    if not CONTEXT_POOL_PATH.exists():
        raise FileNotFoundError(
            f"Context pool not found: {CONTEXT_POOL_PATH}\n"
            f"Run `python build_context_pool.py` first."
        )
    log(f"  Loading context pool: {CONTEXT_POOL_PATH.name}")
    ctx = anndata.read_h5ad(CONTEXT_POOL_PATH)

    # Reorder columns to requested gene order (should already match)
    if list(ctx.var_names) != list(gene_names):
        ctx = ctx[:, gene_names].copy()

    if n_context > 0 and ctx.n_obs > n_context:
        rng = np.random.RandomState(seed)
        sel = rng.choice(ctx.n_obs, n_context, replace=False)
        ctx = ctx[sel].copy()
        log(f"    Subsampled context to {ctx.n_obs}")

    if issparse(ctx.X):
        ctx.X = ctx.X.toarray()
    log(f"    Context pool: {ctx.shape} (actual observed expression)")
    return ctx


def prepare_tangram_inputs(
    sp_adata: anndata.AnnData,
    role: np.ndarray,
    args: argparse.Namespace,
) -> tuple:
    """
    Prepare adata_sc (reference) and adata_sp (non-masked spatial) for Tangram.

    adata_sp holds ONLY the 5,660 experiment non-masked cells. The ~324k
    unassigned/KO spatial-context cells are NOT mapped by Tangram (infeasible
    memory + they are observed, not to be re-predicted); they enter the K-NN
    pool later with their actual expression (see impute_masked_cells_knn).

    Returns:
        adata_sc: reference AnnData (with cell_type in .obs)
        adata_sp: experiment non-masked cells with spatial coords
        training_genes: list of marker genes for training
        n_exp_nonmasked: number of experiment non-masked cells
    """
    log("Preparing Tangram inputs...")

    # Spatial data: non-masked experiment cells only
    non_masked_mask = role != 1  # role 0 + role 2
    adata_sp = sp_adata[non_masked_mask].copy()
    n_exp_nonmasked = adata_sp.shape[0]
    log(f"  adata_sp (non-masked experiment): {adata_sp.shape}")

    # Ensure dense X
    if issparse(adata_sp.X):
        adata_sp.X = adata_sp.X.toarray()

    # Reference: either 30k ref or non-masked cells themselves
    if args.use_ref and REF_PATH.exists():
        log("  Loading 30k reference (Ref_SPAC_2-2.h5ad)...")
        adata_sc = anndata.read_h5ad(REF_PATH)
        log(f"  adata_sc (30k ref): {adata_sc.shape}")
    else:
        log("  Using non-masked experiment cells as reference...")
        adata_sc = sp_adata[non_masked_mask].copy()
        if issparse(adata_sc.X):
            adata_sc.X = adata_sc.X.toarray()
        log(f"  adata_sc (experiment ref): {adata_sc.shape}")

    if issparse(adata_sc.X):
        adata_sc.X = adata_sc.X.toarray()

    # Ensure cell_type is present
    if "cell_type" not in adata_sc.obs.columns:
        raise ValueError("adata_sc missing 'cell_type' in obs")

    # Select training genes (marker genes per cell type)
    log("  Selecting marker genes for training...")
    adata_sc_norm = adata_sc.copy()
    sc.pp.normalize_total(adata_sc_norm, target_sum=1e4)
    sc.pp.log1p(adata_sc_norm)

    sc.tl.rank_genes_groups(
        adata_sc_norm, groupby="cell_type", method="wilcoxon", use_raw=False,
    )

    n_per_ct = args.n_marker_genes
    markers_df = pd.DataFrame(adata_sc_norm.uns["rank_genes_groups"]["names"])
    markers = list(
        np.unique(markers_df.iloc[:n_per_ct, :].values.flatten())
    )

    shared_genes = set(adata_sc.var_names) & set(adata_sp.var_names)
    training_genes = [g for g in markers if g in shared_genes]
    log(f"  Training genes: {len(training_genes)} "
        f"(from {n_per_ct} markers/cell_type)")

    return adata_sc, adata_sp, training_genes, n_exp_nonmasked


def run_tangram_mapping(
    adata_sc: anndata.AnnData,
    adata_sp: anndata.AnnData,
    training_genes: list,
    args: argparse.Namespace,
) -> tuple:
    """
    Run Tangram mapping and gene projection.

    Returns:
        ad_map: mapping AnnData (cells x spots)
        ad_ge: projected gene expression AnnData (spots x genes)
    """
    log("Running Tangram mapping...")
    t0 = time.time()

    # Work on copies (pp_adatas modifies in-place)
    sc_copy = adata_sc.copy()
    sp_copy = adata_sp.copy()

    # Temporarily remove 'spatial' from obsm to skip squidpy dependency
    # (squidpy is only needed for neighborhood extensions which we don't use)
    spatial_backup = None
    if "spatial" in sp_copy.obsm:
        spatial_backup = sp_copy.obsm["spatial"].copy()
        del sp_copy.obsm["spatial"]

    # Preprocess
    tg.pp_adatas(sc_copy, sp_copy, genes=training_genes)
    n_train = len(sc_copy.uns["training_genes"])

    # Restore spatial coordinates
    if spatial_backup is not None:
        sp_copy.obsm["spatial"] = spatial_backup
    log(f"  Training genes after pp_adatas: {n_train}")

    # Map cells to space
    ad_map = tg.map_cells_to_space(
        sc_copy,
        sp_copy,
        mode="cells",
        device=args.device,
        learning_rate=0.1,
        num_epochs=args.num_epochs,
        density_prior="rna_count_based",
        lambda_d=1,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        random_state=args.random_seed,
        verbose=True,
    )
    elapsed = time.time() - t0
    log(f"  Mapping done in {elapsed / 60:.1f} min")

    # Training quality
    train_df = ad_map.uns["train_genes_df"]
    log(f"  Mean train score: {train_df['train_score'].mean():.4f}")
    log(f"  Median train score: {train_df['train_score'].median():.4f}")

    # Project ALL genes (not just training genes)
    log("  Projecting gene expression...")
    sc_proj = adata_sc.copy()
    ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=sc_proj)
    log(f"  Projected expression shape: {ad_ge.shape}")

    return ad_map, ad_ge


def impute_masked_cells_knn(
    exp_aligned: np.ndarray,
    exp_coords_nm: np.ndarray,
    context: anndata.AnnData,
    sp_adata: anndata.AnnData,
    role: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Impute masked cells via spatial K-NN over the honest neighbor pool.

    Neighbor pool = experiment non-masked cells (Tangram-projected expression)
    + ~324k retained context cells (ACTUAL observed expression). Masked cells
    are NOT in the pool -> no ground-truth leakage. This preserves the original
    spatial state: masked patches are imputed from the real surrounding tissue.

    Args:
        exp_aligned: (n_exp_nm, n_gt_genes) projected expr for experiment
                     non-masked cells, aligned to ground-truth gene order
        exp_coords_nm: (n_exp_nm, 2) spatial coords of experiment non-masked cells
        context: context pool AnnData (actual expression, gt gene order, spatial)
        sp_adata: original experiment data (10,690 cells) with spatial coords
        role: role array aligned with sp_adata rows
        k: number of nearest neighbors

    Returns:
        imputed: (n_experiment_cells, n_gt_genes) array with imputed expression
    """
    log(f"Imputing masked cells via spatial K-NN (K={k})...")

    ctx_X = context.X
    if issparse(ctx_X):
        ctx_X = ctx_X.toarray()
    ctx_X = ctx_X.astype(np.float32)
    ctx_coords = np.asarray(context.obsm["spatial"], dtype=np.float32)

    n_exp_nm = exp_aligned.shape[0]
    n_genes = exp_aligned.shape[1]

    # Neighbor pool: experiment non-masked (projected) + context (actual)
    pool_expr = np.vstack([exp_aligned.astype(np.float32), ctx_X])
    pool_coords = np.vstack([exp_coords_nm.astype(np.float32), ctx_coords])
    log(f"  Neighbor pool: {n_exp_nm} experiment (projected) + "
        f"{ctx_X.shape[0]} context (actual) = {pool_expr.shape[0]} cells")

    tree = cKDTree(pool_coords)

    exp_coords = sp_adata.obsm["spatial"]  # (10690, 2)
    masked_idx = np.where(role == 1)[0]
    non_masked_idx = np.where(role != 1)[0]
    n_total = len(role)

    # Output: experiment non-masked keep their projected values
    imputed = np.zeros((n_total, n_genes), dtype=np.float32)
    imputed[non_masked_idx] = exp_aligned

    # For each masked cell, K nearest from pool (exp + context)
    masked_coords = exp_coords[masked_idx]
    distances, indices = tree.query(masked_coords, k=k)

    weights = 1.0 / np.maximum(distances, 1e-10)
    weights_normalized = weights / weights.sum(axis=1, keepdims=True)

    for i in range(len(masked_idx)):
        neighbor_expr = pool_expr[indices[i]]  # (K, n_genes)
        imputed[masked_idx[i]] = (
            weights_normalized[i, :, np.newaxis] * neighbor_expr
        ).sum(axis=0)

    # Report neighbor composition (context cells are indices >= n_exp_nm)
    n_ctx_neighbors = (indices >= n_exp_nm).sum()
    n_total_neighbors = indices.size
    log(f"  Imputed {len(masked_idx)} masked cells")
    log(f"  K-NN neighbors: {n_ctx_neighbors}/{n_total_neighbors} "
        f"({100 * n_ctx_neighbors / n_total_neighbors:.1f}%) from context tissue")
    log(f"  Imputed expression range: "
        f"[{imputed[masked_idx].min():.2f}, {imputed[masked_idx].max():.2f}]")

    return imputed


def evaluate_imputation(
    imputed: np.ndarray,
    ground_truth: anndata.AnnData,
    role: np.ndarray,
    group: np.ndarray,
    matched_gene_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate imputed expression against ground truth for masked cells.

    Metrics: Pearson correlation, Spearman correlation, RMSE.
    Computed overall and per perturbation group.
    """
    log("Evaluating imputation...")

    gt_X = ground_truth.X
    if issparse(gt_X):
        gt_X = gt_X.toarray()

    masked_idx = np.where(role == 1)[0]

    # Only evaluate on matched genes (avoid penalizing unmatched genes)
    if matched_gene_mask is not None:
        gt_X = gt_X[:, matched_gene_mask]
        imputed = imputed[:, matched_gene_mask]
        log(f"  Evaluating on {matched_gene_mask.sum()} matched genes")

    gt_masked = gt_X[masked_idx].astype(np.float64)
    imp_masked = imputed[masked_idx].astype(np.float64)

    # Clip negative values
    imp_masked = np.clip(imp_masked, 0, None)

    results = []

    def compute_metrics(gt_vals: np.ndarray, imp_vals: np.ndarray, label: str) -> dict:
        gt_flat = gt_vals.flatten()
        imp_flat = imp_vals.flatten()

        # Filter out entries where both are zero
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

    # Overall metrics
    overall = compute_metrics(gt_masked, imp_masked, "OVERALL")
    results.append(overall)
    log(f"  OVERALL: PCC={overall['pcc']:.4f}, SCC={overall['scc']:.4f}, "
        f"RMSE={overall['rmse']:.4f} ({overall['n_cells']} cells)")

    # Per-group metrics
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


def align_genes(
    imputed: np.ndarray,
    ad_ge: anndata.AnnData,
    ground_truth: anndata.AnnData,
) -> tuple:
    """
    Align imputed gene expression to match ground truth gene order.

    Tangram's project_genes lowercases gene names, so we need to
    match them back to ground truth genes.

    Returns:
        aligned: (n_cells, n_gt_genes) array with imputed values
        matched_gene_mask: boolean mask of which gt genes were matched
    """
    log("Aligning gene order between projected and ground truth...")

    # Tangram lowercases gene names
    projected_genes = list(ad_ge.var_names)
    gt_genes = list(ground_truth.var_names)
    gt_genes_lower = [g.lower() for g in gt_genes]

    # Build mapping: gt_gene_index -> projected_gene_index
    proj_gene_to_idx = {g: i for i, g in enumerate(projected_genes)}

    n_cells = imputed.shape[0]
    n_gt_genes = len(gt_genes)
    aligned = np.zeros((n_cells, n_gt_genes), dtype=np.float32)
    matched_gene_mask = np.zeros(n_gt_genes, dtype=bool)

    matched = 0
    for i, g_lower in enumerate(gt_genes_lower):
        if g_lower in proj_gene_to_idx:
            aligned[:, i] = imputed[:, proj_gene_to_idx[g_lower]]
            matched_gene_mask[i] = True
            matched += 1

    log(f"  Matched {matched}/{n_gt_genes} genes")
    return aligned, matched_gene_mask


def main() -> None:
    args = parse_args()
    log("=" * 60)
    log("Tangram Imputation Baseline")
    log(f"  device={args.device}, epochs={args.num_epochs}, "
        f"K={args.k_neighbors}, markers/ct={args.n_marker_genes}, "
        f"n_context={args.n_context if args.n_context > 0 else 'ALL'}")
    log("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Phase 1: Load data
    sp_adata, ground_truth, role, group = load_data()

    # Phase 2: Prepare Tangram inputs
    adata_sc, adata_sp, training_genes, n_exp_nonmasked = prepare_tangram_inputs(
        sp_adata, role, args,
    )

    # Phase 3: Run Tangram mapping + projection (experiment spots only)
    ad_map, ad_ge = run_tangram_mapping(
        adata_sc, adata_sp, training_genes, args,
    )

    # Align experiment projected expression to ground-truth gene order
    exp_projected = ad_ge.X.toarray() if issparse(ad_ge.X) else np.asarray(ad_ge.X)
    exp_aligned, matched_gene_mask = align_genes(exp_projected, ad_ge, ground_truth)

    # Match projected rows back to experiment non-masked coords via obs_names
    sp_name_to_row = {n: i for i, n in enumerate(adata_sp.obs_names)}
    ge_rows = [sp_name_to_row[n] for n in ad_ge.obs_names]
    exp_coords_nm = np.asarray(adata_sp.obsm["spatial"])[ge_rows]

    # Phase 4: Load honest spatial-context pool (~324k observed cells)
    context = load_context_pool(
        list(ground_truth.var_names),
        n_context=args.n_context,
        seed=args.random_seed,
    )

    # Phase 5: Impute masked cells via spatial K-NN over the pool
    imputed_aligned = impute_masked_cells_knn(
        exp_aligned, exp_coords_nm, context, sp_adata, role,
        k=args.k_neighbors,
    )

    # Phase 6: Evaluate
    results_df = evaluate_imputation(
        imputed_aligned, ground_truth, role, group,
        matched_gene_mask=matched_gene_mask,
    )

    # Save results
    results_path = OUT_DIR / "imputation_results.csv"
    results_df.to_csv(results_path, index=False)
    log(f"Results saved to {results_path}")

    # Save imputed expression for masked cells
    masked_idx = np.where(role == 1)[0]
    imputed_adata = anndata.AnnData(
        X=imputed_aligned[masked_idx],
        obs=sp_adata[masked_idx].obs.copy(),
        var=ground_truth.var.copy(),
    )
    imputed_path = OUT_DIR / "imputed_masked.h5ad"
    imputed_adata.write(imputed_path)
    log(f"Imputed expression saved to {imputed_path}")

    # Save mapping object
    map_path = OUT_DIR / "tangram_mapping.h5ad"
    ad_map.write(map_path)
    log(f"Mapping saved to {map_path}")

    log("=" * 60)
    log("Tangram imputation complete!")
    log("=" * 60)


if __name__ == "__main__":
    main()
