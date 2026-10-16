"""
Phase 3-4: impute the SPAC-seq 2-2 masked cells with the trained SpatialScope model
- 5 sampling runs (replicates)
- uses Decomposition_imputation.py (patched version)
- evaluates imputed vs ground truth
"""
import sys
import os
import numpy as np
import torch
import anndata
import pandas as pd
import glob

SPATIALSCOPE_SRC = '/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines/SpatialScope/src'
sys.path.insert(0, SPATIALSCOPE_SRC)

# Configuration
CKPT_DIR = '/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines/SpatialScope/SpatialScope/Ckpts_scRefs/SPAC_2-2'
SCREF_PATH = os.path.join(CKPT_DIR, 'Ref_SPAC_2-2.h5ad')
IMPUTATION_DIR = '/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines/SpatialScope/test_output/spac_2-2'
GT_PATH = os.path.join(IMPUTATION_DIR, 'ground_truth.h5ad')
N_REPLICATES = 5
BATCH_SIZE = 500  # spots per batch to avoid OOM


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint"""
    pts = sorted(glob.glob(os.path.join(ckpt_dir, 'model_*.pt')))
    if not pts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return pts[-1]


def run_imputation():
    print("=" * 60, flush=True)
    print("SpatialScope Imputation on SPAC-seq 2-2", flush=True)
    print("=" * 60, flush=True)

    # Find checkpoint
    ckpt_path = find_latest_checkpoint(CKPT_DIR)
    print(f"[1] Checkpoint: {ckpt_path}", flush=True)

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2] Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Import after setting CUDA_VISIBLE_DEVICES
    from Decomposition_imputation import GeneExpDecompositionImputation
    from utils import ConfigWrapper

    # Config - keep SpatialScope defaults
    config = ConfigWrapper()
    config.data = ConfigWrapper()
    config.data.cell_class_column = 'cell_type'
    config.data.tissue = 'spac_2-2'
    config.data.out_dir = os.path.dirname(IMPUTATION_DIR)
    config.data.SC_Data = SCREF_PATH

    config.decomposition = ConfigWrapper()
    config.decomposition.leave_out_test = False
    config.decomposition.test_genes = None
    config.decomposition.save_process = False
    config.decomposition.power = 1
    config.decomposition.verbose = False
    config.decomposition.replicates = N_REPLICATES
    config.decomposition.spot_range = None

    config.ckpt_path = ckpt_path
    config.device = device

    # Initialize and load
    print("[3] Loading model and data...", flush=True)
    decom = GeneExpDecompositionImputation(config)
    decom.LoadScData()
    decom.LoadModel()
    decom.LoadSTdata()

    n_spots = decom.spatial_adata_reorder.shape[0]
    print(f"    Total spots: {n_spots}", flush=True)
    print(f"    Masked spots detected: {len(decom.masked_spot_ids)}", flush=True)
    print(f"    Batch size: {BATCH_SIZE} spots per batch", flush=True)

    # Run 5 replicates, each batched over spot_range
    for i in range(N_REPLICATES):
        print(f"\n[4.{i}] Running replicate {i}...", flush=True)

        for start in range(0, n_spots, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_spots)
            config.decomposition.spot_range = [start, end]
            decom.config.decomposition.spot_range = [start, end]
            print(f"    Batch [{start}-{end}]...", flush=True)

            decom.single_decomposition(rep_No=i)
            torch.cuda.empty_cache()

        print(f"    Replicate {i} done.", flush=True)

        # Quick NaN check on all batch outputs
        rep_files = sorted(glob.glob(os.path.join(IMPUTATION_DIR, f'decomposition/rep{i}/*.h5ad')))
        total_nan = 0
        total_cells = 0
        for f in rep_files:
            r = anndata.read_h5ad(f)
            total_nan += np.isnan(r.X).sum()
            total_cells += r.X.size
        nan_ratio = total_nan / total_cells if total_cells > 0 else 0
        print(f"    NaN ratio: {nan_ratio:.6f} ({len(rep_files)} batch files)", flush=True)

    # Evaluation
    print("\n[5] Evaluation...", flush=True)
    evaluate_imputation()


def evaluate_imputation():
    """Compare imputed vs ground truth"""
    from scipy.stats import pearsonr, spearmanr

    # Load ground truth
    gt = anndata.read_h5ad(GT_PATH)
    gt_X = np.array(gt.X)

    # Load mask info
    mask_info = np.load(os.path.join(IMPUTATION_DIR, 'mask_info.npz'), allow_pickle=True)
    role = mask_info['role']
    group = mask_info['group']
    masked_idx = np.where(role == 1)[0]

    # Ground truth for masked cells only
    gt_masked = gt_X[masked_idx]

    # Collect all replicates (each rep has multiple batch files)
    all_imputed = []
    for i in range(N_REPLICATES):
        rep_files = sorted(glob.glob(os.path.join(IMPUTATION_DIR, f'decomposition/rep{i}/*.h5ad')))
        if not rep_files:
            print(f"  WARNING: rep{i} not found", flush=True)
            continue

        # Concatenate all batch files for this replicate
        batch_parts = []
        for f in rep_files:
            r = anndata.read_h5ad(f)
            batch_parts.append(r)
        r_concat = anndata.concat(batch_parts)
        r_X = np.array(r_concat.X)

        # r_X is in log space (x + mu), convert to count space
        r_count = np.exp(r_X) - 1
        r_count[r_count < 0] = 0

        print(f"  Rep{i}: {r_X.shape[0]} cells from {len(rep_files)} batches", flush=True)

        # Extract masked cells
        r_masked = r_count[masked_idx]
        all_imputed.append(r_masked)

    if not all_imputed:
        print("  ERROR: No replicate results found!", flush=True)
        return

    # Mean across replicates
    mean_imputed = np.mean(all_imputed, axis=0)

    # Overall metrics
    gt_flat = gt_masked.flatten()
    imp_flat = mean_imputed.flatten()

    # Remove zero pairs for correlation
    nonzero = (gt_flat > 0) | (imp_flat > 0)
    if nonzero.sum() > 0:
        pcc, _ = pearsonr(gt_flat[nonzero], imp_flat[nonzero])
        scc, _ = spearmanr(gt_flat[nonzero], imp_flat[nonzero])
    else:
        pcc, scc = 0.0, 0.0

    rmse = np.sqrt(np.mean((gt_masked - mean_imputed) ** 2))

    print(f"\n  === Overall Metrics (5030 masked cells x {gt_masked.shape[1]} genes) ===", flush=True)
    print(f"  Pearson correlation:  {pcc:.4f}", flush=True)
    print(f"  Spearman correlation: {scc:.4f}", flush=True)
    print(f"  RMSE:                 {rmse:.4f}", flush=True)

    # Per-group metrics
    print(f"\n  === Per-group Metrics ===", flush=True)
    groups_unique = np.unique(group[role == 1])
    results = []
    for g in groups_unique:
        g_mask = group[role == 1] == g
        gt_g = gt_masked[g_mask].flatten()
        imp_g = mean_imputed[g_mask].flatten()
        nz = (gt_g > 0) | (imp_g > 0)
        if nz.sum() > 100:
            pcc_g, _ = pearsonr(gt_g[nz], imp_g[nz])
            scc_g, _ = spearmanr(gt_g[nz], imp_g[nz])
        else:
            pcc_g, scc_g = 0.0, 0.0
        rmse_g = np.sqrt(np.mean((gt_masked[g_mask] - mean_imputed[g_mask]) ** 2))
        results.append({'group': g, 'n_cells': g_mask.sum(), 'pcc': pcc_g, 'scc': scc_g, 'rmse': rmse_g})
        print(f"  {g:15s} | n={g_mask.sum():5d} | PCC={pcc_g:.4f} | SCC={scc_g:.4f} | RMSE={rmse_g:.4f}", flush=True)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(IMPUTATION_DIR, 'imputation_results.csv'), index=False)

    # Save mean imputed values
    mean_adata = anndata.AnnData(
        X=mean_imputed,
        obs=pd.DataFrame({'group': group[role == 1]}, index=[f'cell_{i}' for i in masked_idx]),
        var=gt.var.copy()
    )
    mean_adata.write(os.path.join(IMPUTATION_DIR, 'mean_imputed_masked.h5ad'))

    print(f"\n  Results saved to: {IMPUTATION_DIR}/imputation_results.csv", flush=True)
    print(f"  Mean imputed saved to: {IMPUTATION_DIR}/mean_imputed_masked.h5ad", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    run_imputation()
