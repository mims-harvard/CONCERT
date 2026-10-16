#!/usr/bin/env python3
"""
Recompute the full metric suite (E-distance, null-aware LFC signature,
unpert-vs-pred SCC) for an updated imputed_masked.h5ad. Run after the
leakage-free / full-context imputation.

LFC is the null-aware, local-control perturbation signature from
`perturbation_metrics.py` (method / pure-control null / delta), NOT the old
global-baseline LFC (which was inflated by regional tissue baseline).

Usage:
  python compute_full_metrics.py --method tangram
  python compute_full_metrics.py --method spaim
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import argparse
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perturbation_metrics as pm

SPAC_DIR = Path(
    f"{_rp.WORKDIR}/spatial_baselines/concert_baselines"
    "/SpatialScope/test_output/spac_2-2"
)
RAW_PATH = Path(
    f"{_rp.WORKDIR}/concert_data"
    "/processed/SPAC/raw_output_2-2.h5ad"
)
OUT_ROOT = Path(
    f"{_rp.WORKDIR}/spatial_baselines/concert_baselines"
)
PERTURB_GROUPS = ["Ccn1", "Cttn", "H2-Ea", "Iqgap1", "Phgr1"]


def dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def energy_distance(X, Y, max_samples=2000, seed=42):
    rng = np.random.RandomState(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]
    xy = cdist(X, Y, metric="euclidean").mean()
    xx = cdist(X, X, metric="euclidean").mean()
    yy = cdist(Y, Y, metric="euclidean").mean()
    return 2 * xy - xx - yy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["tangram", "spaim"])
    args = ap.parse_args()

    out_dir = OUT_ROOT / f"{args.method}_output" / "spac_2-2"
    imputed_path = out_dir / "imputed_masked.h5ad"
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    gt = anndata.read_h5ad(SPAC_DIR / "ground_truth.h5ad")
    gene_names = list(gt.var_names)
    gt_X = dense(gt.X)

    mask = np.load(SPAC_DIR / "mask_info.npz", allow_pickle=True)
    role = mask["role"]
    group = mask["group"]
    exp_raw_idx = mask["original_cell_index"]

    masked_idx = np.where(role == 1)[0]
    masked_groups = group[masked_idx]
    gt_masked = gt_X[masked_idx].astype(np.float64)

    # Spatial coordinates of the masked cells (for local control / spatial null)
    sp_adata = anndata.read_h5ad(SPAC_DIR / "sp_adata.h5ad")
    masked_coords = np.asarray(sp_adata.obsm["spatial"])[masked_idx]

    imp = anndata.read_h5ad(imputed_path)
    imp_masked = dense(imp.X).astype(np.float64)
    imp_masked = np.clip(imp_masked, 0, None)
    assert imp_masked.shape[0] == len(masked_idx), "row mismatch"
    print(f"[{args.method}] imputed masked: {imp_masked.shape}")

    # Unperturbed baseline: full-slide unassigned, EXCLUDING masked cells
    raw = anndata.read_h5ad(RAW_PATH)
    raw.var_names_make_unique()
    pert = raw.obs["perturbation"].values
    unpert_idx = np.where(pert == "unassigned")[0]
    unpert_idx = np.setdiff1d(unpert_idx, exp_raw_idx)  # drop masked patch cells
    rng = np.random.RandomState(42)
    sub = rng.choice(unpert_idx, min(5000, len(unpert_idx)), replace=False)
    raw_pos = {g: i for i, g in enumerate(raw.var_names)}
    cols = [raw_pos[g] for g in gene_names if g in raw_pos]
    align = [gene_names.index(g) for g in gene_names if g in raw_pos]
    U = dense(raw[sub].X[:, cols])
    gt_unpert = np.zeros((len(sub), len(gene_names)), dtype=np.float64)
    gt_unpert[:, align] = U
    mean_unpert = gt_unpert.mean(axis=0)
    print(f"  unpert baseline cells (masked excluded): {gt_unpert.shape[0]}")

    groups_unique = list(np.unique(masked_groups))

    # E-distance
    np.random.seed(42)
    ed_rows = []
    for g in groups_unique:
        m = masked_groups == g
        ed = energy_distance(gt_masked[m], imp_masked[m])
        ed_rows.append({"group": g, "n_cells": int(m.sum()), "e_distance": ed})
    ed_all = energy_distance(gt_masked, imp_masked)
    ed_rows.append({"group": "OVERALL", "n_cells": len(masked_idx), "e_distance": ed_all})
    ed_df = pd.DataFrame(ed_rows)
    ed_df.to_csv(analysis_dir / "e_distance.csv", index=False)

    # Null-aware LFC signature (local control + pure-control null + delta)
    lfc_df = pm.compute_perturbation_table(
        gt_masked, imp_masked, masked_groups, masked_coords,
        gene_names, PERTURB_GROUPS,
    )
    lfc_df.to_csv(analysis_dir / "lfc_correlation.csv", index=False)

    # SCC unpert vs obs/pred
    sp_rows = []
    for g in PERTURB_GROUPS:
        m = masked_groups == g
        scc_obs, _ = spearmanr(mean_unpert, gt_masked[m].mean(axis=0))
        scc_pred, _ = spearmanr(mean_unpert, imp_masked[m].mean(axis=0))
        sp_rows.append({
            "group": g, "n_cells": int(m.sum()),
            "scc_unpert_vs_obs_pert": scc_obs,
            "scc_unpert_vs_pred_pert": scc_pred,
            "delta": scc_pred - scc_obs,
        })
    sp_df = pd.DataFrame(sp_rows)
    sp_df.to_csv(analysis_dir / "spearman_unpert_vs_pert.csv", index=False)

    # Merge with main PCC/SCC/RMSE
    imp_results = pd.read_csv(out_dir / "imputation_results.csv")
    summary = imp_results.merge(
        ed_df[ed_df["group"] != "OVERALL"][["group", "e_distance"]],
        on="group", how="left")
    summary = summary.merge(lfc_df, on="group", how="left")
    summary = summary.merge(
        sp_df[["group", "scc_unpert_vs_obs_pert", "scc_unpert_vs_pred_pert"]],
        on="group", how="left")
    summary.to_csv(analysis_dir / "full_metrics_summary.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== FULL METRICS SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
