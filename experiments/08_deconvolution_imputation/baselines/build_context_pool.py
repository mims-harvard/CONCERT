#!/usr/bin/env python3
"""
Build the honest spatial-context pool for K-NN imputation baselines.

Professor's requirement: keep the ORIGINAL spatial state. Only hold out the
masked patches; you may drop the unused perturbed cells (we test only 5 KOs);
keep everything else.

Retained context = full slide cells with perturbation in
{unassigned, Ccn1, Cttn, H2-Ea, Iqgap1, Phgr1}, MINUS all 10,690 experiment
patch cells (the 5,030 masked cells -> avoid leakage; the 5,660 non-masked
experiment cells -> added separately by each method with its own predictions).

Output: context_pool.h5ad  (raw counts, 2,000 HVG aligned to sp_adata,
spatial coords) reused by both Tangram and SpaIM.
"""
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse

SPAC_DIR = Path(
    "/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines"
    "/SpatialScope/test_output/spac_2-2"
)
RAW_PATH = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data"
    "/processed/SPAC/raw_output_2-2.h5ad"
)
OUT_PATH = SPAC_DIR / "context_pool.h5ad"

TESTED_KOS = ["Ccn1", "Cttn", "H2-Ea", "Iqgap1", "Phgr1"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("Loading Phase 1 metadata...")
    sp_adata = anndata.read_h5ad(SPAC_DIR / "sp_adata.h5ad")
    gene_names = list(sp_adata.var_names)  # 2,000 HVG, canonical order
    log(f"  HVG genes: {len(gene_names)}")

    mask = np.load(SPAC_DIR / "mask_info.npz", allow_pickle=True)
    exp_raw_idx = mask["original_cell_index"]  # all 10,690 experiment cells
    log(f"  Experiment patch cells (excluded from context): {len(exp_raw_idx)}")

    log("Loading full slide into memory (CSR ~1.3GB)...")
    raw = anndata.read_h5ad(RAW_PATH)
    raw.var_names_make_unique()
    pert = raw.obs["perturbation"].values
    log(f"  Full slide: {raw.shape}")

    # --- Retained context: unassigned + tested KOs, minus ALL experiment cells ---
    keep_labels = ["unassigned"] + TESTED_KOS
    label_mask = np.isin(pert, keep_labels)
    keep_idx = np.where(label_mask)[0]
    keep_idx = np.setdiff1d(keep_idx, exp_raw_idx, assume_unique=False)
    log(f"  Retained context cells: {len(keep_idx)} "
        f"(unassigned+{len(TESTED_KOS)}KO minus {len(exp_raw_idx)} experiment)")

    # --- Align HVG columns ---
    raw_var = list(raw.var_names)
    raw_pos = {g: i for i, g in enumerate(raw_var)}
    hvg_in_raw = [g for g in gene_names if g in raw_pos]
    raw_col_idx = np.array([raw_pos[g] for g in hvg_in_raw])
    gene_align = np.array([gene_names.index(g) for g in hvg_in_raw])
    log(f"  HVG matched in raw: {len(hvg_in_raw)}/{len(gene_names)}")

    # --- Extract expression (raw counts) ---
    log("  Slicing context expression...")
    sub = raw[keep_idx]
    X_sub = sub.X[:, raw_col_idx]
    if issparse(X_sub):
        X_sub = X_sub.toarray()
    X_full = np.zeros((len(keep_idx), len(gene_names)), dtype=np.float32)
    X_full[:, gene_align] = X_sub

    coords = np.asarray(sub.obsm["spatial"], dtype=np.float32)

    obs = pd.DataFrame(
        {
            "perturbation": pert[keep_idx].astype(str),
            "raw_index": keep_idx.astype(np.int64),
        },
        index=[f"ctx_{i}" for i in range(len(keep_idx))],
    )
    ctx = anndata.AnnData(X=X_full, obs=obs)
    ctx.var_names = pd.Index(gene_names)
    ctx.obsm["spatial"] = coords

    log(f"  Context pool: {ctx.shape}, "
        f"nnz frac={np.count_nonzero(X_full) / X_full.size:.4f}")
    ctx.write(OUT_PATH)
    log(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
