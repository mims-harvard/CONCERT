#!/usr/bin/env python3
"""
Null-aware, local-control perturbation-signature metrics.

Replaces the old global-baseline LFC, which was inflated by (a) group-mean
averaging, (b) a GLOBAL control that baked in each patch's regional tissue
baseline, and (c) the absence of a recoverable KO signal on the HVG panel
(the one target gene present, Ccn1, is not even knocked down). A zero-KO
"pure-control" predictor scored the same 0.93-0.98 as the real methods.

The fix, per KO group g:
  - local_ctrl : mean of the unassigned CONTEXT cells spatially nearest to the
                 group's masked cells (removes the regional baseline).
  - pure_ctrl  : each masked cell = distance-weighted mean of its nearest
                 unassigned context cells -> a spatial null with ZERO KO info.
  - lfc_*      : log2((mean+1)/(local_ctrl+1)), evaluated on the top-K DEGs
                 (genes with the largest |lfc_obs|).
  - report lfc_pcc (method), lfc_pcc_null (pure_ctrl), and the HEADLINE
    lfc_pcc_delta = method - null (~0 => no perturbation recovery beyond
    copying local tissue). Same three for SCC.

All expression is raw counts aligned to the ground-truth gene order.
Context cells come from context_pool.h5ad (build_context_pool.py): 324,012
unassigned cells with the 10,690 experiment patch cells excluded (no leakage).
"""
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse

CONTEXT_POOL_PATH = Path(
    "/n/netscratch/mzitnik_lab/Lab/zlkong/spatial_baselines/concert_baselines"
    "/SpatialScope/test_output/spac_2-2/context_pool.h5ad"
)

# Default hyper-parameters (documented in the plan / baseline docs).
K_LOCAL = 30    # nearest context cells per masked cell -> local control pool
K_NULL = 20     # nearest context cells per masked cell -> spatial null predictor
TOP_K_DEG = 100  # genes kept per group (largest |lfc_obs|)


def _dense(x: np.ndarray) -> np.ndarray:
    return x.toarray() if issparse(x) else np.asarray(x)


def load_context(gene_names: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (ctx_expr [n, G] aligned to gene_names, ctx_coords [n, 2])."""
    if not CONTEXT_POOL_PATH.exists():
        raise FileNotFoundError(
            f"Context pool not found: {CONTEXT_POOL_PATH}\n"
            f"Run `python build_context_pool.py` first."
        )
    ctx = anndata.read_h5ad(CONTEXT_POOL_PATH)
    if list(ctx.var_names) != list(gene_names):
        ctx = ctx[:, list(gene_names)].copy()
    ctx_expr = _dense(ctx.X).astype(np.float64)
    ctx_coords = np.asarray(ctx.obsm["spatial"], dtype=np.float64)
    return ctx_expr, ctx_coords


def pure_control_predictor(
    masked_coords: np.ndarray,
    ctx_expr: np.ndarray,
    ctx_coords: np.ndarray,
    k: int = K_NULL,
    tree: cKDTree | None = None,
) -> tuple[np.ndarray, cKDTree]:
    """Spatial null: each masked cell = distance-weighted mean of its k nearest
    unassigned context cells (zero KO information). Returns (n_masked, G)."""
    if tree is None:
        tree = cKDTree(ctx_coords)
    d, idx = tree.query(masked_coords, k=k)
    w = 1.0 / np.maximum(d, 1e-10)
    w /= w.sum(axis=1, keepdims=True)
    pred = np.einsum("ij,ijk->ik", w, ctx_expr[idx])
    return pred, tree


def local_control(
    group_coords: np.ndarray,
    ctx_expr: np.ndarray,
    ctx_coords: np.ndarray,
    k: int = K_LOCAL,
    tree: cKDTree | None = None,
) -> np.ndarray:
    """Local baseline: mean of the unique context cells that are k-NN to any
    cell in group_coords (removes the patch's regional baseline)."""
    if tree is None:
        tree = cKDTree(ctx_coords)
    _, ii = tree.query(group_coords, k=k)
    loc = np.unique(ii.ravel())
    return ctx_expr[loc].mean(axis=0)


def _lfc(mean_pert: np.ndarray, ctrl: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    return np.log2((mean_pert + pseudocount) / (ctrl + pseudocount))


def lfc_signature_metrics(
    gt_g: np.ndarray,
    imp_g: np.ndarray,
    pure_ctrl_g: np.ndarray,
    local_ctrl: np.ndarray,
    top_k: int = TOP_K_DEG,
) -> dict:
    """Null-aware local-control LFC on the top-K DEGs of one group.

    Returns lfc_pcc/scc for the method, the pure-control null, and the
    delta (method - null) = perturbation recovery beyond the spatial null.
    """
    lfc_obs = _lfc(gt_g.mean(axis=0), local_ctrl)
    lfc_pred = _lfc(imp_g.mean(axis=0), local_ctrl)
    lfc_null = _lfc(pure_ctrl_g.mean(axis=0), local_ctrl)

    deg = np.argsort(-np.abs(lfc_obs))[:top_k]
    o, p, n = lfc_obs[deg], lfc_pred[deg], lfc_null[deg]

    pcc_m = pearsonr(o, p)[0]
    pcc_n = pearsonr(o, n)[0]
    scc_m = spearmanr(o, p)[0]
    scc_n = spearmanr(o, n)[0]
    return {
        "lfc_pcc": pcc_m,
        "lfc_pcc_null": pcc_n,
        "lfc_pcc_delta": pcc_m - pcc_n,
        "lfc_scc": scc_m,
        "lfc_scc_null": scc_n,
        "lfc_scc_delta": scc_m - scc_n,
    }


def compute_perturbation_table(
    gt_masked: np.ndarray,
    imp_masked: np.ndarray,
    masked_groups: np.ndarray,
    masked_coords: np.ndarray,
    gene_names: list,
    perturb_groups: list,
    context: tuple[np.ndarray, np.ndarray] | None = None,
    k_local: int = K_LOCAL,
    k_null: int = K_NULL,
    top_k: int = TOP_K_DEG,
) -> pd.DataFrame:
    """Per-group null-aware perturbation-signature table.

    context: optional pre-loaded (ctx_expr, ctx_coords) to avoid re-reading the
    324k context pool when evaluating several methods in one process.
    """
    if context is None:
        ctx_expr, ctx_coords = load_context(gene_names)
    else:
        ctx_expr, ctx_coords = context

    pure_ctrl, tree = pure_control_predictor(masked_coords, ctx_expr, ctx_coords, k=k_null)

    rows = []
    for g in perturb_groups:
        m = masked_groups == g
        lc = local_control(masked_coords[m], ctx_expr, ctx_coords, k=k_local, tree=tree)
        met = lfc_signature_metrics(
            gt_masked[m], imp_masked[m], pure_ctrl[m], lc, top_k=top_k
        )
        rows.append({"group": g, **met})
    return pd.DataFrame(rows)
