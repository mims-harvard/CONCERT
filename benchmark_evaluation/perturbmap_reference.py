"""Observed-LFC reference for the Perturb-map within-niche (seen) arm.

Reference: control cells in the niche bin whose patch role is 0, i.e. outside every held-out
role-1 source patch and every role-2 patch on the slide. When fewer than min_g_cells remain, all
role-0 control cells on the slide. The reference is asserted disjoint from the record's source
cells and from every role-1/2 cell. The knockout side is the knockout cells in the bin, slide-wide
when the bin has fewer than min_g_cells.

The evaluation harness (lfc_scoring.observed_lfc_pm_niche) scores against all control
cells of the niche bin. The role-1 source patch cells carry the "control" label on this assay, so
the rescore in score_perturbmap.py replaces that reference with the one defined here.

Predicted LFCs are never touched here: callers pass the stored vectors unchanged.
"""
from __future__ import annotations

import numpy as np

from lfc_scoring import pseudobulk_lfc, to_dense

CTRL = "control"


def pm_roles(adata) -> np.ndarray:
    if "role" not in adata.obs:
        raise KeyError("the Perturb-map reference needs obs['role'] (patch roles)")
    return adata.obs["role"].to_numpy().astype(int)


def pm_source_idx(adata, niche_key: str) -> np.ndarray:
    """Role-1 source cells of one record (same rule as reference_strategies.pm_source_idx)."""
    b, p = (int(x) for x in str(niche_key).split(":"))
    role = pm_roles(adata)
    patch = adata.obs["patch_id"].to_numpy().astype(int)
    tbin = adata.obs["target_bin"].to_numpy().astype(int)
    return np.where((role == 1) & (patch == p) & (tbin == b))[0]


def pm_ko_idx(adata, gene: str, niche_bin: int, min_g_cells: int) -> tuple[np.ndarray, str]:
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata.obs["bin_id"].to_numpy().astype(int)
    ko = np.where((canon == gene) & (bin_id == niche_bin))[0]
    if len(ko) >= min_g_cells:
        return ko, "niche"
    return np.where(canon == gene)[0], "slide"


def pm_reference(adata, niche_bin: int, min_g_cells: int) -> tuple[np.ndarray, str]:
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata.obs["bin_id"].to_numpy().astype(int)
    role = pm_roles(adata)
    base = (canon == CTRL) & (role == 0)
    ref = np.where(base & (bin_id == niche_bin))[0]
    scope = "niche"
    if len(ref) < min_g_cells:
        ref = np.where(base)[0]
        scope = "slide"
    assert not np.isin(role[ref], (1, 2)).any(), "reference contains a role-1/2 cell"
    return ref, scope


def observed_lfc_pm(adata, gene: str, niche_key: str, min_g_cells: int,
                       source_idx: np.ndarray | None = None):
    """Returns (obs_lfc on adata.var_names or None, info dict)."""
    b = int(str(niche_key).split(":")[0])
    ko, ko_scope = pm_ko_idx(adata, gene, b, min_g_cells)
    ref, ref_scope = pm_reference(adata, b, min_g_cells)
    src = pm_source_idx(adata, niche_key) if source_idx is None else np.asarray(source_idx, int)
    assert len(np.intersect1d(ref, src)) == 0, f"reference overlaps source cells ({gene} {niche_key})"
    assert len(np.intersect1d(ref, ko)) == 0, "reference overlaps knockout cells"
    info = dict(n_ko=int(len(ko)), ko_scope=ko_scope, n_ref=int(len(ref)), ref_scope=ref_scope,
                n_src=int(len(src)))
    if len(ko) < min_g_cells or len(ref) < min_g_cells:
        return None, info
    return pseudobulk_lfc(to_dense(adata[ko].X), to_dense(adata[ref].X)), info


def to_axis(full: np.ndarray, prep_names: np.ndarray, target_gene_names: np.ndarray) -> np.ndarray:
    """Map a full-var LFC vector onto another gene-name axis (as observed_lfc_on_gene_axis)."""
    pmap = {g: i for i, g in enumerate(np.asarray(prep_names).astype(str))}
    out = np.full(len(target_gene_names), np.nan, dtype=np.float64)
    for i, g in enumerate(target_gene_names):
        pi = pmap.get(str(g))
        if pi is not None:
            out[i] = full[pi]
    return out
