"""Benchmark v4 observed-LFC reference for the Perturb-map within-niche (seen) arm.

v3 reference (unchanged, still used for the reproduction check):
  control cells in the niche bin, or all control cells on the slide when the bin has fewer than
  min_g_cells (seen_residual_scoring.observed_lfc_pm_niche; the CONCERT branch,
  observed_lfc_on_gene_axis, has no slide fallback). The role-1 source patch cells are labelled
  "control", so they can enter that reference.

v4 reference:
  control cells in the niche bin whose patch role is 0, i.e. outside every held-out role-1 source
  patch and every role-2 patch on the slide. When fewer than min_g_cells remain, all role-0 control
  cells on the slide. The reference is asserted disjoint from the record's source cells and from
  every role-1/2 cell. The knockout side is the same as v3 (knockout cells in the bin, slide-wide
  when the bin has fewer than min_g_cells).

Predicted LFCs are never touched here: callers pass the stored vectors unchanged.
"""
from __future__ import annotations

import numpy as np

from seen_residual_scoring import pseudobulk_lfc, to_dense

CTRL = "control"


def pm_roles(adata) -> np.ndarray:
    if "role" not in adata.obs:
        raise KeyError("v4 Perturb-map reference needs obs['role'] (patch roles)")
    return adata.obs["role"].to_numpy().astype(int)


def pm_source_idx(adata, niche_key: str) -> np.ndarray:
    """Role-1 source cells of one record (same rule as rescore_collab_naive_baselines.pm_source_idx)."""
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


def pm_reference_v4(adata, niche_bin: int, min_g_cells: int) -> tuple[np.ndarray, str]:
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata.obs["bin_id"].to_numpy().astype(int)
    role = pm_roles(adata)
    base = (canon == CTRL) & (role == 0)
    ref = np.where(base & (bin_id == niche_bin))[0]
    scope = "niche"
    if len(ref) < min_g_cells:
        ref = np.where(base)[0]
        scope = "slide"
    assert not np.isin(role[ref], (1, 2)).any(), "v4 reference contains a role-1/2 cell"
    return ref, scope


def pm_reference_v3(adata, niche_bin: int, min_g_cells: int, slide_fallback: bool) -> tuple[np.ndarray, str]:
    """v3 reference, exactly as observed_lfc_pm_niche (slide_fallback=True) or
    observed_lfc_on_gene_axis (slide_fallback=False, CONCERT branch)."""
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata.obs["bin_id"].to_numpy().astype(int)
    ref = np.where((canon == CTRL) & (bin_id == niche_bin))[0]
    if len(ref) < min_g_cells and slide_fallback:
        return np.where(canon == CTRL)[0], "slide"
    return ref, "niche"


def observed_lfc_pm_v4(adata, gene: str, niche_key: str, min_g_cells: int,
                       source_idx: np.ndarray | None = None):
    """Returns (obs_lfc on adata.var_names or None, info dict)."""
    b = int(str(niche_key).split(":")[0])
    ko, ko_scope = pm_ko_idx(adata, gene, b, min_g_cells)
    ref, ref_scope = pm_reference_v4(adata, b, min_g_cells)
    src = pm_source_idx(adata, niche_key) if source_idx is None else np.asarray(source_idx, int)
    assert len(np.intersect1d(ref, src)) == 0, f"v4 reference overlaps source cells ({gene} {niche_key})"
    assert len(np.intersect1d(ref, ko)) == 0, "v4 reference overlaps knockout cells"
    info = dict(n_ko=int(len(ko)), ko_scope=ko_scope, n_ref_v4=int(len(ref)), ref_scope_v4=ref_scope,
                n_src=int(len(src)))
    if len(ko) < min_g_cells or len(ref) < min_g_cells:
        return None, info
    return pseudobulk_lfc(to_dense(adata[ko].X), to_dense(adata[ref].X)), info


def reference_overlap_v3(adata, niche_key: str, min_g_cells: int, slide_fallback: bool) -> dict:
    b = int(str(niche_key).split(":")[0])
    ref, scope = pm_reference_v3(adata, b, min_g_cells, slide_fallback)
    src = pm_source_idx(adata, niche_key)
    role = pm_roles(adata)
    n_src_in = int(len(np.intersect1d(ref, src)))
    return dict(n_ref_v3=int(len(ref)), ref_scope_v3=scope, n_src_in_ref_v3=n_src_in,
                n_role12_in_ref_v3=int(np.isin(role[ref], (1, 2)).sum()),
                frac_ref_v3_from_src=(n_src_in / len(ref)) if len(ref) else float("nan"))


def to_axis(full: np.ndarray, prep_names: np.ndarray, target_gene_names: np.ndarray) -> np.ndarray:
    """Map a full-var LFC vector onto another gene-name axis (as observed_lfc_on_gene_axis)."""
    pmap = {g: i for i, g in enumerate(np.asarray(prep_names).astype(str))}
    out = np.full(len(target_gene_names), np.nan, dtype=np.float64)
    for i, g in enumerate(target_gene_names):
        pi = pmap.get(str(g))
        if pi is not None:
            out[i] = full[pi]
    return out
