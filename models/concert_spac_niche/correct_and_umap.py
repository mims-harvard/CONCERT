#!/usr/bin/env python3
"""
Control-based gene-space correction for SPAC-seq CONCERT data.

Strategy
--------
1. Load raw h5ad (integer UMI counts in X).
2. Normalize + log1p.
3. Compute baseline mean over ALL genes while X is still sparse -> save as npy.
4. Select HVGs while X is still sparse (seurat_v3 on CSR).
5. Subset to HVGs (632k x 2k, ~5 GB sparse).
6. Save a raw (uncorrected) copy of the HVG subset for the before/after plot.
7. **Gene-space correction**: densify HVG subset and subtract baseline mean for HVGs.
8. Scale -> PCA (50 PCs).
9. Subsample <=100 k cells for UMAP (speed).
10. Build UMAP; save plots + corrected h5ad.

Memory design
-------------
Peak memory is dominated by the dense HVG matrix: 632k x 2k x 4 bytes ~ 5 GB.
The full-gene densification (632k x 20k x 4 ~ 52 GB) is avoided entirely.
Before/after comparison re-uses the saved raw HVG copy (no second full h5ad load).

Why gene space rather than PCA/latent space?
---------------------------------------------
The perturbation-action effect is a systematic additive shift in log
transcript abundance independent of which gene was KO'd.  Correcting
directly in gene space:
  * is lossless and reversible
  * aligns with limma/edgeR/scran batch-correction assumptions
  * avoids removing real biology (tissue gradients) that dominate top PCs

Usage
-----
python correct_and_umap.py \
    --h5ad   /path/to/raw_output_1-1.h5ad \
    --outdir /path/to/analysis/raw_output_1-1 \
    --sample raw_output_1-1 \
    [--n_hvgs 2000] [--n_pcs 50] [--n_umap_cells 100000] [--seed 42]
"""

import argparse, gc, os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad",         required=True)
    p.add_argument("--outdir",       required=True)
    p.add_argument("--sample",       default="sample")
    p.add_argument("--n_hvgs",       type=int, default=2000)
    p.add_argument("--n_pcs",        type=int, default=50)
    p.add_argument("--n_umap_cells", type=int, default=100_000)
    p.add_argument("--n_neighbors",  type=int, default=15)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# helpers
SPECIAL = {"control", "unassigned", "multiple"}


def subsample_for_umap(
    baseline_mask: np.ndarray,
    n_cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified subsample to at most ``n_cap`` cells (baseline vs perturbed)."""
    idx_base = np.where(baseline_mask)[0]
    idx_other = np.where(~baseline_mask)[0]
    n_total = len(idx_base) + len(idx_other)
    if n_total <= n_cap:
        return np.arange(n_total, dtype=int)

    if len(idx_base) <= n_cap:
        n_other = min(n_cap - len(idx_base), len(idx_other))
        return np.sort(np.concatenate([idx_base, rng.choice(idx_other, n_other, replace=False)]))

    # Baseline alone exceeds cap: proportional stratified sample.
    frac_base = len(idx_base) / n_total
    n_base = int(round(n_cap * frac_base))
    n_base = max(1, min(n_base, len(idx_base), n_cap - 1)) if len(idx_other) else min(len(idx_base), n_cap)
    n_other = min(len(idx_other), n_cap - n_base)
    n_base = n_cap - n_other
    parts = [rng.choice(idx_base, n_base, replace=False)]
    if n_other > 0:
        parts.append(rng.choice(idx_other, n_other, replace=False))
    return np.sort(np.concatenate(parts))


def label_cells(adata):
    """Add obs columns: cell_category, is_perturbed, is_baseline."""
    pert = adata.obs["perturbation"].astype(str)

    cat = pert.copy()
    cat[~pert.isin(SPECIAL)] = "gene_KO"
    adata.obs["cell_category"] = pd.Categorical(
        cat, categories=["control", "unassigned", "multiple", "gene_KO"]
    )
    # Baseline = truly unperturbed (non-targeting guide or no guide detected)
    adata.obs["is_baseline"]   = pert.isin({"control", "unassigned"}).values
    # Perturbed = received at least one guide (includes 'multiple')
    adata.obs["is_perturbed"]  = (~pert.isin({"control", "unassigned"})).values
    return adata


def compute_baseline_mean_sparse(X, baseline_mask):
    """
    Compute per-gene mean over baseline cells.  Works on sparse or dense X
    without creating a dense copy of the full matrix.
    Returns a (G,) float32 array.
    """
    if sp.issparse(X):
        # mean() on a CSR row-slice stays sparse internally
        X_base = np.asarray(X[baseline_mask].mean(axis=0)).flatten()
    else:
        X_base = np.asarray(X[baseline_mask].mean(axis=0)).flatten()
    return X_base.astype(np.float32)


def densify_and_subtract(X_sparse, mean_vec):
    """
    Densify a sparse matrix and subtract mean_vec (G,) row-wise.
    Returns dense float32 array of same shape.
    peak memory = shape[0] * shape[1] * 4 bytes.
    """
    X_dense = X_sparse.toarray().astype(np.float32)
    X_dense -= mean_vec.astype(np.float32)
    return X_dense


def top_pert_palette(adata, n_top=12):
    """Return top-N gene KO labels + special labels with a colour palette."""
    gene_ko = adata.obs["perturbation"][~adata.obs["perturbation"].isin(SPECIAL)]
    top = gene_ko.value_counts().index[:n_top].tolist()
    order = ["control", "unassigned", "multiple"] + top + ["other_gene"]

    def simplify(p):
        if p in SPECIAL:
            return p
        if p in top:
            return p
        return "other_gene"

    adata.obs["pert_top"] = pd.Categorical(
        adata.obs["perturbation"].astype(str).map(simplify), categories=order
    )

    base_colours = {
        "control":    "#2ca02c",
        "unassigned": "#aec7e8",
        "multiple":   "#ffbb78",
        "other_gene": "#d3d3d3",
    }
    cmap = plt.cm.get_cmap("tab20", n_top)
    top_colours = {t: matplotlib.colors.to_hex(cmap(i)) for i, t in enumerate(top)}
    palette = {**base_colours, **top_colours}
    return palette


# main
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sc.settings.verbosity = 2
    np.random.seed(args.seed)

    # 1. Load
    print(f"\n[1] Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"    Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    label_cells(adata)

    n_base      = adata.obs["is_baseline"].sum()
    n_perturbed = adata.obs["is_perturbed"].sum()
    n_control   = (adata.obs["cell_category"] == "control").sum()
    n_unassigned= (adata.obs["cell_category"] == "unassigned").sum()
    n_multiple  = (adata.obs["cell_category"] == "multiple").sum()
    n_geneko    = (adata.obs["cell_category"] == "gene_KO").sum()
    print(f"    control={n_control:,}  unassigned={n_unassigned:,}  "
          f"multiple={n_multiple:,}  gene_KO={n_geneko:,}")
    print(f"    Baseline pool: {n_base:,} ({100*n_base/adata.n_obs:.1f}%)")

    baseline_mask = adata.obs["is_baseline"].values.astype(bool)

    # 2. Normalize
    print("\n[2] Normalise + log1p  (X stays sparse)")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Ensure CSR for efficient row-slicing
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()

    # 3. Compute baseline mean over ALL genes (sparse, no densification)
    print("\n[3] Computing baseline mean over all genes (sparse)")
    baseline_mean_all = compute_baseline_mean_sparse(adata.X, baseline_mask)
    np.save(outdir / f"{args.sample}_baseline_mean_all_genes.npy", baseline_mean_all)
    print(f"  Baseline mean (all genes): min={baseline_mean_all.min():.4f}  "
          f"max={baseline_mean_all.max():.4f}  "
          f"nonzero={np.sum(baseline_mean_all != 0):,}/{len(baseline_mean_all):,}")
    print("  Saved: baseline_mean_all_genes.npy  (use this for CONCERT training)")

    # 4. HVG selection on sparse log-normalized data
    print(f"\n[4] Selecting top {args.n_hvgs} HVGs  (on sparse X, no copy)")
    # seurat_v3 flavor uses variance on sparse matrix; works fine on CSR
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, flavor="seurat_v3")
    hvg_mask = adata.var["highly_variable"].values
    print(f"    HVGs selected: {hvg_mask.sum()}")

    # 5. Subset to HVGs while still sparse
    print(f"\n[5] Subsetting to HVGs  ({adata.shape[0]:,} x {hvg_mask.sum():,})")
    adata_hvg = adata[:, hvg_mask].copy()
    # Free full-gene adata -- no longer needed
    del adata
    gc.collect()

    baseline_mean_hvg = baseline_mean_all[hvg_mask]

    # 6. Save raw (uncorrected) HVG copy for before/after comparison
    print("\n[6] Saving raw HVG matrix for before/after comparison")
    # Keep sparse for now; densify later only for the subsampled plot
    X_raw_sparse = adata_hvg.X.copy()   # sparse, ~0.5 GB for 632k x 2k

    # 7. Gene-space correction on HVG subset
    print("\n[7] Gene-space correction (densify HVG subset + subtract baseline)")
    print(f"    Dense HVG matrix size: "
          f"~{adata_hvg.shape[0] * adata_hvg.shape[1] * 4 / 1e9:.1f} GB")
    X_corr_dense = densify_and_subtract(adata_hvg.X, baseline_mean_hvg)
    adata_hvg.X = X_corr_dense   # replace sparse with corrected dense
    print(f"    Correction done.  X range: "
          f"[{adata_hvg.X.min():.3f}, {adata_hvg.X.max():.3f}]")

    # 8. Scale + PCA
    print(f"\n[8] Scale + PCA ({args.n_pcs} PCs)")
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=args.n_pcs, svd_solver="arpack")

    var_exp = adata_hvg.uns["pca"]["variance_ratio"]
    print(f"    PC1-5 variance: {', '.join(f'{v*100:.1f}%' for v in var_exp[:5])}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.cumsum(var_exp) * 100)
    ax.axhline(80, ls="--", color="grey")
    ax.set_xlabel("PC"); ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title(f"{args.sample} -- PCA scree (corrected)")
    fig.tight_layout()
    fig.savefig(outdir / f"{args.sample}_pca_scree.png", dpi=150)
    plt.close(fig)

    # 9. UMAP (subsample for speed)
    print(f"\n[9] UMAP on <={args.n_umap_cells:,} cells")
    if adata_hvg.n_obs > args.n_umap_cells:
        rng = np.random.default_rng(args.seed)
        idx_sub = subsample_for_umap(baseline_mask, args.n_umap_cells, rng)
        n_base_sub = int(baseline_mask[idx_sub].sum())
        adata_sub = adata_hvg[idx_sub].copy()
        print(f"    Subsampled to {adata_sub.n_obs:,} cells "
              f"({n_base_sub:,} baseline + {adata_sub.n_obs - n_base_sub:,} others)")
    else:
        idx_sub = None
        adata_sub = adata_hvg.copy()
        print(f"    Using all {adata_sub.n_obs:,} cells")

    sc.pp.neighbors(adata_sub, n_neighbors=args.n_neighbors, n_pcs=30,
                    random_state=args.seed)
    sc.tl.umap(adata_sub, random_state=args.seed)

    # 10. Plots
    print("\n[10] Saving UMAP plots")

    # Plot A: is_perturbed
    fig, ax = plt.subplots(figsize=(8, 7))
    colours = {False: "#2ca02c", True: "#d62728"}
    for flag, label, zorder in [(False, "baseline\n(ctrl+unassigned)", 2),
                                 (True,  "perturbed", 1)]:
        mask = adata_sub.obs["is_perturbed"].values == flag
        ax.scatter(adata_sub.obsm["X_umap"][mask, 0],
                   adata_sub.obsm["X_umap"][mask, 1],
                   s=0.3, alpha=0.4, color=colours[flag],
                   label=f"{label} (n={mask.sum():,})", rasterized=True,
                   zorder=zorder)
    ax.legend(markerscale=10, loc="upper right", fontsize=9)
    ax.set_title(f"{args.sample} -- gene-space corrected\nbaseline vs. perturbed")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outdir / f"{args.sample}_umap_is_perturbed.png", dpi=150)
    plt.close(fig)
    print("    saved: umap_is_perturbed.png")

    # Plot B: cell category
    cat_colours = {
        "control":    "#2ca02c",
        "unassigned": "#1f77b4",
        "multiple":   "#ff7f0e",
        "gene_KO":    "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8, 7))
    for cat in ["gene_KO", "multiple", "unassigned", "control"]:
        mask = adata_sub.obs["cell_category"].values == cat
        if mask.sum() == 0:
            continue
        ax.scatter(adata_sub.obsm["X_umap"][mask, 0],
                   adata_sub.obsm["X_umap"][mask, 1],
                   s=0.3, alpha=0.5, color=cat_colours[cat],
                   label=f"{cat} (n={mask.sum():,})", rasterized=True)
    ax.legend(markerscale=10, loc="upper right", fontsize=9)
    ax.set_title(f"{args.sample} -- gene-space corrected\ncell categories")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outdir / f"{args.sample}_umap_cell_category.png", dpi=150)
    plt.close(fig)
    print("    saved: umap_cell_category.png")

    # Plot C: top perturbations
    palette = top_pert_palette(adata_sub, n_top=12)
    order = list(adata_sub.obs["pert_top"].cat.categories)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in reversed(order):
        mask = adata_sub.obs["pert_top"].values == lbl
        if mask.sum() == 0:
            continue
        ax.scatter(adata_sub.obsm["X_umap"][mask, 0],
                   adata_sub.obsm["X_umap"][mask, 1],
                   s=0.3 if lbl == "other_gene" else 0.8,
                   alpha=0.25 if lbl == "other_gene" else 0.7,
                   color=palette[lbl], rasterized=True)
    handles = [mpatches.Patch(color=palette[l],
                              label=f"{l} ({(adata_sub.obs['pert_top']==l).sum():,})")
               for l in order if (adata_sub.obs["pert_top"]==l).sum() > 0]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              markerscale=4, ncol=2)
    ax.set_title(f"{args.sample} -- gene-space corrected\ntop 12 perturbations")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outdir / f"{args.sample}_umap_top_perts.png", dpi=150)
    plt.close(fig)
    print("    saved: umap_top_perts.png")

    # 11. Before/after comparison using saved raw HVG copy
    print("\n[11] Before/after comparison  (re-using saved raw HVG matrix)")

    # Build raw sub AnnData for the same subsample indices
    if idx_sub is not None:
        X_raw_sub = X_raw_sparse[idx_sub]
    else:
        X_raw_sub = X_raw_sparse

    # Need obs metadata; extract from adata_sub (same idx_sub order)
    import anndata as ad
    adata_raw_sub = ad.AnnData(X=X_raw_sub, obs=adata_sub.obs.copy(),
                               var=adata_hvg.var.copy())

    sc.pp.scale(adata_raw_sub, max_value=10)
    sc.tl.pca(adata_raw_sub, n_comps=args.n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata_raw_sub, n_neighbors=args.n_neighbors, n_pcs=30,
                    random_state=args.seed)
    sc.tl.umap(adata_raw_sub, random_state=args.seed)

    del X_raw_sparse; gc.collect()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, ad_, title in zip(axes,
                               [adata_raw_sub, adata_sub],
                               ["Raw (no correction)", "Gene-space corrected"]):
        for flag, colour, zo in [(False, "#2ca02c", 2), (True, "#d62728", 1)]:
            mask = ad_.obs["is_perturbed"].values == flag
            ax.scatter(ad_.obsm["X_umap"][mask, 0],
                       ad_.obsm["X_umap"][mask, 1],
                       s=0.2, alpha=0.3, color=colour,
                       label="baseline" if not flag else "perturbed",
                       rasterized=True, zorder=zo)
        ax.legend(markerscale=12, loc="upper right", fontsize=9)
        ax.set_title(f"{args.sample}\n{title}")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outdir / f"{args.sample}_umap_before_after.png", dpi=150)
    plt.close(fig)
    print("    saved: umap_before_after.png")

    # 12. Save corrected subsampled h5ad
    print("\n[12] Saving outputs")
    adata_sub.write_h5ad(outdir / f"{args.sample}_corrected_umap_sub.h5ad",
                         compression="gzip")
    print(f"    Saved: {args.sample}_corrected_umap_sub.h5ad")

    # Summary stats: PC1 separation baseline vs perturbed
    pc1_corr_base = adata_sub.obsm["X_pca"][~adata_sub.obs["is_perturbed"].values, 0]
    pc1_corr_pert = adata_sub.obsm["X_pca"][ adata_sub.obs["is_perturbed"].values, 0]
    pc1_raw_base  = adata_raw_sub.obsm["X_pca"][~adata_raw_sub.obs["is_perturbed"].values, 0]
    pc1_raw_pert  = adata_raw_sub.obsm["X_pca"][ adata_raw_sub.obs["is_perturbed"].values, 0]

    from scipy.stats import mannwhitneyu
    stat_c, pval_c = mannwhitneyu(pc1_corr_base, pc1_corr_pert, alternative="two-sided")
    stat_r, pval_r = mannwhitneyu(pc1_raw_base,  pc1_raw_pert,  alternative="two-sided")

    print(f"\n    PC1 separation (Mann-Whitney U):")
    print(f"      RAW:       baseline={pc1_raw_base.mean():.3f}  "
          f"perturbed={pc1_raw_pert.mean():.3f}  p={pval_r:.3e}")
    print(f"      CORRECTED: baseline={pc1_corr_base.mean():.3f}  "
          f"perturbed={pc1_corr_pert.mean():.3f}  p={pval_c:.3e}")

    print(f"\n[Done] All outputs in {outdir}/")
    print("  baseline_mean_all_genes.npy  -- subtract this per-gene when loading "
          "batches in CONCERT training")


if __name__ == "__main__":
    main()
