#!/usr/bin/env python3
"""
CONCERT-spac post-training UMAP analysis.

Loads the trained model, extracts 10-dim latent z for all cells,
computes UMAP, and saves:
  - umap_concert_latent.pdf   — colored by perturbation class and top KOs
  - umap_coords.npz           — 2D UMAP coords + perturbation labels (lightweight)
  - f_pert_lfc_matrix.npy     — per-pert per-gene LFC matrix (n_pert × n_genes)

Output dir: --outdir (default: datasets/newdata/spac_seq/).
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
import anndata as ad  # needed by scanpy internally
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from sklearn.preprocessing import MinMaxScaler

# ── resolve CONCERT-spac source path ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from preprocess import normalize
from concert_spac import CONCERT
from run_concert_spac import load_h5_dataset, build_attributes_spac, build_inducing_points

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded paths and config — mirrors sub_spac_train_1-1.sh exactly
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE  = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_1-1.h5ad"
MODEL_FILE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/src/CONCERT-spac/outputs/raw_output_1-1/model_spac_1-1_v2.pt"

UNASSIGNED_LABELS  = frozenset(["unassigned", "Unassigned", "No_guide", "no_guide", "unknown"])
CONTROL_LABELS     = frozenset(["Non_targeting", "non_targeting", "NonTargeting",
                                 "control", "Control", "CTRL", "ctrl", "NT"])
EXCLUDED_LABELS    = frozenset(["multiple", "Multiple"])
MIN_CELLS_PER_PERT = 25
HVG_MAX_RANK       = 10000

# Architecture — must match training config exactly
GP_DIM      = 2
NORMAL_DIM  = 8
ENC_DIM     = 256
ENC_LAYERS  = [128, 64]
DEC_LAYERS  = [128]
NOISE       = 0.1
DROPOUT_E   = 0.1
DROPOUT_D   = 0.0
KL_LOSS     = 0.025
INIT_BETA   = 10.0
KERNEL_SCALE = 10.0
LOC_RANGE    = 20.0
IND_STEPS    = 6
MULTI_KERNEL = True
FIX_INDUCING = True
USE_PAD      = False
PAD_ACT_DIM  = 64
USE_ACTION_CORR = True
# Phase 2.7 flags — must match training config
USE_BASAL_ENCODER_ONLY = True
USE_F_PERT             = True
USE_CONSISTENCY_LOSS   = True
CONSISTENCY_LAMBDA     = 1.0
BATCH_SIZE_INFER = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",
                   default="/n/holylabs/mzitnik_lab/Users/xianglin226/CONCERT")
    p.add_argument("--model_file",  default=MODEL_FILE)
    p.add_argument("--data_file",   default=DATA_FILE)
    p.add_argument("--umap_seed",   type=int,   default=42)
    p.add_argument("--n_neighbors", type=int,   default=30)
    p.add_argument("--min_dist",    type=float, default=0.3)
    p.add_argument("--top_k_perts", type=int,   default=20)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (mirrors run_concert_spac.run() exactly)
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_file: str, hvg_max_rank: int = HVG_MAX_RANK,
              min_cells_per_pert: int = MIN_CELLS_PER_PERT,
              holdout_indices: "np.ndarray | None" = None,
              holdout_keep_only: bool = False,
              force_gene_order: "np.ndarray | None" = None):
    logging.info("Loading %s  (hvg_max_rank=%d, min_cells_per_pert=%d)",
                 data_file, hvg_max_rank, min_cells_per_pert)
    X, pos_raw, tissue_raw, perturb_raw, has_guide, is_real_KO, gene_names = load_h5_dataset(
        data_file,
        unassigned_labels=UNASSIGNED_LABELS,
        control_labels=CONTROL_LABELS,
        hvg_max_rank=hvg_max_rank,
        force_gene_order=force_gene_order,
    )

    # Filter 1: excluded labels ("multiple")
    keep = np.array([p not in EXCLUDED_LABELS for p in perturb_raw], dtype=bool)
    X, pos_raw, perturb_raw = X[keep], pos_raw[keep], perturb_raw[keep]
    has_guide, is_real_KO   = has_guide[keep], is_real_KO[keep]
    if tissue_raw is not None:
        tissue_raw = tissue_raw[keep]
    logging.info("After excluding 'multiple': %d cells", X.shape[0])

    # Holdout filter (applied here, before rare-pert filter, matching training pipeline order).
    # holdout_indices are relative to this post-multiple-exclusion array.
    if holdout_indices is not None:
        hold_mask = np.zeros(len(X), dtype=bool)
        valid = holdout_indices[holdout_indices < len(X)]
        hold_mask[valid] = True
        # keep_only=True  → keep ONLY held-out cells (the unseen eval source set)
        # keep_only=False → remove held-out cells (training/seen set, default)
        keep_ho = hold_mask if holdout_keep_only else ~hold_mask
        X, pos_raw, perturb_raw = X[keep_ho], pos_raw[keep_ho], perturb_raw[keep_ho]
        has_guide, is_real_KO   = has_guide[keep_ho], is_real_KO[keep_ho]
        if tissue_raw is not None:
            tissue_raw = tissue_raw[keep_ho]
        logging.info("[holdout] %s %d held-out cells → %d remaining",
                     "Kept ONLY" if holdout_keep_only else "Removed",
                     int(hold_mask.sum()), len(X))

    # Filter 2: rare perturbations
    pert_counts = Counter(perturb_raw.tolist())
    keep2 = np.array([pert_counts[p] >= min_cells_per_pert for p in perturb_raw], dtype=bool)
    X, pos_raw, perturb_raw = X[keep2], pos_raw[keep2], perturb_raw[keep2]
    has_guide, is_real_KO   = has_guide[keep2], is_real_KO[keep2]
    if tissue_raw is not None:
        tissue_raw = tissue_raw[keep2]
    logging.info("After rare-pert filter: %d cells, %d unique perts",
                 X.shape[0], len(set(perturb_raw.tolist())))

    # Attributes + batch one-hot (identical to run_concert_spac.run())
    tissue_idx, tissue_dict, perturb_idx, pert_map = build_attributes_spac(
        perturb_raw, tissue_raw=tissue_raw)
    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)
    pert_name = {code: name for name, code in pert_map.items()}

    # max(code)+1, NOT count-of-distinct: build_attributes_spac reserves code 0 for the
    # basal slot, which may hold zero cells on a given slide (then count-of-distinct would
    # be one short and the one-hot would go out of bounds).
    n_batch = int(perturb_idx.max()) + 1
    batch   = np.eye(n_batch, dtype=np.float32)[perturb_idx]

    scaler     = MinMaxScaler()
    pos_scaled = scaler.fit_transform(pos_raw) * LOC_RANGE
    cutoff     = np.full(pos_scaled.shape[0], 0.5, dtype=np.float32)
    pos_batched = np.concatenate([pos_scaled, batch], axis=1).astype(np.float32)

    # AnnData + normalize
    adata = sc.AnnData(X.astype("float32"))
    if gene_names is not None:
        adata.var_names = gene_names
    adata = normalize(adata, filter_min_counts=False, size_factors=True,
                      normalize_input=True, logtrans_input=True)

    # After normalize, companion arrays may need re-indexing if rows were dropped
    N_after = adata.n_obs
    if N_after < len(perturb_raw):
        survived = np.array(adata.obs_names.astype(int))
        pos_batched  = pos_batched[survived]
        cell_atts    = cell_atts[survived]
        batch        = batch[survived]
        cutoff       = cutoff[survived]
        perturb_raw  = perturb_raw[survived]
        has_guide    = has_guide[survived]
        is_real_KO   = is_real_KO[survived]
        perturb_idx  = perturb_idx[survived]
        logging.info("[normalize] Re-indexed to %d cells", N_after)

    cutoff_t = torch.tensor(cutoff, dtype=torch.float32)

    return (adata, pos_batched, cell_atts, batch, pert_name, cutoff_t,
            perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Model reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def build_model(adata, pos_batched, cell_atts, cutoff_t, n_batch,
                use_basal_encoder_only: bool = USE_BASAL_ENCODER_ONLY, **extra_kwargs):
    spatial_dims = pos_batched.shape[1] - n_batch
    kernel_scale = np.full((n_batch, spatial_dims), KERNEL_SCALE, dtype=np.float32)

    ind_steps = int(extra_kwargs.pop("ind_steps", IND_STEPS))
    inducing = build_inducing_points(
        pos_batched=pos_batched,
        n_batch=n_batch,
        steps=ind_steps,
        loc_range=LOC_RANGE,
        grid=True,
        k_clusters=None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Device: %s", device)

    model = CONCERT(
        cell_atts=cell_atts,
        num_genes=adata.n_vars,
        encoder_dim=ENC_DIM,
        GP_dim=GP_DIM,
        Normal_dim=NORMAL_DIM,
        n_batch=n_batch,
        encoder_layers=ENC_LAYERS,
        decoder_layers=DEC_LAYERS,
        noise=NOISE,
        encoder_dropout=DROPOUT_E,
        decoder_dropout=DROPOUT_D,
        shared_dispersion=False,
        fixed_inducing_points=FIX_INDUCING,
        initial_inducing_points=inducing,
        fixed_gp_params=False,
        kernel_scale=kernel_scale,
        multi_kernel_mode=MULTI_KERNEL,
        N_train=adata.n_obs,
        KL_loss=KL_LOSS,
        dynamicVAE=True,
        init_beta=INIT_BETA,
        min_beta=5.0,
        max_beta=25.0,
        mask_cutoff=cutoff_t,
        dtype=torch.float32,
        device=device,
        use_pad=USE_PAD,
        pad_action_dim=PAD_ACT_DIM,
        pad_lambda_mmd=0.1,
        pad_lambda_adv=0.01,
        use_action_correction=extra_kwargs.pop("use_action_correction", USE_ACTION_CORR),
        use_basal_encoder_only=use_basal_encoder_only,
        use_f_pert=USE_F_PERT,
        use_consistency_loss=USE_CONSISTENCY_LOSS,
        consistency_lambda=CONSISTENCY_LAMBDA,
        **extra_kwargs,
    )
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "unassigned": "#aaaaaa",
    "control":    "#2166ac",
    "gene_KO":    "#d73027",
}

def classify_perts(perturb_raw):
    out = []
    for p in perturb_raw:
        if p in UNASSIGNED_LABELS:
            out.append("unassigned")
        elif p in CONTROL_LABELS:
            out.append("control")
        else:
            out.append("gene_KO")
    return np.array(out)


def _scatter(ax, xy, mask, color, s=0.3, alpha=0.5, label=None):
    ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=s, alpha=alpha,
               linewidths=0, rasterized=True, label=label)


def plot_umap(umap_coords, pert_class, perturb_raw, top_k, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Panel A: 3-class ────────────────────────────────────────────────────
    ax = axes[0]
    # Plot in order so gene_KO is on top of unassigned
    for cls in ["unassigned", "control", "gene_KO"]:
        mask = pert_class == cls
        _scatter(ax, umap_coords, mask,
                 CLASS_COLORS[cls], s=0.3, alpha=0.4, label=f"{cls} ({mask.sum():,})")
    ax.set_title("Latent UMAP — perturbation class", fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, markerscale=5, loc="best", framealpha=0.8,
              handler_map={plt.Line2D: None})
    # Use Patch legend since scatter labels don't render well
    patches = [mpatches.Patch(color=CLASS_COLORS[c],
                               label=f"{c} ({(pert_class==c).sum():,})")
               for c in ["unassigned", "control", "gene_KO"]]
    ax.legend(handles=patches, fontsize=8, loc="best", framealpha=0.8)

    # ── Panel B: top-K gene KOs ─────────────────────────────────────────────
    ax = axes[1]
    gene_ko_perts = [p for p in perturb_raw if p not in UNASSIGNED_LABELS and p not in CONTROL_LABELS]
    top = [p for p, _ in Counter(gene_ko_perts).most_common(top_k)]
    cmap_colors = plt.cm.tab20(np.linspace(0, 1, len(top)))
    top_cmap = {p: to_hex(cmap_colors[i]) for i, p in enumerate(top)}

    # Background: other KOs + unassigned (grey)
    other_ko = np.array([p not in UNASSIGNED_LABELS and p not in CONTROL_LABELS
                         and p not in top_cmap for p in perturb_raw])
    _scatter(ax, umap_coords, other_ko,    "#e0e0e0", s=0.2, alpha=0.15)
    _scatter(ax, umap_coords, pert_class=="unassigned", CLASS_COLORS["unassigned"], s=0.3, alpha=0.3)
    _scatter(ax, umap_coords, pert_class=="control",    CLASS_COLORS["control"],    s=0.5, alpha=0.5)
    # Top KOs on top
    for p in top:
        mask = np.array([pp == p for pp in perturb_raw])
        _scatter(ax, umap_coords, mask, top_cmap[p], s=1.0, alpha=0.8)

    ax.set_title(f"Latent UMAP — top {top_k} gene KOs", fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)

    legend_h = [mpatches.Patch(color=CLASS_COLORS["unassigned"], label="unassigned"),
                mpatches.Patch(color=CLASS_COLORS["control"],    label="control"),
                mpatches.Patch(color="#e0e0e0",                  label="other KO")]
    legend_h += [mpatches.Patch(color=top_cmap[p], label=p) for p in top]
    ax.legend(handles=legend_h, fontsize=5.5, ncol=2, loc="upper right", framealpha=0.8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure → %s", outpath)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load & prepare data
    (adata, pos_batched, cell_atts, batch, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(args.data_file)

    # 2) Reconstruct model + load checkpoint
    model, device = build_model(adata, pos_batched, cell_atts, cutoff_t, n_batch)
    logging.info("Loading checkpoint: %s", args.model_file)
    model.load_model(args.model_file)
    model.eval()

    # 2b) Inspect f_pert: compute per-pert LFC via LORD "perturbation" embedding → f_pert
    # LORD attribute key is "perturbation". f_pert.weight is (num_genes, encoder_dim)
    # in PyTorch convention, so per-pert LFC = f_pert(pert_emb) not weight row norms.
    if USE_F_PERT and hasattr(model, "f_pert") and model.f_pert is not None:
        with torch.no_grad():
            all_codes = torch.arange(n_batch, dtype=torch.long, device=device)
            pert_emb  = model.lord_encoder.s_encoder["perturbation"](all_codes)  # (n_batch, enc_dim)
            lfc_mat   = model.f_pert(pert_emb).cpu().float().numpy()             # (n_batch, num_genes)
        lfc_norm = np.linalg.norm(lfc_mat, axis=1)             # (n_batch,) per-pert magnitude
        top_idx  = np.argsort(lfc_norm)[::-1][:15]
        logging.info("f_pert per-pert LFC norms (top 15 by ||LFC||):")
        for i in top_idx:
            name = pert_name.get(i, f"code_{i}")
            logging.info("  %-20s  ||lfc||=%.4f", name, lfc_norm[i])
        np.save(outdir / "f_pert_lfc_matrix.npy", lfc_mat)
        np.save(outdir / "f_pert_weights.npy",
                model.f_pert.weight.detach().cpu().float().numpy())
        logging.info("Saved f_pert LFC matrix → %s/f_pert_lfc_matrix.npy", outdir)

    # 3) Extract latent z
    N = adata.n_obs
    logging.info("Extracting latent z for %d cells (batch_size=%d)...", N, BATCH_SIZE_INFER)
    sample_indices = torch.arange(N, dtype=torch.int)
    Z = model.batching_latent_samples(
        pos_batched, sample_indices, cell_atts, batch_size=BATCH_SIZE_INFER
    )
    logging.info("Latent z: %s  (GP dims 0:2, Normal dims 2:10)", Z.shape)

    # 4) UMAP on latent z (10 dims — no PCA needed)
    logging.info("Computing UMAP (n_neighbors=%d, min_dist=%.2f, seed=%d)...",
                 args.n_neighbors, args.min_dist, args.umap_seed)
    adata_z = ad.AnnData(X=Z.astype("float32"))
    adata_z.obsm["X_concert"] = Z.astype("float32")
    sc.pp.neighbors(adata_z, use_rep="X_concert",
                    n_neighbors=args.n_neighbors, random_state=args.umap_seed)
    sc.tl.umap(adata_z, min_dist=args.min_dist, random_state=args.umap_seed)
    umap_coords = adata_z.obsm["X_umap"]

    # Save only the 2D coords + pert labels — skip the bulky h5ad
    np.savez_compressed(
        outdir / "umap_coords.npz",
        umap=umap_coords.astype("float32"),
        perturb=perturb_raw,
    )
    logging.info("Saved UMAP coords → %s/umap_coords.npz", outdir)

    # 5) Plot
    pert_class = np.array(classify_perts(perturb_raw))
    plot_umap(umap_coords, pert_class, perturb_raw, args.top_k_perts,
              outdir / "umap_concert_latent.pdf")

    # 6) Health report
    n_una = (pert_class == "unassigned").sum()
    n_ctrl = (pert_class == "control").sum()
    n_ko   = (pert_class == "gene_KO").sum()
    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("  Cells in UMAP   : %d", N)
    logging.info("  unassigned      : %d (%.1f%%)", n_una,  100*n_una/N)
    logging.info("  control         : %d (%.1f%%)", n_ctrl, 100*n_ctrl/N)
    logging.info("  gene_KO         : %d (%.1f%%)", n_ko,   100*n_ko/N)
    logging.info("  Unique KO genes : %d", len(set(perturb_raw[pert_class=="gene_KO"].tolist())))
    logging.info("  Outputs in      : %s", outdir)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
