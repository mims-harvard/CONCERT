#!/usr/bin/env python3
"""
Spatial footprint diagnostics for CONCERT-spac Phase 2.7 model.

Answers: does each perturbation learn a different spatial footprint?
  - per-pert kernel scale (learned Cauchy spatial radius)
  - per-pert f_pert LFC magnitude
  - tissue maps of where each KO's cells sit

Uses make_umap.load_data (matches training pipeline) so LORD's cell-indexed
basal table aligns with the checkpoint (off-by-one normalize drop matters).

Outputs (all in --outdir):
  kernel_scales.npz      -- (n_pert, 2) learned Cauchy scales + pert name array
  f_pert_lfc_matrix.npy  -- (n_pert, n_genes) per-pert gene-space LFC
  f_pert_pert_names.npy  -- pert name for each row
  spatial_footprint.pdf  -- 5-panel diagnostic figure
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from make_umap import load_data, build_model as build_umap_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

# --- Config ---
DATA_FILE  = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_1-1.h5ad"
MODEL_FILE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/src/CONCERT-spac/outputs/raw_output_1-1/model_spac_1-1_v2.pt"
OUTDIR     = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/analysis/umap_v2"

UNASSIGNED = frozenset(["unassigned","Unassigned","No_guide","no_guide","unknown"])
CONTROL    = frozenset(["Non_targeting","non_targeting","NonTargeting","control","Control","CTRL","ctrl","NT"])
EXCLUDED   = frozenset(["multiple","Multiple"])
MIN_CELLS  = 25
LOC_RANGE  = 20.0
IND_STEPS  = 6
ENC_DIM    = 256
N_GENES_PLACEHOLDER = 9998   # only needed for model constructor; not used here


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",     default=OUTDIR)
    p.add_argument("--data_file",  default=DATA_FILE)
    p.add_argument("--model_file", default=MODEL_FILE)
    p.add_argument("--top_k",      type=int, default=20)
    return p.parse_args()


# --- Extract diagnostics ---
def get_kernel_scales(model):
    """(n_batch, 2) learned Cauchy spatial scales."""
    return model.svgp.kernel.scale.detach().cpu().float().numpy().clip(1e-6, 1e6)


def get_fpert_lfc(model, n_batch):
    """(n_batch, n_genes) per-pert gene-space LFC via LORD pert embedding -> f_pert."""
    with torch.no_grad():
        device   = model.lord_encoder.s_encoder["perturbation"].weight.device
        codes    = torch.arange(n_batch, dtype=torch.long, device=device)
        pert_emb = model.lord_encoder.s_encoder["perturbation"](codes)
        lfc_mat  = model.f_pert(pert_emb).detach().cpu().float().numpy()
    return lfc_mat


def get_mu_action(model):
    """Per-gene global guide-delivery shift."""
    if hasattr(model, "mu_action") and model.mu_action is not None:
        return model.mu_action.detach().cpu().float().numpy()
    return None


# --- Plot helpers ---
def _cmap(n):
    return [to_hex(plt.cm.tab20(i % 20)) for i in range(n)]


def plot_scale_scatter(ax, scales, pert_names, top_idx):
    mag = np.linalg.norm(scales, axis=1)
    ax.scatter(scales[:, 0], scales[:, 1], c="#cccccc", s=8, alpha=0.5, linewidths=0, zorder=1)
    colors = _cmap(len(top_idx))
    patches = []
    for i, c in zip(top_idx, colors):
        ax.scatter(scales[i, 0], scales[i, 1], c=c, s=40, zorder=3, linewidths=0)
        ax.annotate(pert_names[i], (scales[i, 0], scales[i, 1]),
                    fontsize=4.5, xytext=(2, 2), textcoords="offset points")
        patches.append(mpatches.Patch(color=c, label=f"{pert_names[i]} ({scales[i,0]:.2f},{scales[i,1]:.2f})"))
    ax.set_xlabel("Scale X", fontsize=8); ax.set_ylabel("Scale Y", fontsize=8)
    ax.set_title("Per-pert kernel scale (spatial correlation radius)", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(handles=patches, fontsize=4, ncol=2, loc="upper left",
              framealpha=0.8, title="top by ||scale||", title_fontsize=4)


def plot_scale_hist(ax, scales):
    ax.hist(scales[:, 0], bins=40, alpha=0.6, label="X axis", color="#2166ac")
    ax.hist(scales[:, 1], bins=40, alpha=0.6, label="Y axis", color="#d73027")
    ax.set_xlabel("Scale value", fontsize=8); ax.set_ylabel("# perts", fontsize=8)
    ax.set_title("Scale distribution\n(diversity = perts differ spatially)", fontsize=8)
    ax.legend(fontsize=7); ax.tick_params(labelsize=6)


def plot_scale_vs_lfc(ax, scale_mag, lfc_norm, pert_names, top_lfc_idx):
    ax.scatter(scale_mag, lfc_norm, c="#888888", s=5, alpha=0.4, linewidths=0)
    for i in top_lfc_idx[:10]:
        ax.scatter(scale_mag[i], lfc_norm[i], c="#d73027", s=25, zorder=3)
        ax.annotate(pert_names[i], (scale_mag[i], lfc_norm[i]),
                    fontsize=4.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Kernel scale ||s||", fontsize=8)
    ax.set_ylabel("LFC ||w||", fontsize=8)
    ax.set_title("Spatial spread vs gene effect size\n(independence = signal not confounded)", fontsize=8)
    ax.tick_params(labelsize=6)


def plot_spatial_cells(ax, pos_scaled, perturb_idx, pert_names, top_codes, title):
    is_top = np.isin(perturb_idx, top_codes)
    ax.scatter(pos_scaled[~is_top, 0], pos_scaled[~is_top, 1],
               c="#e8e8e8", s=0.05, alpha=0.15, linewidths=0, rasterized=True)
    colors = _cmap(len(top_codes))
    patches = []
    for code, c in zip(top_codes, colors):
        mask = perturb_idx == code
        ax.scatter(pos_scaled[mask, 0], pos_scaled[mask, 1],
                   c=c, s=0.4, alpha=0.7, linewidths=0, rasterized=True)
        patches.append(mpatches.Patch(color=c,
                       label=f"{pert_names.get(code,'?')} (n={mask.sum():,})"))
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(handles=patches, fontsize=4, ncol=2, loc="best",
              framealpha=0.8, markerscale=6)
    ax.set_aspect("equal")


# --- Main ---
def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    top_k  = args.top_k

    # 1) Load via make_umap.load_data -- matches training pipeline (alignment safe)
    logging.info("Loading data via make_umap.load_data...")
    (adata, pos_batched, cell_atts, batch_oh, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(args.data_file)
    pos_scaled     = pos_batched[:, :2]
    pert_names_arr = np.array([pert_name.get(i, f"code_{i}") for i in range(n_batch)])
    logging.info("Cells: %d  perts: %d", adata.n_obs, n_batch)

    logging.info("Building model...")
    model, _ = build_umap_model(adata, pos_batched, cell_atts, cutoff_t, n_batch)
    model.load_model(args.model_file)
    model.eval()
    del adata
    import gc; gc.collect()

    # 2) Kernel scales
    scales    = get_kernel_scales(model)           # (n_batch, 2)
    scale_mag = np.linalg.norm(scales, axis=1)

    np.savez_compressed(outdir / "kernel_scales.npz",
                        scales=scales, pert_names=pert_names_arr)
    logging.info("Saved kernel_scales.npz  shape=%s", scales.shape)

    top_scale_idx = np.argsort(scale_mag)[::-1][:top_k]
    bot_scale_idx = np.argsort(scale_mag)[:5]
    logging.info("Top %d perts by kernel scale (diffuse):", top_k)
    for i in top_scale_idx:
        logging.info("  %-25s  (%.3f, %.3f)  ||s||=%.3f",
                     pert_names_arr[i], scales[i,0], scales[i,1], scale_mag[i])
    logging.info("Bottom 5 perts (cell-autonomous):")
    for i in bot_scale_idx:
        logging.info("  %-25s  (%.3f, %.3f)  ||s||=%.3f",
                     pert_names_arr[i], scales[i,0], scales[i,1], scale_mag[i])
    scale_std = scales.std(axis=0)
    logging.info("Scale std X=%.4f  Y=%.4f  (>0 = perts have different spatial footprints)",
                 scale_std[0], scale_std[1])

    # 3) f_pert LFC
    lfc_mat  = get_fpert_lfc(model, n_batch)      # (n_batch, n_genes)
    lfc_norm = np.linalg.norm(lfc_mat, axis=1)

    np.save(outdir / "f_pert_lfc_matrix.npy",  lfc_mat)
    np.save(outdir / "f_pert_pert_names.npy",  pert_names_arr)
    logging.info("Saved f_pert_lfc_matrix.npy  shape=%s", lfc_mat.shape)

    top_lfc_idx = np.argsort(lfc_norm)[::-1][:top_k]
    logging.info("Top %d perts by ||LFC||:", top_k)
    for i in top_lfc_idx:
        logging.info("  %-25s  ||lfc||=%.4f", pert_names_arr[i], lfc_norm[i])

    # 4) mu_action
    mu_action = get_mu_action(model)
    if mu_action is not None:
        logging.info("mu_action: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                     mu_action.mean(), mu_action.std(), mu_action.min(), mu_action.max())

    # 5) Spatial maps -- KO cells only
    ko_codes = set(perturb_idx[
        np.array([p not in UNASSIGNED and p not in CONTROL for p in perturb_raw])
    ].tolist())
    top_lfc_ko   = [i for i in top_lfc_idx   if i in ko_codes][:top_k]
    top_scale_ko = [i for i in top_scale_idx  if i in ko_codes][:top_k]

    # 6) Figure
    fig = plt.figure(figsize=(22, 14))
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    plot_scale_scatter(fig.add_subplot(gs[0, 0]), scales, pert_names_arr, top_scale_idx[:top_k])
    plot_scale_hist(fig.add_subplot(gs[0, 1]), scales)
    plot_scale_vs_lfc(fig.add_subplot(gs[0, 2]), scale_mag, lfc_norm, pert_names_arr, top_lfc_idx)
    plot_spatial_cells(
        fig.add_subplot(gs[1, 0:2]), pos_scaled, perturb_idx, pert_name, top_lfc_ko,
        f"Tissue positions -- top-{len(top_lfc_ko)} KOs by gene-space LFC\n"
        "(spatial clustering = pert enriched in tissue region)")
    plot_spatial_cells(
        fig.add_subplot(gs[1, 2]), pos_scaled, perturb_idx, pert_name, top_scale_ko,
        f"Tissue positions -- top-{len(top_scale_ko)} KOs by kernel scale\n"
        "(largest spatial correlation radius)")

    fig.suptitle("CONCERT-spac Phase 2.7 -- Per-perturbation spatial footprint",
                 fontsize=11, y=1.01)
    out_pdf = outdir / "spatial_footprint.pdf"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved -> %s", out_pdf)

    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("  n_perts        : %d", n_batch)
    logging.info("  scale range X  : [%.4f, %.4f]", scales[:,0].min(), scales[:,0].max())
    logging.info("  scale range Y  : [%.4f, %.4f]", scales[:,1].min(), scales[:,1].max())
    logging.info("  scale std X/Y  : %.4f / %.4f", scale_std[0], scale_std[1])
    logging.info("  LFC norm range : [%.4f, %.4f]", lfc_norm.min(), lfc_norm.max())
    logging.info("  LFC norm std   : %.4f", lfc_norm.std())
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
