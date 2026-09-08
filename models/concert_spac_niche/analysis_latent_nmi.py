#!/usr/bin/env python3
"""Part 1 -- spatial autocorrelation of CONCERT v6 latent spaces (slide 1-1).

Extract every latent representation the v6 model exposes for all cells, cluster
each (KMeans, k = #spatial bins = 36), and measure how strongly each clustering
recovers the cell's physical 6x6 grid bin (NMI / ARI). High NMI ⇒ that latent
space is organized by physical location (spatial autocorrelation).

Latent spaces evaluated:
  - z_full   : 10-D posterior-mean latent  [GP spatial(2) | Normal cell-state(8)]
  - z_gp     : 2-D GP latent           (expected HIGH -- explicitly spatial; positive ctrl)
  - z_normal : 8-D Normal cell-state latent
  - basal    : 256-D amortized basal latent (expression encoder; pert-blind)
  - pert_emb : 256-D LORD perturbation embedding per cell (the "learned perturbation space")

Reference: NMI(raw perturbation identity, bin) -- does KO identity itself colocalize?

No model retraining. Mirrors decompose_levels.py for data/model alignment.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from make_umap import load_data
from concert_spac import CONCERT
from run_concert_spac import build_inducing_points
from eval_utils import build_niche_grid

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    normalized_mutual_info_score as nmi,
    adjusted_rand_score as ari,
    adjusted_mutual_info_score as ami,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_FILE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_1-1.h5ad"
MODEL_FILE = "/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT/outputs/raw_output_1-1/model_spac_1-1_v6.pt"

# v6 architecture (from logs/spac_train_1-1_v6_19076996.err Config dump)
GP_DIM, NORMAL_DIM, ENC_DIM = 2, 8, 256
ENC_LAYERS, DEC_LAYERS = [128, 64], [128]
NOISE, DROPOUT_E, DROPOUT_D = 0.1, 0.1, 0.0
KL_LOSS, KERNEL_SCALE, LOC_RANGE = 0.025, 10.0, 20.0
IND_STEPS = 6
HVG_MAX_RANK, MIN_CELLS = 2000, 25
BATCH_INF = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", default=DATA_FILE)
    p.add_argument("--model_file", default=MODEL_FILE)
    p.add_argument("--outdir", default=str(SCRIPT_DIR / "outputs" / "latent_nmi_v6_1-1"))
    p.add_argument("--grid", type=int, default=6, help="grid per axis; k = grid^2 = 36")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_v6_model(adata, pos_batched, cell_atts, cutoff_t, n_batch):
    """Construct CONCERT with the exact v6 flag set (use_f_pert=False)."""
    spatial_dims = pos_batched.shape[1] - n_batch
    kernel_scale = np.full((n_batch, spatial_dims), KERNEL_SCALE, dtype=np.float32)
    inducing = build_inducing_points(pos_batched=pos_batched, n_batch=n_batch,
                                     steps=IND_STEPS, loc_range=LOC_RANGE, grid=True, k_clusters=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CONCERT(
        cell_atts=cell_atts, num_genes=adata.n_vars, encoder_dim=ENC_DIM,
        GP_dim=GP_DIM, Normal_dim=NORMAL_DIM, n_batch=n_batch,
        encoder_layers=ENC_LAYERS, decoder_layers=DEC_LAYERS, noise=NOISE,
        encoder_dropout=DROPOUT_E, decoder_dropout=DROPOUT_D, shared_dispersion=False,
        fixed_inducing_points=True, initial_inducing_points=inducing, fixed_gp_params=False,
        kernel_scale=kernel_scale, multi_kernel_mode=True, N_train=adata.n_obs, KL_loss=KL_LOSS,
        dynamicVAE=False, init_beta=1.0, min_beta=5.0, max_beta=25.0, mask_cutoff=cutoff_t,
        dtype=torch.float32, device=device, use_pad=False, pad_action_dim=64,
        pad_lambda_mmd=0.1, pad_lambda_adv=0.01, use_action_correction=True,
        use_basal_encoder_only=True, use_f_pert=False,
        use_consistency_loss=True, consistency_lambda=1.0,
        use_expr_encoder=True, expr_hidden_dims=[512, 256],
        use_niche_interaction=True, niche_rank=8,
        use_pert_gate=True, gate_hidden=64,
    )
    return model, device


def cluster_nmi(name, feats, bin_id, k, seed):
    """Standardize -> MiniBatchKMeans(k) -> NMI/AMI/ARI vs spatial bins."""
    Xs = StandardScaler().fit_transform(np.asarray(feats, dtype=np.float64))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                         batch_size=4096, max_iter=300)
    lab = km.fit_predict(Xs)
    res = dict(space=name, dim=int(Xs.shape[1]),
               nmi=float(nmi(bin_id, lab)),
               ami=float(ami(bin_id, lab)),
               ari=float(ari(bin_id, lab)),
               n_clusters_used=int(len(np.unique(lab))))
    log.info("[%-9s dim=%-4d] NMI=%.4f  AMI=%.4f  ARI=%.4f",
             name, res["dim"], res["nmi"], res["ami"], res["ari"])
    return res, lab


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    k = args.grid * args.grid

    log.info("Loading data (hvg_max_rank=%d, min_cells_per_pert=%d)...", HVG_MAX_RANK, MIN_CELLS)
    (adata, pos_batched, cell_atts, batch_oh, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(
        args.data_file, hvg_max_rank=HVG_MAX_RANK, min_cells_per_pert=MIN_CELLS)
    N = adata.n_obs
    pos_scaled = pos_batched[:, :2]
    log.info("Cells=%d  perts=%d", N, n_batch)

    # 36 spatial bins
    bin_id, n_bins = build_niche_grid(pos_scaled, grid=args.grid, loc_range=LOC_RANGE)
    log.info("Spatial bins: k=%d, occupied=%d", n_bins, len(np.unique(bin_id)))

    # expression matrix for the amortized basal encoder
    Yexpr = adata.X
    if sp.issparse(Yexpr):
        Yexpr = Yexpr.toarray()
    Yexpr = np.asarray(Yexpr, dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Building v6 model on %s...", device)
    model, device = build_v6_model(adata, pos_batched, cell_atts, cutoff_t, n_batch)
    model.load_model(args.model_file)
    model.eval()

    # ---- z_full (10-D posterior mean): GP(2) + Normal(8) ----
    log.info("Extracting 10-D latent z for all cells...")
    Z = model.batching_latent_samples(pos_batched, torch.arange(N), cell_atts,
                                       batch_size=BATCH_INF, Y=Yexpr)
    z_gp, z_normal = Z[:, :GP_DIM], Z[:, GP_DIM:]

    # ---- basal (256-D amortized) ----
    log.info("Extracting 256-D basal latent (expression encoder)...")
    basal = np.empty((N, ENC_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, BATCH_INF):
            e = min(s + BATCH_INF, N)
            xb = torch.as_tensor(Yexpr[s:e], dtype=torch.float32, device=device)
            basal[s:e] = model.lord_encoder.basal_encoder(xb).cpu().numpy()

    # ---- per-cell perturbation embedding (256-D LORD lookup) ----
    log.info("Extracting 256-D perturbation embedding per cell...")
    with torch.no_grad():
        codes = torch.arange(n_batch, dtype=torch.long, device=device)
        pert_table = model.lord_encoder.s_encoder["perturbation"](codes).cpu().numpy()
    pert_emb = pert_table[perturb_idx]

    # ---- cluster each space and score vs spatial bins ----
    log.info("=== KMeans (k=%d) + NMI vs spatial bins ===", k)
    spaces = {"z_full": Z, "z_gp": z_gp, "z_normal": z_normal,
              "basal": basal, "pert_emb": pert_emb}
    results, labels = [], {}
    for nm, feats in spaces.items():
        res, lab = cluster_nmi(nm, feats, bin_id, k, args.seed)
        results.append(res); labels[nm] = lab

    # reference: raw perturbation identity vs location (no clustering)
    ref = dict(space="pert_identity(raw)", dim=1,
               nmi=float(nmi(bin_id, perturb_idx)),
               ami=float(ami(bin_id, perturb_idx)),
               ari=float(ari(bin_id, perturb_idx)),
               n_clusters_used=int(n_batch))
    log.info("[%-9s] NMI=%.4f  AMI=%.4f  ARI=%.4f (raw KO identity vs bin)",
             "pert_id", ref["nmi"], ref["ami"], ref["ari"])
    results.append(ref)

    # random-label NMI floor (k random clusters)
    rng = np.random.default_rng(args.seed)
    rand_lab = rng.integers(0, k, size=N)
    floor = dict(space="random_floor", dim=0,
                 nmi=float(nmi(bin_id, rand_lab)),
                 ami=float(ami(bin_id, rand_lab)),
                 ari=float(ari(bin_id, rand_lab)),
                 n_clusters_used=k)
    results.append(floor)
    log.info("[random   ] NMI=%.4f (floor)", floor["nmi"])

    # ---- save ----
    import csv
    with (outdir / "latent_nmi.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["space", "dim", "nmi", "ami", "ari", "n_clusters_used"])
        w.writeheader(); w.writerows(results)
    json.dump({"k": k, "n_bins": n_bins, "N": int(N), "n_perts": int(n_batch),
               "results": results}, (outdir / "latent_nmi.json").open("w"), indent=2)
    np.savez_compressed(outdir / "cluster_labels.npz",
                        bin_id=bin_id, perturb_idx=perturb_idx, **labels)

    # ---- bar plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["z_gp", "z_full", "z_normal", "basal", "pert_emb", "pert_identity(raw)", "random_floor"]
    rmap = {r["space"]: r for r in results}
    names = [o for o in order if o in rmap]
    vals = [rmap[o]["nmi"] for o in names]
    colors = ["#1a9850" if o == "z_gp" else "#999999" if o == "random_floor"
              else "#4575b4" if o == "pert_emb" else "#d73027" if o == "pert_identity(raw)"
              else "#fc8d59" for o in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(names)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"NMI vs {k} spatial bins")
    ax.set_title("CONCERT v6 (slide 1-1): spatial autocorrelation of latent spaces\n"
                 "higher = latent organized by physical location")
    fig.tight_layout()
    fig.savefig(outdir / "latent_nmi.png", dpi=150)
    log.info("Wrote %s/latent_nmi.{csv,json,png}", outdir)
    log.info("DONE")


if __name__ == "__main__":
    main()
