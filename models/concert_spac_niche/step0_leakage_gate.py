#!/usr/bin/env python
"""
Step-0 leakage gate (spatial-basal disentanglement design, sec 3 Step-0).

Question: does the basal latent `z` already carry the KO effect?
Method: reconstruct has-guide cells with f_pert AND mu_action OFF (force_basal=True),
then compute the BASAL-ONLY per-niche KO-vs-NTC LFC and correlate it with the OBSERVED
per-niche KO-vs-NTC LFC. If they correlate, `z` leaks the perturbation -> Change A justified.

Reuses make_umap.load_data + build_model (cell-index alignment gotcha: idx is GLOBAL,
mask_cutoff is sliced by row -- replicated exactly from decompose_levels.py).
"""
import argparse, logging, json, os
import numpy as np
import torch
from scipy.stats import spearmanr

from make_umap import load_data, build_model as build_umap_model

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

DATA_FILE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_1-1.h5ad"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_file", required=True)
    p.add_argument("--data_file", default=DATA_FILE)
    p.add_argument("--outdir", required=True)
    p.add_argument("--niche_grid", type=int, default=6)
    p.add_argument("--loc_range", type=float, default=20.0)
    p.add_argument("--min_g_cells", type=int, default=20, help="min KO cells of a pert in a niche")
    p.add_argument("--ntc_floor", type=int, default=30, help="min NTC cells in a niche")
    p.add_argument("--hvg_max_rank", type=int, default=2000)
    p.add_argument("--min_cells_per_pert", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    # model arch flags (MUST match v7fa training/eval)
    p.add_argument("--use_expr_encoder", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--expr_hidden_dims", nargs="+", type=int, default=[512, 256])
    p.add_argument("--use_basal_encoder_only", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--use_nngp", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--nngp_lambda", type=float, default=0.0)
    p.add_argument("--use_paracrine", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--use_niche_interaction", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--niche_rank", type=int, default=8)
    p.add_argument("--use_pert_gate", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--gate_hidden", type=int, default=64)
    p.add_argument("--spatial_basal_only", type=lambda x: x.lower() == "true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (adata, pos_batched, cell_atts, batch, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(
        args.data_file, hvg_max_rank=args.hvg_max_rank,
        min_cells_per_pert=args.min_cells_per_pert)
    N = adata.n_obs
    spatial_dims = pos_batched.shape[1] - n_batch

    # capture normalized expression for the expr encoder + as observed counts
    Yexpr = adata.X
    import scipy.sparse as sp
    if sp.issparse(Yexpr):
        Yexpr = Yexpr.toarray()
    Yexpr = np.asarray(Yexpr, dtype=np.float32)
    coords = np.asarray(pos_batched[:, :spatial_dims], dtype=np.float32)

    extra = dict(use_expr_encoder=args.use_expr_encoder, expr_hidden_dims=args.expr_hidden_dims,
                 use_nngp=args.use_nngp, nngp_lambda=args.nngp_lambda,
                 use_paracrine=args.use_paracrine, use_niche_interaction=args.use_niche_interaction,
                 niche_rank=args.niche_rank, use_pert_gate=args.use_pert_gate,
                 gate_hidden=args.gate_hidden, spatial_basal_only=args.spatial_basal_only)
    model, device = build_umap_model(adata, pos_batched, cell_atts, cutoff_t, n_batch,
                                     use_basal_encoder_only=args.use_basal_encoder_only, **extra)
    model.load_model(args.model_file)
    model.eval()
    del adata

    # restrict to has-guide cells (NTC + KO); the LFC set
    sub = np.where(has_guide)[0]
    sub.sort()
    logging.info("has-guide cells (NTC+KO): %d", len(sub))
    pos_b_sub = pos_batched[sub]
    atts_sub = cell_atts[sub]
    idx_sub = torch.arange(N)[sub]      # GLOBAL index (LORD)
    cut_sub = cutoff_t[sub]
    model.mask_cutoff = torch.nn.Parameter(cut_sub.to(device), requires_grad=False)

    logging.info("Computing BASAL-ONLY mu (force_basal=True; f_pert + mu_action OFF)...")
    mu_basal, _ = model.batching_denoise_counts(
        pos_b_sub, idx_sub, atts_sub, n_samples=1, batch_size=args.batch_size,
        force_basal=True, Y=(Yexpr[sub] if args.use_expr_encoder else None))
    mu_basal = np.asarray(mu_basal, dtype=np.float32)        # (n_sub, G)
    obs = Yexpr[sub]                                          # (n_sub, G) observed normalized
    logging.info("mu_basal non-finite frac=%.4g  obs non-finite frac=%.4g",
                 float(np.mean(~np.isfinite(mu_basal))), float(np.mean(~np.isfinite(obs))))

    rk = is_real_KO[sub].astype(bool)
    hg = has_guide[sub].astype(bool)
    is_ntc = hg & (~rk)
    pert = perturb_raw[sub]
    cc = coords[sub]

    # niche grid
    step = args.loc_range / args.niche_grid
    bx = np.clip((cc[:, 0] / step).astype(int), 0, args.niche_grid - 1)
    by = np.clip((cc[:, 1] / step).astype(int), 0, args.niche_grid - 1)
    niche = bx * args.niche_grid + by

    eps = 1e-6
    def lfc(mat, ko_mask, ntc_mask):
        return np.log(mat[ko_mask].mean(0) + eps) - np.log(mat[ntc_mask].mean(0) + eps)

    per_pb = []        # (pert, niche, spearman_basal_vs_obs, n_ko, n_ntc)
    basal_stack, obs_stack = [], []
    ko_perts = sorted(set(pert[rk].tolist()))
    for b in range(args.niche_grid ** 2):
        in_b = niche == b
        ntc_b = in_b & is_ntc
        if ntc_b.sum() < args.ntc_floor:
            continue
        for p in ko_perts:
            ko_b = in_b & rk & (pert == p)
            if ko_b.sum() < args.min_g_cells:
                continue
            bl = lfc(mu_basal, ko_b, ntc_b)
            ol = lfc(obs, ko_b, ntc_b)
            m = np.isfinite(bl) & np.isfinite(ol)
            if m.sum() < 10 or np.std(bl[m]) == 0 or np.std(ol[m]) == 0:
                continue
            bl, ol = bl[m], ol[m]
            rho = spearmanr(bl, ol).correlation
            per_pb.append((str(p), int(b), float(rho), int(ko_b.sum()), int(ntc_b.sum())))
            basal_stack.append(bl); obs_stack.append(ol)

    rhos = np.array([r[2] for r in per_pb], dtype=float)
    pooled = spearmanr(np.concatenate(basal_stack), np.concatenate(obs_stack)).correlation \
        if basal_stack else float("nan")
    summary = dict(
        n_pairs=len(per_pb),
        mean_spearman=float(np.nanmean(rhos)) if len(rhos) else float("nan"),
        median_spearman=float(np.nanmedian(rhos)) if len(rhos) else float("nan"),
        frac_positive=float(np.mean(rhos > 0)) if len(rhos) else float("nan"),
        pooled_spearman=float(pooled),
        model_file=args.model_file,
    )
    logging.info("RESULT: %s", json.dumps(summary, indent=2))
    leak = (summary["mean_spearman"] is not None and summary["mean_spearman"] > 0.2)
    logging.info("LEAKAGE %s (mean basal-vs-obs LFC Spearman = %.3f; >0.2 => z leaks KO)",
                 "PROVEN" if leak else "NOT shown", summary["mean_spearman"])
    with open(os.path.join(args.outdir, "step0_leakage_summary.json"), "w") as f:
        json.dump({"summary": summary, "per_pair": per_pb}, f, indent=2)
    logging.info("wrote %s/step0_leakage_summary.json", args.outdir)


if __name__ == "__main__":
    main()
