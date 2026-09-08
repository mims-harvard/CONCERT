#!/usr/bin/env python3
"""Comprehensive one-shot diagnostic for why v5a imputation/counterfactual is weak.

Measures every competing hypothesis in ONE pass (no back-and-forth tuning):
  H1 decoder output collapse   -> per-gene std of dec_mean ~ 0
  H2 metric-axis artifact      -> per-cell & pseudobulk corr HIGH while per-gene LOW
  H4 force_basal strips signal -> full-forward >> force_basal
  H5 latent collapse (encoder) -> latent std per dim ~ 0
  H6 sparsity tie-domination   -> high zero-fraction; top-expressed-gene recompute
Reuses eval_spatial_counterfactual.load_eval_context (same loader/flags as v5a).
"""
from __future__ import annotations
import argparse
import logging
from types import SimpleNamespace

import numpy as np
import torch

import eval_utils as eu
import eval_spatial_counterfactual as E

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
DATA_FILE = E.DATA_FILE


def _bool(x):
    return str(x).lower() == "true"


def per_cell_spearman_mean(pred, obs):
    """Mean over cells of per-cell Spearman across genes (does the model rank genes
    correctly WITHIN a cell). Rank along axis=1."""
    pred = np.asarray(pred, np.float64)
    obs = np.asarray(obs, np.float64)
    rp = pred.argsort(axis=1).argsort(axis=1).astype(np.float64)
    ro = obs.argsort(axis=1).argsort(axis=1).astype(np.float64)
    rp -= rp.mean(axis=1, keepdims=True)
    ro -= ro.mean(axis=1, keepdims=True)
    num = (rp * ro).sum(axis=1)
    den = np.sqrt((rp ** 2).sum(axis=1) * (ro ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = num / den
    return float(np.nanmean(c))


def pseudobulk_corr(pred, obs):
    """Correlation of the mean-expression PROFILE across genes (does the model get the
    average right, ignoring cell-to-cell variation)."""
    mp = np.asarray(pred, np.float64).mean(axis=0)
    mo = np.asarray(obs, np.float64).mean(axis=0)
    from scipy.stats import spearmanr, pearsonr
    return float(spearmanr(mp, mo).correlation), float(pearsonr(mp, mo)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_file", required=True)
    ap.add_argument("--data_file", default=DATA_FILE)
    ap.add_argument("--graph_file", default=None)
    ap.add_argument("--n_cells", type=int, default=10000)
    ap.add_argument("--top_genes", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_expr_encoder", type=_bool, default=True)
    ap.add_argument("--expr_hidden_dims", nargs="+", type=int, default=[512, 256])
    ap.add_argument("--use_nngp", type=_bool, default=False)
    ap.add_argument("--nngp_lambda", type=float, default=0.0)
    ap.add_argument("--use_paracrine", type=_bool, default=True)
    ap.add_argument("--paracrine_l1", type=float, default=1e-4)
    ap.add_argument("--use_niche_interaction", type=_bool, default=True)
    ap.add_argument("--niche_rank", type=int, default=8)
    a = ap.parse_args()

    ctx = E.load_eval_context(SimpleNamespace(**vars(a)))
    model, N = ctx["model"], ctx["N"]
    Y_enc, Y_obs = ctx["Y_expr"], ctx["Y_obs"]
    cls, coords = ctx["cls"], ctx["coords"]
    pos, atts, idx = ctx["pos_batched"], ctx["cell_atts"], ctx["sample_indices"]

    rng = np.random.RandomState(a.seed)
    sub = np.sort(rng.choice(N, min(a.n_cells, N), replace=False))
    rest = np.setdiff1d(np.arange(N), sub, assume_unique=True)
    hg = (cls != "unassigned").astype(np.float32)
    rk = (cls == "gene_KO").astype(np.float32)
    logging.info("Subsample %d cells (control=%d gene_KO=%d unassigned=%d)", len(sub),
                 int((cls[sub] == "control").sum()), int((cls[sub] == "gene_KO").sum()),
                 int((cls[sub] == "unassigned").sum()))

    # --- H5: latent variance per dim ---
    Z = model.batching_latent_samples(pos[sub], idx[sub], atts[sub],
                                       batch_size=a.batch_size, Y=Y_enc[sub])
    z_std = Z.std(axis=0)
    logging.info("[H5 latent] per-dim std (GP first): %s", np.array2string(z_std, precision=4))

    # --- predictions: basal (clean) and full (with shifts) ---
    pred_basal = model.batching_denoise_counts(
        X=pos[sub], sample_index=idx[sub], cell_atts=atts[sub], force_basal=True,
        n_samples=10, batch_size=a.batch_size, Y=Y_enc[sub])[0]
    pred_full = model.batching_denoise_counts(
        X=pos[sub], sample_index=idx[sub], cell_atts=atts[sub], force_basal=False,
        has_guide=hg[sub], is_real_KO=rk[sub],
        n_samples=10, batch_size=a.batch_size, Y=Y_enc[sub])[0]
    obs = Y_obs[sub]

    # --- H1: decoder output variance vs observed ---
    g_std_basal = pred_basal.std(axis=0)
    g_std_full = pred_full.std(axis=0)
    g_std_obs = obs.std(axis=0)
    logging.info("[H1 output-var] median per-gene std  basal=%.4g full=%.4g obs=%.4g  "
                 "ratio(full/obs)=%.3g", np.median(g_std_basal), np.median(g_std_full),
                 np.median(g_std_obs), np.median(g_std_full) / (np.median(g_std_obs) + 1e-12))

    # --- H6: sparsity ---
    zero_frac = float((obs == 0).mean())
    logging.info("[H6 sparsity] observed raw-count zero fraction = %.3f", zero_frac)

    # --- C + baseline: per-gene-across-cells Spearman ---
    base = eu.spatial_knn_mean(coords[sub], coords[rest], Y_obs[rest], k=a.knn_k)
    pg_basal = E._per_gene_spearman_mean(pred_basal, obs)
    pg_full = E._per_gene_spearman_mean(pred_full, obs)
    pg_knn = E._per_gene_spearman_mean(base, obs)
    logging.info("[per-gene/cells] basal=%.3f full=%.3f  kNN-baseline=%.3f  (eval's axis)",
                 pg_basal, pg_full, pg_knn)

    # --- H2: per-cell-across-genes + pseudobulk profile ---
    pc_basal = per_cell_spearman_mean(pred_basal, obs)
    pc_full = per_cell_spearman_mean(pred_full, obs)
    pb_s_basal, pb_p_basal = pseudobulk_corr(pred_basal, obs)
    pb_s_full, pb_p_full = pseudobulk_corr(pred_full, obs)
    logging.info("[H2 per-cell/genes] basal=%.3f full=%.3f", pc_basal, pc_full)
    logging.info("[H2 pseudobulk profile] basal: spearman=%.3f pearson=%.3f | "
                 "full: spearman=%.3f pearson=%.3f", pb_s_basal, pb_p_basal, pb_s_full, pb_p_full)

    # --- H6: recompute per-gene on top-expressed genes (fewer zero ties) ---
    top = np.argsort(-obs.mean(axis=0))[:a.top_genes]
    pg_basal_top = E._per_gene_spearman_mean(pred_basal[:, top], obs[:, top])
    pg_knn_top = E._per_gene_spearman_mean(base[:, top], obs[:, top])
    top_zero = float((obs[:, top] == 0).mean())
    logging.info("[H6 top-%d genes] per-gene/cells basal=%.3f kNN=%.3f (zero-frac=%.3f)",
                 a.top_genes, pg_basal_top, pg_knn_top, top_zero)

    # --- Verdict ---
    logging.info("================= VERDICT =================")
    gp_flat = z_std[:2].mean() < 1e-2
    out_flat = (np.median(g_std_full) / (np.median(g_std_obs) + 1e-12)) < 0.05
    logging.info("H5 latent collapse?   %s (GP std=%.3g, Normal std=%.3g)",
                 "YES" if gp_flat else "no", float(z_std[:2].mean()), float(z_std[2:].mean()))
    logging.info("H1 output collapse?   %s (pred/obs std ratio=%.3g)",
                 "YES" if out_flat else "no", np.median(g_std_full) / (np.median(g_std_obs) + 1e-12))
    logging.info("H4 force_basal issue? %s (full %.3f vs basal %.3f per-gene)",
                 "YES" if pg_full - pg_basal > 0.05 else "no", pg_full, pg_basal)
    logging.info("H2 metric-axis?       %s (per-cell=%.3f, pseudobulk-spearman=%.3f >> per-gene=%.3f)",
                 "LIKELY" if (pc_full > 0.3 or pb_s_full > 0.5) and pg_full < 0.15 else "no",
                 pc_full, pb_s_full, pg_full)
    logging.info("H6 sparsity?          %s (zero-frac=%.3f; top-gene basal=%.3f vs all=%.3f)",
                 "LIKELY" if zero_frac > 0.8 and pg_basal_top - pg_basal > 0.05 else "no",
                 zero_frac, pg_basal_top, pg_basal)
    logging.info("==========================================")


if __name__ == "__main__":
    main()
