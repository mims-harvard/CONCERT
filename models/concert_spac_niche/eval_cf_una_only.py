#!/usr/bin/env python3
"""Per-niche unassigned-only counterfactual eval — TWO setups (seen + unseen).

No NTC required. Source cells are UNASSIGNED (true basal). For each KO gene we
predict unassigned -> KO per niche and compare pseudobulk vs observed KO cells.

  SEEN  : source = unassigned cells the model trained on (non-holdout).
  UNSEEN: source = held-out unassigned cells the model never saw (generalization).

Both setups score against the SAME observed KO cells (KO cells are never held
out — only unassigned are). The v6 model amortizes the basal latent from each
cell's own expression (use_expr_encoder), so it can encode held-out cells; the
GP mask_cutoff is sliced positionally, so any source set with <= N_train cells
is valid. The two setups therefore differ ONLY in the source unassigned set.

For each gene with >=min_cells_per_pert KO cells, for each niche with
>=min_g_cells KO cells AND >=una_floor source cells:
    pred = _predict_swap(source_in_niche, gene_code)   # una -> KO
    obs  = observed KO cells in that niche
    score: abs_rho (pseudobulk Spearman over HVGs), e_dist (energy distance)

Writes per_niche.csv / per_gene.csv / summary.json per setup to --outdir.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

SPAC_DIR = Path(__file__).parent
sys.path.insert(0, str(SPAC_DIR))

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_FILE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_1-1.h5ad"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_file", required=True)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--data_file", default=DATA_FILE)
    p.add_argument("--holdout_unassigned_file", default=None,
                   help="holdout_unassigned_indices.npy saved during training. "
                        "Removes held-out cells for the SEEN setup; the same file "
                        "is used to load the held-out cells for the UNSEEN setup.")
    p.add_argument("--setups", default="seen,unseen",
                   help="comma list: which setups to run (seen, unseen).")
    p.add_argument("--hvg_max_rank",    type=int, default=2000)
    p.add_argument("--min_cells_per_pert", type=int, default=25,
                   help="load_data rare-pert filter; MUST match training (sets N / mask_cutoff).")
    p.add_argument("--panel_min_cells", type=int, default=200,
                   help="slide-level threshold to ENTER the prediction panel (eval-time only).")
    p.add_argument("--max_panel",       type=int, default=30,
                   help="cap: keep at most this many perturbations (top by KO-cell count).")
    p.add_argument("--una_floor",       type=int, default=50)
    p.add_argument("--min_g_cells",     type=int, default=20)
    p.add_argument("--max_src_per_niche", type=int, default=100,
                   help="cap on unassigned source cells PREDICTED per niche (0=all).")
    p.add_argument("--niche_grid",      type=int, default=6)
    p.add_argument("--knn_k",           type=int, default=20, help="k for global spatial-kNN baseline (closest k KO cells)")
    p.add_argument("--n_samples",       type=int, default=10)
    p.add_argument("--batch_size",      type=int, default=512)
    p.add_argument("--n_hvg",           type=int, default=2000)
    # Architecture — must match training
    p.add_argument("--use_expr_encoder",      type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--expr_hidden_dims",      type=int, nargs="+", default=[512, 256])
    p.add_argument("--use_basal_encoder_only",type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--use_nngp",              type=lambda x: x.lower()=="true", default=False)
    p.add_argument("--nngp_lambda",           type=float, default=0.0)
    p.add_argument("--use_paracrine",         type=lambda x: x.lower()=="true", default=False)
    p.add_argument("--paracrine_l1",          type=float, default=0.0)
    p.add_argument("--use_niche_interaction", type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--niche_rank",            type=int, default=8)
    p.add_argument("--use_pert_gate",         type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--gate_hidden",           type=int, default=64)
    return p.parse_args()


def pseudobulk(X: np.ndarray) -> np.ndarray:
    """Log-mean pseudobulk profile (1D, length G)."""
    return np.log1p(X.mean(axis=0))


def load_unseen_ctx(args, seen_ctx, holdout_indices):
    """Load the held-out unassigned cells as a prediction source set, reusing the
    SEEN model. Returns a ctx-like dict with the arrays _predict_swap needs.

    Uses make_umap.load_data(holdout_keep_only=True) so the exact same filter
    pipeline runs and the returned cells are precisely the held-out set — no
    fragile re-indexing across the rare-pert filter.
    """
    import scipy.sparse as sp
    import torch
    from make_umap import load_data
    from eval_spatial_counterfactual import classify_perts

    (adata, pos_batched, cell_atts, _batch, _pert_name, _cutoff_t,
     perturb_raw, _has_guide, _is_real_KO, _n_batch, _perturb_idx) = load_data(
         args.data_file,
         hvg_max_rank=args.hvg_max_rank,
         min_cells_per_pert=args.min_cells_per_pert,
         holdout_indices=holdout_indices,
         holdout_keep_only=True,
     )
    N = adata.n_obs
    Y_expr = adata.X
    Y_expr = (Y_expr.toarray() if sp.issparse(Y_expr) else np.asarray(Y_expr)).astype(np.float32)
    xy = pos_batched[:, :2].astype(np.float32)      # scaled coords (collapsed-load scaler; fine)
    cls = classify_perts(perturb_raw)

    # CRITICAL: the held-out load collapses the perturbation vocab (all cells are
    # unassigned), so its batch one-hot width != the model's n_batch and the GP
    # kernel's per-batch scale matmul fails. All held-out cells ARE unassigned and
    # share ONE perturbation index, so reuse the SEEN vocab's unassigned one-hot row.
    seen_una = np.where(seen_ctx["cls"] == "unassigned")[0]
    onehot = seen_ctx["pos_batched"][seen_una[0], 2:]   # (n_batch,) full-width
    sample = seen_ctx["pos_batched"][seen_una[:2000], 2:]
    assert np.allclose(sample, onehot), "seen unassigned cells do not share one batch one-hot"
    pos_full = np.concatenate([xy, np.tile(onehot, (N, 1)).astype(np.float32)], axis=1)
    log.info("[unseen] pos_batched: %s (reused seen unassigned one-hot, width=%d)",
             pos_full.shape, onehot.shape[0])

    return dict(model=seen_ctx["model"], device=seen_ctx["device"], N=N,
                Y_expr=Y_expr, pos_batched=pos_full,
                sample_indices=torch.arange(N), cell_atts=cell_atts.astype(int),
                coords=xy.astype(np.float64), cls=cls)


def run_setup(label, src_ctx, ko_ref, args, eu, _predict_swap):
    """Score one setup. src_ctx supplies prediction inputs + source unassigned
    cells; ko_ref supplies observed KO cells, their niche bins, Y_obs, name2code,
    hvg_mask, and panel (all from the seen ctx — KO cells are never held out)."""
    src_bin, n_bins = eu.build_niche_grid(src_ctx["coords"], args.niche_grid)
    una_all = np.where(src_ctx["cls"] == "unassigned")[0]
    log.info("[%s] source unassigned cells: %d  (%d niches)", label, len(una_all), n_bins)

    Y_obs    = ko_ref["Y_obs"]
    ko_bin   = ko_ref["bin_id"]
    hvg_mask = ko_ref["hvg_mask"]
    name2code = ko_ref["name2code"]
    una_obs_idx = ko_ref["una_obs_idx"]   # SEEN observed unassigned (LFC reference, both setups)
    seen_coords = ko_ref["coords"]        # SEEN coords for kNN reference pools
    src_coords  = src_ctx["coords"]

    # Global kNN baseline is model-free (observed data + source coords) → identical for
    # seen/unseen up to the coord frame; compute it only in the SEEN run (matched frame).
    # Held-out-source drivers set args.force_knn to get the baseline in a single run.
    compute_knn = getattr(args, "force_knn", False) or (label == "seen")

    # Source cells per niche: cap at max_src_per_niche. Deterministic per niche (seed=42+b)
    # so pred_g, pred0 and seen/unseen all use a consistent subsample. Floor check below
    # uses the FULL niche population; we then predict on at most max_src_per_niche of them.
    src_cache: dict[int, np.ndarray] = {}
    def src_for_niche(b):
        if b not in src_cache:
            full = una_all[src_bin[una_all] == b]
            if args.max_src_per_niche > 0 and len(full) > args.max_src_per_niche:
                rng = np.random.default_rng(42 + int(b))
                full = np.sort(rng.choice(full, args.max_src_per_niche, replace=False))
            src_cache[b] = full
        return src_cache[b]
    # pred(src→0) depends only on the niche's (subsampled) source cells → cache per niche.
    pred0_cache: dict[int, np.ndarray] = {}
    def pred0_for(b):
        if b not in pred0_cache:
            pred0_cache[b] = _predict_swap(src_ctx, src_for_niche(b), 0, args).astype(np.float64)
        return pred0_cache[b]
    # kNN unassigned base = closest k observed unassigned cells (global, whole slide)
    # per query coord. Query depends only on niche b → cache per niche.
    knn_base_cache: dict[int, np.ndarray] = {}
    def knn_base_for(b, qxy):
        if b not in knn_base_cache:
            knn_base_cache[b] = eu.spatial_knn_mean(qxy, seen_coords[una_obs_idx],
                                                    Y_obs[una_obs_idx], k=args.knn_k).astype(np.float64)
        return knn_base_cache[b]

    rows = []
    for g in ko_ref["panel"]:
        g_code = name2code.get(g)
        if g_code is None:
            continue
        ko_idx = ko_ref["ko_idx"][g]
        # Predict this gene ONLY in its MOST ABUNDANT niche (max observed KO cells) that
        # also clears min_g_cells (KO) and una_floor (source unassigned).
        ko_counts = np.bincount(ko_bin[ko_idx], minlength=n_bins)
        b = -1
        for cand in np.argsort(-ko_counts):
            if ko_counts[cand] < args.min_g_cells:
                break    # remaining niches have even fewer KO cells
            if len(una_all[src_bin[una_all] == cand]) >= args.una_floor:
                b = int(cand); break
        if b < 0:
            continue

        ko_b  = ko_idx[ko_bin[ko_idx] == b]
        una_b = src_for_niche(b)                          # <= max_src_per_niche source cells
        pred_raw = _predict_swap(src_ctx, una_b, g_code, args).astype(np.float64)
        obs_raw  = Y_obs[ko_b].astype(np.float64)
        pb_pred = pseudobulk(pred_raw)[hvg_mask]
        pb_obs  = pseudobulk(obs_raw)[hvg_mask]
        rho, _ = stats.spearmanr(pb_pred, pb_obs)
        edist = eu.energy_distance(pred_raw, obs_raw, hvg_mask)

        # LFC (with-noise): model LFC = pred(una→g) − pred(una→0); obs = KO − observed unassigned.
        # lfc_rho = rank agreement (scale-free); lfc_dist = RMSE between LFC vectors (magnitude).
        lfc_rho = float("nan"); lfc_dist = float("nan"); obs_lfc = None
        una_obs_b = una_obs_idx[ko_bin[una_obs_idx] == b]
        if len(una_obs_b) >= args.una_floor:
            pred0 = pred0_for(b)
            model_lfc = eu.pseudobulk_lfc(pred_raw, pred0)[hvg_mask]
            obs_lfc   = eu.pseudobulk_lfc(obs_raw, Y_obs[una_obs_b].astype(np.float64))[hvg_mask]
            lfc_rho = float(stats.spearmanr(model_lfc, obs_lfc)[0])
            _fin = np.isfinite(model_lfc) & np.isfinite(obs_lfc)
            if _fin.any():
                lfc_dist = float(np.sqrt(np.mean((model_lfc[_fin] - obs_lfc[_fin]) ** 2)))

        # ── Global kNN baseline (model-free): predict each source cell as the mean
        #    of its k spatially-nearest KO cells of this gene, sampled across the
        #    WHOLE slide (no leave-one-niche-out). Source cells are unassigned and
        #    references are KO → always distinct cells, so no self-leakage. ──────
        knn_abs_rho = knn_lfc_rho = knn_e_dist = float("nan")
        if compute_knn:
            g_glob = ko_idx                       # all KO cells of gene g, global
            if len(g_glob) >= args.min_g_cells:
                qxy = src_coords[una_b]
                knn_g = eu.spatial_knn_mean(qxy, seen_coords[g_glob],
                                            Y_obs[g_glob], k=args.knn_k).astype(np.float64)
                knn_abs_rho = float(stats.spearmanr(pseudobulk(knn_g)[hvg_mask], pb_obs)[0])
                knn_e_dist  = float(eu.energy_distance(knn_g, obs_raw, hvg_mask))
                if obs_lfc is not None:
                    knn_base = knn_base_for(b, qxy)
                    knn_lfc = eu.pseudobulk_lfc(knn_g, knn_base)[hvg_mask]
                    knn_lfc_rho = float(stats.spearmanr(knn_lfc, obs_lfc)[0])

        rows.append(dict(gene=g, niche=b, n_ko=len(ko_b), n_una=len(una_b),
                         abs_rho=float(rho), lfc_rho=lfc_rho, lfc_dist=lfc_dist,
                         e_dist=float(edist),
                         knn_abs_rho=knn_abs_rho, knn_lfc_rho=knn_lfc_rho,
                         knn_e_dist=knn_e_dist))
        log.info("[%s]   %s: niche %d (n_ko=%d n_una=%d) abs=%.3f lfc=%.3f",
                 label, g, b, len(ko_b), len(una_b), rho, lfc_rho)

    if not rows:
        log.warning("[%s] No niches scored", label)
        return None

    df = pd.DataFrame(rows)
    out = args.outdir / label
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_niche.csv", index=False)
    agg = df.groupby("gene").agg(
        n_niches=("niche", "count"),
        abs_rho_mean=("abs_rho", "mean"), abs_rho_sd=("abs_rho", "std"),
        lfc_rho_mean=("lfc_rho", "mean"), lfc_rho_sd=("lfc_rho", "std"),
        lfc_dist_mean=("lfc_dist", "mean"), lfc_dist_sd=("lfc_dist", "std"),
        e_dist_mean=("e_dist", "mean"),  e_dist_sd=("e_dist", "std"),
        knn_abs_rho_mean=("knn_abs_rho", "mean"),
        knn_lfc_rho_mean=("knn_lfc_rho", "mean"),
        knn_e_dist_mean=("knn_e_dist", "mean"),
    ).reset_index()
    agg.to_csv(out / "per_gene.csv", index=False)
    summary = dict(setup=label, n_genes=int(agg.shape[0]), n_niche_rows=int(len(df)),
                   abs_rho_mean=float(df["abs_rho"].mean()),
                   lfc_rho_mean=float(df["lfc_rho"].mean()),
                   lfc_rho_n=int(df["lfc_rho"].notna().sum()),
                   lfc_dist_mean=float(df["lfc_dist"].mean()),
                   e_dist_mean=float(df["e_dist"].mean()),
                   knn_abs_rho_mean=float(df["knn_abs_rho"].mean()),
                   knn_lfc_rho_mean=float(df["knn_lfc_rho"].mean()),
                   knn_e_dist_mean=float(df["knn_e_dist"].mean()),
                   per_gene=agg.to_dict(orient="records"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("[%s] DONE %d genes × %d rows | MODEL abs=%.3f lfc=%.3f e=%.3f | kNN abs=%.3f lfc=%.3f e=%.3f → %s",
             label, agg.shape[0], len(df), summary["abs_rho_mean"], summary["lfc_rho_mean"],
             summary["e_dist_mean"], summary["knn_abs_rho_mean"], summary["knn_lfc_rho_mean"],
             summary["knn_e_dist_mean"], out)
    return summary


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    setups = [s.strip() for s in args.setups.split(",") if s.strip()]

    import eval_utils as eu
    from eval_spatial_counterfactual import load_eval_context, _predict_swap

    # SEEN ctx — builds + loads the model (holdout removed to match N_train).
    ctx = load_eval_context(args)
    log.info("Seen ctx: %d cells, %d unique perts", ctx["N"], ctx["n_batch"])

    cls = ctx["cls"]
    perturb_raw = ctx["perturb_raw"]
    Y_obs = ctx["Y_obs"]

    # Niche bins + KO reference (shared observed targets for both setups).
    ko_bin, n_bins = eu.build_niche_grid(ctx["coords"], args.niche_grid)
    hvg_mask = eu.select_hvgs(Y_obs, n_top=args.n_hvg)
    log.info("HVG genes for scoring: %d / %d", hvg_mask.sum(), len(hvg_mask))

    ko_mask = cls == "gene_KO"
    pert_counts = Counter(perturb_raw[ko_mask].tolist())
    # Panel rule: gene-KO with >= panel_min_cells on the slide, top max_panel by count.
    eligible = {g: n for g, n in pert_counts.items() if n >= args.panel_min_cells}
    panel = sorted(eligible, key=lambda g: -eligible[g])[:args.max_panel]
    log.info("Panel: %d genes (>=%d KO cells, top %d by count): %s",
             len(panel), args.panel_min_cells, args.max_panel,
             [(g, eligible[g]) for g in panel])
    ko_idx = {g: np.where(ko_mask & (perturb_raw == g))[0] for g in panel}
    una_obs_idx = np.where(cls == "unassigned")[0]   # observed unassigned (LFC reference)

    ko_ref = dict(Y_obs=Y_obs, bin_id=ko_bin, hvg_mask=hvg_mask,
                  name2code=ctx["name2code"], panel=panel, ko_idx=ko_idx,
                  una_obs_idx=una_obs_idx, coords=ctx["coords"])

    results = {}
    if "seen" in setups:
        results["seen"] = run_setup("seen", ctx, ko_ref, args, eu, _predict_swap)

    if "unseen" in setups:
        if not getattr(args, "holdout_unassigned_file", None):
            log.warning("UNSEEN requested but no --holdout_unassigned_file — skipping")
        else:
            holdout_indices = np.load(args.holdout_unassigned_file)
            unseen_ctx = load_unseen_ctx(args, ctx, holdout_indices)
            results["unseen"] = run_setup("unseen", unseen_ctx, ko_ref, args, eu, _predict_swap)

    # Combined summary
    combined = {k: dict(abs_rho_mean=v["abs_rho_mean"], lfc_rho_mean=v["lfc_rho_mean"],
                        lfc_rho_n=v["lfc_rho_n"], e_dist_mean=v["e_dist_mean"],
                        knn_abs_rho_mean=v["knn_abs_rho_mean"], knn_lfc_rho_mean=v["knn_lfc_rho_mean"],
                        knn_e_dist_mean=v["knn_e_dist_mean"],
                        n_genes=v["n_genes"], n_niche_rows=v["n_niche_rows"])
                for k, v in results.items() if v}
    (args.outdir / "summary_both.json").write_text(json.dumps(combined, indent=2))
    for k, v in combined.items():
        log.info("== %-6s MODEL abs=%.3f lfc=%.3f e=%.3f | kNN abs=%.3f lfc=%.3f e=%.3f (%d genes, %d rows)",
                 k, v["abs_rho_mean"], v["lfc_rho_mean"], v["e_dist_mean"],
                 v["knn_abs_rho_mean"], v["knn_lfc_rho_mean"], v["knn_e_dist_mean"],
                 v["n_genes"], v["n_niche_rows"])


if __name__ == "__main__":
    main()
