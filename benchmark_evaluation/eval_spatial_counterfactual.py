#!/usr/bin/env python3
"""Spatial-counterfactual evaluation harness for CONCERT-spac (goal-3).

Evaluates the model's impute -> counterfactual predictions with a spatial-aware,
perturbation-noise-cancelled methodology across the three goal-3 scenarios:
  (c) seen-cell pert swap   -- Task 6 (two source levels: NTC + unassigned)
  (b) held-out cell         -- Task 7
  (a) coord-only patches    -- Task 7

Data + model are loaded via ``make_umap.load_data`` / ``build_model`` -- the SAME
preprocessing as training, so LORD global cell indices align with the trained
basal-embedding table (panel/grid/NTC counts are therefore computed in-driver on
the filtered arrays, never hard-coded). No retraining, no model-code changes.
"""
from __future__ import annotations
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import eval_utils as eu
from make_umap import (load_data, classify_perts, build_model, normalize, LOC_RANGE,
                       EXCLUDED_LABELS, UNASSIGNED_LABELS, CONTROL_LABELS)
from run_concert_spac import load_h5ad_multi, build_attributes_spac

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

DATA_FILE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_1-1.h5ad"


def _bool(x):
    return str(x).lower() == "true"


def _add_fpert_kwargs(extra, args):
    """Thread the f_pert footprint-field flags into build_model's CONCERT kwargs so the
    evaluated graph matches the trained checkpoint (rff/svgp/hybrid add field weights)."""
    extra["fpert_field"] = args.fpert_field
    extra["f_pert_clamp"] = args.f_pert_clamp
    # basal spatial field (svgp legacy | rff deterministic). MUST match the trained
    # checkpoint so the basal_rff submodule (or SVGP prior) is present for state_dict load.
    extra["basal_field"] = args.basal_field
    # Bayes-linear basal head + multi-bandwidth fpert -- MUST match training for state_dict load.
    extra["basal_bayes"] = args.basal_bayes
    if args.fpert_field == "mb_rff" and args.fpert_mb_bandwidths:
        extra["fpert_mb_bandwidths"] = [float(x) for x in args.fpert_mb_bandwidths.split(",") if x.strip()]
    if args.fpert_field != "constant":
        extra["fpert_rff_features"] = args.fpert_rff_features
        extra["fpert_rff_bandwidth"] = args.fpert_rff_bandwidth
        extra["fpert_hidden"] = args.fpert_hidden
        extra["fpert_svgp_inducing"] = args.fpert_svgp_inducing
        extra["fpert_svgp_ls_init"] = args.fpert_svgp_ls_init
        extra["fpert_coord_scale"] = args.fpert_coord_scale
        # v4 collapse-proof band: gp_pp squashes its per-pert lengthscale into
        # [ls_min,ls_max] via sigmoid. This band MUST match the trained checkpoint or the
        # SAME ls_head weights render a DIFFERENT lengthscale at inference (silent corruption).
        # Left unset => CONCERT defaults (3,8), which mismatches v4's [3,18].
        if getattr(args, "ls_prior_min", None) is not None:
            extra["ls_prior_min"] = args.ls_prior_min
        if getattr(args, "ls_prior_max", None) is not None:
            extra["ls_prior_max"] = args.ls_prior_max
        # UNITS FIX: gp_pp consumes coords/coord_scale, so its band lives in NORMALIZED
        # units, unlike ls_prior_min/max (loc units, tuned for the SVGP kernel scale).
        # Checkpoints trained with --fpert_ls_min/max MUST be evaluated with the same
        # values: the band is a plain float on FpertField, not a state_dict entry, so a
        # mismatch silently renders a different lengthscale from the same ls_head weights.
        # Left unset => legacy behaviour (reuse ls_prior_min/max unconverted).
        if getattr(args, "fpert_ls_min", None) is not None:
            extra["fpert_ls_min"] = args.fpert_ls_min
        if getattr(args, "fpert_ls_max", None) is not None:
            extra["fpert_ls_max"] = args.fpert_ls_max


def parse_args():
    p = argparse.ArgumentParser(description="CONCERT-spac spatial counterfactual eval")
    p.add_argument("--model_file", required=True)
    p.add_argument("--data_file", default=DATA_FILE)
    p.add_argument("--outdir", required=True)
    # niche grid / scoring
    p.add_argument("--niche_grid", type=int, default=6)
    p.add_argument("--una_floor", type=int, default=50,
                   help="min unassigned cells per niche to declare it evaluable (primary gate)")
    p.add_argument("--ntc_floor", type=int, default=30,
                   help="min NTC cells per niche to compute LFC-without-noise (secondary, per-metric)")
    p.add_argument("--loc_range", type=float, default=20.0)
    p.add_argument("--min_g_cells", type=int, default=20, help="min real-g cells in a niche to score")
    p.add_argument("--n_hvg", type=int, default=2000,
                   help="restrict all scoring to top-N HVGs recomputed on normalized data (sparsity-robust)")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--knn_k", type=int, default=20, help="k for spatial-kNN baseline")
    p.add_argument("--patch_impute_cap", type=int, default=5000,
                   help="cap on patch test cells scored in scenario (a) imputation (NN loop is O(N))")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit_panel", type=int, default=0, help="0 = full panel; else cap for subset runs")
    p.add_argument("--panel_genes", default="", help="comma-separated gene names; bypasses auto panel selection")
    p.add_argument("--scenarios", default="c", help="comma list of scenarios to run: c,b,a")
    p.add_argument("--setup_only", action="store_true", help="load + grid + panel, then return (smoke)")
    # --- v5a model flags (MUST match training keys, mirrors sub_decompose_levels_v5a.sh) ---
    p.add_argument("--use_expr_encoder", type=_bool, default=True)
    p.add_argument("--expr_hidden_dims", nargs="+", type=int, default=[512, 256])
    p.add_argument("--use_nngp", type=_bool, default=False)
    p.add_argument("--nngp_lambda", type=float, default=0.0)
    p.add_argument("--graph_file", default=None)
    p.add_argument("--use_paracrine", type=_bool, default=True)
    p.add_argument("--paracrine_l1", type=float, default=1e-4)
    p.add_argument("--use_niche_interaction", type=_bool, default=True)
    p.add_argument("--niche_rank", type=int, default=8)
    # per-cell scalar gate (must match training to load v5d_gate checkpoints)
    p.add_argument("--use_pert_gate", type=_bool, default=False)
    p.add_argument("--gate_hidden", type=int, default=64)
    # spatial-basal disentanglement (no new params; routing flag for consistency)
    p.add_argument("--spatial_basal_only", type=_bool, default=False)
    # basal spatial field type -- MUST match training (svgp|rff) for state_dict load
    p.add_argument("--basal_field", default="svgp", help="svgp (legacy) | rff (deterministic)")
    p.add_argument("--basal_bayes", type=_bool, default=False, help="M2: Bayes-linear basal head")
    p.add_argument("--fpert_mb_bandwidths", default="", help="comma list for mb_rff, e.g. 0.05,0.15,0.5")
    # data filtering -- must match training to get same gene set and cell count
    p.add_argument("--hvg_max_rank", type=int, default=10000)
    p.add_argument("--min_cells_per_pert", type=int, default=25)
    # encoder architecture -- must match training (default False = 3-block encoder)
    p.add_argument("--use_basal_encoder_only", type=_bool, default=False)

    # f_pert spatial footprint field -- MUST match training so load_model state_dict aligns
    p.add_argument("--fpert_field", default="constant",
                   help="constant|rff|svgp|hybrid (default constant = legacy core only)")
    p.add_argument("--fpert_rff_features", type=int, default=128)
    p.add_argument("--fpert_rff_bandwidth", type=float, default=0.15)
    p.add_argument("--fpert_hidden", type=int, default=256)
    p.add_argument("--fpert_svgp_inducing", type=int, default=32)
    p.add_argument("--fpert_svgp_ls_init", type=float, default=5.0)
    p.add_argument("--fpert_coord_scale", type=float, default=1.0)
    p.add_argument("--f_pert_clamp", type=float, default=5.0)
    # MUST match training: pmap trained with action correction OFF (no mu_action param),
    # spac with it ON. Wrong value -> build_model constructs a mismatched mu_action.
    p.add_argument("--use_action_correction", type=_bool, default=True)
    # GP-latent inducing grid steps -- training used 4 (pmap) / 6 (spac). Wrong value ->
    # inducing-point shape mismatch on load.
    p.add_argument("--inducing_point_steps", type=int, default=6)
    # v5c holdout: path to holdout_unassigned_indices.npy saved during training
    p.add_argument("--holdout_unassigned_file", type=str, default=None)
    # multi-slide vocab: pass all training slide files so the pert embedding matches the checkpoint
    p.add_argument("--multi_sample_files", default="",
                   help="comma-separated paths to all training slides (needed for multi-slide models)")
    p.add_argument("--target_slide_idx", type=int, default=0,
                   help="0-indexed slide to evaluate (only used with --multi_sample_files)")
    p.add_argument("--within_slide_panel", type=_bool, default=False,
                   help="In multi-slide mode, select the panel by WITHIN-target-slide per-niche "
                        "abundance (adequate n) instead of the default cross-slide ranking.")
    return p.parse_args()


def load_eval_context_multi(args):
    """Multi-slide variant: reconstruct the same pert vocab as the multi model,
    filter to target slide, then build and load the model.
    Returns the same dict shape as load_eval_context."""
    import gc
    import scipy.sparse as sp
    from collections import Counter
    from sklearn.preprocessing import MinMaxScaler
    import scanpy as sc

    paths = [p.strip() for p in args.multi_sample_files.split(",") if p.strip()]
    logging.info("Multi-slide eval: %d slides, target slide_idx=%d", len(paths), args.target_slide_idx)

    pm_path = Path(args.model_file).parent / "pert_map.json"
    if not pm_path.exists():
        raise SystemExit(f"pert_map.json not found at {pm_path}; retrain to persist it "
                         "(eval must not rebuild the vocab).")
    with open(pm_path) as _fh:
        _pm = json.load(_fh)
    pert_map          = {k: int(v) for k, v in _pm["pert_map"].items()}
    unassigned_labels = frozenset(_pm.get("unassigned_labels") or UNASSIGNED_LABELS)
    control_labels    = frozenset(_pm.get("control_labels") or CONTROL_LABELS)
    excluded_labels   = frozenset(_pm.get("excluded_labels") or EXCLUDED_LABELS)
    tissue_dict       = {str(k): int(v) for k, v in (_pm.get("tissue_dict") or {}).items()}
    non_pert          = unassigned_labels | control_labels
    logging.info("[vocab] pert_map.json: %d perts (incl basal), tissue_dict=%s",
                 len(pert_map), tissue_dict)

    # Load all slides with shared HVG mask + the SAME label sets as training
    X_all, pos_all, tissue_all, perturb_all, has_guide_all, is_real_KO_all, gene_names, sample_id_all = \
        load_h5ad_multi(paths, unassigned_labels=unassigned_labels,
                        control_labels=control_labels, hvg_max_rank=args.hvg_max_rank)

    # Filter excluded labels across all slides (json's set -- matches training)
    keep = np.array([p not in excluded_labels for p in perturb_all], dtype=bool)
    X_all, pos_all, perturb_all = X_all[keep], pos_all[keep], perturb_all[keep]
    has_guide_all = has_guide_all[keep]
    is_real_KO_all = is_real_KO_all[keep]
    sample_id_all = sample_id_all[keep]
    if tissue_all is not None:
        tissue_all = tissue_all[keep]

    # Filter rare perturbs on COMBINED counts -- exactly as training did. basal/control
    # are NOT rare-filtered: they collapse into the reserved code 0 regardless of count.
    real_counts = Counter([p for p in perturb_all.tolist() if p not in non_pert])
    keep2 = np.array([(p in non_pert) or (real_counts[p] >= args.min_cells_per_pert)
                      for p in perturb_all], dtype=bool)
    X_all, pos_all, perturb_all = X_all[keep2], pos_all[keep2], perturb_all[keep2]
    has_guide_all = has_guide_all[keep2]
    is_real_KO_all = is_real_KO_all[keep2]
    sample_id_all = sample_id_all[keep2]
    if tissue_all is not None:
        tissue_all = tissue_all[keep2]

    # Map to trained codes via the persisted vocab: basal/control/unassigned + any pert
    # absent from the trained vocab (e.g. rare-filtered) -> reserved basal code 0.
    perturb_idx_all = np.array([pert_map.get(p, 0) for p in perturb_all], dtype=int)
    n_batch = max(pert_map.values()) + 1
    logging.info("Trained vocab: n_batch=%d; %d cells mapped to basal code 0 (of %d)",
                 n_batch, int((perturb_idx_all == 0).sum()), len(perturb_idx_all))

    # Filter to target slide AFTER vocab is built (so indices remain valid)
    tmask = (sample_id_all == args.target_slide_idx)

    # Cross-slide counts: KO cells per pert in all OTHER slides.
    # Used by setup_grid_and_panel to rank the panel by "how well-learned" each pert is.
    other_ko_mask = (~tmask) & is_real_KO_all
    cross_slide_counts = Counter(perturb_all[other_ko_mask].tolist())
    top5_cross = sorted(cross_slide_counts.items(), key=lambda kv: -kv[1])[:5]
    logging.info("Cross-slide KO counts (top 5 other slides): %s", top5_cross)

    X           = X_all[tmask];         del X_all
    pos_raw     = pos_all[tmask];       del pos_all
    perturb_raw = perturb_all[tmask]
    has_guide   = has_guide_all[tmask]
    is_real_KO  = is_real_KO_all[tmask]
    perturb_idx = perturb_idx_all[tmask]
    tis_target  = tissue_all[tmask] if tissue_all is not None else None
    logging.info("Slide %d after filter: %d cells, %d unique perts present",
                 args.target_slide_idx, X.shape[0], len(np.unique(perturb_idx)))

    pert_name = {code: name for name, code in pert_map.items()}
    # Tissue axis: thread real codes via the trained tissue_dict (perturbmap
    # tumor/normal/periphery) so the model rebuilds the SAME tissue encoder the
    # checkpoint carries; all-zeros for spac (no tissue -> no tissue encoder).
    if tis_target is not None and tissue_dict:
        tissue_idx = np.array([tissue_dict.get(str(t), 0) for t in tis_target], dtype=int)
    else:
        tissue_idx = np.zeros(len(perturb_raw), dtype=int)
    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)
    batch = np.eye(n_batch, dtype=np.float32)[perturb_idx]

    scaler = MinMaxScaler()
    pos_scaled = scaler.fit_transform(pos_raw) * LOC_RANGE
    cutoff = np.full(pos_scaled.shape[0], 0.5, dtype=np.float32)
    pos_batched = np.concatenate([pos_scaled, batch], axis=1).astype(np.float32)

    adata = sc.AnnData(X.astype("float32"))
    if gene_names is not None:
        adata.var_names = gene_names
    adata = normalize(adata, filter_min_counts=False, size_factors=True,
                      normalize_input=True, logtrans_input=True)

    N = adata.n_obs
    if N < len(perturb_raw):
        survived = np.array(adata.obs_names.astype(int))
        pos_batched = pos_batched[survived]
        cell_atts   = cell_atts[survived]
        batch       = batch[survived]
        cutoff      = cutoff[survived]
        perturb_raw = perturb_raw[survived]
        has_guide   = has_guide[survived]
        is_real_KO  = is_real_KO[survived]
        perturb_idx = perturb_idx[survived]
        pos_raw     = pos_raw[survived]  # keep raw (pixel) coords aligned to model rows

    cutoff_t = torch.tensor(cutoff, dtype=torch.float32)
    logging.info("Cells: %d  combined perts: %d", N, n_batch)

    extra = {}
    if args.use_nngp:
        extra["use_nngp"] = True; extra["nngp_lambda"] = args.nngp_lambda
    if args.use_paracrine:
        extra["use_paracrine"] = True; extra["paracrine_l1"] = args.paracrine_l1
    if args.use_niche_interaction:
        extra["use_niche_interaction"] = True; extra["niche_rank"] = args.niche_rank
    if getattr(args, "use_niche_film", False):
        extra["use_niche_film"] = True
        extra["niche_film_dim"] = getattr(args, "niche_film_dim", 8)
        extra["niche_film_k"] = getattr(args, "niche_film_k", 25)
        extra["niche_film_gcap"] = getattr(args, "niche_film_gcap", 3.0)
        extra["niche_film_l2"] = getattr(args, "niche_film_l2", 1e-3)
        extra["niche_film_rank"] = getattr(args, "niche_film_rank", 8)
        extra["niche_film_dispersion"] = getattr(args, "niche_film_dispersion", False)
    if args.use_pert_gate:
        extra["use_pert_gate"] = True; extra["gate_hidden"] = args.gate_hidden
    if args.use_expr_encoder:
        extra["use_expr_encoder"] = True; extra["expr_hidden_dims"] = args.expr_hidden_dims
    if getattr(args, "spatial_basal_only", False):
        extra["spatial_basal_only"] = True
    # MUST match training: action correction (pmap OFF / spac ON) and GP inducing steps.
    extra["use_action_correction"] = args.use_action_correction
    extra["ind_steps"] = args.inducing_point_steps
    _add_fpert_kwargs(extra, args)
    model, device = build_model(adata, pos_batched, cell_atts, cutoff_t, n_batch,
                                use_basal_encoder_only=args.use_basal_encoder_only, **extra)
    model.load_model(args.model_file, allow_cell_size_mismatch=True)
    model.eval()

    Y_expr = adata.X
    Y_expr = (Y_expr.toarray() if sp.issparse(Y_expr) else np.asarray(Y_expr)).astype(np.float32)
    Y_raw = adata.raw.X if adata.raw is not None else adata.X
    Y_obs = (Y_raw.toarray() if sp.issparse(Y_raw) else np.asarray(Y_raw)).astype(np.float32)
    gnames = (np.asarray(adata.var_names) if adata.var_names is not None
              else np.array([f"g{i}" for i in range(adata.n_vars)]))
    del adata; gc.collect()

    cls = classify_perts(perturb_raw)
    name2code = {v: k for k, v in pert_name.items()}
    coords = pos_batched[:, :2].astype(np.float64)

    return dict(model=model, device=device, N=N, Y_expr=Y_expr, Y_obs=Y_obs, gene_names=gnames,
                pos_batched=pos_batched.astype(np.float32),
                sample_indices=torch.arange(N), cell_atts=cell_atts.astype(int),
                coords=coords, coords_raw=np.asarray(pos_raw).astype(np.float64),
                perturb_raw=np.asarray(perturb_raw), cls=cls,
                pert_name=pert_name, name2code=name2code, n_batch=n_batch,
                cross_slide_counts=cross_slide_counts)


def load_eval_context(args):
    """Load model + data exactly as training/decompose did, returning everything
    the scenarios need. Returns a dict (an explicit namespace)."""
    logging.info("Loading data via make_umap.load_data (matches training pipeline)...")
    holdout_indices = None
    if getattr(args, "holdout_unassigned_file", None):
        holdout_indices = np.load(args.holdout_unassigned_file)
        logging.info("[holdout] Loaded %d holdout indices from %s",
                     len(holdout_indices), args.holdout_unassigned_file)
    (adata, pos_batched, cell_atts, batch_oh, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(
         args.data_file,
         hvg_max_rank=args.hvg_max_rank,
         min_cells_per_pert=args.min_cells_per_pert,
         holdout_indices=holdout_indices,
     )
    N = adata.n_obs
    logging.info("Cells: %d  perts: %d", N, n_batch)

    # Build model -- flags MUST match training keys so load_model state_dict matches.
    extra = {}
    if args.use_nngp:
        extra["use_nngp"] = True
        extra["nngp_lambda"] = args.nngp_lambda
    if args.use_paracrine:
        extra["use_paracrine"] = True
        extra["paracrine_l1"] = args.paracrine_l1
    if args.use_niche_interaction:
        extra["use_niche_interaction"] = True
        extra["niche_rank"] = args.niche_rank
    if getattr(args, "use_niche_film", False):
        extra["use_niche_film"] = True
        extra["niche_film_dim"] = getattr(args, "niche_film_dim", 8)
        extra["niche_film_k"] = getattr(args, "niche_film_k", 25)
        extra["niche_film_gcap"] = getattr(args, "niche_film_gcap", 3.0)
        extra["niche_film_l2"] = getattr(args, "niche_film_l2", 1e-3)
        extra["niche_film_rank"] = getattr(args, "niche_film_rank", 8)
        extra["niche_film_dispersion"] = getattr(args, "niche_film_dispersion", False)
    if args.use_pert_gate:
        extra["use_pert_gate"] = True
        extra["gate_hidden"] = args.gate_hidden
    if args.use_expr_encoder:
        extra["use_expr_encoder"] = True
        extra["expr_hidden_dims"] = args.expr_hidden_dims
    if getattr(args, "spatial_basal_only", False):
        extra["spatial_basal_only"] = True
    _add_fpert_kwargs(extra, args)
    model, device = build_model(adata, pos_batched, cell_atts, cutoff_t, n_batch,
                                use_basal_encoder_only=args.use_basal_encoder_only,
                                **extra)
    model.load_model(args.model_file, allow_cell_size_mismatch=True)
    if args.use_nngp:
        if not args.graph_file:
            raise ValueError("--graph_file required when --use_nngp True")
        from spatial_graph import SpatialGraph
        graph = SpatialGraph.load(args.graph_file, device=device)
        model.attach_graph(graph)
        logging.info("Graph attached (N=%d)", graph.knn_idx.shape[0])
    model.eval()

    import scipy.sparse as sp
    Y_expr = adata.X
    Y_expr = Y_expr.toarray() if sp.issparse(Y_expr) else np.asarray(Y_expr)
    Y_expr = Y_expr.astype(np.float32)
    Y_raw = adata.raw.X if adata.raw is not None else adata.X
    Y_obs = Y_raw.toarray() if sp.issparse(Y_raw) else np.asarray(Y_raw)
    Y_obs = Y_obs.astype(np.float32)
    n_vars = adata.n_vars
    gene_names = (np.asarray(adata.var_names) if adata.var_names is not None else
                  np.array([f"g{i}" for i in range(n_vars)]))
    del adata
    import gc
    gc.collect()

    cls = classify_perts(perturb_raw)               # {"unassigned","control","gene_KO"} per cell
    name2code = {v: k for k, v in pert_name.items()}
    coords = pos_batched[:, :2].astype(np.float64)  # scaled x/y in [0, loc_range]

    return dict(model=model, device=device, N=N, Y_expr=Y_expr, Y_obs=Y_obs, gene_names=gene_names,
                pos_batched=pos_batched.astype(np.float32),
                sample_indices=torch.arange(N), cell_atts=cell_atts.astype(int),
                coords=coords, perturb_raw=np.asarray(perturb_raw), cls=cls,
                pert_name=pert_name, name2code=name2code, n_batch=n_batch)


def setup_grid_and_panel(ctx, args):
    """Build the niche grid + evaluable mask + perturbation panel on the FILTERED arrays."""
    bin_id, n_bins = eu.build_niche_grid(ctx["coords"], args.niche_grid, args.loc_range)
    una_mask = ctx["cls"] == "unassigned"
    ntc_mask = ctx["cls"] == "control"
    una_per_bin = np.bincount(bin_id[una_mask], minlength=n_bins)
    ntc_per_bin = np.bincount(bin_id[ntc_mask], minlength=n_bins)
    evaluable = una_per_bin >= args.una_floor   # gate on unassigned (source cells)
    n_eval = int(evaluable.sum())
    frac_below = 1.0 - n_eval / n_bins
    logging.info("Evaluable niches: %d/%d (una_floor=%d). unassigned/bin: %s",
                 n_eval, n_bins, args.una_floor, una_per_bin.tolist())
    logging.info("NTC/bin (secondary, LFC-w/o gate=%d): %s", args.ntc_floor, ntc_per_bin.tolist())
    if frac_below > 0.25:
        logging.warning("%.0f%% of bins below una_floor -- consider a coarser --niche_grid",
                        100 * frac_below)

    # Panel: only gene_KO labels are candidates (NTC/unassigned mapped to an excluded sentinel).
    labels_for_panel = np.where(ctx["cls"] == "gene_KO", ctx["perturb_raw"], "unassigned")
    cross_counts = ctx.get("cross_slide_counts")
    if args.panel_genes:
        forced = [g.strip() for g in args.panel_genes.split(",") if g.strip()]
        present_counts = {g: int((labels_for_panel == g).sum()) for g in forced}
        panel = [g for g in forced if present_counts.get(g, 0) >= args.min_cells_per_pert]
        logging.info("Forced panel (%d/%d genes present with >=%d cells): %s",
                     len(panel), len(forced), args.min_cells_per_pert, panel)
    elif cross_counts and not getattr(args, "within_slide_panel", False):
        # Cross-slide panel: rank by abundance in OTHER slides; require >= min_cells_per_pert
        # in THIS slide so ground-truth KO cells exist for evaluation.
        from eval_utils import EXCLUDE_LABELS as _EX
        target_counts = Counter(labels_for_panel.tolist())
        candidates = [
            g for g in cross_counts
            if g not in _EX and target_counts.get(g, 0) >= args.min_cells_per_pert
        ]
        panel = sorted(candidates, key=lambda g: -cross_counts[g])[:12]
        logging.info("Cross-slide panel (%d genes, ranked by other-slide abundance): %s",
                     len(panel), panel)
        for g in panel:
            logging.info("  %s: target=%d, other_slides=%d", g, target_counts[g], cross_counts[g])
    else:
        panel = eu.select_perturbation_panel(labels_for_panel, bin_id, n_eval,
                                             min_cells_per_niche=args.min_g_cells)
    if args.limit_panel and len(panel) > args.limit_panel:
        panel = panel[:args.limit_panel]
    logging.info("Selected panel (%d): %s", len(panel), panel)
    # Sparsity-robust eval gene set: HVGs recomputed on normalized data (the raw counts
    # are ~96% zero, so per-gene metrics on all genes measure dropout, not signal).
    hvg_mask = eu.select_hvgs(ctx["Y_obs"], n_top=args.n_hvg)
    logging.info("HVG eval genes: %d/%d (top-%d by normalized dispersion)",
                 int(hvg_mask.sum()), len(hvg_mask), args.n_hvg)
    ctx.update(bin_id=bin_id, n_bins=n_bins, evaluable=evaluable, ntc_per_bin=ntc_per_bin,
               panel=panel, hvg_mask=hvg_mask)
    return ctx


def _predict_swap(ctx, idx, target_code, args):
    """Predicted expression (n_cells, G) for cells `idx` counterfactually set to
    perturbation `target_code` (uses the trained basal encoder via Y, then re-adds
    the target's f_pert/mu_action shift). Mirrors run_concert_spac's eval call."""
    idx = np.asarray(idx, dtype=int)
    means, _ = ctx["model"].counterfactualPrediction(
        X=ctx["pos_batched"][idx],
        sample_index=ctx["sample_indices"][idx],
        cell_atts=ctx["cell_atts"][idx].copy(),
        perturb_cell_id=torch.arange(len(idx)),
        target_cell_tissue=0,
        target_cell_perturbation=int(target_code),
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        Y=ctx["Y_expr"][idx],
    )
    return means


def run_scenario_c(ctx, args, src_restrict=None):
    """Seen-cell pert swap -- TWO LFC flavours, TWO ceilings, kNN baseline for each.

    Decoder: log μ = f_basal + 1[has_guide]·μ_action + 1[is_real_KO]·f_pert(g)
      target_code > 0  ->  has_guide=1, is_real_KO=1  ->  f_basal + μ_action + f_pert(g)
      target_code = 0  ->  has_guide=0, is_real_KO=0  ->  f_basal only

    LFC WITH noise  (μ_action + f_pert(g)):
      model:    pred(src->g, code>0) - pred(src->0)          [decoder diff; encoder cancels]
      observed: gene_KO - unassigned                        [= μ_action + f_pert(g)]
      ceiling:  split-half reliability of gene_KO vs unassigned

    LFC WITHOUT noise  (f_pert(g) only):
      model:    (pred(src->g) - pred(src->0)) - μ_action     [subtract learned global param]
      observed: gene_KO - NTC                               [= f_pert(g), μ_action cancels]
      ceiling:  split-half reliability of gene_KO vs NTC

    src_restrict: optional boolean mask restricting SOURCE cells (scenario b).
    """
    bin_id, evaluable = ctx["bin_id"], ctx["evaluable"]
    cls, perturb_raw, gene_names = ctx["cls"], ctx["perturb_raw"], ctx["gene_names"]
    Y_obs  = ctx["Y_obs"]
    coords = ctx["coords"]
    eval_bins = np.where(evaluable)[0]
    restrict  = np.ones(ctx["N"], dtype=bool) if src_restrict is None else np.asarray(src_restrict)
    src_masks = {"L1": (cls == "control") & restrict, "L2": (cls == "unassigned") & restrict}

    # Global μ_action parameter -- used to compute "without noise" model LFC.
    mu_action = ctx["model"].mu_action.detach().cpu().numpy()  # (G,)

    # Reference pools for observed LFC and kNN baselines.
    ntc_all = np.where(cls == "control")[0]
    una_all = np.where(cls == "unassigned")[0]
    g_real_all = {g: np.where(perturb_raw == g)[0] for g in ctx["panel"]}

    # Per-level per-gene accumulators for BOTH LFC flavours.
    lvs = list(src_masks.keys())
    scores_with    = {lv: {g: [] for g in ctx["panel"]} for lv in lvs}
    scores_without = {lv: {g: [] for g in ctx["panel"]} for lv in lvs}
    knn_scores_with    = {lv: {g: [] for g in ctx["panel"]} for lv in lvs}
    knn_scores_without = {lv: {g: [] for g in ctx["panel"]} for lv in lvs}
    model_lfc_with    = {lv: {g: {} for g in ctx["panel"]} for lv in lvs}
    model_lfc_without = {lv: {g: {} for g in ctx["panel"]} for lv in lvs}
    # Absolute pseudobulk Spearman: pred(unassigned->g) vs observed KO -- L2 source only.
    # Captures both spatial f_basal and f_pert; meaningful for vs-baseline comparison.
    abs_scores     = {g: [] for g in ctx["panel"]}
    knn_abs_scores = {g: [] for g in ctx["panel"]}
    # Energy distance: population-level distributional similarity (not pseudobulk).
    # Lower = predicted and observed KO distributions more similar.
    edist_scores     = {g: [] for g in ctx["panel"]}
    knn_edist_scores = {g: [] for g in ctx["panel"]}

    for lv, src_mask in src_masks.items():
        for b in eval_bins:
            src_b = np.where(src_mask & (bin_id == b))[0]
            if len(src_b) < args.min_g_cells:
                continue
            ntc_b = np.where((cls == "control")   & (bin_id == b))[0]
            una_b = np.where((cls == "unassigned") & (bin_id == b))[0]
            if len(una_b) < args.min_g_cells:   # gate on unassigned (source cells)
                continue
            has_ntc = len(ntc_b) >= args.ntc_floor  # secondary: enables LFC-without-noise
            # Empirical NTC re-reference offset = log(mean_src) - log(mean_NTC), observed-scale LFC.
            # Used to put the model's KO prediction on the NTC baseline WITHOUT trusting model.mu_action.
            emp_offset = eu.pseudobulk_lfc(Y_obs[una_b], Y_obs[ntc_b]) if has_ntc else None

            # Shared base prediction: target_code=0 -> pure f_basal, no CRISPR terms.
            pred_base = _predict_swap(ctx, src_b, 0, args)

            # Leave-one-niche-out pools: exclude test niche b so kNN baseline
            # cannot copy from the same cells that define the observed LFC.
            ntc_lono = ntc_all[bin_id[ntc_all] != b]
            una_lono = una_all[bin_id[una_all] != b]

            # kNN baselines -- one per reference population (shared across all g).
            knn_base_ntc = eu.spatial_knn_mean(coords[src_b], coords[ntc_lono],
                                               Y_obs[ntc_lono], k=args.knn_k) if has_ntc else None
            knn_base_una = eu.spatial_knn_mean(coords[src_b], coords[una_lono],
                                               Y_obs[una_lono], k=args.knn_k)

            for g in ctx["panel"]:
                g_code    = ctx["name2code"][g]
                g_real_b  = np.where((perturb_raw == g) & (bin_id == b))[0]
                if len(g_real_b) < args.min_g_cells:
                    continue

                pred_g = _predict_swap(ctx, src_b, g_code, args)
                keep   = ctx["hvg_mask"] & (gene_names != g)

                # --- Model LFCs ---
                m_with    = eu.pseudobulk_lfc(pred_g, pred_base)   # μ_action + f_pert(g)
                # Clean KO-vs-NTC LFC via EMPIRICAL re-reference (robust to mis-estimated model.mu_action):
                #   predicted_clean = stored_LFC(model KO-vs-basal) + (log mean_src - log mean_NTC)
                # vs observed true_clean = log(mean_realKO) - log(mean_NTC) = o_without below.
                m_without = (m_with + emp_offset) if has_ntc else (m_with - mu_action)

                # --- Observed LFCs ---
                o_with    = eu.pseudobulk_lfc(Y_obs[g_real_b], Y_obs[una_b])   # μ_action + f_pert(g)
                o_without = eu.pseudobulk_lfc(Y_obs[g_real_b], Y_obs[ntc_b]) if has_ntc else None

                # --- Score both flavours ---
                sc_w  = eu.score_lfc_agreement(m_with[keep], o_with[keep]); sc_w["niche"] = int(b)
                scores_with[lv][g].append(sc_w)
                model_lfc_with[lv][g][int(b)] = m_with
                if has_ntc and o_without is not None:
                    sc_wo = eu.score_lfc_agreement(m_without[keep], o_without[keep]); sc_wo["niche"] = int(b)
                    scores_without[lv][g].append(sc_wo)
                    model_lfc_without[lv][g][int(b)] = m_without
                    # DIAG: decompose the without-noise score -- is it model skill or the
                    # shared log-mean(NTC) baseline term in (m_with+emp_offset) vs o_without?
                    import builtins as _b
                    _d = getattr(_b, "_CF_DIAG", None)
                    if _d is None:
                        _d = _b._CF_DIAG = {"emp": [], "model_clean": [], "null_offset": []}
                        import atexit, numpy as _np
                        atexit.register(lambda: logging.info(
                            "[DIAG without-noise n=%d] reported(m_with+emp_offset)=%.3f  "
                            "model_clean(m_with-mu_action)=%.3f  null(emp_offset_alone)=%.3f",
                            len(_d["emp"]), _np.nanmean(_d["emp"]),
                            _np.nanmean(_d["model_clean"]), _np.nanmean(_d["null_offset"])))
                    _d["emp"].append(sc_wo["spearman"])
                    _d["model_clean"].append(
                        eu.score_lfc_agreement((m_with - mu_action)[keep], o_without[keep])["spearman"])
                    _d["null_offset"].append(
                        eu.score_lfc_agreement(emp_offset[keep], o_without[keep])["spearman"])

                # --- kNN LFCs ---
                g_lono = g_real_all[g][bin_id[g_real_all[g]] != b]
                has_knn = len(g_lono) >= args.min_g_cells
                if has_knn:
                    knn_g = eu.spatial_knn_mean(coords[src_b], coords[g_lono],
                                                Y_obs[g_lono], k=args.knn_k)
                    k_w  = eu.pseudobulk_lfc(knn_g, knn_base_una)
                    ksc_w = eu.score_lfc_agreement(k_w[keep], o_with[keep]); ksc_w["niche"] = int(b)
                    knn_scores_with[lv][g].append(ksc_w)
                    if has_ntc and knn_base_ntc is not None and o_without is not None:
                        k_wo = eu.pseudobulk_lfc(knn_g, knn_base_ntc)
                        ksc_wo = eu.score_lfc_agreement(k_wo[keep], o_without[keep]); ksc_wo["niche"] = int(b)
                        knn_scores_without[lv][g].append(ksc_wo)

                # --- Absolute pseudobulk Spearman (L2/unassigned source only) ---
                if lv == "L2":
                    hvg = ctx["hvg_mask"]
                    pred_mean = pred_g[:, hvg].mean(axis=0).astype(np.float64)
                    obs_mean  = Y_obs[g_real_b][:, hvg].mean(axis=0).astype(np.float64)
                    abs_scores[g].append(dict(spearman=_pseudobulk_spearman(pred_mean, obs_mean),
                                             niche=int(b)))
                    ed_m = eu.energy_distance(pred_g, Y_obs[g_real_b], hvg)
                    edist_scores[g].append(dict(edist=ed_m, niche=int(b)))
                    if has_knn:
                        knn_mean = knn_g[:, hvg].mean(axis=0).astype(np.float64) if knn_g.ndim == 2 \
                                   else knn_g[hvg].astype(np.float64)
                        knn_abs_scores[g].append(dict(spearman=_pseudobulk_spearman(knn_mean, obs_mean),
                                                      niche=int(b)))
                        knn_pop = knn_g if knn_g.ndim == 2 else knn_g[np.newaxis, :]
                        ed_k = eu.energy_distance(knn_pop, Y_obs[g_real_b], hvg)
                        knn_edist_scores[g].append(dict(edist=ed_k, niche=int(b)))

    # --- Aggregate ---
    def _agg(score_dict):
        out = {lv: {} for lv in lvs}
        for lv in lvs:
            for g in ctx["panel"]:
                rows = score_dict[lv][g]
                if not rows:
                    continue
                sp = np.array([r["spearman"] for r in rows], dtype=float)
                out[lv][g] = dict(n_niches=len(rows),
                                  spearman_mean=float(np.nanmean(sp)),
                                  spearman_sd=float(np.nanstd(sp)),
                                  sign_mean=float(np.nanmean([r["sign_agreement"] for r in rows])),
                                  topk_mean=float(np.nanmean([r["topk_overlap"] for r in rows])))
        return out

    agg_with    = _agg(scores_with)
    agg_without = _agg(scores_without)
    knn_agg_with    = _agg(knn_scores_with)
    knn_agg_without = _agg(knn_scores_without)

    def _agg_abs(score_dict):
        out = {}
        for g in ctx["panel"]:
            rows = score_dict[g]
            if not rows:
                continue
            sp = np.array([r["spearman"] for r in rows], dtype=float)
            out[g] = dict(n_niches=len(rows),
                          spearman_mean=float(np.nanmean(sp)),
                          spearman_sd=float(np.nanstd(sp)))
        return out

    abs_agg     = _agg_abs(abs_scores)
    knn_abs_agg = _agg_abs(knn_abs_scores)
    for g in ctx["panel"]:
        av = abs_agg.get(g); kv = knn_abs_agg.get(g)
        if av:
            logging.info("[c/abs_spearman] %-12s n=%2d model=%.3f±%.3f knn=%.3f",
                         g, av["n_niches"], av["spearman_mean"], av["spearman_sd"],
                         kv["spearman_mean"] if kv else float("nan"))

    def _agg_edist(score_dict):
        out = {}
        for g in ctx["panel"]:
            rows = score_dict[g]
            if not rows:
                continue
            ed = np.array([r["edist"] for r in rows], dtype=float)
            out[g] = dict(n_niches=len(rows),
                          edist_mean=float(np.nanmean(ed)),
                          edist_sd=float(np.nanstd(ed)))
        return out

    edist_agg     = _agg_edist(edist_scores)
    knn_edist_agg = _agg_edist(knn_edist_scores)
    for g in ctx["panel"]:
        ev = edist_agg.get(g); kev = knn_edist_agg.get(g)
        if ev:
            logging.info("[c/edist] %-12s n=%2d model=%.3f±%.3f knn=%.3f (lower=better)",
                         g, ev["n_niches"], ev["edist_mean"], ev["edist_sd"],
                         kev["edist_mean"] if kev else float("nan"))

    for lv in lvs:
        for g in ctx["panel"]:
            aw = agg_with[lv].get(g, {}); awo = agg_without[lv].get(g, {})
            kw = knn_agg_with[lv].get(g, {}); kwo = knn_agg_without[lv].get(g, {})
            if not aw:
                continue
            logging.info("[c/%s/with_noise]    %-12s n=%2d model=%.3f±%.3f knn=%.3f sign=%.3f",
                         lv, g, aw["n_niches"], aw["spearman_mean"], aw["spearman_sd"],
                         kw.get("spearman_mean", float("nan")), aw["sign_mean"])
            logging.info("[c/%s/without_noise] %-12s n=%2d model=%.3f±%.3f knn=%.3f sign=%.3f",
                         lv, g, awo.get("n_niches", 0), awo.get("spearman_mean", float("nan")),
                         awo.get("spearman_sd", float("nan")),
                         kwo.get("spearman_mean", float("nan")), awo.get("sign_mean", float("nan")))

    # --- L1↔L2 convergence on the "with noise" model LFC ---
    conv = {}
    for g in ctx["panel"]:
        shared = sorted(set(model_lfc_with["L1"][g]) & set(model_lfc_with["L2"][g]))
        rhos = [eu.score_lfc_agreement(model_lfc_with["L2"][g][b],
                                       model_lfc_with["L1"][g][b])["spearman"] for b in shared]
        if rhos:
            conv[g] = dict(n_shared_niches=len(shared), l1_l2_spearman_mean=float(np.nanmean(rhos)))
            logging.info("[c/convergence] %-12s shared=%2d L1<->L2=%.3f",
                         g, len(shared), conv[g]["l1_l2_spearman_mean"])

    # --- Identity: pred(NTC -> 0) should recover unassigned basal profile ---
    # target=0 -> f_basal only (strips μ_action); ground truth = raw unassigned.
    identity = {}
    id_scores = []
    for b in eval_bins:
        ntc_b = np.where((cls == "control")   & (bin_id == b))[0]
        una_b = np.where((cls == "unassigned") & (bin_id == b))[0]
        if len(ntc_b) < args.min_g_cells or len(una_b) < args.min_g_cells:
            continue
        pred_basal = _predict_swap(ctx, ntc_b, 0, args)   # strip μ_action from NTC
        # Populations differ in size -> compare pseudobulk mean profiles over HVGs
        hvg = ctx["hvg_mask"]
        pred_mean = pred_basal[:, hvg].mean(axis=0).astype(np.float64)
        obs_mean  = Y_obs[una_b][:, hvg].mean(axis=0).astype(np.float64)
        rp = pred_mean.argsort().argsort().astype(np.float64)
        ro = obs_mean.argsort().argsort().astype(np.float64)
        rp -= rp.mean(); ro -= ro.mean()
        denom = np.sqrt((rp**2).sum() * (ro**2).sum())
        id_scores.append(float((rp * ro).sum() / denom) if denom > 0 else float("nan"))
    if id_scores:
        identity = dict(denoise_ntc_vs_unassigned_spearman=float(np.nanmean(id_scores)))
        logging.info("[c/identity] denoise(NTC->0) vs observed-unassigned Spearman=%.3f "
                     "(want >raw-NTC-vs-unassigned to confirm μ_action stripping works)",
                     identity["denoise_ntc_vs_unassigned_spearman"])

    # --- Cell-state consistency (data-only signal detection) ---
    cell_consistency = eu.cell_state_consistency_test(
        Y_obs, perturb_raw, bin_id, ctx["panel"],
        ntc_label="control", gene_mask=ctx["hvg_mask"], n_pairs=500, n_seeds=5)
    for g, v in cell_consistency.items():
        logging.info("[c/cell-consist] %-12s n=%2d same=%.4f cross=%.4f effect=%.4f p=%.3g",
                     g, v["n_niches"], v["same_mean"], v["cross_mean"],
                     v["effect_size"], v["mw_pvalue_median"])

    # --- Reliability ceilings -- one per LFC flavour ---
    ceiling_with    = {}   # gene_KO vs unassigned
    ceiling_without = {}   # gene_KO vs NTC
    for g in ctx["panel"]:
        keep = ctx["hvg_mask"] & (gene_names != g)
        rw, rwo = [], []
        for b in eval_bins:
            g_b   = np.where((perturb_raw == g) & (bin_id == b))[0]
            ntc_b = np.where((cls == "control")   & (bin_id == b))[0]
            una_b = np.where((cls == "unassigned") & (bin_id == b))[0]
            if len(g_b) < args.min_g_cells:
                continue
            if len(una_b) >= args.min_g_cells:
                r, sb = eu.split_half_lfc_reliability(Y_obs[g_b], Y_obs[una_b], gene_mask=keep, n_seeds=5)
                if np.isfinite(r):
                    rw.append((r, sb))
            if len(ntc_b) >= args.min_g_cells:
                r, sb = eu.split_half_lfc_reliability(Y_obs[g_b], Y_obs[ntc_b], gene_mask=keep, n_seeds=5)
                if np.isfinite(r):
                    rwo.append((r, sb))
        if rw:
            ceiling_with[g] = dict(n_niches=len(rw),
                                   split_half_mean=float(np.nanmean([x[0] for x in rw])),
                                   spearman_brown_mean=float(np.nanmean([x[1] for x in rw])))
        if rwo:
            ceiling_without[g] = dict(n_niches=len(rwo),
                                      split_half_mean=float(np.nanmean([x[0] for x in rwo])),
                                      spearman_brown_mean=float(np.nanmean([x[1] for x in rwo])))
        logging.info("[c/ceiling] %-12s with_noise SB=%.3f  without_noise SB=%.3f",
                     g, ceiling_with.get(g, {}).get("spearman_brown_mean", float("nan")),
                     ceiling_without.get(g, {}).get("spearman_brown_mean", float("nan")))

    return dict(
        aggregate_with=agg_with, aggregate_without=agg_without,
        knn_aggregate_with=knn_agg_with, knn_aggregate_without=knn_agg_without,
        abs_agg=abs_agg, knn_abs_agg=knn_abs_agg,
        edist_agg=edist_agg, knn_edist_agg=knn_edist_agg,
        ceiling_with=ceiling_with, ceiling_without=ceiling_without,
        cell_consistency=cell_consistency,
        convergence=conv, identity=identity,
        model_lfc_with=model_lfc_with, model_lfc_without=model_lfc_without,
        per_niche_scores_with=scores_with, per_niche_scores_without=scores_without,
        knn_per_niche_scores_with=knn_scores_with,
        per_niche_edist=edist_scores, knn_per_niche_edist=knn_edist_scores,
    )


def _pseudobulk_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two 1-D pseudobulk mean vectors (over genes)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = a.argsort().argsort().astype(np.float64); ra -= ra.mean()
    rb = b.argsort().argsort().astype(np.float64); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _per_gene_spearman_mean(pred, obs):
    """Mean over genes of the per-gene Spearman correlation across cells.
    Rank-based, so robust to the count-scale (pred) vs log-normalized (obs) mismatch
    (monotone). Ordinal ranks (ties broken arbitrarily) -- fine for a comparison metric."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if pred.shape[0] < 3:
        return float("nan")
    rp = pred.argsort(axis=0).argsort(axis=0).astype(np.float64)
    ro = obs.argsort(axis=0).argsort(axis=0).astype(np.float64)
    rp -= rp.mean(axis=0)
    ro -= ro.mean(axis=0)
    num = (rp * ro).sum(axis=0)
    den = np.sqrt((rp ** 2).sum(axis=0) * (ro ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / den
    return float(np.nanmean(corr))


def run_scenario_b(ctx, args):
    """Held-out cell (random 5%). Imputation scored on held-out NTC (clean basal vs
    observed); counterfactual = held-out source->g via the niche-LFC machinery."""
    rng = np.random.RandomState(args.seed)
    heldout = rng.rand(ctx["N"]) < 0.05
    insample = ~heldout
    cls, coords = ctx["cls"], ctx["coords"]
    Y_enc, Y_obs = ctx["Y_expr"], ctx["Y_obs"]   # encoder input vs observed (positive) scoring
    model = ctx["model"]
    ho_ntc = np.where(heldout & (cls == "control"))[0]

    imp = {}
    if len(ho_ntc) >= args.min_g_cells:
        pred_clean = model.batching_denoise_counts(
            X=ctx["pos_batched"][ho_ntc], sample_index=ctx["sample_indices"][ho_ntc],
            cell_atts=ctx["cell_atts"][ho_ntc], force_basal=True,
            n_samples=args.n_samples, batch_size=args.batch_size, Y=Y_enc[ho_ntc])[0]
        base = eu.spatial_knn_mean(coords[ho_ntc], coords[insample], Y_obs[insample], k=args.knn_k)
        # in-sample gap: same imputation on an equal-sized in-sample NTC subset
        is_ntc = np.where(insample & (cls == "control"))[0]
        if len(is_ntc) > len(ho_ntc):
            is_ntc = rng.choice(is_ntc, len(ho_ntc), replace=False)
        pred_is = model.batching_denoise_counts(
            X=ctx["pos_batched"][is_ntc], sample_index=ctx["sample_indices"][is_ntc],
            cell_atts=ctx["cell_atts"][is_ntc], force_basal=True,
            n_samples=args.n_samples, batch_size=args.batch_size, Y=Y_enc[is_ntc])[0]
        hvg = ctx["hvg_mask"]
        zero_bin = np.zeros(len(ho_ntc), dtype=int)
        # primary metrics = sparsity-robust (per-cell across HVGs, single-bulk pseudobulk on HVGs)
        imp = dict(n_heldout_ntc=int(len(ho_ntc)),
                   model_spearman=eu.per_cell_spearman_mean(pred_clean, Y_obs[ho_ntc], hvg),
                   knn_baseline_spearman=eu.per_cell_spearman_mean(base, Y_obs[ho_ntc], hvg),
                   insample_spearman=eu.per_cell_spearman_mean(pred_is, Y_obs[is_ntc], hvg),
                   pseudobulk_spearman=eu.pseudobulk_profile_score(
                       pred_clean, Y_obs[ho_ntc], zero_bin, 1, hvg, min_cells=1)[0],
                   diag_pergene_allgenes=_per_gene_spearman_mean(pred_clean, Y_obs[ho_ntc]))
        logging.info("[b/impute] held-out NTC=%d  per-cell/HVG model=%.3f knn=%.3f in-sample=%.3f"
                     "  pseudobulk/HVG=%.3f  (diag per-gene/all=%.3f)",
                     imp["n_heldout_ntc"], imp["model_spearman"], imp["knn_baseline_spearman"],
                     imp["insample_spearman"], imp["pseudobulk_spearman"], imp["diag_pergene_allgenes"])

    logging.info("[b] counterfactual: held-out source -> g (niche-LFC, both levels)")
    cf = run_scenario_c(ctx, args, src_restrict=heldout)
    return dict(imputation=imp, counterfactual=cf)


def run_scenario_a(ctx, args):
    """Coord-only: held-out contiguous patches. Imputation at patch coords vs observed;
    counterfactual via impute_and_counterfactual_fun2, scored per niche where >= min_g
    real g cells fall in the patch (else qualitative)."""
    coords = ctx["coords"]
    Y_enc, Y_obs = ctx["Y_expr"], ctx["Y_obs"]   # encoder input vs observed (positive) scoring
    pos, atts, idx = ctx["pos_batched"], ctx["cell_atts"], ctx["sample_indices"]
    bin_id, perturb_raw, gene_names = ctx["bin_id"], ctx["perturb_raw"], ctx["gene_names"]
    model = ctx["model"]
    patch = eu.make_spatial_patches(coords, k=4, loc_range=args.loc_range,
                                    area_frac=0.12, seed=args.seed)
    train = ~patch
    patch_ids = np.where(patch)[0]
    rng = np.random.RandomState(args.seed)
    score_ids = patch_ids
    if len(patch_ids) > args.patch_impute_cap:   # bound the O(N_test) nearest-neighbor loop
        score_ids = np.sort(rng.choice(patch_ids, args.patch_impute_cap, replace=False))
    logging.info("[a] patch cells=%d (scoring %d), train=%d", len(patch_ids),
                 len(score_ids), int(train.sum()))

    # imputation() expects X_test = coords only (2 cols); it appends the nearest-train
    # batch columns itself. X_train stays full pos_batched. Y_train = encoder input.
    _, pred_imp = model.imputation(
        X_test=pos[score_ids][:, :2], X_train=pos[train], Y_sample_index=idx[train],
        Y_cell_atts=atts[train], Y_train=Y_enc[train],
        n_samples=args.n_samples, batch_size=args.batch_size)
    base = eu.spatial_knn_mean(coords[score_ids], coords[train], Y_obs[train], k=args.knn_k)
    hvg = ctx["hvg_mask"]
    sc_bins = bin_id[score_ids]
    pb_model, n_nb = eu.pseudobulk_profile_score(pred_imp, Y_obs[score_ids], sc_bins,
                                                 ctx["n_bins"], hvg, min_cells=args.min_g_cells)
    pb_knn, _ = eu.pseudobulk_profile_score(base, Y_obs[score_ids], sc_bins,
                                            ctx["n_bins"], hvg, min_cells=args.min_g_cells)
    imp = dict(n_patch=int(len(patch_ids)), n_scored=int(len(score_ids)),
               model_spearman=eu.per_cell_spearman_mean(pred_imp, Y_obs[score_ids], hvg),
               knn_baseline_spearman=eu.per_cell_spearman_mean(base, Y_obs[score_ids], hvg),
               pseudobulk_spearman=pb_model, pseudobulk_knn=pb_knn, n_niche_bulk=n_nb,
               diag_pergene_allgenes=_per_gene_spearman_mean(pred_imp, Y_obs[score_ids]))
    logging.info("[a/impute] per-cell/HVG model=%.3f knn=%.3f  pseudobulk/HVG model=%.3f knn=%.3f "
                 "(%d niches)  (diag per-gene/all=%.3f)", imp["model_spearman"],
                 imp["knn_baseline_spearman"], pb_model, pb_knn, n_nb, imp["diag_pergene_allgenes"])

    # Counterfactual: control reference at patch coords, then each panel g.
    ctrl_code = ctx["name2code"]["control"]
    # fun2 (unlike imputation) does NOT re-append train batch cols -> needs FULL pos_batched X_test.
    pred_ctrl = model.impute_and_counterfactual_fun2(
        target=ctrl_code, tissue=0, X_test=pos[score_ids], X_train=pos[train],
        Y_sample_index=idx[train], Y_cell_atts=atts[train], Y_train=Y_enc[train],
        n_samples=args.n_samples, batch_size=args.batch_size)[1]
    patch_bins = bin_id[score_ids]
    cf = {}
    for g in ctx["panel"]:
        g_code = ctx["name2code"][g]
        pred_g = model.impute_and_counterfactual_fun2(
            target=g_code, tissue=0, X_test=pos[score_ids], X_train=pos[train],
            Y_sample_index=idx[train], Y_cell_atts=atts[train], Y_train=Y_enc[train],
            n_samples=args.n_samples, batch_size=args.batch_size)[1]
        keep = ctx["hvg_mask"] & (gene_names != g)
        rows = []
        for b in np.unique(patch_bins):
            in_b = patch_bins == b
            if in_b.sum() < args.min_g_cells:
                continue
            g_real_b = np.where((perturb_raw == g) & (bin_id == b))[0]
            ntc_b = np.where((ctx["cls"] == "control") & (bin_id == b))[0]
            if len(g_real_b) < args.min_g_cells or len(ntc_b) < args.min_g_cells:
                continue
            m_lfc = eu.pseudobulk_lfc(pred_g[in_b], pred_ctrl[in_b])
            o_lfc = eu.pseudobulk_lfc(Y_obs[g_real_b], Y_obs[ntc_b])
            sc = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
            sc["niche"] = int(b)
            rows.append(sc)
        if rows:
            sp = np.array([r["spearman"] for r in rows], dtype=float)
            cf[g] = dict(n_niches=len(rows), spearman_mean=float(np.nanmean(sp)))
            logging.info("[a/cf] %-12s niches=%2d spearman=%.3f", g, len(rows), cf[g]["spearman_mean"])
        else:
            logging.info("[a/cf] %-12s qualitative only (no niche with >=%d real-g + NTC in patch)",
                         g, args.min_g_cells)
    return dict(imputation=imp, counterfactual=cf)


def write_report(results, ctx, args, outdir):
    """Write eval_summary.json, lfc_per_niche.npz (footprint arrays), eval_report.md,
    and log a one-line PASS/CONCERN per scenario (PASS = model beats its baseline /
    identity controls hold)."""
    summary = {"panel": ctx["panel"], "n_evaluable_niches": int(ctx["evaluable"].sum()),
               "n_bins": int(ctx["n_bins"]), "ntc_per_bin": ctx["ntc_per_bin"].tolist(),
               "args": {k: v for k, v in vars(args).items()}}
    npz = {}
    verdicts = {}
    lines = ["# CONCERT-spac spatial counterfactual eval", "",
             f"- model: `{args.model_file}`",
             f"- evaluable niches: {summary['n_evaluable_niches']}/{summary['n_bins']}",
             f"- panel ({len(ctx['panel'])}): {', '.join(ctx['panel'])}", ""]

    if "c" in results:
        c = results["c"]
        summary["c"] = {
            "aggregate_with": c["aggregate_with"], "aggregate_without": c["aggregate_without"],
            "knn_aggregate_with": c["knn_aggregate_with"], "knn_aggregate_without": c["knn_aggregate_without"],
            "ceiling_with": c["ceiling_with"], "ceiling_without": c["ceiling_without"],
            "cell_consistency": c.get("cell_consistency", {}),
            "convergence": c["convergence"], "identity": c["identity"],
        }
        for lv in c["model_lfc_with"]:
            for g, by_b in c["model_lfc_with"][lv].items():
                for b, arr in by_b.items():
                    npz[f"c_{lv}_{g}_niche{b}_with"]    = np.asarray(arr, dtype=np.float32)
            for g, by_b in c["model_lfc_without"][lv].items():
                for b, arr in by_b.items():
                    npz[f"c_{lv}_{g}_niche{b}_without"] = np.asarray(arr, dtype=np.float32)

        def _mean_sp(agg, lv="L1"):
            vals = [v["spearman_mean"] for v in agg.get(lv, {}).values()]
            return float(np.nanmean(vals)) if vals else float("nan")

        mw  = _mean_sp(c["aggregate_with"])
        mwo = _mean_sp(c["aggregate_without"])
        kw  = _mean_sp(c["knn_aggregate_with"])
        kwo = _mean_sp(c["knn_aggregate_without"])
        cw  = float(np.nanmean([v["spearman_brown_mean"] for v in c["ceiling_with"].values()])) \
              if c["ceiling_with"] else float("nan")
        cwo = float(np.nanmean([v["spearman_brown_mean"] for v in c["ceiling_without"].values()])) \
              if c["ceiling_without"] else float("nan")
        id_sp = c["identity"].get("denoise_ntc_vs_unassigned_spearman") if c["identity"] else None
        id_ok = id_sp is not None and id_sp > 0
        passed = (mwo > 0) and id_ok   # primary verdict on noise-free flavour
        verdicts["c"] = dict(passed=bool(passed),
                             l1_with_model=mw, l1_with_knn=kw, ceiling_with=cw,
                             l1_without_model=mwo, l1_without_knn=kwo, ceiling_without=cwo,
                             identity_denoise_spearman=id_sp)
        lines += [f"## (c) seen-cell pert swap -- **{'PASS' if passed else 'CONCERN'}**",
                  "",
                  "| | model ρ (L1) | kNN ρ | ceiling (SB) |",
                  "|---|---|---|---|",
                  f"| **With noise** (gene_KO - unassigned) | {mw:.3f} | {kw:.3f} | {cw:.3f} |",
                  f"| **Without noise** (gene_KO - NTC) | {mwo:.3f} | {kwo:.3f} | {cwo:.3f} |",
                  "",
                  f"- Identity: denoise(NTC->basal) vs observed-unassigned Spearman = "
                  f"{(id_sp if id_sp is not None else float('nan')):.3f} "
                  f"(want > raw-NTC-vs-unassigned; confirms μ_action stripping)",
                  "- L1↔L2 convergence: " +
                  ", ".join(f"{g}={v['l1_l2_spearman_mean']:.2f}" for g, v in c["convergence"].items()),
                  "",
                  "### Per-perturbation (LFC: L1/NTC source; abs ρ + e-dist: L2/unassigned source)",
                  "| gene | n | LFC ρ (w/) | ceil (w/) | LFC ρ (w/o) | ceil (w/o) | abs ρ | kNN abs ρ | e-dist | kNN e-dist | cell-sig p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        cc    = c.get("cell_consistency", {})
        aw    = c["aggregate_with"].get("L1",  {})
        awo   = c["aggregate_without"].get("L1", {})
        cew   = c["ceiling_with"];  cewo = c["ceiling_without"]
        absg  = c.get("abs_agg", {}); kabsg  = c.get("knn_abs_agg", {})
        edg   = c.get("edist_agg", {}); kedg  = c.get("knn_edist_agg", {})
        for g in ctx["panel"]:
            vw = aw.get(g); vwo = awo.get(g); cwg = cew.get(g); cwog = cewo.get(g)
            cv = cc.get(g); va = absg.get(g); kva = kabsg.get(g)
            ev = edg.get(g); kev = kedg.get(g)
            if vw is None:
                continue
            lines.append(
                f"| {g} | {vw['n_niches']} "
                f"| {vw['spearman_mean']:.3f} | {(cwg['spearman_brown_mean'] if cwg else float('nan')):.3f} "
                f"| {(vwo['spearman_mean'] if vwo else float('nan')):.3f} "
                f"| {(cwog['spearman_brown_mean'] if cwog else float('nan')):.3f} "
                f"| {(va['spearman_mean'] if va else float('nan')):.3f} "
                f"| {(kva['spearman_mean'] if kva else float('nan')):.3f} "
                f"| {(ev['edist_mean'] if ev else float('nan')):.3f} "
                f"| {(kev['edist_mean'] if kev else float('nan')):.3f} "
                f"| {(cv['mw_pvalue_median'] if cv else float('nan')):.3g} |")
        lines += ["",
                  "_LFC w/_ = μ_action+f_pert(g) (KO vs unassigned); _LFC w/o_ = f_pert(g) (KO vs NTC); "
                  "_abs ρ_ = pseudobulk Spearman(pred, obs KO) over HVGs; "
                  "_e-dist_ = energy distance in PCA-50 space (lower=better, population-level); "
                  "_ceil_ = Spearman-Brown reliability; _cell-sig p_ = Mann-Whitney p.", ""]

        # --- Top perturbations by e-distance (lower = better) ---
        if edg:
            ranked_ed = sorted([(g, v["edist_mean"]) for g, v in edg.items()],
                               key=lambda x: x[1])
            lines += ["### Top perturbations by energy distance (lower = more similar distributions)",
                      "| rank | gene | model e-dist | kNN e-dist | advantage |",
                      "|---|---|---|---|---|"]
            for rank, (g, ed) in enumerate(ranked_ed, 1):
                ked = kedg.get(g, {}).get("edist_mean", float("nan"))
                lines.append(f"| {rank} | {g} | {ed:.3f} | {ked:.3f} | {ked - ed:+.3f} |")
            lines += ["", "_advantage_ = kNN e-dist - model e-dist (positive = model closer to observed).", ""]

    for sc_key, label in [("b", "(b) held-out cell"), ("a", "(a) coord-only patches")]:
        if sc_key not in results:
            continue
        sc = results[sc_key]
        imp = sc.get("imputation", {})
        summary[sc_key] = {"imputation": imp,
                           "counterfactual": (sc["counterfactual"] if sc_key == "a"
                                              else {"aggregate": sc["counterfactual"]["aggregate"],
                                                    "convergence": sc["counterfactual"]["convergence"]})}
        m, base = imp.get("model_spearman"), imp.get("knn_baseline_spearman")
        pb, pbk = imp.get("pseudobulk_spearman"), imp.get("pseudobulk_knn")
        # PASS if the model matches/beats the baseline on EITHER sparsity-robust axis
        # (per-cell across HVGs, or pseudobulk profile on HVGs).
        passed = (m is not None and base is not None and m >= base) or \
                 (pb is not None and pbk is not None and pb >= pbk)
        verdicts[sc_key] = dict(passed=bool(passed), model_spearman=m, knn_baseline_spearman=base,
                                pseudobulk_spearman=pb, pseudobulk_knn=pbk,
                                diag_pergene_allgenes=imp.get("diag_pergene_allgenes"))
        lines += [f"## {label} -- **{'PASS' if passed else 'CONCERN'}**",
                  f"- per-cell/HVG Spearman: model {m} vs kNN {base}",
                  f"- pseudobulk/HVG Spearman: model {pb} vs kNN {pbk}",
                  f"- (diagnostic) per-gene/all-genes Spearman: {imp.get('diag_pergene_allgenes')} "
                  f"-- expected ~0 on 96%-sparse data, not a quality signal", ""]

    summary["verdicts"] = verdicts
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if npz:
        np.savez_compressed(outdir / "lfc_per_niche.npz", **npz)
    (outdir / "eval_report.md").write_text("\n".join(lines))

    # --- Per-niche performance table (L2/unassigned source only) ---
    # Rows: one per (gene, niche). Columns: model_lfc_rho, knn_lfc_rho,
    # model_edist, knn_edist. Saves per-niche CSV for cross-model comparison.
    import csv
    c_sc = results.get("c", {})
    pn_scores   = c_sc.get("per_niche_scores_with",     {}).get("L2", {})
    knn_pn      = c_sc.get("knn_per_niche_scores_with", {}).get("L2", {})
    pn_edist    = c_sc.get("per_niche_edist",    {})
    knn_pn_edist= c_sc.get("knn_per_niche_edist",{})
    if pn_scores:
        rows = []
        for g in pn_scores:
            lfc_by_niche  = {r["niche"]: r["spearman"] for r in pn_scores.get(g, [])}
            klfc_by_niche = {r["niche"]: r["spearman"] for r in knn_pn.get(g, [])}
            ed_by_niche   = {r["niche"]: r["edist"]    for r in pn_edist.get(g, [])}
            ked_by_niche  = {r["niche"]: r["edist"]    for r in knn_pn_edist.get(g, [])}
            niches = sorted(set(lfc_by_niche) | set(ed_by_niche))
            for b in niches:
                rows.append(dict(
                    gene=g, niche=b,
                    model_lfc_rho=lfc_by_niche.get(b, float("nan")),
                    knn_lfc_rho=klfc_by_niche.get(b, float("nan")),
                    model_edist=ed_by_niche.get(b, float("nan")),
                    knn_edist=ked_by_niche.get(b, float("nan")),
                ))
        csv_path = outdir / "per_niche_performance.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["gene","niche","model_lfc_rho","knn_lfc_rho","model_edist","knn_edist"])
            w.writeheader(); w.writerows(rows)
        logging.info("Wrote per_niche_performance.csv (%d rows) to %s", len(rows), outdir)

    for k, v in verdicts.items():
        logging.info("[VERDICT] scenario %s: %s", k, "PASS" if v["passed"] else "CONCERN")
    logging.info("Wrote eval_summary.json, lfc_per_niche.npz (%d arrays), eval_report.md to %s",
                 len(npz), outdir)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ctx = (load_eval_context_multi(args) if args.multi_sample_files
           else load_eval_context(args))
    ctx = setup_grid_and_panel(ctx, args)

    if args.setup_only:
        logging.info("[setup_only] done -- niches=%d/%d panel=%d",
                     int(ctx["evaluable"].sum()), ctx["n_bins"], len(ctx["panel"]))
        return

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    results = {}
    if "c" in scenarios:
        logging.info("=== Scenario (c): seen-cell pert swap (L1 NTC + L2 unassigned) ===")
        results["c"] = run_scenario_c(ctx, args)
    if "b" in scenarios:
        logging.info("=== Scenario (b): held-out cell ===")
        results["b"] = run_scenario_b(ctx, args)
    if "a" in scenarios:
        logging.info("=== Scenario (a): coord-only patches ===")
        results["a"] = run_scenario_a(ctx, args)

    write_report(results, ctx, args, outdir)
    return results


if __name__ == "__main__":
    main()
