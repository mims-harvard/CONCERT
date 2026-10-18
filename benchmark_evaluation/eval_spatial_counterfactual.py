#!/usr/bin/env python3
"""Load a trained CONCERT-spac checkpoint and decode counterfactual predictions.

Data + model are loaded via ``make_umap.load_data`` / ``build_model`` (single slide) or
``run_concert_spac.load_h5ad_multi`` (multi-slide), the SAME preprocessing as training,
so LORD global cell indices align with the trained basal-embedding table.
``_predict_swap`` decodes the expression of a set of cells counterfactually assigned to
one perturbation. The benchmark scorers in ``benchmark_evaluation/`` import these loaders.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import torch

from make_umap import (load_data, classify_perts, build_model, normalize, LOC_RANGE,
                       EXCLUDED_LABELS, UNASSIGNED_LABELS, CONTROL_LABELS)
from run_concert_spac import load_h5ad_multi, build_attributes_spac

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")


def _add_fpert_kwargs(extra, args):
    """Thread the f_pert footprint-field flags into build_model's CONCERT kwargs so the
    evaluated graph matches the trained checkpoint (rff/svgp/hybrid add field weights)."""
    extra["fpert_field"] = args.fpert_field
    extra["f_pert_clamp"] = args.f_pert_clamp
    # P0: basal spatial field (svgp legacy | rff deterministic). MUST match the trained
    # checkpoint so the basal_rff submodule (or SVGP prior) is present for state_dict load.
    extra["basal_field"] = args.basal_field
    # P1: Bayes-linear basal head + multi-bandwidth fpert -- MUST match training for state_dict load.
    extra["basal_bayes"] = args.basal_bayes
    # Basal RFF geometry is a hard architecture dim (basal_rff.rff_W is (2, features)), so a
    # checkpoint trained off-default cannot load without it. Optional: callers that never set
    # these keep the CONCERT defaults, so existing evals are unchanged.
    for _k in ("basal_rff_features", "basal_rff_bandwidth"):
        _v = getattr(args, _k, None)
        if _v is not None:
            extra[_k] = _v
    if args.fpert_field == "mb_rff" and args.fpert_mb_bandwidths:
        extra["fpert_mb_bandwidths"] = [float(x) for x in args.fpert_mb_bandwidths.split(",") if x.strip()]
    if args.fpert_field != "constant":
        extra["fpert_rff_features"] = args.fpert_rff_features
        extra["fpert_rff_bandwidth"] = args.fpert_rff_bandwidth
        extra["fpert_hidden"] = args.fpert_hidden
        extra["fpert_svgp_inducing"] = args.fpert_svgp_inducing
        extra["fpert_svgp_ls_init"] = args.fpert_svgp_ls_init
        extra["fpert_coord_scale"] = args.fpert_coord_scale
        # Collapse-proof band: gp_pp squashes its per-pert lengthscale into
        # [ls_min,ls_max] via sigmoid. This band MUST match the trained checkpoint or the
        # SAME ls_head weights render a DIFFERENT lengthscale at inference (silent corruption).
        # Left unset => CONCERT defaults (3,8), which mismatches the reported models' [3,18].
        if getattr(args, "ls_prior_min", None) is not None:
            extra["ls_prior_min"] = args.ls_prior_min
        if getattr(args, "ls_prior_max", None) is not None:
            extra["ls_prior_max"] = args.ls_prior_max
        # gp_pp consumes coords/coord_scale, so its band lives in NORMALIZED
        # units, unlike ls_prior_min/max (loc units, tuned for the SVGP kernel scale).
        # Checkpoints trained with --fpert_ls_min/max MUST be evaluated with the same
        # values: the band is a plain float on FpertField, not a state_dict entry, so a
        # mismatch silently renders a different lengthscale from the same ls_head weights.
        # Left unset => the band reuses ls_prior_min/max unconverted.
        if getattr(args, "fpert_ls_min", None) is not None:
            extra["fpert_ls_min"] = args.fpert_ls_min
        if getattr(args, "fpert_ls_max", None) is not None:
            extra["fpert_ls_max"] = args.fpert_ls_max


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

    # --- Load the TRAINED perturbation vocabulary -- the single source of truth persisted
    #    at training (run_concert_spac.py). Eval MUST reproduce the exact codes + label
    #    sets; rebuilding the vocab here mis-indexes the embedding table (basal/control
    #    would become real perts and shift every code). ---
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
    # Stored in ctx["cross_slide_counts"] ("how well-learned" each pert is).
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
    set_target_slide_rows(model, args.model_file, args.target_slide_idx,
                          force_slide_bias=bool(getattr(args, "cf_slide_bias", False)))

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


def set_target_slide_rows(model, model_file, target_slide_idx, force_slide_bias=False):
    """Attach the target slide's per-slide rows to a single-slide eval model.

    The eval model is built for one slide, so multi-slide rows (slide_bias, mu_action_slide)
    are absent from its state_dict and load_model skips them. Checkpoints trained with
    --slide_guide_shift always get both rows (the model is incomplete without them); older
    checkpoints get slide_bias only when force_slide_bias is set (ablation).
    """
    st = torch.load(str(model_file), map_location="cpu")
    t = int(target_slide_idx)
    gs, sb = st.get("mu_action_slide"), st.get("slide_bias")
    if gs is not None:
        model.cf_guide_shift = gs[t].float().clone()
    if sb is not None and (gs is not None or force_slide_bias):
        model.cf_slide_bias = sb[t].float().clone()
    logging.info("[cf rows] target slide %d: guide_shift=%s (|.|=%.3f) slide_bias=%s (|.|=%.3f)", t,
                 model.cf_guide_shift is not None,
                 float(model.cf_guide_shift.norm()) if model.cf_guide_shift is not None else 0.0,
                 model.cf_slide_bias is not None,
                 float(model.cf_slide_bias.norm()) if model.cf_slide_bias is not None else 0.0)
    del st


def load_eval_context(args):
    """Load model + data exactly as training/decompose did, returning everything
    the scorers need. Returns a dict (an explicit namespace)."""
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

    # Two matrices, different roles:
    #  - Y_expr = adata.X (z-scored, can be negative): the basal ENCODER input fed to the
    #    model via Y= (matches training input). NEVER use for LFC (log of negatives = NaN).
    #  - Y_obs  = adata.raw.X (raw counts, positive): the OBSERVED expression used for
    #    pseudobulk LFC and imputation scoring. LFC/Spearman are scale/rank robust.
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
