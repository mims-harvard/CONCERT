#!/usr/bin/env python3
"""CONCERT perturb-map SEEN eval: dominant-KO-per-niche on held-out KP-tumor patch
cells (role==1), scored via perturbmap_common.niche_lfc_seen_loop.

pmap differs from spac: the eval prepared panel (panel1053, 1056 genes) does NOT
share barcodes/coords with the model's spacfmt h5ad, so we score in CONCERT
MODEL-ROW space (X = ctx Y_obs raw counts, 2000 concert genes). The patch npz is
the bridge to the model: its cell_index -> spacfmt row -> array_coords -> ctx
coords_raw -> model row. obs (role/target_ko/target_bin/patch_id/bin_id/canon)
come from the patch npz mapped to model rows.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Scoring code (this directory) and the model source it loads checkpoints into, both resolved
# from this file's position so the codebase can be moved without editing paths. The import
# order below matters: eval_spatial_counterfactual binds the model modules from CC before the
# other scoring modules extend sys.path themselves.
EVAL = Path(__file__).resolve().parent
CC = EVAL.parent / "models" / "concert_benchmark"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(EVAL))
import eval_spatial_counterfactual as ecf  # noqa: E402  bind rffbasal before sec pollutes path
import spacseq_eval_common as sec  # noqa: E402
import spacseq_multi_common as smc  # noqa: E402  (add_bootstrap_args)
import perturbmap_common as pm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets")
PMAP = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map")
DEF_MODEL = Path("/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT/outputs/ms_pmap_gp_pp_bayesTrue/model.pt")
SLIDES = ["0", "1", "2", "3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_file", type=Path, default=DEF_MODEL)
    p.add_argument("--patch_dir", type=Path, default=PMAP / "patches")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--slides", nargs="*", default=SLIDES)
    p.add_argument("--hvg_max_rank", type=int, default=2000)
    p.add_argument("--min_cells_per_pert", type=int, default=0)
    p.add_argument("--min_g_cells", type=int, default=5)
    p.add_argument("--niche_grid", type=int, default=6)
    p.add_argument("--loc_range", type=float, default=20.0)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_src_cells", type=int, default=2000)
    # pmap arch (sub_train_multislide pmap): ACTCORR=False IND=4 NRANK=4 MINCELLS=0
    _b = lambda x: x.lower() == "true"
    p.add_argument("--use_action_correction", type=_b, default=False)
    p.add_argument("--inducing_point_steps", type=int, default=4)
    p.add_argument("--niche_rank", type=int, default=4)
    p.add_argument("--use_expr_encoder", type=_b, default=True)
    p.add_argument("--expr_hidden_dims", type=int, nargs="+", default=[512, 256])
    p.add_argument("--use_basal_encoder_only", type=_b, default=True)
    p.add_argument("--use_nngp", type=_b, default=False)
    p.add_argument("--nngp_lambda", type=float, default=0.0)
    p.add_argument("--use_paracrine", type=_b, default=False)
    p.add_argument("--paracrine_l1", type=float, default=0.0)
    p.add_argument("--use_niche_interaction", type=_b, default=True)
    p.add_argument("--use_pert_gate", type=_b, default=True)
    p.add_argument("--gate_hidden", type=int, default=64)
    p.add_argument("--spatial_basal_only", type=_b, default=True)
    p.add_argument("--basal_field", default="rff")
    p.add_argument("--basal_bayes", type=_b, default=True)
    p.add_argument("--fpert_field", default="gp_pp")
    p.add_argument("--fpert_rff_features", type=int, default=128)
    p.add_argument("--fpert_rff_bandwidth", type=float, default=0.15)
    p.add_argument("--fpert_coord_scale", type=float, default=20.0)
    p.add_argument("--fpert_hidden", type=int, default=256)
    p.add_argument("--fpert_svgp_inducing", type=int, default=16)
    p.add_argument("--fpert_svgp_ls_init", type=float, default=5.0)
    # v4 gp_pp lengthscale band -- MUST match the trained checkpoint (v4 used [3,18]).
    # Left as None => CONCERT construction defaults (3,8); pass explicitly for v4 models.
    p.add_argument("--ls_prior_min", type=float, default=None)
    p.add_argument("--ls_prior_max", type=float, default=None)
    # lsfix: gp_pp footprint band in NORMALIZED units. Must match the checkpoint --
    # the band defines the sigmoid squash inside FpertField, not just a training penalty,
    # so evaluating an lsfix ckpt without these reconstructs the wrong lengthscales.
    p.add_argument("--fpert_ls_min", type=float, default=None)
    p.add_argument("--fpert_ls_max", type=float, default=None)
    p.add_argument("--fpert_mb_bandwidths", default="")
    p.add_argument("--f_pert_clamp", type=float, default=5.0)
    smc.add_bootstrap_args(p, default_n=10)
    p.add_argument("--setup_only", action="store_true")
    # v3 UNSEEN: score with the unseen-trained model; GT restricted to the held-out
    # (masked) KO cells named in this coords file. Non-masked genes auto-drop (empty
    # ko_idx&holdout). When set, run_seen writes to unseen_pert and donor run_unseen is skipped.
    p.add_argument("--v3_unseen_ko_file", type=Path, default=None)
    # Diagnostic: disable the coord-conditioned FpertField at inference (core f_pert /
    # niche / gate kept). Tests whether the spatial field corrupts unseen/multimodal perts.
    p.add_argument("--fpert_off", action="store_true")
    return p.parse_args()


def build_pm_ctx(args, sid: str) -> dict:
    """Load pmap CONCERT model + slide (model-row space) via the shared rffbasal loader."""
    all_paths = [str(DATA / f"perturbmap_s{s}_spacfmt.h5ad") for s in SLIDES]
    fwd = SimpleNamespace(
        multi_sample_files=",".join(all_paths),
        target_slide_idx=SLIDES.index(sid),
        model_file=str(args.model_file),
        hvg_max_rank=args.hvg_max_rank,
        min_cells_per_pert=args.min_cells_per_pert,
        use_expr_encoder=args.use_expr_encoder,
        expr_hidden_dims=args.expr_hidden_dims,
        use_basal_encoder_only=args.use_basal_encoder_only,
        use_nngp=args.use_nngp,
        nngp_lambda=args.nngp_lambda,
        use_paracrine=args.use_paracrine,
        paracrine_l1=args.paracrine_l1,
        use_niche_interaction=args.use_niche_interaction,
        niche_rank=args.niche_rank,
        use_pert_gate=args.use_pert_gate,
        gate_hidden=args.gate_hidden,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        use_action_correction=args.use_action_correction,
        inducing_point_steps=args.inducing_point_steps,
        spatial_basal_only=args.spatial_basal_only,
        basal_field=args.basal_field,
        basal_bayes=args.basal_bayes,
        fpert_field=args.fpert_field,
        fpert_rff_features=args.fpert_rff_features,
        fpert_rff_bandwidth=args.fpert_rff_bandwidth,
        fpert_coord_scale=args.fpert_coord_scale,
        fpert_hidden=args.fpert_hidden,
        fpert_svgp_inducing=args.fpert_svgp_inducing,
        fpert_svgp_ls_init=args.fpert_svgp_ls_init,
        fpert_mb_bandwidths=args.fpert_mb_bandwidths,
        f_pert_clamp=args.f_pert_clamp,
        ls_prior_min=args.ls_prior_min,
        ls_prior_max=args.ls_prior_max,
        fpert_ls_min=args.fpert_ls_min,
        fpert_ls_max=args.fpert_ls_max,
    )
    # Robustness: the gp_pp inducing-grid size is a hard architecture dim (fpert_field.Z).
    # A CLI/checkpoint mismatch aborts load_model. Read the true count from the checkpoint
    # so the eval self-configures (v3=16, v4=64) and can never mismatch again.
    try:
        import torch as _torch
        _z = _torch.load(str(args.model_file), map_location="cpu").get("fpert_field.Z")
        if _z is not None and int(_z.shape[0]) != int(fwd.fpert_svgp_inducing):
            log.info("auto-set fpert_svgp_inducing %s -> %s (from checkpoint Z)",
                     fwd.fpert_svgp_inducing, int(_z.shape[0]))
            fwd.fpert_svgp_inducing = int(_z.shape[0])
    except Exception as _e:  # non-fatal: fall back to the CLI value
        log.warning("could not auto-detect inducing count from checkpoint: %s", _e)
    # Niche-FiLM auto-detect: if the checkpoint carries the FiLM head, build with it and turn the
    # legacy additive niche interaction OFF (matching training). niche_film_k / gcap are plain-float
    # attrs NOT saved in the state_dict, so pin them to the training defaults -- they MUST match or
    # gamma renders differently at inference (silent corruption, same class as the ls-band bug).
    try:
        import torch as _t2
        _sd = _t2.load(str(args.model_file), map_location="cpu")
        if any(k.endswith("g_scale_head.weight") for k in _sd):
            fwd.use_niche_film = True
            fwd.use_niche_interaction = False
            _bv = _sd.get("beta_V.weight")
            fwd.niche_film_dim = int(_bv.shape[1]) if _bv is not None else 8
            # beta_V is (rank, niche_film_dim) -> rank = shape[0]; MUST match or beta_* shape-mismatch.
            fwd.niche_film_rank = int(_bv.shape[0]) if _bv is not None else 8
            # dispersion grows NicheDescriptor pool input 2H->3H; detect from pool.0 vs proj widths.
            _proj = _sd.get("niche_descriptor.proj.weight")
            _pool0 = _sd.get("niche_descriptor.pool.0.weight")
            fwd.niche_film_dispersion = bool(
                _proj is not None and _pool0 is not None
                and int(_pool0.shape[1]) == 3 * int(_proj.shape[0])
            )
            fwd.niche_film_k = 25
            fwd.niche_film_gcap = 3.0
            fwd.niche_film_l2 = 1e-3
            log.info("auto-detected niche-FiLM checkpoint: dim=%s rank=%s dispersion=%s, interaction OFF",
                     fwd.niche_film_dim, fwd.niche_film_rank, fwd.niche_film_dispersion)
    except Exception as _e:
        log.warning("niche-FiLM auto-detect failed: %s", _e)
    ctx = ecf.load_eval_context_multi(fwd)
    # Field-off diagnostic: flag consumed in counterfactualPrediction (coords=None path).
    if getattr(args, "fpert_off", False):
        ctx["model"]._disable_fpert_field = True
        log.info("[fpert_off] FpertField DISABLED at inference for slide %s", sid)
    return ctx


def _patch_obs_on_model_rows(args, sid: str, ctx: dict) -> dict:
    """Map patch-npz per-cell fields onto CONCERT model rows by coordinate.

    cell_index is NOT reliable (indexes a pre-filter cell set that differs from the
    model's spacfmt h5ad). Bridge purely by coordinate:
      patch (pos_x,pos_y) == spacfmt obsm['spatial']  ->  spacfmt obsm['array_coords']
      ==  ctx['coords_raw'] (model rows).
    Returns model-row arrays: role, target_ko, target_bin, patch_id, bin_id, canon.
    """
    P = np.load(args.patch_dir / f"slide{sid}.npz", allow_pickle=True)
    with h5py.File(str(DATA / f"perturbmap_s{sid}_spacfmt.h5ad"), "r") as f:
        sp_all = np.asarray(f["obsm"]["spatial"][:], dtype=np.float64)
        ac_all = np.asarray(f["obsm"]["array_coords"][:], dtype=np.float64)
    # spacfmt: pixel-spatial -> array_coord
    spatial2ac = {(round(float(x), 1), round(float(y), 1)): (round(float(a), 1), round(float(b), 1))
                  for (x, y), (a, b) in zip(sp_all, ac_all)}
    px = P["pos_x"].astype(np.float64)
    py = P["pos_y"].astype(np.float64)
    # patch cell k -> its array_coord (via spacfmt spatial); then array_coord -> k
    coord2k = {}
    for k in range(len(px)):
        ack = spatial2ac.get((round(float(px[k]), 1), round(float(py[k]), 1)))
        if ack is not None:
            coord2k[ack] = k

    cr = np.round(np.asarray(ctx["coords_raw"], dtype=np.float64), 1)
    N = ctx["N"]
    role = P["role"].astype(int)
    tko = P["target_ko"].astype(str)
    tb = P["target_bin"].astype(int)
    pid = P["patch_id"].astype(int)
    binid = P["bin_id"].astype(int)
    canon = P["perturbation_canon"].astype(str)

    role_m = np.zeros(N, dtype=int)
    tko_m = np.array([""] * N, dtype=object)
    tb_m = np.full(N, -1, dtype=int)
    pid_m = np.full(N, -1, dtype=int)
    bin_m = np.zeros(N, dtype=int)
    canon_m = np.array(["unassigned"] * N, dtype=object)
    nmap = 0
    for i in range(N):
        k = coord2k.get((float(cr[i, 0]), float(cr[i, 1])))
        if k is None:
            continue
        nmap += 1
        role_m[i], tko_m[i], tb_m[i] = role[k], tko[k], tb[k]
        pid_m[i], bin_m[i], canon_m[i] = pid[k], binid[k], canon[k]
    log.info("slide %s: mapped %d/%d model rows to patch cells (%.1f%%)",
             sid, nmap, N, 100.0 * nmap / max(1, N))
    return dict(role=role_m, target_ko=tko_m, target_bin=tb_m, patch_id=pid_m,
                bin_id=bin_m, perturbation_canon=canon_m, n_mapped=nmap)


def _v3_ko_gt_rows(args, sid: str, ctx: dict) -> np.ndarray:
    """Model-row indices of the HELD-OUT (masked) KO cells for this slide, from the
    v3 unseen coords file. Matched by (array_coords) against ctx['coords_raw'].
    Coords are pooled across masked genes; the per-gene GT falls out of the
    canon==g & ko_gt_rows intersection in niche_lfc_seen_loop."""
    hv = json.load(open(args.v3_unseen_ko_file))
    sid_key = str(hv.get("slide_to_sid", {}).get(sid, sid))
    coords = hv.get("ko", {}).get(sid_key, [])
    if not coords:
        return np.array([], dtype=int)
    want = set((int(x), int(y)) for x, y in coords)
    cr = np.rint(np.asarray(ctx["coords_raw"], dtype=np.float64)).astype(int)
    rows = np.array([i for i in range(len(cr)) if (int(cr[i, 0]), int(cr[i, 1])) in want], dtype=int)
    return rows


def _v3_units_by_slide(args) -> dict:
    """Map target_slide -> [gene] from the v3 eval_panel. Panel-driven UNSEEN units:
    each gene is MASKED on its target slide (GT = same-slide masked KO) and KEPT on
    >=1 donor slide (so the model learns it). ONLY these (gene,target_slide) pairs are
    valid unseen units -- nothing else may be scored as unseen."""
    panel_path = pm.DEFAULT_PREPARED / "benchmark_v3" / "eval_panel.json"
    ep = json.load(open(panel_path))
    out: dict[str, list[str]] = {}
    for u in ep.get("eval_units", []):
        out.setdefault(str(u["target_slide"]), []).append(u["gene"])
    log.info("v3 eval panel target_slide->genes: %s", out)
    return out


def run_seen(args, sid: str, v3_units=None):
    v3_unseen = v3_units is not None
    sub = "unseen_pert" if v3_unseen else "seen_pert"
    out = args.outdir / sub / f"slide_{sid}"
    if (out / "lfc_scores.json").exists():
        log.info("slide %s %s exists -- skip", sid, sub)
        return
    ctx = build_pm_ctx(args, sid)
    N = ctx["N"]
    gene_names = np.asarray(ctx["gene_names"])
    Yobs = np.asarray(ctx["Y_obs"], dtype=np.float32)
    obs = _patch_obs_on_model_rows(args, sid, ctx)

    adata_target = ad.AnnData(
        X=sp.csr_matrix(Yobs),
        obs=pd.DataFrame({
            "perturbation_canon": obs["perturbation_canon"],
            "bin_id": obs["bin_id"],
            "role": obs["role"],
            "target_ko": obs["target_ko"],
            "target_bin": obs["target_bin"],
            "patch_id": obs["patch_id"],
        }, index=np.arange(N).astype(str)),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    # Scorer reads observed LFC / e-dist off adata_target.X, and normalize_pred_cache
    # puts CONCERT predictions into log-CP10K. Match that space on the OBSERVED side
    # too (competitors do this inside perturbmap_common: normalize_total(1e4)+log1p,
    # raw kept in layers["counts"]). Without this the observed X stays raw counts and
    # e-dist is ~10x inflated vs the log-CP10K competitors.
    import scanpy as sc
    adata_target.layers["counts"] = adata_target.X.copy()
    sc.pp.normalize_total(adata_target, target_sum=1e4)
    sc.pp.log1p(adata_target)
    raw_mm = Yobs
    lib_sizes = Yobs.sum(axis=1).astype(np.float64)

    global_src_idx = pm.source_indices(adata_target, src_role=1)
    if len(global_src_idx) == 0:
        log.warning("slide %s: no role==1 source cells -- skip", sid)
        return
    if v3_unseen:
        # Panel-driven: score ONLY the masked eval-unit genes on this (target) slide.
        # GT = same-slide masked KO = ALL canon==G cells (holdout == all target KO per
        # eval_panel: n_ko_holdout == n_ko_target). Niche = dominant bin of those masked
        # cells; predict G on the WHOLE held-out unassigned patch. The source patch's own
        # assigned target_ko is IRRELEVANT here (that niche-driven path scored the wrong
        # gene, e.g. Jak2, and missed the real units).
        canon_all = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
        bin_all = adata_target.obs["bin_id"].astype(int).to_numpy()
        n_src = len(global_src_idx)
        groups = {}
        for g in v3_units:
            if g not in ctx["name2code"]:
                log.warning("slide %s v3-unseen: gene %s has no model code -- skip", sid, g)
                continue
            gi = np.where(canon_all == g)[0]
            if len(gi) == 0:
                log.warning("slide %s v3-unseen: gene %s has no KO cells on target -- skip", sid, g)
                continue
            # APPLES-TO-APPLES with the competitor unseen harness (benchmark_v3_eval_lib.
            # unseen_v3_pm): competitors score masked-KO-of-G vs WHOLE-SLIDE KP control with
            # NO niche restriction (niche=-1). Register the group with b=-1 so _ref_ko_idx /
            # _ref_ctrl_idx (which fall back to slide-level when the niche has < min_g_cells)
            # use the whole-slide masked KO and whole-slide control -- identical GT/control to
            # the competitor. (Previously used the dominant bin -> niche-restricted GT, which
            # made CONCERT's unseen numbers not directly comparable to competitors'.)
            groups[(-1, g, 0)] = np.arange(n_src, dtype=int)
        if not groups:
            log.warning("slide %s v3-unseen: no valid panel-unit genes -- skip", sid)
            return
        panel = sorted({g for (_, g, _) in groups})
        log.info("slide %s v3-UNSEEN panel-driven units=%s (masked-GT = same-slide canon==G)", sid, panel)
    else:
        groups = pm.seen_niche_groups(adata_target, src_role=1)
        if not groups:
            log.warning("slide %s: no seen niches -- skip", sid)
            return
        panel = sorted({g for (_, g, _) in groups})
    log.info("slide %s: %d groups, panel=%s, %d source cells", sid, len(groups), panel, len(global_src_idx))
    if args.setup_only:
        return

    # Score only the genes that live in the 1053-gene KO panel competitors use.
    # CONCERT predicts its 2000 HVG (highly-VARIABLE genes); scoring e-dist/LFC over
    # all 2000 measures distributional spread over max-variance genes and inflates
    # e-dist ~10-16x vs competitors, who score the 1053-gene panel. Restrict to the
    # panel-overlap (~380-470 genes) so both metrics are apples-to-apples.
    with h5py.File(str(DATA / f"perturbmap_s{sid}_spacfmt.h5ad"), "r") as _vf:
        _vnames = _vf["var"]["_index"][:].astype(str)
        _inpanel = _vf["var"]["in_panel_1053"][:].astype(bool)
    _panel_set = set(_vnames[_inpanel])
    hvg_mask = np.array([g in _panel_set for g in gene_names], dtype=bool)
    log.info("slide %s: scoring %d/%d panel-overlap genes (was all %d HVG)",
             sid, int(hvg_mask.sum()), len(gene_names), len(gene_names))
    from eval_spatial_counterfactual import _predict_swap
    pred_g_cache = {}
    for g in panel:
        code = ctx["name2code"].get(g)
        if code is None:
            pred_g_cache[g] = None
            continue
        pred_g_cache[g] = np.asarray(_predict_swap(ctx, global_src_idx, code, args), dtype=np.float32)
    unpred = {g for g, v in pred_g_cache.items() if v is None}

    # Basal prediction (code=0) on the SAME source cells -> DELTA scoring
    # pred_KO - pred_basal cancels basal reconstruction error.
    pred_basal = np.asarray(_predict_swap(ctx, global_src_idx, 0, args), dtype=np.float32)

    # v3-unseen GT needs no coord filter: groups already scope observed LFC to
    # canon==G (the masked KO) per _ref_ko_idx, and all target-slide KO of G was masked.
    scores, agg, mlfc = pm.niche_lfc_seen_loop(
        adata_target, raw_mm, lib_sizes, global_src_idx, groups, pred_g_cache, hvg_mask, args,
        unpredictable_genes=unpred, scoring_method="concert", pred_basal=pred_basal,
    )
    _, sa = sec.import_eval_utils()
    pm.save_results(
        args.outdir, sid, sub, panel, scores, agg, mlfc, "concert",
        hvg_info=sa.hvg_info_dict(hvg_mask),
        extra={"niche_dominant_perts": {f"{b}": g for (b, g, _r) in sorted(groups)},
               "n_replicate_patches": len(groups), "src_role": 1,
               "n_boot": args.n_boot, "boot_frac": args.boot_frac,
               "n_src": int(len(global_src_idx)), "n_mapped": int(obs["n_mapped"]),
               "unpredictable_genes": sorted(unpred)},
    )
    log.info("slide %s seen: written -> %s", sid, out)


def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def _model_adata(args, sid: str):
    """Build model-row-space AnnData (X=log-CP10K, layers['counts']=raw, obs.canon from
    patch, obs.n_counts=raw lib size) + return ctx, gene_names, raw Yobs. Same recipe
    run_seen uses inline; shared by run_unseen + donor pooling."""
    import scanpy as sc
    ctx = build_pm_ctx(args, sid)
    N = ctx["N"]
    gene_names = np.asarray(ctx["gene_names"])
    Yobs = np.asarray(ctx["Y_obs"], dtype=np.float32)
    obs = _patch_obs_on_model_rows(args, sid, ctx)
    adata = ad.AnnData(
        X=sp.csr_matrix(Yobs),
        obs=pd.DataFrame({
            "perturbation_canon": obs["perturbation_canon"],
            "bin_id": obs["bin_id"],
            "role": obs["role"],
            "target_ko": obs["target_ko"],
            "target_bin": obs["target_bin"],
            "patch_id": obs["patch_id"],
            "n_counts": Yobs.sum(axis=1).astype(np.float64),
        }, index=np.arange(N).astype(str)),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata, ctx, gene_names, Yobs


def run_unseen(args, sid: str):
    """Unseen pert (Option A): absent-KO genes on this slide, donor-pooled reference
    across the OTHER slides. Scored in MODEL-ROW space with the same in-panel gene
    mask as seen; observed LFC = donor real-KO vs donor control(KP)."""
    out = args.outdir / "unseen_pert" / f"slide_{sid}"
    if (out / "lfc_scores.json").exists():
        log.info("slide %s unseen exists -- skip", sid)
        return
    eval_genes = pm.load_manifest_genes(pm.DEFAULT_PREPARED, sid)
    adata_target, ctx, gene_names, Yobs = _model_adata(args, sid)
    name2code = ctx["name2code"]
    eval_genes = [g for g in eval_genes if g in name2code]
    if not eval_genes:
        log.info("slide %s: no unseen genes with a model code", sid)
        return
    global_src_idx = pm.source_indices(adata_target, src_role=1)
    if len(global_src_idx) == 0:
        log.warning("slide %s: no role==1 source cells for unseen -- skip", sid)
        return
    with h5py.File(str(DATA / f"perturbmap_s{sid}_spacfmt.h5ad"), "r") as _vf:
        _vnames = _vf["var"]["_index"][:].astype(str)
        _inpanel = _vf["var"]["in_panel_1053"][:].astype(bool)
    _panel_set = set(_vnames[_inpanel])
    hvg_mask = np.array([g in _panel_set for g in gene_names], dtype=bool)
    log.info("slide %s UNSEEN: genes=%s, %d src cells, scoring %d/%d panel genes",
             sid, eval_genes, len(global_src_idx), int(hvg_mask.sum()), len(gene_names))
    if args.setup_only:
        return

    from eval_spatial_counterfactual import _predict_swap
    pred_g_cache = {}
    for g in eval_genes:
        code = name2code.get(g)
        pred_g_cache[g] = (np.asarray(_predict_swap(ctx, global_src_idx, code, args), dtype=np.float32)
                           if code is not None else None)
    unpred = {g for g, v in pred_g_cache.items() if v is None}

    # Basal prediction (code=0) on the SAME source cells -> DELTA scoring.
    pred_basal = np.asarray(_predict_swap(ctx, global_src_idx, 0, args), dtype=np.float32)

    # Donor pools from the OTHER slides (model-row space, same gene order).
    donor_ko_log = {g: [] for g in eval_genes}
    donor_ko_raw = {g: [] for g in eval_genes}
    donor_ctrl_log = []
    for sid2 in SLIDES:
        if sid2 == sid:
            continue
        a2, _c2, gn2, Y2 = _model_adata(args, sid2)
        assert list(gn2) == list(gene_names), f"donor {sid2} gene order != target {sid}"
        c2 = a2.obs["perturbation_canon"].astype(str).to_numpy()
        ci = np.where(c2 == pm.CTRL_CANON)[0]
        if len(ci):
            donor_ctrl_log.append(_dense(a2[ci].X))
        for g in eval_genes:
            gi = np.where(c2 == g)[0]
            if len(gi):
                donor_ko_log[g].append(_dense(a2[gi].X))
                donor_ko_raw[g].append(Y2[gi])
    donor_ctrl = np.vstack(donor_ctrl_log) if donor_ctrl_log else None
    log.info("slide %s UNSEEN donors: ctrl=%d, per-gene KO=%s", sid,
             0 if donor_ctrl is None else donor_ctrl.shape[0],
             {g: int(sum(a.shape[0] for a in donor_ko_raw[g])) for g in eval_genes})

    scores, agg, mlfc = pm.niche_lfc_unseen_score(
        adata_target, global_src_idx, eval_genes, pred_g_cache, hvg_mask, args,
        donor_ko_log, donor_ko_raw, donor_ctrl,
        unpredictable_genes=unpred, scoring_method="concert", pred_basal=pred_basal,
    )
    _, sa = sec.import_eval_utils()
    pm.save_results(
        args.outdir, sid, "unseen_pert", eval_genes, scores, agg, mlfc, "concert",
        hvg_info=sa.hvg_info_dict(hvg_mask),
        extra={"eval_genes": eval_genes, "src_role": 1,
               "n_boot": args.n_boot, "boot_frac": args.boot_frac,
               "n_src": int(len(global_src_idx)),
               "n_donor_ctrl": int(donor_ctrl.shape[0]) if donor_ctrl is not None else 0,
               "unpredictable_genes": sorted(unpred),
               "reference": "donor real-KO vs donor control(KP), model-row space, in-panel genes"},
    )
    log.info("slide %s unseen: written -> %s", sid, out)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    v3_unseen = args.v3_unseen_ko_file is not None
    v3_units_by_slide = _v3_units_by_slide(args) if v3_unseen else {}
    for sid in args.slides:
        if v3_unseen:
            # ONLY target slides with a valid eval unit are scored as unseen.
            units = v3_units_by_slide.get(str(sid), [])
            if not units:
                log.info("=== pmap v3-unseen slide %s: not a target slide -- skip ===", sid)
                continue
            log.info("=== pmap v3-unseen slide %s units=%s ===", sid, units)
            run_seen(args, sid, v3_units=units)
        else:
            log.info("=== pmap seen slide %s ===", sid)
            run_seen(args, sid)
            log.info("=== pmap unseen(donor, legacy) slide %s ===", sid)
            run_unseen(args, sid)
    log.info("Done -> %s", args.outdir)


if __name__ == "__main__":
    main()
