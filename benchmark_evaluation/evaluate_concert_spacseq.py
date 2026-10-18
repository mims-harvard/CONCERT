#!/usr/bin/env python3
"""CONCERT SPAC-seq benchmark evaluation (same harness as the competing methods).

Seen: patch-dominant holdout (role==1) via niche_lfc_patch_holdout_loop.
Unseen: benchmark/eval_panel.json units via spacseq_cross_slide_scoring.unseen_spac (reference
excludes the patch-holdout unassigned pool; model_lfc.json = the scored vector).

With the default output flags (--cf_output mean, --cf_seen_ref pred_basal) the seen arm scores
the predicted mean against CONCERT's own basal prediction; the SI sensitivity sweep
(experiments/06_sensitivity_analysis) uses these defaults. The reported benchmark passes
--cf_output poisson_depth --cf_seen_ref observed (evaluate.sh).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import release_paths as rp  # noqa: E402
from types import SimpleNamespace

import h5py
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
CONCERT_DIR = EVAL_DIR.parent / "models" / "concert_benchmark"
sys.path.insert(0, str(CONCERT_DIR))
sys.path.insert(0, str(EVAL_DIR))

import eval_spatial_counterfactual as ecf  # noqa: E402

import spacseq_common as sec
import spacseq_slides as smc
from spacseq_cross_slide_scoring import (
    aggregate_unseen,
    filter_units,
    ko_holdout_indices,
    load_eval_panel,
    load_holdout_unassigned,
    load_source_cells,
    save_unseen_unit_results,
    unseen_spac,
)
from spacseq_common import UNASSIGNED_LABELS
from spacseq_slides import (
    DEFAULT_SLIDES,
    load_patch_roles,
    load_prepared_raw_resources,
    load_prepared_slide,
    niche_lfc_patch_holdout_loop,
    patch_dominant_perts,
    patch_source_indices,
    save_multi_seen_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = rp.SPACSEQ_RAW
DEFAULT_PREP = rp.SPACSEQ_PREPARED
CHECKPOINTS = rp.CHECKPOINTS
DEFAULT_SEEN_CKPT = CHECKPOINTS / "spacseq_within_niche" / "model.pt"
DEFAULT_UNSEEN_CKPT = CHECKPOINTS / "spacseq_cross_slide" / "model.pt"
METHOD = "concert"


def parse_args():
    p = argparse.ArgumentParser(description="CONCERT SPAC-seq benchmark evaluation.")
    p.add_argument("--regime", required=True, choices=["seen", "unseen"])
    p.add_argument("--prepared_dir", type=Path, default=DEFAULT_PREP)
    p.add_argument("--benchmark_dir", type=Path, default=None, help="Default: prepared_dir/benchmark")
    p.add_argument("--units", nargs="*", default=None, help="Optional unseen subset: Gene or Gene@slide")
    p.add_argument("--model_file", type=Path, default=None)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--target_slide", default=None, choices=DEFAULT_SLIDES,
                   help="Required for seen (one slide per invocation).")
    p.add_argument("--n_hvgs", type=int, default=2000)
    p.add_argument("--min_g_cells", type=int, default=20)
    p.add_argument("--max_src_cells", type=int, default=500)
    p.add_argument("--max_src_cells_per_niche", type=int, default=500)
    p.add_argument("--patch_role", type=int, default=1, choices=[1, 2])
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hvg_max_rank", type=int, default=2000)
    p.add_argument("--min_cells_per_pert", type=int, default=25)
    p.add_argument("--force", action="store_true")
    _b = lambda x: x.lower() == "true"
    p.add_argument("--use_expr_encoder", type=_b, default=True)
    p.add_argument("--expr_hidden_dims", type=int, nargs="+", default=[512, 256])
    p.add_argument("--use_basal_encoder_only", type=_b, default=True)
    p.add_argument("--use_nngp", type=_b, default=False)
    p.add_argument("--nngp_lambda", type=float, default=0.0)
    p.add_argument("--use_paracrine", type=_b, default=False)
    p.add_argument("--paracrine_l1", type=float, default=0.0)
    p.add_argument("--use_niche_interaction", type=_b, default=False)
    p.add_argument("--niche_rank", type=int, default=8)
    p.add_argument("--use_pert_gate", type=_b, default=True)
    p.add_argument("--gate_hidden", type=int, default=64)
    p.add_argument("--use_action_correction", type=_b, default=True)
    p.add_argument("--inducing_point_steps", type=int, default=6)
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
    p.add_argument("--fpert_mb_bandwidths", default="")
    p.add_argument("--f_pert_clamp", type=float, default=5.0)
    # gp_pp footprint lengthscale band. MUST match the trained checkpoint: the band is a
    # plain float on FpertField (not a state_dict entry), so the SAME ls_head weights render
    # a DIFFERENT lengthscale if it differs. ls_prior_* are in loc units (the reported models use [3,18]);
    # fpert_ls_* are in normalized units and, when given, override ls_prior_* for gp_pp only.
    p.add_argument("--ls_prior_min", type=float, default=3.0)
    p.add_argument("--ls_prior_max", type=float, default=18.0)
    p.add_argument("--fpert_ls_min", type=float, default=None)
    p.add_argument("--fpert_ls_max", type=float, default=None)
    # niche_film_k is a plain attr, absent from the state_dict, and MUST match training or the
    # FiLM gate renders differently at inference. Default 25 = the training default, so every
    # existing checkpoint is unaffected; sweeps pass their own value.
    p.add_argument("--niche_film_k", type=int, default=25)
    # --cf_slide_bias: add the checkpoint's slide_bias row of the target slide to every prediction
    # (automatic for --slide_guide_shift checkpoints). --cf_output poisson_depth: Poisson-sample
    # each source spot's predicted profile at that spot's observed library size (model genes);
    # genes outside the model keep the observed counts so the CP10k denominator is the spot's own.
    p.add_argument("--cf_slide_bias", action="store_true")
    p.add_argument("--cf_output", choices=["mean", "poisson_depth"], default="mean")
    # Offset added to every Poisson seed, for repeat draws of the same evaluation.
    p.add_argument("--cf_poisson_seed_base", type=int, default=0)
    # Seen-arm LFC reference: pred_basal (CONCERT own control) or the observed source cells,
    # which is the rule the competitors are scored under.
    p.add_argument("--cf_seen_ref", choices=["pred_basal", "observed"], default="pred_basal")
    smc.add_bootstrap_args(p, default_n=10)
    return p.parse_args()


def _raw_barcode_to_pixel(sid: str) -> dict:
    path = str(DATA_DIR / f"raw_output_{sid}.h5ad")
    with h5py.File(path, "r") as f:
        bc = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs"]["_index"][:]]
        sp_ = np.asarray(f["obsm"]["array_coords"][:], dtype=np.float64)
    return {b: (round(float(x), 1), round(float(y), 1)) for b, (x, y) in zip(bc, sp_)}


def _auto_arch_from_checkpoint(fwd: SimpleNamespace, model_file: Path, niche_film_k: int = 25) -> None:
    try:
        import torch

        sd = torch.load(str(model_file), map_location="cpu")
        z = sd.get("fpert_field.Z")
        if z is not None and int(z.shape[0]) != int(fwd.fpert_svgp_inducing):
            log.info("auto-set fpert_svgp_inducing %s -> %s", fwd.fpert_svgp_inducing, int(z.shape[0]))
            fwd.fpert_svgp_inducing = int(z.shape[0])
        _w = sd.get("basal_rff.rff_W")
        if _w is not None and int(_w.shape[1]) != int(getattr(fwd, "basal_rff_features", 128)):
            log.info("auto-set basal_rff_features %s -> %s (from checkpoint rff_W)",
                     getattr(fwd, "basal_rff_features", None), int(_w.shape[1]))
            fwd.basal_rff_features = int(_w.shape[1])
        if any(k.endswith("g_scale_head.weight") for k in sd):
            fwd.use_niche_film = True
            fwd.use_niche_interaction = False
            bv = sd.get("beta_V.weight")
            fwd.niche_film_dim = int(bv.shape[1]) if bv is not None else 8
            fwd.niche_film_rank = int(bv.shape[0]) if bv is not None else 8
            proj = sd.get("niche_descriptor.proj.weight")
            pool0 = sd.get("niche_descriptor.pool.0.weight")
            fwd.niche_film_dispersion = bool(
                proj is not None and pool0 is not None
                and int(pool0.shape[1]) == 3 * int(proj.shape[0])
            )
            fwd.niche_film_k = int(niche_film_k)
            fwd.niche_film_gcap = 3.0
            fwd.niche_film_l2 = 1e-3
            log.info(
                "niche-FiLM checkpoint: dim=%s rank=%s dispersion=%s",
                fwd.niche_film_dim, fwd.niche_film_rank, fwd.niche_film_dispersion,
            )
    except Exception as exc:
        log.warning("checkpoint auto-detect failed: %s", exc)


def build_concert_ctx(args, target_slide: str) -> dict:
    all_paths = [str(DATA_DIR / f"raw_output_{s}.h5ad") for s in DEFAULT_SLIDES]
    fwd = SimpleNamespace(
        multi_sample_files=",".join(all_paths),
        target_slide_idx=DEFAULT_SLIDES.index(target_slide),
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
        cf_slide_bias=args.cf_slide_bias,
    )
    _auto_arch_from_checkpoint(fwd, args.model_file, niche_film_k=args.niche_film_k)
    ctx = ecf.load_eval_context_multi(fwd)
    eu, _ = sec.import_eval_utils()
    bin_id, _ = eu.build_niche_grid(ctx["coords"], 6, 20.0)
    ctx["bin_id"] = bin_id
    return ctx


def map_prep_to_model_rows(
    sid: str,
    prep_bc: np.ndarray,
    prep_indices: np.ndarray,
    ctx: dict,
) -> tuple[np.ndarray, np.ndarray]:
    bc2px = _raw_barcode_to_pixel(sid)
    cr = np.round(np.asarray(ctx["coords_raw"], dtype=np.float64), 1)
    px2row = {(float(x), float(y)): i for i, (x, y) in enumerate(cr)}
    kept, model_rows = [], []
    for pi in np.asarray(prep_indices, dtype=int):
        px = bc2px.get(str(prep_bc[pi]))
        if px is None:
            continue
        r = px2row.get((float(px[0]), float(px[1])))
        if r is not None:
            kept.append(int(pi))
            model_rows.append(int(r))
    return np.asarray(kept, dtype=int), np.asarray(model_rows, dtype=int)


def poisson_at_observed_depth(pr: np.ndarray, concert_to_prep: np.ndarray, raw_src: np.ndarray,
                              seed: int) -> np.ndarray:
    raw_src = np.asarray(raw_src, dtype=np.float64)
    lib_c = raw_src[:, concert_to_prep].sum(1)
    p = np.asarray(pr, dtype=np.float64)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    full = raw_src.copy()
    full[:, concert_to_prep] = np.random.default_rng(seed).poisson(p * lib_c[:, None])
    return full.astype(np.float32)


def _cf_output(pr, concert_to_prep, n_prep_genes, args, raw_src, seed):
    if args.cf_output == "poisson_depth":
        seed = int(args.cf_poisson_seed_base) + int(seed)
        if raw_src is None:
            raise ValueError("--cf_output poisson_depth needs the source spots' raw counts")
        return poisson_at_observed_depth(pr, concert_to_prep, raw_src, seed)
    full = np.zeros((pr.shape[0], n_prep_genes), dtype=np.float32)
    full[:, concert_to_prep] = np.asarray(pr, dtype=np.float32)
    return full


def predict_in_prepared_space(
    ctx: dict,
    model_rows: np.ndarray,
    panel: list[str],
    prep_genes: np.ndarray,
    args,
    raw_src: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray | None], set[str]]:
    from eval_spatial_counterfactual import _predict_swap

    concert_genes = np.asarray(ctx["gene_names"])
    p2i = {g: i for i, g in enumerate(prep_genes)}
    concert_to_prep = np.array([p2i[g] for g in concert_genes], dtype=int)
    pred_g_cache: dict[str, np.ndarray | None] = {}
    unpredictable: set[str] = set()
    for g in panel:
        code = ctx["name2code"].get(g)
        if code is None:
            pred_g_cache[g] = None
            unpredictable.add(g)
            continue
        try:
            pr = _predict_swap(ctx, model_rows, code, args)
            pred_g_cache[g] = _cf_output(pr, concert_to_prep, len(prep_genes), args, raw_src,
                                         seed=int(code))
        except Exception as exc:
            log.exception("[CONCERT/%s] prediction failed: %s", g, exc)
            pred_g_cache[g] = None
            unpredictable.add(g)
    return pred_g_cache, unpredictable


def predict_basal_in_prepared_space(
    ctx: dict,
    model_rows: np.ndarray,
    prep_genes: np.ndarray,
    args,
    raw_src: np.ndarray | None = None,
) -> np.ndarray:
    from eval_spatial_counterfactual import _predict_swap

    concert_genes = np.asarray(ctx["gene_names"])
    p2i = {g: i for i, g in enumerate(prep_genes)}
    concert_to_prep = np.array([p2i[g] for g in concert_genes], dtype=int)
    pr_b = _predict_swap(ctx, model_rows, 0, args)
    return _cf_output(pr_b, concert_to_prep, len(prep_genes), args, raw_src, seed=0)


def eval_seen_slide(args, slide_id: str) -> None:
    out_path = args.outdir / "patch" / f"slide_{slide_id}" / "lfc_agg.json"
    if out_path.exists() and not args.force:
        log.info("Slide %s: existing results -- skip", slide_id)
        return

    log.info("=== CONCERT seen patch eval slide %s ===", slide_id)
    args.slide_id = slide_id
    _, sa = sec.import_eval_utils()
    adata_target = load_prepared_slide(args.prepared_dir, slide_id)

    role = load_patch_roles(args.prepared_dir, slide_id, adata_target)
    if role is None:
        log.warning("Slide %s: no patch holdout file -- skip", slide_id)
        return

    niche_dom = patch_dominant_perts(
        adata_target, role, args.min_g_cells, src_role=args.patch_role,
    )
    if not niche_dom:
        log.warning("Slide %s: no evaluable patch niches -- skip", slide_id)
        return

    panel = sorted(set(niche_dom.values()))
    src_idx = patch_source_indices(
        adata_target, role, niche_dom, args.max_src_cells_per_niche, seed=42,
        src_role=args.patch_role,
    )
    prep_bc = np.asarray(adata_target.obs_names)
    ctx = build_concert_ctx(args, slide_id)
    kept_idx, model_rows = map_prep_to_model_rows(slide_id, prep_bc, src_idx, ctx)
    log.info(
        "Slide %s: %d niches, %d genes, %d/%d source cells mapped to model",
        slide_id, len(niche_dom), len(panel), len(kept_idx), len(src_idx),
    )
    if len(kept_idx) < args.min_g_cells:
        log.warning("Slide %s: too few mapped source cells (%d) -- skip", slide_id, len(kept_idx))
        return

    pert = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    una = np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"}))
    patch_exclude_idx = np.where(np.isin(role, (1, 2)) & una)[0]

    prep_genes = np.asarray(adata_target.var_names)
    raw_mm, lib_sizes = load_prepared_raw_resources(args.prepared_dir, slide_id)
    raw_src = np.asarray(raw_mm[kept_idx]) if args.cf_output == "poisson_depth" else None
    pred_g_cache, unpredictable = predict_in_prepared_space(
        ctx, model_rows, panel, prep_genes, args, raw_src=raw_src,
    )
    pred_basal = (
        predict_basal_in_prepared_space(ctx, model_rows, prep_genes, args, raw_src=raw_src)
        if args.cf_seen_ref == "pred_basal" else None
    )

    hvg_mask = sec.select_hvgs(adata_target, args.n_hvgs, sa)
    hvg_info = sa.hvg_info_dict(hvg_mask)

    scores, agg, mlfc = niche_lfc_patch_holdout_loop(
        niche_dom,
        adata_target,
        kept_idx,
        pred_g_cache,
        hvg_mask,
        raw_mm,
        lib_sizes,
        args,
        unpredictable_genes=unpredictable,
        exclude_from_baseline_idx=patch_exclude_idx,
        scoring_method=METHOD,
        pred_basal=pred_basal,
    )

    extra = {
        "method": METHOD,
        "benchmark": True,
        "regime": "seen",
        "niche_dominant_perts": {str(b): g for b, g in sorted(niche_dom.items())},
        "cf_output": args.cf_output,
        "cf_poisson_seed_base": int(args.cf_poisson_seed_base),
        "cf_seen_ref": args.cf_seen_ref,
        "patch_role": int(args.patch_role),
        "patch_role_meaning": "1=holdout_patch(test1); 0=seen_patch(test2)",
        "patch_index": str(smc.patch_holdout_path(args.prepared_dir, slide_id)),
        "n_src_mapped": int(len(kept_idx)),
        "n_src_total": int(len(src_idx)),
        "unpredictable_genes": sorted(unpredictable),
        "model_file": str(args.model_file),
        "n_boot": args.n_boot,
        "boot_frac": args.boot_frac,
    }
    save_multi_seen_results(
        args.outdir,
        slide_id,
        panel,
        scores,
        agg,
        mlfc,
        hvg_info,
        METHOD,
        extra=extra,
        subdir="patch",
        eval_type=f"multi_patch_dominant_role{args.patch_role}",
    )
    log.info("Seen slide %s done -> %s", slide_id, out_path.parent)


def eval_unseen_all(args) -> None:
    bench = args.benchmark_dir or (args.prepared_dir / "benchmark")
    panel = load_eval_panel(bench)
    args.outdir.mkdir(parents=True, exist_ok=True)
    ctx_cache: dict[str, dict] = {}

    for unit in filter_units(panel, args.units):
        gene = unit["gene"]
        slide_id = unit["target_slide"]
        unit_dir = args.outdir / "unseen_pert" / f"{gene}__slide_{slide_id}"
        if (unit_dir / "lfc_scores.json").exists() and not args.force:
            log.info("Skip existing %s", unit_dir)
            continue

        log.info("=== CONCERT unseen %s @ slide %s ===", gene, slide_id)
        args.slide_id = slide_id
        if slide_id not in ctx_cache:
            ctx_cache[slide_id] = build_concert_ctx(args, slide_id)
        ctx = ctx_cache[slide_id]

        adata = load_prepared_slide(args.prepared_dir, slide_id)
        _, sa = sec.import_eval_utils()
        hvg_mask = sec.select_hvgs(adata, args.n_hvgs, sa)
        hvg_info = sa.hvg_info_dict(hvg_mask)

        # shared source cells (same 5000 cells as every competitor); no CONCERT-specific subsample
        src_idx = load_source_cells(bench, slide_id)

        prep_bc = np.asarray(adata.obs_names)
        kept_idx, model_rows = map_prep_to_model_rows(slide_id, prep_bc, src_idx, ctx)
        if len(kept_idx) != len(src_idx):
            raise SystemExit(f"{gene}@{slide_id}: only {len(kept_idx)}/{len(src_idx)} shared source cells map "
                             "to model rows; the shared source file must contain mappable cells only")
        log.info(
            "Unseen %s @ %s: %d/%d holdout-unassigned cells mapped",
            gene, slide_id, len(kept_idx), len(src_idx),
        )

        prep_genes = np.asarray(adata.var_names)
        if len(kept_idx) < args.min_g_cells:
            pred_g = None
            unpredictable = True
        else:
            raw_src = None
            if args.cf_output == "poisson_depth":
                raw_mm_t, _ = smc.load_prepared_raw_resources(args.prepared_dir, slide_id)
                raw_src = np.asarray(raw_mm_t[kept_idx])
            pred_cache, unpredictable_set = predict_in_prepared_space(
                ctx, model_rows, [gene], prep_genes, args, raw_src=raw_src,
            )
            pred_g = pred_cache.get(gene)
            unpredictable = gene in unpredictable_set

        ko_idx = ko_holdout_indices(bench, gene, slide_id)
        score, m_lfc = unseen_spac(
            adata, gene, kept_idx, ko_idx, pred_g, hvg_mask, args,
            unpredictable=unpredictable or pred_g is None, scoring_method=METHOD,
        )
        save_unseen_unit_results(
            args.outdir,
            unit,
            score,
            m_lfc,
            hvg_info,
            extra={
                "method": METHOD,
                "dataset": "spacseq",
                "benchmark_dir": str(bench),
                "n_src": int(len(kept_idx)),
                "n_src_total": int(len(src_idx)),
                "n_ko_gt": int(len(ko_idx)),
                "donor_slides": unit["donor_slides"],
                "model_file": str(args.model_file),
                "cf_output": args.cf_output,
                        "cf_poisson_seed_base": int(args.cf_poisson_seed_base),
                "cf_slide_bias": bool(args.cf_slide_bias),
            },
        )
        log.info(
            "%s @ %s: spearman=%.3f n_ko_gt=%d",
            gene, slide_id, score.get("spearman", float("nan")), len(ko_idx),
        )

    aggregate_unseen(args.outdir, panel)
    log.info("unseen eval done -> %s", args.outdir)


def main():
    args = parse_args()
    if args.model_file is None:
        args.model_file = DEFAULT_SEEN_CKPT if args.regime == "seen" else DEFAULT_UNSEEN_CKPT

    if args.regime == "seen":
        if args.target_slide is None:
            raise SystemExit("--target_slide required for seen regime")
        eval_seen_slide(args, args.target_slide)
    else:
        eval_unseen_all(args)


if __name__ == "__main__":
    main()
