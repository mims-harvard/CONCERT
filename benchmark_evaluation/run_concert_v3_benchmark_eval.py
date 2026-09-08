#!/usr/bin/env python3
"""CONCERT SPAC-seq eval on the benchmark v3 protocol (same harness as baselines).

Seen: patch-dominant holdout (role==1) via niche_lfc_patch_holdout_loop.
Unseen: eval_panel.json units via unseen_v3_spac.

Writes to multi_1-5_ho_v3/concert/{seen,unseen}/ in the same layout as
run_multi_seen_eval.py / run_benchmark_v3_unseen_eval.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

CONCERT_DIR = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/CONCERT-spac-niche")
EVAL_DIR = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval")
sys.path.insert(0, str(CONCERT_DIR))
sys.path.insert(0, str(EVAL_DIR))

import eval_spatial_counterfactual as ecf  # noqa: E402

import spacseq_eval_common as sec
import spacseq_multi_common as smc
from benchmark_v3_eval_lib import (
    aggregate_unseen_v3,
    ko_holdout_indices,
    load_eval_panel,
    load_holdout_unassigned,
    save_unseen_unit_results,
    unseen_v3_spac,
)
from spacseq_eval_common import UNASSIGNED_LABELS
from spacseq_multi_common import (
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

DATA_DIR = Path("/n/holylabs/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq")
DEFAULT_PREP = EVAL_DIR / "multi_1-5/prepared"
DEFAULT_SEEN_CKPT = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_spac1_niche_seen/model.pt"
)
DEFAULT_UNSEEN_CKPT = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_spac1_niche_unseen/model.pt"
)
METHOD = "concert"


def parse_args():
    p = argparse.ArgumentParser(description="CONCERT benchmark v3 SPAC-seq eval.")
    p.add_argument("--regime", required=True, choices=["seen", "unseen"])
    p.add_argument("--prepared_dir", type=Path, default=DEFAULT_PREP)
    p.add_argument("--benchmark_v3_dir", type=Path, default=None)
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
    # a DIFFERENT lengthscale if it differs. ls_prior_* are in loc units (legacy, v4 = [3,18]);
    # fpert_ls_* are in normalized units and, when given, override ls_prior_* for gp_pp only.
    p.add_argument("--ls_prior_min", type=float, default=3.0)
    p.add_argument("--ls_prior_max", type=float, default=18.0)
    p.add_argument("--fpert_ls_min", type=float, default=None)
    p.add_argument("--fpert_ls_max", type=float, default=None)
    smc.add_bootstrap_args(p, default_n=10)
    return p.parse_args()


def _raw_barcode_to_pixel(sid: str) -> dict:
    path = str(DATA_DIR / f"raw_output_{sid}.h5ad")
    with h5py.File(path, "r") as f:
        bc = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs"]["_index"][:]]
        sp_ = np.asarray(f["obsm"]["array_coords"][:], dtype=np.float64)
    return {b: (round(float(x), 1), round(float(y), 1)) for b, (x, y) in zip(bc, sp_)}


def _auto_arch_from_checkpoint(fwd: SimpleNamespace, model_file: Path) -> None:
    try:
        import torch

        sd = torch.load(str(model_file), map_location="cpu")
        z = sd.get("fpert_field.Z")
        if z is not None and int(z.shape[0]) != int(fwd.fpert_svgp_inducing):
            log.info("auto-set fpert_svgp_inducing %s -> %s", fwd.fpert_svgp_inducing, int(z.shape[0]))
            fwd.fpert_svgp_inducing = int(z.shape[0])
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
            fwd.niche_film_k = 25
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
    )
    _auto_arch_from_checkpoint(fwd, args.model_file)
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


def predict_in_prepared_space(
    ctx: dict,
    model_rows: np.ndarray,
    panel: list[str],
    prep_genes: np.ndarray,
    args,
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
            full = np.zeros((len(model_rows), len(prep_genes)), dtype=np.float32)
            full[:, concert_to_prep] = np.asarray(pr, dtype=np.float32)
            pred_g_cache[g] = full
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
) -> np.ndarray:
    from eval_spatial_counterfactual import _predict_swap

    concert_genes = np.asarray(ctx["gene_names"])
    p2i = {g: i for i, g in enumerate(prep_genes)}
    concert_to_prep = np.array([p2i[g] for g in concert_genes], dtype=int)
    pr_b = _predict_swap(ctx, model_rows, 0, args)
    full = np.zeros((len(model_rows), len(prep_genes)), dtype=np.float32)
    full[:, concert_to_prep] = np.asarray(pr_b, dtype=np.float32)
    return full


def eval_seen_slide(args, slide_id: str) -> None:
    out_path = args.outdir / "patch" / f"slide_{slide_id}" / "lfc_agg.json"
    if out_path.exists() and not args.force:
        log.info("Slide %s: existing results -- skip", slide_id)
        return

    log.info("=== CONCERT v3 seen patch eval slide %s ===", slide_id)
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
    pred_g_cache, unpredictable = predict_in_prepared_space(
        ctx, model_rows, panel, prep_genes, args,
    )
    pred_basal = predict_basal_in_prepared_space(ctx, model_rows, prep_genes, args)

    raw_mm, lib_sizes = load_prepared_raw_resources(args.prepared_dir, slide_id)
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
        "benchmark_v3": True,
        "regime": "seen",
        "niche_dominant_perts": {str(b): g for b, g in sorted(niche_dom.items())},
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
    b3 = args.benchmark_v3_dir or (args.prepared_dir / "benchmark_v3")
    panel = load_eval_panel(b3)
    args.outdir.mkdir(parents=True, exist_ok=True)
    ctx_cache: dict[str, dict] = {}

    for unit in panel["eval_units"]:
        gene = unit["gene"]
        slide_id = unit["target_slide"]
        unit_dir = args.outdir / "unseen_pert" / f"{gene}__slide_{slide_id}"
        if (unit_dir / "lfc_scores.json").exists() and not args.force:
            log.info("Skip existing %s", unit_dir)
            continue

        log.info("=== CONCERT v3 unseen %s @ slide %s ===", gene, slide_id)
        args.slide_id = slide_id
        if slide_id not in ctx_cache:
            ctx_cache[slide_id] = build_concert_ctx(args, slide_id)
        ctx = ctx_cache[slide_id]

        adata = load_prepared_slide(args.prepared_dir, slide_id)
        _, sa = sec.import_eval_utils()
        hvg_mask = sec.select_hvgs(adata, args.n_hvgs, sa)
        hvg_info = sa.hvg_info_dict(hvg_mask)

        src_idx = load_holdout_unassigned(b3, slide_id)
        if args.max_src_cells > 0 and len(src_idx) > args.max_src_cells:
            rng = np.random.default_rng(42)
            src_idx = rng.choice(src_idx, args.max_src_cells, replace=False)

        prep_bc = np.asarray(adata.obs_names)
        kept_idx, model_rows = map_prep_to_model_rows(slide_id, prep_bc, src_idx, ctx)
        log.info(
            "Unseen %s @ %s: %d/%d holdout-unassigned cells mapped",
            gene, slide_id, len(kept_idx), len(src_idx),
        )

        prep_genes = np.asarray(adata.var_names)
        if len(kept_idx) < args.min_g_cells:
            pred_g = None
            unpredictable = True
        else:
            pred_cache, unpredictable_set = predict_in_prepared_space(
                ctx, model_rows, [gene], prep_genes, args,
            )
            pred_g = pred_cache.get(gene)
            unpredictable = gene in unpredictable_set

        ko_idx = ko_holdout_indices(b3, gene, slide_id)
        score = unseen_v3_spac(
            adata, gene, kept_idx, ko_idx, pred_g, hvg_mask, args,
            unpredictable=unpredictable or pred_g is None, scoring_method=METHOD,
        )

        eu, _ = sec.import_eval_utils()
        src_expr = smc.to_dense(adata[kept_idx].X) if len(kept_idx) else np.empty((0, adata.n_vars))
        m_lfc = eu.pseudobulk_lfc(pred_g, src_expr) if pred_g is not None else []
        save_unseen_unit_results(
            args.outdir,
            unit,
            score,
            m_lfc.tolist() if hasattr(m_lfc, "tolist") else [],
            hvg_info,
            extra={
                "method": METHOD,
                "dataset": "spacseq",
                "benchmark_v3_dir": str(b3),
                "n_src": int(len(kept_idx)),
                "n_src_total": int(len(src_idx)),
                "n_ko_gt": int(len(ko_idx)),
                "donor_slides": unit["donor_slides"],
                "model_file": str(args.model_file),
            },
        )
        log.info(
            "%s @ %s: spearman=%.3f e_dist=%.3f n_ko_gt=%d",
            gene, slide_id, score.get("spearman", float("nan")),
            score.get("e_dist", float("nan")), len(ko_idx),
        )

    aggregate_unseen_v3(args.outdir, panel)
    log.info("v3 unseen eval done -> %s", args.outdir)


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
