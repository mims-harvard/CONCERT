#!/usr/bin/env python3
"""Multi-slide counterfactual eval on HELD-OUT unassigned source patches.

Companion to eval_cf_una_multi.py (SEEN-source). Here the source unassigned cells
are restricted to the v3 held-out patches (holdout_v3_*.json -> same coords the
training removed), so the score is a clean generalization test:

  seen-task   ckpt (ms_spac1_{arch}_seen)   : gene-KO effect WAS trained; predict on
                                               held-out source patches.
  unseen-task ckpt (ms_spac1_{arch}_unseen) : gene-KO cells were ALSO masked from
                                               training -> the effect itself is unseen.

Both hand the model held-out source patches (keep_only), never memorized cells.
The v3 (additive niche) and niche-FiLM checkpoints are both supported -- pass the
matching --use_niche_interaction/--niche_rank OR --use_niche_film flags.

Held-out cells are matched exactly as training did: (sample_id, rint(raw_coords))
against holdout_v3_*.json["unassigned"][str(sid)]. The KO reference (observed KO
cells + observed-unassigned LFC reference) is taken from the FULL target-slide ctx
-- KO cells are never held out (seen-task) or are simply present in the eval h5ad
even when masked from training (unseen-task).

Metrics per genexniche (via run_setup, single source of truth):
  abs_rho   pseudobulk Spearman over HVGs (spatial advantage vs baselines)
  lfc_rho   LFC Spearman  (with-noise: pred(una->g)-pred(una->0) vs KO-una)
  lfc_dist  LFC-RMSE      (magnitude agreement; lower is better)
  e_dist    energy distance
  knn_*     LONO-ish global spatial-kNN baseline (forced on via args.force_knn)
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

import numpy as np

SPAC_DIR = Path(__file__).parent
sys.path.insert(0, str(SPAC_DIR))

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq"
DEFAULT_SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_slide", required=True, choices=DEFAULT_SLIDES)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--model_file", required=True)
    p.add_argument("--holdout_v3_file", required=True,
                   help="holdout_v3_spac_{seen,unseen}.json -- coord-keyed held-out source patches.")
    p.add_argument("--task_label", default="heldout",
                   help="tag for the output subdir / summary (e.g. v3_seen, niche_unseen).")
    p.add_argument("--multi_sample_files", default=None,
                   help="comma list of all 5 slide h5ads (default: raw_output_1-1..1-5).")
    p.add_argument("--hvg_max_rank",    type=int, default=2000)
    p.add_argument("--min_cells_per_pert", type=int, default=25,
                   help="load filter; MUST match training (sets vocab/N).")
    p.add_argument("--panel_min_cells", type=int, default=200)
    p.add_argument("--max_panel",       type=int, default=30)
    p.add_argument("--una_floor",       type=int, default=50)
    p.add_argument("--min_g_cells",     type=int, default=20)
    p.add_argument("--max_src_per_niche", type=int, default=100)
    p.add_argument("--niche_grid",      type=int, default=6)
    p.add_argument("--knn_k",           type=int, default=20)
    p.add_argument("--n_samples",       type=int, default=10)
    p.add_argument("--batch_size",      type=int, default=512)
    p.add_argument("--n_hvg",           type=int, default=2000)
    # Architecture -- MUST match training (state_dict load)
    p.add_argument("--use_expr_encoder",      type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--expr_hidden_dims",      type=int, nargs="+", default=[512, 256])
    p.add_argument("--use_basal_encoder_only",type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--use_nngp",              type=lambda x: x.lower()=="true", default=False)
    p.add_argument("--nngp_lambda",           type=float, default=0.0)
    p.add_argument("--use_paracrine",         type=lambda x: x.lower()=="true", default=False)
    p.add_argument("--paracrine_l1",          type=float, default=0.0)
    # v3 additive niche interaction
    p.add_argument("--use_niche_interaction", type=lambda x: x.lower()=="true", default=False)
    p.add_argument("--niche_rank",            type=int, default=8)
    # niche-FiLM head
    p.add_argument("--use_niche_film",        type=lambda x: x.lower()=="true", default=False)
    p.add_argument("--niche_film_dim",        type=int, default=8)
    p.add_argument("--niche_film_k",          type=int, default=25)
    p.add_argument("--niche_film_gcap",       type=float, default=3.0)
    p.add_argument("--niche_film_l2",         type=float, default=1e-3)
    p.add_argument("--niche_film_rank",       type=int, default=8)
    p.add_argument("--niche_film_dispersion", type=lambda x: x.lower()=="true", default=False)
    p.add_argument("--use_pert_gate",         type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--gate_hidden",           type=int, default=64)
    # f_pert footprint field + basal field -- MUST match training
    p.add_argument("--use_action_correction", type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--inducing_point_steps",  type=int, default=6)
    p.add_argument("--spatial_basal_only",    type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--basal_field",           default="rff")
    p.add_argument("--basal_bayes",           type=lambda x: x.lower()=="true", default=True)
    p.add_argument("--f_pert_clamp",          type=float, default=5.0)
    p.add_argument("--fpert_field",           default="gp_pp")
    p.add_argument("--fpert_rff_features",    type=int, default=128)
    p.add_argument("--fpert_rff_bandwidth",   type=float, default=0.15)
    p.add_argument("--fpert_hidden",          type=int, default=256)
    p.add_argument("--fpert_svgp_inducing",   type=int, default=16)
    p.add_argument("--fpert_svgp_ls_init",    type=float, default=5.0)
    p.add_argument("--fpert_coord_scale",     type=float, default=20.0)
    p.add_argument("--fpert_mb_bandwidths",   default=None)
    p.add_argument("--ls_prior_min",          type=float, default=3.0)
    p.add_argument("--ls_prior_max",          type=float, default=18.0)
    # gp_pp band in NORMALIZED units; when set, overrides ls_prior_* for the footprint
    # field only. MUST match training -- the band is not stored in the state_dict, so a
    # mismatch renders a different lengthscale from the same ls_head weights.
    p.add_argument("--fpert_ls_min",          type=float, default=None)
    p.add_argument("--fpert_ls_max",          type=float, default=None)
    return p.parse_args()


def heldout_unassigned_mask(ctx, holdout_json, target_sid):
    """Boolean mask over ctx cells: unassigned AND in the held-out coord set for
    this slide. Matches training's (sid, rint(raw_coord)) keying exactly."""
    coords_raw = np.asarray(ctx["coords_raw"])
    kx = np.rint(coords_raw[:, 0]).astype(np.int64)
    ky = np.rint(coords_raw[:, 1]).astype(np.int64)
    want = set()
    for xy in holdout_json.get("unassigned", {}).get(str(target_sid), []):
        want.add((int(xy[0]), int(xy[1])))
    in_hold = np.fromiter(((int(x), int(y)) in want for x, y in zip(kx.tolist(), ky.tolist())),
                          dtype=bool, count=len(kx))
    is_una = ctx["cls"] == "unassigned"
    return in_hold & is_una, int(is_una.sum()), len(want)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.force_knn = True

    import eval_utils as eu
    from eval_spatial_counterfactual import load_eval_context_multi, _predict_swap
    from eval_cf_una_only import run_setup

    paths = (args.multi_sample_files.split(",") if args.multi_sample_files
             else [f"{DATA_DIR}/raw_output_{s}.h5ad" for s in DEFAULT_SLIDES])
    target_idx = DEFAULT_SLIDES.index(args.target_slide)

    # Full model-build namespace -- every flag that shapes the checkpoint graph.
    fwd = SimpleNamespace(
        multi_sample_files=",".join(paths), target_slide_idx=target_idx,
        model_file=args.model_file, hvg_max_rank=args.hvg_max_rank,
        min_cells_per_pert=args.min_cells_per_pert,
        use_expr_encoder=args.use_expr_encoder, expr_hidden_dims=args.expr_hidden_dims,
        use_basal_encoder_only=args.use_basal_encoder_only,
        use_nngp=args.use_nngp, nngp_lambda=args.nngp_lambda,
        use_paracrine=args.use_paracrine, paracrine_l1=args.paracrine_l1,
        use_niche_interaction=args.use_niche_interaction, niche_rank=args.niche_rank,
        use_niche_film=args.use_niche_film, niche_film_dim=args.niche_film_dim,
        niche_film_k=args.niche_film_k, niche_film_gcap=args.niche_film_gcap,
        niche_film_l2=args.niche_film_l2, niche_film_rank=args.niche_film_rank,
        niche_film_dispersion=args.niche_film_dispersion,
        use_pert_gate=args.use_pert_gate, gate_hidden=args.gate_hidden,
        use_action_correction=args.use_action_correction,
        inducing_point_steps=args.inducing_point_steps,
        spatial_basal_only=args.spatial_basal_only,
        basal_field=args.basal_field, basal_bayes=args.basal_bayes,
        f_pert_clamp=args.f_pert_clamp, fpert_field=args.fpert_field,
        fpert_rff_features=args.fpert_rff_features, fpert_rff_bandwidth=args.fpert_rff_bandwidth,
        fpert_hidden=args.fpert_hidden, fpert_svgp_inducing=args.fpert_svgp_inducing,
        fpert_svgp_ls_init=args.fpert_svgp_ls_init, fpert_coord_scale=args.fpert_coord_scale,
        fpert_mb_bandwidths=args.fpert_mb_bandwidths,
        ls_prior_min=args.ls_prior_min, ls_prior_max=args.ls_prior_max,
        fpert_ls_min=args.fpert_ls_min, fpert_ls_max=args.fpert_ls_max,
        n_samples=args.n_samples, batch_size=args.batch_size, graph_file=None,
    )
    ctx = load_eval_context_multi(fwd)
    log.info("Multi ctx: target %s (sid=%d) -> %d cells, %d combined perts",
             args.target_slide, target_idx, ctx["N"], ctx["n_batch"])

    # Restrict SOURCE to held-out unassigned patches
    holdout = json.load(open(args.holdout_v3_file))
    hold_mask, n_una_total, n_coords = heldout_unassigned_mask(ctx, holdout, target_idx)
    n_hold = int(hold_mask.sum())
    log.info("[holdout] %s slide %s: %d held-out source cells "
             "(of %d unassigned; %d coords listed for this slide)",
             Path(args.holdout_v3_file).name, args.target_slide, n_hold, n_una_total, n_coords)
    if n_hold < args.una_floor:
        log.warning("[holdout] only %d held-out source cells (< una_floor=%d) -- panel will be thin",
                    n_hold, args.una_floor)

    cls = ctx["cls"]
    perturb_raw = ctx["perturb_raw"]
    Y_obs = ctx["Y_obs"]
    ko_bin, _ = eu.build_niche_grid(ctx["coords"], args.niche_grid)
    hvg_mask = eu.select_hvgs(Y_obs, n_top=args.n_hvg)

    ko_mask = cls == "gene_KO"
    pert_counts = Counter(perturb_raw[ko_mask].tolist())
    eligible = {g: n for g, n in pert_counts.items() if n >= args.panel_min_cells}
    panel = sorted(eligible, key=lambda g: -eligible[g])[:args.max_panel]
    log.info("Panel: %d genes (>=%d KO cells, top %d): %s",
             len(panel), args.panel_min_cells, args.max_panel, [(g, eligible[g]) for g in panel])
    ko_idx = {g: np.where(ko_mask & (perturb_raw == g))[0] for g in panel}

    # KO reference from the FULL ctx (observed KO targets + observed-unassigned LFC ref).
    ko_ref = dict(Y_obs=Y_obs, bin_id=ko_bin, hvg_mask=hvg_mask, name2code=ctx["name2code"],
                  panel=panel, ko_idx=ko_idx,
                  una_obs_idx=np.where(cls == "unassigned")[0], coords=ctx["coords"])

    # SOURCE ctx: same model + arrays, but only held-out unassigned count as source.
    src_cls = np.asarray(cls, dtype=object).copy()
    src_cls[(cls == "unassigned") & (~hold_mask)] = "unassigned_train"  # excluded from source
    src_ctx = dict(ctx)
    src_ctx["cls"] = src_cls

    res = run_setup(args.task_label, src_ctx, ko_ref, args, eu, _predict_swap)
    if res:
        (args.outdir / f"summary_{args.task_label}.json").write_text(json.dumps(
            dict(task=args.task_label, target=args.target_slide,
                 model_file=args.model_file, holdout=Path(args.holdout_v3_file).name,
                 n_heldout_src=n_hold,
                 abs_rho_mean=res["abs_rho_mean"], lfc_rho_mean=res["lfc_rho_mean"],
                 lfc_dist_mean=res["lfc_dist_mean"], e_dist_mean=res["e_dist_mean"],
                 knn_abs_rho_mean=res["knn_abs_rho_mean"], knn_lfc_rho_mean=res["knn_lfc_rho_mean"],
                 knn_e_dist_mean=res["knn_e_dist_mean"],
                 n_genes=res["n_genes"], n_niche_rows=res["n_niche_rows"]), indent=2))
        log.info("== %s [%s] MODEL abs=%.3f lfc=%.3f lfc_rmse=%.3f e=%.3f | kNN abs=%.3f lfc=%.3f (%d genes)",
                 args.task_label, args.target_slide, res["abs_rho_mean"], res["lfc_rho_mean"],
                 res["lfc_dist_mean"], res["e_dist_mean"],
                 res["knn_abs_rho_mean"], res["knn_lfc_rho_mean"], res["n_genes"])


if __name__ == "__main__":
    main()
