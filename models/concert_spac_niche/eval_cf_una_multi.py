#!/usr/bin/env python3
"""Multi-slide model · per-niche unassigned-only counterfactual eval (SEEN only).

Same scoring rule as the single-slide eval (eval_cf_una_only.py), but the model
is the MULTI-slide checkpoint (trained on all 5 slides). For a target slide we
predict on its SEEN unassigned cells (no holdout) → KO, per the top perturbations.

Panel rule (identical to single-slide):
  - gene-KO with >= panel_min_cells on the slide, top max_panel by count
  - each gene scored in its MOST ABUNDANT niche (>= min_g_cells KO, >= una_floor una)
  - predict on at most max_src_per_niche unassigned source cells per niche
  - LONO kNN baseline reported alongside the model

Reuses run_setup() from eval_cf_una_only so the scoring is a single source of truth.
The multi model is loaded via eval_spatial_counterfactual.load_eval_context_multi
(same loader the cross-slide eval used; truncates per-cell mask_cutoff to the target
slide via allow_cell_size_mismatch=True).
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
DEFAULT_MODEL = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/multi_v6/model_spac_multi_v6b.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_slide", required=True, choices=DEFAULT_SLIDES)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--model_file", default=DEFAULT_MODEL)
    p.add_argument("--multi_sample_files", default=None,
                   help="comma list of all 5 slide h5ads (default: raw_output_1-1..1-5).")
    p.add_argument("--hvg_max_rank",    type=int, default=2000)
    p.add_argument("--min_cells_per_pert", type=int, default=25,
                   help="load filter; MUST match multi training (sets vocab/N).")
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
    # Architecture — must match multi_v6b training
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


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import eval_utils as eu
    from eval_spatial_counterfactual import load_eval_context_multi, _predict_swap
    from eval_cf_una_only import run_setup

    paths = (args.multi_sample_files.split(",") if args.multi_sample_files
             else [f"{DATA_DIR}/raw_output_{s}.h5ad" for s in DEFAULT_SLIDES])
    target_idx = DEFAULT_SLIDES.index(args.target_slide)

    fwd = SimpleNamespace(
        multi_sample_files=",".join(paths), target_slide_idx=target_idx,
        model_file=args.model_file, hvg_max_rank=args.hvg_max_rank,
        min_cells_per_pert=args.min_cells_per_pert,
        use_expr_encoder=args.use_expr_encoder, expr_hidden_dims=args.expr_hidden_dims,
        use_basal_encoder_only=args.use_basal_encoder_only,
        use_nngp=args.use_nngp, nngp_lambda=args.nngp_lambda,
        use_paracrine=args.use_paracrine, paracrine_l1=args.paracrine_l1,
        use_niche_interaction=args.use_niche_interaction, niche_rank=args.niche_rank,
        use_pert_gate=args.use_pert_gate, gate_hidden=args.gate_hidden,
        n_samples=args.n_samples, batch_size=args.batch_size, graph_file=None,
    )
    ctx = load_eval_context_multi(fwd)
    log.info("Multi ctx: target %s → %d cells, %d combined perts", args.target_slide, ctx["N"], ctx["n_batch"])

    cls, perturb_raw, Y_obs = ctx["cls"], ctx["perturb_raw"], ctx["Y_obs"]
    ko_bin, _ = eu.build_niche_grid(ctx["coords"], args.niche_grid)
    hvg_mask = eu.select_hvgs(Y_obs, n_top=args.n_hvg)

    ko_mask = cls == "gene_KO"
    pert_counts = Counter(perturb_raw[ko_mask].tolist())
    eligible = {g: n for g, n in pert_counts.items() if n >= args.panel_min_cells}
    panel = sorted(eligible, key=lambda g: -eligible[g])[:args.max_panel]
    log.info("Panel: %d genes (>=%d KO cells, top %d): %s",
             len(panel), args.panel_min_cells, args.max_panel, [(g, eligible[g]) for g in panel])
    ko_idx = {g: np.where(ko_mask & (perturb_raw == g))[0] for g in panel}

    ko_ref = dict(Y_obs=Y_obs, bin_id=ko_bin, hvg_mask=hvg_mask, name2code=ctx["name2code"],
                  panel=panel, ko_idx=ko_idx, una_obs_idx=np.where(cls == "unassigned")[0],
                  coords=ctx["coords"])

    res = run_setup("seen", ctx, ko_ref, args, eu, _predict_swap)
    if res:
        (args.outdir / "summary_seen.json").write_text(json.dumps(
            dict(target=args.target_slide, model="multi_v6b",
                 abs_rho_mean=res["abs_rho_mean"], lfc_rho_mean=res["lfc_rho_mean"],
                 e_dist_mean=res["e_dist_mean"], knn_abs_rho_mean=res["knn_abs_rho_mean"],
                 knn_lfc_rho_mean=res["knn_lfc_rho_mean"], knn_e_dist_mean=res["knn_e_dist_mean"],
                 n_genes=res["n_genes"], n_niche_rows=res["n_niche_rows"]), indent=2))
        log.info("== %s MODEL abs=%.3f lfc=%.3f e=%.3f | kNN abs=%.3f lfc=%.3f e=%.3f (%d genes)",
                 args.target_slide, res["abs_rho_mean"], res["lfc_rho_mean"], res["e_dist_mean"],
                 res["knn_abs_rho_mean"], res["knn_lfc_rho_mean"], res["knn_e_dist_mean"], res["n_genes"])


if __name__ == "__main__":
    main()
