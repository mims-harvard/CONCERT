#!/usr/bin/env python3
"""
CONCERT-spac model decomposition into interpretable levels.

Decoder structure:
    log mu = f_basal(z) + 1[has_guide]*mu_action + 1[real_KO]*f_pert(p)
             \_basal_/   \_pert NOISE (drop)_/    \_pert EFFECT_/
    z = [GP dims (spatial), Normal dims (cell-state)]

Produces 4 levels:
  1) BASAL              mu_basal = f_basal(z), [force_basal] -> UMAP (expect mixed by pert)
  2) PERT EFFECT        f_pert(p) LFC matrix (n_pert x G) -> per-pert UMAP (expect perts separate)
  3) SPATIAL EFFECT     GP latent dims (z[:,:2]) -> tissue x/y map (smooth spatial field)
  4) COMBINED, NO NOISE mu_basal*exp(f_pert), mu_action dropped -> UMAP (pert structure emerges)

IMPORTANT -- cell alignment:
  LORD's basal embedding table is sized to the TRAINING cell count and indexed
  by global cell position. Training ran normalize() which can drop cells, so we
  MUST replicate that exact pipeline. We reuse make_umap.load_data() (identical
  to the working UMAP job) to guarantee alignment, then free the expression
  matrix before the heavy GPU compute (inference itself never needs expression).

Outputs in --outdir:
  decompose_levels.pdf
  level_arrays.npz   (umap coords for L1/L4, gp latent for L3, pert umap for L2)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from run_concert_spac import build_inducing_points
from concert_spac import CONCERT
from make_umap import load_data, classify_perts, build_model as build_umap_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

DATA_FILE  = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_1-1.h5ad"
MODEL_FILE = "/n/holylabs/mzitnik_lab/Users/xianglin226/CONCERT/outputs/raw_output_1-1/model_spac_1-1_v2.pt"
OUTDIR     = "/n/holylabs/mzitnik_lab/Users/xianglin226/CONCERT/umap_v2"

UNASSIGNED = frozenset(["unassigned","Unassigned","No_guide","no_guide","unknown"])
CONTROL    = frozenset(["Non_targeting","non_targeting","NonTargeting","control","Control","CTRL","ctrl","NT"])
EXCLUDED   = frozenset(["multiple","Multiple"])
MIN_CELLS  = 25
LOC_RANGE  = 20.0
IND_STEPS  = 6
ENC_DIM    = 256
N_GENES    = 9998         # final HVG count after normalize (matches training)
BATCH_INF  = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",     default=OUTDIR)
    p.add_argument("--data_file",  default=DATA_FILE)
    p.add_argument("--model_file", default=MODEL_FILE)
    p.add_argument("--subsample",  type=int, default=120000, help="cells for expression UMAPs (L1/L4)")
    p.add_argument("--top_k",      type=int, default=20)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--use_nngp",   type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--nngp_lambda",type=float, default=1.0)
    p.add_argument("--graph_file", default=None, help="path to graph_1-1.npz (required if --use_nngp True)")
    # Phase 2.8 full-model modules -- MUST match the flags the checkpoint was trained with,
    # otherwise load_model fails on missing/unexpected state_dict keys.
    p.add_argument("--use_paracrine",         type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--paracrine_l1",          type=float, default=0.0)
    p.add_argument("--use_niche_interaction", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--niche_rank",            type=int, default=8)
    # Step 0: per-cell scalar gate (must match training to load v5d_gate checkpoints)
    p.add_argument("--use_pert_gate",         type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--gate_hidden",           type=int, default=64)
    # Phase 2.9: inductive expression encoder (must match training to load v4+ checkpoints)
    p.add_argument("--use_expr_encoder",      type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--expr_hidden_dims",      nargs="+", type=int, default=[512, 256])
    return p.parse_args()


def quick_umap(mat, seed, n_neighbors=30, min_dist=0.3, n_pcs=50):
    a = ad.AnnData(mat.astype("float32"))
    sc.pp.pca(a, n_comps=min(n_pcs, mat.shape[1]-1), random_state=seed)
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(a, min_dist=min_dist, random_state=seed)
    return a.obsm["X_umap"]


# Plot helpers
CLASS_COLORS = {"unassigned": "#bbbbbb", "control": "#2166ac", "gene_KO": "#d73027"}

def _cmap(n):
    return [to_hex(plt.cm.tab20(i % 20)) for i in range(n)]

def umap_by_class(ax, umap, cls, title):
    for c in ["unassigned", "control", "gene_KO"]:
        m = cls == c
        ax.scatter(umap[m,0], umap[m,1], c=CLASS_COLORS[c], s=0.4, alpha=0.4,
                   linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6)
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.legend(handles=[mpatches.Patch(color=CLASS_COLORS[c],
              label=f"{c} ({(cls==c).sum():,})") for c in ["unassigned","control","gene_KO"]],
              fontsize=6, loc="best", framealpha=0.8)

def umap_by_topk(ax, umap, perturb, cls, top, title):
    other = np.array([p not in UNASSIGNED and p not in CONTROL and p not in set(top) for p in perturb])
    ax.scatter(umap[other,0], umap[other,1], c="#e8e8e8", s=0.2, alpha=0.15, linewidths=0, rasterized=True)
    ax.scatter(umap[cls=="unassigned",0], umap[cls=="unassigned",1], c="#bbbbbb", s=0.2, alpha=0.2, linewidths=0, rasterized=True)
    colors = _cmap(len(top))
    pat = []
    for p, c in zip(top, colors):
        m = perturb == p
        ax.scatter(umap[m,0], umap[m,1], c=c, s=1.0, alpha=0.8, linewidths=0, rasterized=True)
        pat.append(mpatches.Patch(color=c, label=p))
    ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6)
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.legend(handles=pat, fontsize=4.5, ncol=2, loc="best", framealpha=0.8)


# Main
def main():
    args   = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng    = np.random.RandomState(args.seed)

    # Reuse make_umap.load_data -- IDENTICAL preprocessing to the working UMAP job,
    # so LORD global cell indices align with the trained basal embedding table.
    logging.info("Loading data via make_umap.load_data (matches training pipeline)...")
    (adata, pos_batched, cell_atts, batch_oh, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(args.data_file)
    N = adata.n_obs
    pos_scaled     = pos_batched[:, :2]                       # first 2 cols = scaled x/y
    pert_names_arr = np.array([pert_name.get(i, f"code_{i}") for i in range(n_batch)])
    cls_all        = classify_perts(perturb_raw)
    logging.info("Cells: %d  perts: %d", N, n_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Building model on %s...", device)
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
    if args.use_pert_gate:
        extra["use_pert_gate"] = True
        extra["gate_hidden"] = args.gate_hidden
    if args.use_expr_encoder:
        extra["use_expr_encoder"] = True
        extra["expr_hidden_dims"] = args.expr_hidden_dims
    model, device = build_umap_model(adata, pos_batched, cell_atts, cutoff_t, n_batch, **extra)
    model.load_model(args.model_file)
    if args.use_nngp:
        if not args.graph_file:
            raise ValueError("--graph_file required when --use_nngp True")
        from spatial_graph import SpatialGraph
        logging.info("Attaching spatial graph for NNGP: %s", args.graph_file)
        graph = SpatialGraph.load(args.graph_file, device=device)
        model.attach_graph(graph)
        logging.info("Graph attached (N=%d)", graph.knn_idx.shape[0])
    model.eval()

    # Phase 2.9: the inductive expr encoder needs the normalized expression at inference
    # (the old transductive lookup did not). Capture it before freeing adata.
    n_vars = adata.n_vars
    if args.use_expr_encoder:
        Yexpr = adata.X
        try:
            import scipy.sparse as _sp
            if _sp.issparse(Yexpr):
                Yexpr = Yexpr.toarray()
        except Exception:
            pass
        Yexpr = np.asarray(Yexpr, dtype=np.float32)
        logging.info("[expr-encoder] captured expression matrix %s for inference", Yexpr.shape)
    else:
        Yexpr = None
    del adata
    import gc; gc.collect()

    # Subsample for expression-level UMAPs (L1, L4)
    n_sub = min(args.subsample, N)
    sub = rng.choice(N, n_sub, replace=False); sub.sort()
    logging.info("Subsample for L1/L4 UMAP: %d cells", n_sub)

    pos_b_sub  = pos_batched[sub]
    atts_sub   = cell_atts[sub]
    idx_sub    = torch.arange(N)[sub]    # NOTE: LORD indexes by global cell index
    cut_sub    = cutoff_t[sub]
    hg_sub     = has_guide[sub]
    rk_sub     = is_real_KO[sub]
    cls_sub    = cls_all[sub]
    pert_sub   = perturb_raw[sub]

    # Temporarily set model.mask_cutoff to the subsample for denoise (it slices by row)
    model.mask_cutoff = torch.nn.Parameter(cut_sub.to(device), requires_grad=False)

    # LEVEL 1: basal expression
    logging.info("[L1] Computing basal expression (force_basal=True)...")
    mu_basal, _ = model.batching_denoise_counts(
        pos_b_sub, idx_sub, atts_sub, n_samples=1, batch_size=BATCH_INF, force_basal=True,
        Y=(Yexpr[sub] if Yexpr is not None else None))
    logging.info("[L1] mu_basal %s -> log1p + UMAP", mu_basal.shape)
    umap_basal = quick_umap(np.log1p(mu_basal), args.seed)

    # LEVEL 4: combined, no pert noise (basal*exp(f_pert), no mu_action)
    logging.info("[L4] Computing combined no-noise (has_guide=None, is_real_KO set)...")
    mu_clean, _ = model.batching_denoise_counts(
        pos_b_sub, idx_sub, atts_sub, n_samples=1, batch_size=BATCH_INF,
        has_guide=None, is_real_KO=rk_sub, force_basal=False,
        Y=(Yexpr[sub] if Yexpr is not None else None))
    logging.info("[L4] mu_clean %s -> log1p + UMAP", mu_clean.shape)
    umap_clean = quick_umap(np.log1p(mu_clean), args.seed)

    # restore full cutoff for any later full-data ops
    model.mask_cutoff = torch.nn.Parameter(cutoff_t.to(device), requires_grad=False)

    # LEVEL 3: spatial effect = GP latent dims (all cells)
    logging.info("[L3] Extracting latent z (GP dims = spatial) for all %d cells...", N)
    Z = model.batching_latent_samples(pos_batched, torch.arange(N), cell_atts, batch_size=BATCH_INF,
                                      Y=Yexpr)
    gp_latent = Z[:, :2]   # spatial GP dims
    logging.info("[L3] GP latent %s", gp_latent.shape)

    # Phase 2.9 collapse + disentanglement diagnostics
    gp_std = gp_latent.std(axis=0)
    logging.info("[collapse-check] GP-dim per-dim std: %s  (COLLAPSED if < 1e-3)",
                 np.array2string(gp_std, precision=6))
    np.save(f"{args.outdir}/gp_latent_std.npy", gp_std)
    try:
        from sklearn.metrics import silhouette_score
        ko_mask_all = np.isin(cls_all, ["control", "gene_KO"])
        if int(ko_mask_all.sum()) > 100 and len(np.unique(cls_all[ko_mask_all])) > 1:
            ss = min(5000, int(ko_mask_all.sum()))
            sil = silhouette_score(gp_latent[ko_mask_all], cls_all[ko_mask_all],
                                   sample_size=ss, random_state=args.seed)
            logging.info("[disentangle-check] class silhouette in GP latent: %.4f  "
                         "(want ~0; high = pert leakage into z)", sil)
    except Exception as e:
        logging.info("[disentangle-check] skipped: %s", e)

    # Step 0: per-cell scalar gate histogram (over real-KO cells)
    # gate_i = sigmoid(pert_gate([gp_latent ; basal_latent])).  Z (full posterior-mean
    # latent) == cat([gp_p_m, gs_mu]) == the gate's input, so we reuse it directly.
    # Degenerate (all ~1) -> cell-specificity earned nothing; spread -> location/state
    # modulation is real.
    if getattr(model, "use_pert_gate", False) and model.pert_gate is not None:
        with torch.no_grad():
            gate_all = torch.sigmoid(
                model.pert_gate(torch.as_tensor(Z, dtype=torch.float32, device=device))
            ).squeeze(-1).cpu().numpy()
        ko_mask = np.asarray(is_real_KO, dtype=bool).reshape(-1)
        gate_ko = gate_all[ko_mask]
        np.save(f"{args.outdir}/pert_gate_values_ko.npy", gate_ko)
        if gate_ko.size > 0:
            logging.info("[gate] KO-cell gate: n=%d  mean=%.4f  std=%.4f  "
                         "min=%.4f  p5=%.4f  median=%.4f  p95=%.4f  max=%.4f",
                         gate_ko.size, gate_ko.mean(), gate_ko.std(),
                         gate_ko.min(), np.percentile(gate_ko, 5),
                         np.median(gate_ko), np.percentile(gate_ko, 95), gate_ko.max())
            figg, axg = plt.subplots(figsize=(6, 4))
            axg.hist(gate_ko, bins=50, range=(0, 1), color="#4C72B0", edgecolor="white")
            axg.axvline(float(np.median(gate_ko)), color="crimson", ls="--",
                        label=f"median={np.median(gate_ko):.3f}")
            axg.set_xlabel("gate = sigmoid(MLP([gp; basal]))  in [0,1]")
            axg.set_ylabel("KO cells")
            axg.set_title("Step 0 per-cell gate (real-KO cells)\nall~1 -> no modulation; spread -> real")
            axg.legend()
            gate_pdf = f"{args.outdir}/pert_gate_hist.pdf"
            figg.savefig(gate_pdf, dpi=150, bbox_inches="tight")
            plt.close(figg)
            logging.info("[gate] histogram -> %s", gate_pdf)

    # LEVEL 2: pert effect = f_pert LFC per pert -> per-pert UMAP
    logging.info("[L2] Computing f_pert LFC matrix (%d perts)...", n_batch)
    with torch.no_grad():
        codes    = torch.arange(n_batch, dtype=torch.long, device=device)
        pert_emb = model.lord_encoder.s_encoder["perturbation"](codes)
        lfc_mat  = model.f_pert(pert_emb).cpu().float().numpy()
    lfc_norm = np.linalg.norm(lfc_mat, axis=1)
    # per-pert UMAP (drop unassigned/control codes for clarity)
    ko_codes = np.array([i for i in range(n_batch)
                         if pert_names_arr[i] not in UNASSIGNED and pert_names_arr[i] not in CONTROL])
    pert_umap = quick_umap(lfc_mat[ko_codes], args.seed, n_neighbors=15, n_pcs=min(50, len(ko_codes)-1))

    # top-K perts by LFC for coloring expression UMAPs
    top_lfc = [pert_names_arr[i] for i in np.argsort(lfc_norm)[::-1]
               if pert_names_arr[i] not in UNASSIGNED and pert_names_arr[i] not in CONTROL][:args.top_k]

    # Figure
    fig = plt.figure(figsize=(22, 16))
    gs  = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.28)

    # Row 1: L1 basal (class + topK)
    umap_by_class(fig.add_subplot(gs[0,0]), umap_basal, cls_sub,
                  "L1 BASAL -- f_basal(z)\ncolored by class (expect MIXED = pert removed)")
    umap_by_topk(fig.add_subplot(gs[0,1]), umap_basal, pert_sub, cls_sub, top_lfc,
                 f"L1 BASAL -- top-{args.top_k} KOs (expect mixed)")
    # L2 pert effect per-pert UMAP
    axp = fig.add_subplot(gs[0,2])
    scp = axp.scatter(pert_umap[:,0], pert_umap[:,1], c=lfc_norm[ko_codes],
                      cmap="viridis", s=25, linewidths=0)
    for rank, i in enumerate(np.argsort(lfc_norm[ko_codes])[::-1][:15]):
        axp.annotate(pert_names_arr[ko_codes][i], (pert_umap[i,0], pert_umap[i,1]),
                     fontsize=4.5, xytext=(2,2), textcoords="offset points")
    plt.colorbar(scp, ax=axp, label="||LFC||", fraction=0.046)
    axp.set_title("L2 PERT EFFECT -- f_pert(p) per-pert UMAP\n(1 dot/pert; expect SEPARATION)", fontsize=8)
    axp.set_xlabel("UMAP1", fontsize=7); axp.set_ylabel("UMAP2", fontsize=7); axp.tick_params(labelsize=6)

    # Row 2: L3 spatial GP fields (2 dims) on tissue
    for d in range(2):
        ax = fig.add_subplot(gs[1, d])
        # FIXED absolute color scale (+/-2.5 latent units). A collapsed near-constant
        # field renders uniformly mid-color (flat); a real field varies. Prevents the
        # per-panel autoscale artifact that produced the false "L3 pass" in v3a/v3b.
        sc_ = ax.scatter(pos_scaled[:,0], pos_scaled[:,1], c=gp_latent[:,d],
                         cmap="coolwarm", s=0.3, alpha=0.7, linewidths=0, rasterized=True,
                         vmin=-2.5, vmax=2.5)
        plt.colorbar(sc_, ax=ax, fraction=0.046)
        ax.set_title(f"L3 SPATIAL EFFECT -- GP latent dim {d} on tissue\n(smooth field = spatial structure)", fontsize=8)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.tick_params(labelsize=6); ax.set_aspect("equal")
    # tissue colored by class for reference
    axc = fig.add_subplot(gs[1,2])
    for c in ["unassigned","control","gene_KO"]:
        m = cls_all == c
        axc.scatter(pos_scaled[m,0], pos_scaled[m,1], c=CLASS_COLORS[c], s=0.2, alpha=0.4, linewidths=0, rasterized=True)
    axc.set_title("Tissue -- perturbation class (reference)", fontsize=8)
    axc.set_xlabel("X", fontsize=7); axc.set_ylabel("Y", fontsize=7)
    axc.tick_params(labelsize=6); axc.set_aspect("equal")

    # Row 3: L4 combined no-noise (class + topK) + spacer
    umap_by_class(fig.add_subplot(gs[2,0]), umap_clean, cls_sub,
                  "L4 COMBINED no-noise -- basal*exp(f_pert)\nmu_action dropped (expect pert STRUCTURE)")
    umap_by_topk(fig.add_subplot(gs[2,1]), umap_clean, pert_sub, cls_sub, top_lfc,
                 f"L4 COMBINED no-noise -- top-{args.top_k} KOs (expect separation)")
    # compare panel: same cells, basal vs clean side note
    axn = fig.add_subplot(gs[2,2]); axn.axis("off")
    axn.text(0.0, 0.95,
             "Decomposition:\n"
             "  log mu = f_basal(z) + mu_action + f_pert(p)\n\n"
             "L1 basal  = f_basal(z)            -> mixed\n"
             "L2 pert   = f_pert(p)             -> separates\n"
             "L3 spatial= GP latent (z[:,:2])   -> smooth field\n"
             "L4 clean  = basal*exp(f_pert)     -> pert structure\n"
             "            (mu_action 'pert noise' removed)\n\n"
             f"cells (L1/L4 UMAP): {n_sub:,} of {N:,}\n"
             f"perts: {n_batch}  |  KO perts: {len(ko_codes)}",
             fontsize=7, family="monospace", va="top")

    model_tag = Path(args.model_file).stem
    nngp_tag = " + NNGP" if args.use_nngp else ""
    fig.suptitle(f"CONCERT-spac -- model decomposition by level ({model_tag}{nngp_tag})", fontsize=12, y=1.005)
    out_pdf = outdir / "decompose_levels.pdf"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure -> %s", out_pdf)

    np.savez_compressed(
        outdir / "level_arrays.npz",
        umap_basal=umap_basal.astype("float32"),
        umap_clean=umap_clean.astype("float32"),
        gp_latent=gp_latent.astype("float32"),
        pos_scaled=pos_scaled.astype("float32"),
        pert_umap=pert_umap.astype("float32"),
        pert_umap_codes=ko_codes,
        lfc_norm=lfc_norm.astype("float32"),
        sub=sub, cls_sub=cls_sub, pert_sub=pert_sub,
        pert_names=pert_names_arr,
    )
    logging.info("Saved arrays -> %s/level_arrays.npz", outdir)
    logging.info("DONE")


if __name__ == "__main__":
    main()
