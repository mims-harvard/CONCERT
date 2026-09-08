"""Disentanglement diagnostic (D1-D3) for the RFF-basal gate: svgp vs rff.

Computed on a trained checkpoint, on UNASSIGNED/unperturbed cells (has_guide<0.5 —
the cells that define the spatial basal field in `spatial_basal_only` routing):

  D1  footprint-scale spatial leakage in z_basal  = _moran_i_scale(gs_mu, coords, radius)
        LOWER is better disentangled (PRIMARY gate quantity).
  D2  spatial signal captured by the basal field  = gp_p_m.var(dim=0)
        must be > 0 (RFF) / healthy (svgp) -> confirms the spatial field is NOT collapsed.
  D3  reconstruction = mean NB NLL on a held-out split  -> must not degrade vs svgp.

Loader/build contract mirrors eval_pmap_patches.py. Coords fed to the model and to
_moran_i_scale are MinMax-scaled to [0, LOC_RANGE=20] in make_umap.load_data, so `radius`
is in that scaled space (4.0 across all datasets; gate compares rff-vs-svgp at the SAME R).

Usage:
  python diag_disentangle.py --dataset 2-2 --field rff \
      --data_file <h5ad> --ckpt <model.pt> --outdir <dir> --radius 4.0 \
      --niche_rank 8 --ind_steps 6 --min_cells_per_pert 25
"""
import argparse, json, logging, os
import numpy as np
import torch
from make_umap import load_data, build_model
from concert_spac import _moran_i_scale, NBLoss

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--field", required=True, choices=["svgp", "rff"])
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--niche_rank", type=int, default=8)
    ap.add_argument("--ind_steps", type=int, default=6)
    ap.add_argument("--min_cells_per_pert", type=int, default=25)
    ap.add_argument("--hvg_max_rank", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--moran_subsample", type=int, default=5000)
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--heldout_cap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── data ──────────────────────────────────────────────────────────────────
    (adata, pos_batched, cell_atts, batch_oh, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(
        args.data_file, hvg_max_rank=args.hvg_max_rank,
        min_cells_per_pert=args.min_cells_per_pert)
    N = adata.n_obs
    coords = pos_batched[:, :2].astype(np.float32)            # scaled [0,20]
    Yexpr = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=np.float32)
    has_guide = np.asarray(has_guide).astype(float)
    is_real_KO = np.asarray(is_real_KO).astype(float)
    perturb_raw = np.asarray(perturb_raw)
    # Basal/unassigned cells = those defining the spatial basal field (model routing:
    # is_una = has_guide<0.5). make_umap.load_data uses HARDCODED UNASSIGNED_LABELS that
    # do NOT include perturbmap's "unperturbed" (so has_guide is all-True there); fall back
    # to a name-based basal set in that case so D1/D2 are measured on the true basal cells.
    una = has_guide < 0.5
    una_src = "has_guide<0.5"
    if int(una.sum()) == 0:
        BASAL_NAMES = {"unperturbed", "unassigned", "Unassigned", "no_guide",
                       "No_guide", "unknown"}
        una = np.isin(perturb_raw, list(BASAL_NAMES))
        una_src = "name-based basal set"
    logging.info("[%s/%s] N=%d unassigned=%d (via %s) n_batch=%d radius=%.2f",
                 args.dataset, args.field, N, int(una.sum()), una_src, n_batch, args.radius)

    # ── model ─────────────────────────────────────────────────────────────────
    extra = dict(basal_field=args.field, disentangle_radius=args.radius,
                 fpert_field="rff", fpert_coord_scale=20.0,
                 fpert_rff_features=128, fpert_rff_bandwidth=0.15)
    model, device = build_model(
        adata, pos_batched, cell_atts, cutoff_t, n_batch,
        use_basal_encoder_only=True, spatial_basal_only=True,
        use_pert_gate=True, gate_hidden=64,
        use_expr_encoder=True, expr_hidden_dims=[512, 256],
        use_niche_interaction=True, niche_rank=args.niche_rank,
        ind_steps=args.ind_steps, f_pert_clamp=5.0, **extra)
    model.load_model(args.ckpt, allow_cell_size_mismatch=True)
    model.eval()

    # ── latents (gp_p_m | gs_mu) ───────────────────────────────────────────────
    si = torch.arange(N)
    latents = model.batching_latent_samples(
        X=pos_batched, sample_index=si, cell_atts=cell_atts,
        batch_size=args.batch_size, Y=Yexpr)                  # (N, GP_dim+Normal_dim)
    GP = model.GP_dim
    gp_p_m = latents[:, :GP]
    gs_mu = latents[:, GP:]

    # ── D1: footprint-scale leakage in z_basal (unassigned only) ───────────────
    ui = np.where(una)[0]
    if len(ui) > args.moran_subsample:
        ui = rng.choice(ui, args.moran_subsample, replace=False)
    gs_u = torch.tensor(gs_mu[ui], dtype=torch.float32)
    co_u = torch.tensor(coords[ui], dtype=torch.float32)
    with torch.no_grad():
        D1 = float(_moran_i_scale(gs_u, co_u, args.radius).item())

    # ── D2: spatial field variance (not collapsed?) ────────────────────────────
    field_u = gp_p_m[ui]
    D2_per_dim = field_u.var(axis=0).tolist()
    D2 = float(np.mean(D2_per_dim))

    # ── D3: held-out NB NLL (mean per element) ─────────────────────────────────
    D3 = None
    try:
        order = rng.permutation(N)
        n_ho = min(args.heldout_cap, max(1, int(args.heldout_frac * N)))
        ho = np.sort(order[:n_ho])
        mu_ho, _ = model.batching_denoise_counts(
            X=pos_batched[ho], sample_index=si[ho], cell_atts=cell_atts[ho],
            n_samples=1, batch_size=args.batch_size,
            has_guide=has_guide[ho], is_real_KO=is_real_KO[ho],
            force_basal=False, Y=Yexpr[ho])
        mu_ho = torch.tensor(np.asarray(mu_ho), dtype=torch.float32)
        if adata.raw is not None:
            rawX = adata.raw.X
        else:
            rawX = adata.X
            logging.warning("adata.raw is None; using adata.X as counts (may be normalized)")
        rawX = np.asarray(rawX.todense() if hasattr(rawX, "todense") else rawX, dtype=np.float32)
        raw_ho = torch.tensor(rawX[ho], dtype=torch.float32)
        if "size_factors" in adata.obs:
            sf = torch.tensor(np.asarray(adata.obs["size_factors"])[ho], dtype=torch.float32)
        else:
            sf = torch.ones(len(ho), dtype=torch.float32)
            logging.warning("no size_factors in adata.obs; using 1.0")
        # per-cell dispersion (matches forward :1133-1136)
        dec_disp = model.dec_disp.detach().cpu()
        if model.shared_dispersion:
            disp = torch.exp(torch.clamp(dec_disp, -15.0, 15.0)).expand(len(ho), -1)
        else:
            b_ho = torch.tensor(batch_oh[ho], dtype=torch.float32)        # (n,n_batch)
            disp = torch.exp(torch.clamp(b_ho @ dec_disp.T, -15.0, 15.0))  # (n,G)
        nb = NBLoss()
        nll_sum = float(nb(x=raw_ho, mean=mu_ho, disp=disp, scale_factor=sf).item())
        D3 = nll_sum / (raw_ho.shape[0] * raw_ho.shape[1])
    except Exception as e:
        import traceback; logging.error("D3 failed: %s", e); traceback.print_exc()

    metrics = dict(
        dataset=args.dataset, field=args.field, ckpt=args.ckpt, radius=args.radius,
        N=int(N), n_unassigned=int(una.sum()), moran_n=int(len(ui)),
        D1_footprint_leakage=D1, D2_field_var_mean=D2, D2_field_var_per_dim=D2_per_dim,
        D3_heldout_nb_nll=D3,
    )
    out = os.path.join(args.outdir, "disentangle_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== disentangle metrics (%s / basal=%s) =====" % (args.dataset, args.field))
    print(f"{'metric':<28s} {'value':>14s}")
    print(f"{'D1 footprint-leakage':<28s} {D1:>14.6f}   (lower=better)")
    print(f"{'D2 field-var (mean)':<28s} {D2:>14.6f}   (>0 = not collapsed)")
    print(f"{'D2 field-var per-dim':<28s} {str([round(x,5) for x in D2_per_dim]):>14s}")
    print(f"{'D3 heldout NB NLL':<28s} {('%.6f'%D3) if D3 is not None else 'NA':>14s}   (lower=better)")
    print("wrote", out)


if __name__ == "__main__":
    main()
