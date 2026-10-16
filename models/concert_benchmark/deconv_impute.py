#!/usr/bin/env python3
"""Deconvolution (Task 1) and patch imputation (Task 2) inference for CONCERT-spac.

Imported by run_concert_spac.py once the model is built + checkpoint loaded
(stages "deconv" / "impute_patch"), so it inherits the EXACT coord framing,
normalization, model architecture, and perturbation-code map used at train time.

DECONVOLUTION (Task 1, slide 2-1)
  A pseudo-bulk spot = mean counts of ~9 spatially-proximal cells (a mixture of a
  local KO patch + surrounding unassigned cells). We recover cell resolution via
  counterfactual-signature NNLS:
    - Encode each spot's expression -> basal latent (amortized by the expr encoder).
    - For each candidate perturbation p in the spot's local panel {KO_g, unassigned,
      control}, decode a signature mu_p = model's predicted PER-CELL mean counts for
      a cell of perturbation p at the spot's location (run with X's kernel one-hot =
      p AND target_cell_perturbation = p, so both the GP group and the
      mu_action+f_pert shift are p's).
    - Mixture model: y_spot ~ sum_p w_p * mu_p, solved by non-negative least squares
      (weights then L1-normalized -> composition). Because the basal cancels in
      mu_KO - mu_un (= the KO effect) and the mean of a cell mixture is linear in
      counts, w_KO is identifiable.  Output:
        * composition w  (scored vs the wholeslide frac_* ground truth)
        * cell-level expression = the mu_p signatures (scored vs the TRUE cells)

  We restrict to the top-K KO patches + surrounding cells (per-patch ROI); per ROI
  the candidate panel is {that patch's KO, unassigned, control} -> a clean 2-3
  component problem (this is how the "target perturbation state" is decided: locally,
  by which patch the spot sits in/near).
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

from preprocess import normalize

log = logging.getLogger(__name__)

UNASSIGNED = frozenset(["unassigned", "Unassigned", "No_guide", "no_guide", "unknown"])
CONTROL = frozenset(["Non_targeting", "non_targeting", "NonTargeting", "control",
                     "Control", "CTRL", "ctrl", "NT"])


# helpers
def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def _align_genes(pb: ad.AnnData, model_genes):
    """Subset pseudo-bulk to the model's gene order; returns (counts[n_spots,G], ok_mask).
    Missing model genes are filled with zeros (rare; logged)."""
    pb_names = np.asarray(pb.var_names)
    pos = {g: i for i, g in enumerate(pb_names)}      # first occurrence wins (dup var names)
    G = len(model_genes)
    cols = np.full(G, -1, dtype=np.int64)
    for j, g in enumerate(model_genes):
        if g in pos:
            cols[j] = pos[g]
    miss = int((cols < 0).sum())
    if miss:
        log.warning("[deconv] %d/%d model genes absent from pseudo-bulk -> zero-filled", miss, G)
    Xpb = _dense(pb.X).astype(np.float32)
    out = np.zeros((pb.n_obs, G), dtype=np.float32)
    good = cols >= 0
    out[:, good] = Xpb[:, cols[good]]
    return out, good


def _normalize_like_train(counts, var_names):
    """Run the SAME normalize() the trainer used; returns normalized log expr (n,G)."""
    a = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)),
                   var=pd.DataFrame(index=list(var_names)))
    a = normalize(a, filter_min_counts=False, size_factors=True,
                  normalize_input=True, logtrans_input=True)
    return _dense(a.X).astype(np.float32)


def _signature(model, pos_scaled, Y_norm, n_batch, pert_code, tissue_code, batch_size, n_samples):
    """Model per-cell mean counts if every spot were perturbation `pert_code` at its
    own location/basal. Builds X with kernel one-hot = pert_code; sets mask_cutoff to a
    constant 0.5 vector sized to the spots (it is the 0.5 init at train, indexed per row)."""
    import torch
    n = pos_scaled.shape[0]
    onehot = np.zeros((n, n_batch), dtype=np.float32)
    onehot[:, int(pert_code)] = 1.0
    X = np.concatenate([pos_scaled, onehot], axis=1).astype(np.float32)
    cell_atts = np.c_[np.full(n, int(tissue_code)), np.full(n, int(pert_code))].astype(int)
    sample_index = torch.arange(n, dtype=torch.int)

    saved = model.mask_cutoff
    try:
        # mask_cutoff is a registered nn.Parameter, so it cannot be replaced with a
        # plain tensor; mirror the (requires_grad=False) Parameter pattern used in
        # decompose_levels / step0_leakage_gate, on the model's device.
        cut = torch.full((n,), 0.5, dtype=torch.get_default_dtype()).to(model.device)
        model.mask_cutoff = torch.nn.Parameter(cut, requires_grad=False)
        mu, _ = model.counterfactualPrediction(
            X=X, sample_index=sample_index, cell_atts=cell_atts,
            perturb_cell_id=np.arange(n), target_cell_tissue=int(tissue_code),
            target_cell_perturbation=int(pert_code), n_samples=n_samples,
            batch_size=batch_size, Y=Y_norm,
        )
    finally:
        model.mask_cutoff = saved
    return np.asarray(mu, dtype=np.float32)


# deconvolution (Task 1)
def run_deconvolution(model, adata, pos_raw, pert_map, cfg, outdir):
    """adata = normalized TRAIN AnnData (var_names = model genes); pos_raw = filtered
    train raw coords (for the MinMax framing); pert_map = {name: code}."""
    outdir = Path(outdir)
    model_genes = list(adata.var_names)
    G = len(model_genes)

    # candidate KO patches: user-supplied (cfg.deconv_perts) or top-K by patch size
    pb = ad.read_h5ad(cfg.pseudobulk_file)
    cen = np.asarray(pb.obsm["spatial"], dtype=np.float64)
    dom = pb.obs["dominant_pert"].astype(str).to_numpy()
    ncell = pb.obs["n_cells"].astype(int).to_numpy() if "n_cells" in pb.obs else np.full(pb.n_obs, 9)

    if cfg.deconv_perts:
        ko_list = [g.strip() for g in cfg.deconv_perts.split(",") if g.strip()]
    else:
        is_ko = np.array([d not in UNASSIGNED and d not in CONTROL for d in dom])
        vals, cnts = np.unique(dom[is_ko], return_counts=True)
        ko_list = [v for v, _ in sorted(zip(vals, cnts), key=lambda t: -t[1])][: cfg.deconv_top_k]
    ko_list = [g for g in ko_list if g in pert_map]
    log.info("[deconv] KO patches: %s", ko_list)

    una_code = next((pert_map[n] for n in pert_map if n in UNASSIGNED), 0)
    ctrl_code = next((pert_map[n] for n in pert_map if n in CONTROL), None)
    n_batch = int(max(pert_map.values())) + 1

    # ROI = patch spots (dominant == g) | spots within radius R of the patch
    tree = cKDTree(cen)
    R = float(cfg.deconv_roi_radius)
    roi_mask = np.zeros(pb.n_obs, dtype=bool)
    roi_gene = np.empty(pb.n_obs, dtype=object)
    for g in ko_list:
        patch = np.where(dom == g)[0]
        if patch.size == 0:
            continue
        near = set()
        for s in patch:
            near.update(tree.query_ball_point(cen[s], R))
        idx = np.array(sorted(near), dtype=int)
        new = idx[~roi_mask[idx]]                 # first patch claims overlaps
        roi_mask[new] = True
        roi_gene[new] = g
    roi = np.where(roi_mask)[0]
    log.info("[deconv] %d ROI spots over %d patches (R=%.0f)", roi.size, len(ko_list), R)

    # frame coords + normalize expression (TRAIN conventions)
    scaler = MinMaxScaler().fit(pos_raw)
    cen_scaled = (scaler.transform(cen[roi]) * cfg.loc_range).astype(np.float32)
    counts_roi, _ = _align_genes(pb[roi], model_genes)
    Y_norm = _normalize_like_train(counts_roi, model_genes)

    # signatures for every candidate perturbation that appears in any ROI panel
    panel_codes = {int(una_code)}
    if ctrl_code is not None:
        panel_codes.add(int(ctrl_code))
    for g in ko_list:
        panel_codes.add(int(pert_map[g]))
    sig = {}
    for pc in sorted(panel_codes):
        log.info("[deconv] signature for pert_code=%d ...", pc)
        sig[pc] = _signature(model, cen_scaled, Y_norm, n_batch, pc, tissue_code=0,
                             batch_size=512, n_samples=cfg.deconv_n_samples)

    # per-spot NNLS over its local panel {KO_g, unassigned, (control)}
    comp = np.zeros((roi.size, n_batch), dtype=np.float32)   # composition over pert codes
    cell_pred = np.zeros((roi.size, G), dtype=np.float32)    # reconstructed (sum_p w_p mu_p)
    for i in range(roi.size):
        g = roi_gene[roi[i]]
        codes = [int(una_code), int(pert_map[g])]
        if ctrl_code is not None:
            codes.append(int(ctrl_code))
        A = np.stack([sig[c][i] for c in codes], axis=1)     # (G, n_comp)
        try:
            w, _ = nnls(A, counts_roi[i], maxiter=10 * A.shape[1])
        except (ValueError, RuntimeError):
            # scipy nnls can crash ("zero-size array to reduction") on
            # ill-conditioned/degenerate ROI spots -> clipped LS fallback
            w, *_ = np.linalg.lstsq(A, counts_roi[i], rcond=None)
            w = np.clip(w, 0.0, None)
        s = w.sum()
        wf = w / s if s > 0 else np.full(len(codes), 1.0 / len(codes))
        for c, wc in zip(codes, wf):
            comp[i, c] = wc
        cell_pred[i] = A @ w

    # save everything for the (model-free) scoring/plotting script
    np.savez(
        outdir / "deconv_results.npz",
        spot_index=roi.astype(np.int64),
        roi_gene=roi_gene[roi].astype(object),
        centroid=cen[roi].astype(np.float32),
        n_cells=ncell[roi].astype(np.int64),
        composition=comp,                       # (n_roi, n_batch) NNLS fractions
        cell_pred=cell_pred,                    # (n_roi, G) reconstructed mean counts
        obs_counts=counts_roi,                  # (n_roi, G) observed spot mean counts
        signatures=np.stack([sig[c] for c in sorted(panel_codes)], axis=0),  # (P, n_roi, G)
        signature_codes=np.array(sorted(panel_codes), dtype=np.int64),
        genes=np.array(model_genes, dtype=object),
        pert_names=np.array([f"{c}:{n}" for n, c in pert_map.items()], dtype=object),
    )
    log.info("[deconv] wrote %s", outdir / "deconv_results.npz")

    # cell-level reconstruction: expand each spot into its members
    # Resolution recovery: allocate w_p * n_cells members to each perturbation p in the
    # spot's panel, give each member that spot's type signature mu_p, place it at the spot
    # centroid (sub-spot position is not recoverable from pseudobulk). Output = a cells x
    # genes slide with a perturbation label + coordinate, comparable to the TRUE cells.
    code2name = {v: k for k, v in pert_map.items()}
    ce, cp, cxy, csrc = [], [], [], []
    for i in range(roi.size):
        g = roi_gene[roi[i]]
        codes = [int(una_code), int(pert_map[g])] + ([int(ctrl_code)] if ctrl_code is not None else [])
        nc = int(max(1, ncell[roi[i]]))
        w = comp[i, codes].astype(float); s = w.sum()
        if s <= 0:
            continue
        raw_k = w / s * nc                       # largest-remainder allocation -> sums to nc
        k = np.floor(raw_k).astype(int)
        rem = nc - int(k.sum())
        if rem > 0:
            k[np.argsort(-(raw_k - k))[:rem]] += 1
        for c, kc in zip(codes, k):
            if kc <= 0:
                continue
            ce.append(np.repeat(sig[c][i][None, :], kc, axis=0))
            cp.extend([code2name.get(c, str(c))] * kc)
            cxy.append(np.repeat(cen[roi[i]][None, :], kc, axis=0))
            csrc.extend([int(roi[i])] * kc)
    if ce:
        Xc = np.concatenate(ce, axis=0).astype(np.float32)
        rec = ad.AnnData(Xc, var=pd.DataFrame(index=list(model_genes)))
        rec.obs["perturbation"] = pd.Categorical(cp)
        rec.obs["src_spot"] = np.asarray(csrc, dtype=np.int64)
        rec.obsm["spatial"] = np.concatenate(cxy, axis=0).astype(np.float32)
        rec.write(outdir / "deconv_celllevel.h5ad")
        log.info("[deconv] wrote cell-level reconstruction %s (%d cells)",
                 outdir / "deconv_celllevel.h5ad", Xc.shape[0])


# patch imputation (Task 2)
def run_impute_patch(model, adata, pos_raw, pos_batched, cell_atts,
                     sample_indices, pert_map, cfg, outdir):
    """Impute the masked patch cells (role==1) at their held-out coords from the
    surrounding TRAINED tissue, per perturbation group.

    For KO groups this is impute + counterfactual (the KO cells were never seen, so
    we predict 'what a KO cell looks like here' = impute_and_counterfactual_fun1 with
    target=KO code, which pulls spatial neighbors of that perturbation). For the
    unassigned group it is pure imputation (target=unassigned, no pert shift).
    model = trained on the MASKED slide; adata = its normalized TRAIN AnnData.
    """
    import torch
    outdir = Path(outdir)
    model_genes = list(adata.var_names)
    G = len(model_genes)
    n_batch = int(max(pert_map.values())) + 1

    npz = np.load(cfg.mask_index_file, allow_pickle=True)
    role = npz["role"]; on = npz["obs_name"].astype(str); grp = npz["group"].astype(str)
    masked = role == 1

    raw = ad.read_h5ad(cfg.raw_slide_file)
    raw_obs = np.asarray(raw.obs_names)
    obs_pos = {o: i for i, o in enumerate(raw_obs)}
    coords_raw = np.asarray(raw.obsm["spatial"], dtype=np.float64)
    raw_counts_full, _ = _align_genes(raw, model_genes)   # (n_raw, G) counts for GT

    scaler = MinMaxScaler().fit(pos_raw)               # train (masked-slide) framing
    X_train = pos_batched.astype(np.float32)
    Y_train = _dense(adata.X).astype(np.float32)
    Y_sidx = np.asarray(sample_indices)
    Y_atts = np.asarray(cell_atts)

    rows_imp, rows_gt, rows_grp, rows_coord = [], [], [], []
    for g in np.unique(grp[masked]):
        sel = masked & (grp == g)
        names = on[sel]
        ridx = np.array([obs_pos[o] for o in names if o in obs_pos], dtype=int)
        if ridx.size == 0:
            continue
        tcode = int(pert_map.get(g, next((pert_map[n] for n in pert_map if n in UNASSIGNED), 0)))
        framed = (scaler.transform(coords_raw[ridx]) * cfg.loc_range).astype(np.float32)
        onehot = np.zeros((ridx.size, n_batch), dtype=np.float32); onehot[:, tcode] = 1.0
        X_test = np.concatenate([framed, onehot], axis=1).astype(np.float32)
        log.info("[impute] group=%s code=%d n=%d", g, tcode, ridx.size)
        _, imp = model.impute_and_counterfactual_fun1(   # (latents, denoised_means); want means
            target=tcode, tissue=0, X_test=X_test, X_train=X_train,
            Y_sample_index=Y_sidx, Y_cell_atts=Y_atts,
            n_samples=cfg.deconv_n_samples, batch_size=512, Y_train=Y_train,
        )
        imp = np.asarray(imp, dtype=np.float32)
        rows_imp.append(imp)
        rows_gt.append(raw_counts_full[ridx])
        rows_grp.append(np.array([g] * ridx.size, dtype=object))
        rows_coord.append(coords_raw[ridx].astype(np.float32))

    np.savez(
        outdir / "impute_results.npz",
        imputed=np.concatenate(rows_imp, axis=0),
        gt_counts=np.concatenate(rows_gt, axis=0),
        group=np.concatenate(rows_grp, axis=0),
        coord=np.concatenate(rows_coord, axis=0),
        genes=np.array(model_genes, dtype=object),
    )
    log.info("[impute] wrote %s", outdir / "impute_results.npz")
