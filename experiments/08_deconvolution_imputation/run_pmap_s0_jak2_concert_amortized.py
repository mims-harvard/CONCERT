#!/usr/bin/env python3
"""Paper-grade pmap s0 Jak2 deconv -- SAME method as the SPAC-seq 2-1 deconvolution.

Caller: sub_pmap_s0_jak2_concert_amortized.sh

Pipeline (mirrors concert_deconv_methods.py):
  1. Full contiguous non-normal patch around Jak2.
  2. Location-matched CONCERT signatures mu_KO(x), mu_un(x).
  3. Amortized head on synthetic mixtures with those location mu's.
  4. Compare to naive NNLS on the same CONCERT signatures.
  5. Primary metrics = composition Pearson/MAE/win-rate on held-out synth
     at patch coordinates (same claim style as 2-1).

Writes: location_signatures.npz {spot_ids, mu_ko, mu_un, mu_normal, y_obs, genes},
        summary.json, real_patch_mixed_spots.csv, PNGs.
Never store empirical panel means as CONCERT signatures.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

NICHE_DIR = Path(f"{_rp.REPO}/models/concert_spac_niche")
CKPT_DEFAULT = Path(f"{_rp.WORKDIR}/outputs/ms_pmap_niche_seen")
DATA = Path(f"{_rp.DATA}perturbmap/raw/perturbmap_s0_spacfmt.h5ad")
OUT_DEFAULT = Path(
    f"{_rp.WORKDIR}/paper/Outputs/"
    "spac_seq/pmap_s0_jak2_concert_amortized"
)
LOC_RANGE = 20.0
KO_GENE = "Jak2"
N_HVG = 1500


def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def logcpm(X):
    lib = np.asarray(X).sum(1, keepdims=True)
    lib[lib <= 0] = 1.0
    return np.log1p(np.asarray(X) / lib * 1e4).astype(np.float32)


def safe_corr(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(spearmanr(a, b).correlation)


def hex_neighbors(rows, cols):
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    out = []
    for r, c in zip(rows, cols):
        r, c = int(r), int(c)
        offs = (
            [(0, 1), (0, -1), (-1, 0), (-1, 1), (1, 0), (1, 1)]
            if r % 2 == 0
            else [(0, 1), (0, -1), (-1, -1), (-1, 0), (1, -1), (1, 0)]
        )
        out.append([pos[k] for dr, dc in offs if (k := (r + dr, c + dc)) in pos])
    return out


def jak2_contiguous_patch(is_jak2, is_normal, nbrs):
    layer = np.full(len(is_jak2), -1, dtype=int)
    q = deque()
    for i in np.where(is_jak2)[0]:
        layer[int(i)] = 0
        q.append(int(i))
    while q:
        u = q.popleft()
        for v in nbrs[u]:
            if layer[v] >= 0 or is_normal[v]:
                continue
            layer[v] = layer[u] + 1
            q.append(v)
    return layer


def unmix_nnls(y, mu_ko, mu_una):
    A = np.stack([mu_ko, mu_una], axis=1)
    try:
        w, _ = nnls(A, y.astype(np.float64))
    except Exception:
        w, *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
        w = np.clip(w, 0, None)
    s = float(w.sum())
    return (w / s) if s > 0 else np.array([0.5, 0.5])


def unmix_effect(y, mu_ko, mu_una):
    delta = mu_ko - mu_una
    resid = y - mu_una
    denom = float(np.dot(delta, delta))
    if denom < 1e-12:
        return np.array([0.5, 0.5])
    w = float(np.clip(np.dot(resid, delta) / denom, 0.0, 1.0))
    return np.array([w, 1.0 - w])


def make_features(y, mu_ko, mu_una):
    resid = y - mu_una
    effect = mu_ko - mu_una
    w_nnls = unmix_nnls(y, mu_ko, mu_una)
    w_eff = unmix_effect(y, mu_ko, mu_una)

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    prop = np.array([w_nnls[0], w_eff[0], 0.5 * (w_nnls[0] + w_eff[0])], dtype=np.float32)
    sims = np.array(
        [
            cos(y, mu_ko), cos(y, mu_una), cos(resid, effect),
            float(np.mean(resid)), float(np.std(resid)),
            float(np.mean(effect)), float(np.std(effect)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([y, effect, resid, prop, sims]).astype(np.float32)


class DeconvHead(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_amortized(X, w_true, epochs=80, lr=1e-3, batch=256, seed=42, device="cpu"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = DeconvHead(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(w_true, dtype=torch.float32, device=device)
    n = X.shape[0]
    n_val = max(100, n // 10)
    perm0 = rng.permutation(n)
    val_idx, tr_idx = perm0[:n_val], perm0[n_val:]
    best_state, best_mae = None, 1e9
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        for s in range(0, len(perm), batch):
            idx = perm[s : s + batch]
            pred = model(Xt[idx])
            loss = (
                nn.functional.mse_loss(pred, yt[idx])
                + 0.25 * nn.functional.l1_loss(pred, yt[idx])
                + 0.1 * nn.functional.binary_cross_entropy(pred, yt[idx])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            mae = float(np.mean(np.abs(model(Xt[val_idx]).cpu().numpy() - w_true[val_idx])))
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  epoch {ep+1} val_MAE={mae:.4f} best={best_mae:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_w(model, feats, device="cpu"):
    return model(torch.tensor(feats, device=device)).cpu().numpy()


def cf_query(model, ctx, query_xy_raw, y_parent, pert_code, tissue_code, is_ko_sig,
             n_samples, batch_size):
    xy = ctx["pos_raw"]
    tissue = ctx["tissue_raw"]
    pert = ctx["perturb_raw"]
    pert_map = ctx["pert_map"]
    tissue_dict = ctx["tissue_dict"]
    scaler = ctx["scaler"]
    n_batch = ctx["n_batch"]
    Y_norm = ctx["Y_norm"]
    comb_scaled = ctx["pos_scaled"]
    n_all = len(xy)
    n_q = len(query_xy_raw)
    q_scaled = (scaler.transform(query_xy_raw) * LOC_RANGE).astype(np.float32)
    big_xy = np.vstack([comb_scaled, q_scaled])
    big_Y = np.vstack([Y_norm, np.repeat(y_parent, n_q, axis=0)])
    slide_pert_idx = np.array([pert_map.get(p, 0) for p in pert], dtype=int)
    onehot = np.zeros((big_xy.shape[0], n_batch), dtype=np.float32)
    onehot[:n_all] = np.eye(n_batch, dtype=np.float32)[slide_pert_idx]
    onehot[n_all:, int(pert_code)] = 1.0
    atts = np.zeros((big_xy.shape[0], 2), dtype=int)
    for j in range(n_all):
        atts[j, 0] = tissue_dict.get(str(tissue[j]), 0)
        atts[j, 1] = slide_pert_idx[j]
    atts[n_all:, 0] = int(tissue_code)
    atts[n_all:, 1] = int(pert_code)
    X = np.concatenate([big_xy, onehot], axis=1).astype(np.float32)
    q_ids = list(range(n_all, n_all + n_q))
    mu, _ = model.counterfactualPrediction(
        X=X, sample_index=torch.arange(len(X), dtype=torch.int),
        cell_atts=atts, perturb_cell_id=q_ids,
        target_cell_tissue=int(tissue_code), target_cell_perturbation=int(pert_code),
        n_samples=n_samples, batch_size=batch_size, Y=big_Y,
        target_has_guide=False, target_is_real_KO=bool(is_ko_sig),
    )
    return mu[n_all:]


def load_concert_slide0(data_file, ckpt_dir, hvg_path):
    sys.path.insert(0, str(NICHE_DIR))
    from make_umap import build_model, normalize
    from run_concert_spac import load_h5_dataset

    with open(ckpt_dir / "pert_map.json") as fh:
        pm = json.load(fh)
    pert_map = {k: int(v) for k, v in pm["pert_map"].items()}
    tissue_dict = {str(k): int(v) for k, v in pm["tissue_dict"].items()}
    gene_order = np.array([ln.strip() for ln in open(hvg_path) if ln.strip()], dtype=object)
    X, pos_raw, tissue_raw, perturb_raw, _, _, gene_names = load_h5_dataset(
        str(data_file), unassigned_labels=frozenset(["unperturbed"]),
        control_labels=frozenset(["__NONE__"]), hvg_max_rank=2000,
        force_gene_order=gene_order,
    )
    tissue_idx = np.array([tissue_dict.get(str(t), 0) for t in tissue_raw], dtype=int)
    perturb_idx = np.array([pert_map.get(p, 0) for p in perturb_raw], dtype=int)
    n_batch = int(max(pert_map.values())) + 1
    batch = np.eye(n_batch, dtype=np.float32)[perturb_idx]
    scaler = MinMaxScaler().fit(pos_raw)
    pos_scaled = (scaler.transform(pos_raw) * LOC_RANGE).astype(np.float32)
    pos_batched = np.concatenate([pos_scaled, batch], axis=1).astype(np.float32)
    cutoff_t = torch.full((len(pos_raw),), 0.5, dtype=torch.float32)
    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)
    adata = ad.AnnData(X.astype(np.float32))
    adata.var_names = list(gene_names)
    adata = normalize(adata, filter_min_counts=False, size_factors=True,
                      normalize_input=True, logtrans_input=True)
    Y_norm = _dense(adata.X).astype(np.float32)
    extra = dict(
        use_niche_interaction=False, use_niche_film=True,
        niche_film_dim=8, niche_film_k=25, niche_film_gcap=3.0, niche_film_l2=1e-3,
        use_pert_gate=True, gate_hidden=64,
        use_expr_encoder=True, expr_hidden_dims=[512, 256],
        spatial_basal_only=True, use_action_correction=False,
        ind_steps=4, basal_field="rff", basal_bayes=True,
        fpert_field="gp_pp", fpert_coord_scale=20.0,
        fpert_rff_features=128, fpert_rff_bandwidth=0.15,
        fpert_svgp_inducing=64, fpert_svgp_ls_init=5.0,
        ls_prior_min=3, ls_prior_max=18, f_pert_clamp=5.0,
    )
    model, device = build_model(adata, pos_batched, cell_atts, cutoff_t, n_batch,
                                use_basal_encoder_only=True, **extra)
    model.load_model(str(ckpt_dir / "model.pt"), allow_cell_size_mismatch=True)
    model.eval()
    print(f"CONCERT loaded on {device}, N={adata.n_obs}, G={adata.n_vars}", flush=True)
    return dict(
        model=model, scaler=scaler, pert_map=pert_map, tissue_dict=tissue_dict,
        n_batch=n_batch, counts=X.astype(np.float32), Y_norm=Y_norm,
        pos_raw=pos_raw, pos_scaled=pos_scaled, perturb_raw=perturb_raw,
        tissue_raw=tissue_raw, tissue_idx=tissue_idx, gene_names=np.asarray(gene_names),
        device=device,
    )


def display_coords(xy):
    x, y = xy[:, 0].astype(float), xy[:, 1].astype(float)
    return np.column_stack([y, -x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--ckpt_dir", type=Path, default=CKPT_DEFAULT)
    ap.add_argument("--outdir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n_synth", type=int, default=4000)
    ap.add_argument("--n_synth_test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sig_cache", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = ad.read_h5ad(args.data)
    pert = meta.obs["perturbation"].astype(str).to_numpy()
    tissue = meta.obs["tissue"].astype(str).to_numpy()
    xy = np.asarray(meta.obsm["spatial"], dtype=np.float64)
    rows = meta.obs["array_row"].astype(int).to_numpy()
    cols = meta.obs["array_col"].astype(int).to_numpy()

    is_jak2 = pert == KO_GENE
    is_peri = tissue == "periphery"
    is_unpert = (pert == "unperturbed") & (tissue == "tumor")
    is_normal = tissue == "normal"
    nbrs = hex_neighbors(rows, cols)
    layer = jak2_contiguous_patch(is_jak2, is_normal, nbrs)
    in_patch = layer >= 0
    mixed_mask = in_patch & (~is_jak2) & (is_peri | is_unpert)
    jak2_idx = np.where(is_jak2)[0]
    mixed_idx = np.where(mixed_mask)[0]
    loc_idx = np.where(in_patch)[0]

    print(
        f"FULL contiguous patch: total={in_patch.sum()} Jak2={is_jak2.sum()} "
        f"mixed={mixed_idx.size} (peri={(mixed_mask & is_peri).sum()}, "
        f"unpert={(mixed_mask & is_unpert).sum()})",
        flush=True,
    )

    cache_path = args.sig_cache or (outdir / "location_signatures.npz")
    cache_ok = False
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        if "mu_normal" in z.files:
            print(f"loading signature cache {cache_path}", flush=True)
            spot_ids = z["spot_ids"].astype(int)
            mu_ko_all = z["mu_ko"].astype(np.float32)
            mu_un_all = z["mu_un"].astype(np.float32)
            mu_normal_all = z["mu_normal"].astype(np.float32)
            y_obs = z["y_obs"].astype(np.float32)
            id_to_row = {int(s): i for i, s in enumerate(spot_ids)}
            cache_ok = True
        else:
            print(
                f"signature cache missing mu_normal -> requery CONCERT "
                f"(do not use empirical means): {cache_path}",
                flush=True,
            )
    if not cache_ok:
        print("loading CONCERT + querying location-matched mu_KO/mu_un/mu_normal ...", flush=True)
        ctx = load_concert_slide0(args.data, args.ckpt_dir, args.ckpt_dir / "hvg_gene_names.txt")
        model = ctx["model"]
        pert_map = ctx["pert_map"]
        tissue_dict = ctx["tissue_dict"]
        ko_code = int(pert_map[KO_GENE])
        unpert_code = int(pert_map.get("unperturbed", pert_map.get("__basal__", 0)))
        t_peri = int(tissue_dict["periphery"])
        t_tumor = int(tissue_dict["tumor"])
        t_normal = int(tissue_dict["normal"])
        assert len(ctx["pos_raw"]) == meta.n_obs
        G = ctx["Y_norm"].shape[1]
        mu_ko_all = np.zeros((len(loc_idx), G), dtype=np.float32)
        mu_un_all = np.zeros_like(mu_ko_all)
        mu_normal_all = np.zeros_like(mu_ko_all)
        y_obs = np.zeros_like(mu_ko_all)
        for j, i in enumerate(loc_idx):
            y_sp = ctx["Y_norm"][i : i + 1]
            tcode = t_peri if is_peri[i] else t_tumor
            mu_ko_all[j] = cf_query(
                model, ctx, xy[i : i + 1], y_sp, ko_code, tcode, True,
                args.n_samples, args.batch_size,
            )[0]
            mu_un_all[j] = cf_query(
                model, ctx, xy[i : i + 1], y_sp, unpert_code, t_tumor, False,
                args.n_samples, args.batch_size,
            )[0]
            mu_normal_all[j] = cf_query(
                model, ctx, xy[i : i + 1], y_sp, unpert_code, t_normal, False,
                args.n_samples, args.batch_size,
            )[0]
            y_obs[j] = y_sp[0]
            if (j + 1) % 5 == 0 or j == 0:
                print(f"  signatures {j+1}/{len(loc_idx)}", flush=True)
        spot_ids = loc_idx.astype(int)
        np.savez_compressed(
            cache_path,
            spot_ids=spot_ids, mu_ko=mu_ko_all, mu_un=mu_un_all,
            mu_normal=mu_normal_all, y_obs=y_obs,
            genes=np.asarray(ctx["gene_names"]).astype(str),
        )
        print(f"wrote signature cache {cache_path}", flush=True)
        id_to_row = {int(s): i for i, s in enumerate(spot_ids)}

    var = y_obs.var(0)
    hvg_local = np.argsort(var)[-min(N_HVG, y_obs.shape[1]) :]
    mu_ko_all = mu_ko_all[:, hvg_local]
    mu_un_all = mu_un_all[:, hvg_local]
    mu_normal_all = mu_normal_all[:, hvg_local]
    y_obs = y_obs[:, hvg_local]

    pure_ko_rows = [id_to_row[i] for i in jak2_idx if i in id_to_row]
    unpert_in_patch = [id_to_row[i] for i in np.where(is_unpert & in_patch)[0] if i in id_to_row]
    unpert_expr = y_obs[unpert_in_patch] if len(unpert_in_patch) >= 3 else mu_un_all
    ko_expr = y_obs[pure_ko_rows] if pure_ko_rows else mu_ko_all

    def sample_synth(n):
        Xtr, wtr, nnls_w = [], [], []
        for _ in range(n):
            loc = int(rng.integers(0, len(loc_idx)))
            i_ko = int(rng.integers(0, len(ko_expr)))
            i_un = int(rng.integers(0, len(unpert_expr)))
            w = float(rng.uniform(0.15, 0.85))
            y = w * ko_expr[i_ko] + (1.0 - w) * unpert_expr[i_un]
            y = y + rng.normal(0, 0.01, size=y.shape).astype(np.float32)
            mu_k, mu_u = mu_ko_all[loc], mu_un_all[loc]
            Xtr.append(make_features(y, mu_k, mu_u))
            wtr.append(w)
            nnls_w.append(float(unmix_nnls(y, mu_k, mu_u)[0]))
        return np.stack(Xtr), np.asarray(wtr, np.float32), np.asarray(nnls_w, np.float32)

    print("building synth train/test with location-matched CONCERT mu ...", flush=True)
    Xtr, wtr, _ = sample_synth(args.n_synth)
    Xte, wte, nnls_te = sample_synth(args.n_synth_test)
    print(f"train {Xtr.shape} test {Xte.shape}", flush=True)

    head = train_amortized(Xtr, wtr, epochs=args.epochs, seed=args.seed, device=args.device)
    torch.save({"state": head.state_dict(), "in_dim": Xtr.shape[1]}, outdir / "amortized_head.pt")
    am_te = predict_w(head, Xte, args.device)

    primary = {
        "n_synth_test": int(len(wte)),
        "naive_nnls_concert": {
            "comp_pearson": safe_corr(wte, nnls_te),
            "comp_MAE": float(np.mean(np.abs(wte - nnls_te))),
        },
        "concert_amortized": {
            "comp_pearson": safe_corr(wte, am_te),
            "comp_MAE": float(np.mean(np.abs(wte - am_te))),
        },
    }
    primary["beats_nnls_MAE"] = bool(
        primary["concert_amortized"]["comp_MAE"] < primary["naive_nnls_concert"]["comp_MAE"]
    )
    primary["beats_nnls_pearson"] = bool(
        primary["concert_amortized"]["comp_pearson"] > primary["naive_nnls_concert"]["comp_pearson"]
    )
    primary["win_rate_vs_nnls"] = float(
        np.mean(np.abs(wte - am_te) < np.abs(wte - nnls_te) - 1e-12)
    )
    print("PRIMARY:", json.dumps(primary, indent=2), flush=True)

    rows_real = []
    for i in mixed_idx:
        if i not in id_to_row:
            continue
        r = id_to_row[i]
        y = y_obs[r]
        mu_k, mu_u = mu_ko_all[r], mu_un_all[r]
        w_nnls = float(unmix_nnls(y, mu_k, mu_u)[0])
        w_am = float(predict_w(head, make_features(y, mu_k, mu_u)[None, :], args.device)[0])
        d = np.linalg.norm(xy[jak2_idx] - xy[i], axis=1)
        loc_j = jak2_idx[d <= 8.0] if (d <= 8.0).any() else jak2_idx
        ref_rows = [id_to_row[j] for j in loc_j if j in id_to_row]
        ref = y_obs[ref_rows].mean(0) if ref_rows else mu_k
        iso_am = (y - (1.0 - w_am) * mu_u) / max(w_am, 1e-6)
        iso_nn = (y - (1.0 - w_nnls) * mu_u) / max(w_nnls, 1e-6)
        rows_real.append(dict(
            spot=str(meta.obs_names[i]), idx=int(i), tissue=str(tissue[i]),
            layer=int(layer[i]), w_nnls=w_nnls, w_amortized=w_am,
            rho_iso_am_vs_localJak2=safe_spearman(iso_am, ref),
            rho_iso_nnls_vs_localJak2=safe_spearman(iso_nn, ref),
            rho_muKO_vs_localJak2=safe_spearman(mu_k, ref),
            rho_recon_am_vs_mix=safe_spearman(w_am * mu_k + (1 - w_am) * mu_u, y),
            rho_recon_nnls_vs_mix=safe_spearman(w_nnls * mu_k + (1 - w_nnls) * mu_u, y),
        ))
    df = pd.DataFrame(rows_real)
    df.to_csv(outdir / "real_patch_mixed_spots.csv", index=False)
    secondary = {
        "n_real_mixed": int(len(df)),
        "n_peri": int((df.tissue == "periphery").sum()) if len(df) else 0,
        "n_unpert": int((df.tissue != "periphery").sum()) if len(df) else 0,
        "mean_w_am": float(df.w_amortized.mean()) if len(df) else None,
        "mean_w_nnls": float(df.w_nnls.mean()) if len(df) else None,
        "mean_rho_iso_am_vs_localJak2": float(df.rho_iso_am_vs_localJak2.mean()) if len(df) else None,
        "mean_rho_iso_nnls_vs_localJak2": float(df.rho_iso_nnls_vs_localJak2.mean()) if len(df) else None,
        "frac_iso_am_beats_nnls": float(
            np.mean(df.rho_iso_am_vs_localJak2 > df.rho_iso_nnls_vs_localJak2)
        ) if len(df) else None,
    }
    summary = {
        "method": "identical_to_spac_2-1: location-matched CONCERT mu + amortized head vs NNLS",
        "cohort": "full_contiguous_non_normal_CC_from_Jak2",
        "n_patch_total": int(in_patch.sum()),
        "n_jak2": int(is_jak2.sum()),
        "primary_synth_composition": primary,
        "secondary_real_patch": secondary,
        "ckpt": str(args.ckpt_dir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].bar(
        ["NNLS\n(CONCERT $\\mu$)", "Amortized\n(CONCERT $\\mu$)"],
        [primary["naive_nnls_concert"]["comp_MAE"], primary["concert_amortized"]["comp_MAE"]],
        color=["#1565c0", "#c62828"],
    )
    axes[0].set_ylabel("Composition MAE $\\downarrow$")
    axes[0].set_title("Primary: synth mixes @ Jak2 patch locs")
    axes[1].bar(
        ["NNLS", "Amortized"],
        [primary["naive_nnls_concert"]["comp_pearson"], primary["concert_amortized"]["comp_pearson"]],
        color=["#1565c0", "#c62828"],
    )
    axes[1].set_ylabel("Composition Pearson $\\uparrow$")
    axes[1].set_title(f"win-rate vs NNLS: {100*primary['win_rate_vs_nnls']:.1f}%")
    fig.suptitle("Pmap s0 Jak2 -- same method as SPAC 2-1 deconv", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "primary_vs_nnls.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    xy_d = display_coords(xy)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(xy_d[:, 0], xy_d[:, 1], s=4, c="#dddddd", lw=0)
    ax.scatter(xy_d[is_jak2, 0], xy_d[is_jak2, 1], s=28, c="#d73027", lw=0, label="Jak2", zorder=3)
    for _, r in df.iterrows():
        i = int(r.idx)
        w = float(r.w_amortized)
        x, y = xy_d[i]
        ax.add_patch(plt.matplotlib.patches.Wedge(
            (x, y), 0.55, 90 - 360 * w, 90, facecolor="#d73027", edgecolor="0.3", lw=0.2, zorder=4))
        ax.add_patch(plt.matplotlib.patches.Wedge(
            (x, y), 0.55, 90 - 360, 90 - 360 * w, facecolor="#4daf4a", edgecolor="0.3", lw=0.2, zorder=4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Amortized $\\hat{{w}}$ -- full contiguous patch (n_mixed={len(df)})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "slide_patch_weights.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
