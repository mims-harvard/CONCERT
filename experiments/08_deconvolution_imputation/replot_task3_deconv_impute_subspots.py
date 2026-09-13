#!/usr/bin/env python3
"""Task-3 paper figs: Jak2 niche deconv -> typed sub-spots (pmap peri style).

Pipeline (same idea as run_pmap_s0_peri_deconv_viz.py):
  1) Neighbor-matched deconv: KO-facing->Jak2+tumor; normal/tumor-facing->tumor+normal.
  2) Place typed sub-spots along 6 spatial-kNN directions using w_hat.
  3) Imputed Jak2 sub-spots get CONCERT mu_KO(parent); compare to true local Jak2.
  4) Slide: raw vs deconv+imputed sub-spots (NO pies). Bars: raw peri vs deconv+impute.

Reuses location_signatures.npz + amortized_head.pt (no CONCERT reload).

location_signatures.npz MUST include mu_ko, mu_un, mu_normal (CONCERT CFs).
Never use empirical panel means as imputed / predicted signatures.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

DATA = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/perturbmap_s0_spacfmt.h5ad")
OUT_DEFAULT = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/"
    "spac_seq/pmap_s0_jak2_concert_amortized"
)
EXT_DEFAULT = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/ext_exp"
)
KO_GENE = "Jak2"
N_HVG = 1500
N_CELLS = 10
SUB_ALPHA = 0.45
K_NEIGHBORS = 6

TYPE_COLORS = {
    "KO_Jak2": "#e31a1c",
    "peri": "#6a3d9a",
    "unpert_tumor": "#33a02c",
    "normal": "#9e9e9e",
    "other": "#8a8a8a",
}


def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def logcpm(X):
    lib = np.asarray(X).sum(1, keepdims=True)
    lib[lib <= 0] = 1.0
    return np.log1p(np.asarray(X) / lib * 1e4).astype(np.float32)


def safe_spearman(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(spearmanr(a, b).correlation)


def display_coords(xy):
    x, y = xy[:, 0].astype(float), xy[:, 1].astype(float)
    return np.column_stack([y, -x])


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


def build_spatial_knn(xy, k=K_NEIGHBORS):
    tree = cKDTree(xy)
    kq = min(k + 1, len(xy))
    _, idx = tree.query(xy, k=kq)
    if idx.ndim == 1:
        return [[] for _ in range(len(xy))]
    return [idx[i, 1:].tolist() for i in range(len(xy))]


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


@torch.no_grad()
def predict_w(model, feats, device="cpu"):
    return model(torch.tensor(feats, device=device)).cpu().numpy()


def spot_label(pert_i, tissue_i, is_jak2_i):
    if is_jak2_i:
        return "KO_Jak2"
    if tissue_i == "periphery":
        return "peri"
    if pert_i == "unperturbed" and tissue_i == "tumor":
        return "unpert_tumor"
    if tissue_i == "normal":
        return "normal"
    return "other"


def ko_facing_spot(i, nbrs_i, labels, xy, jak2_idx, unpert_idx, normal_idx):
    """True if peri is Jak2-adjacent / closer to Jak2 than to unpert or normal."""
    if any(labels[j] == "KO_Jak2" for j in nbrs_i):
        return True
    if len(jak2_idx) == 0:
        return False
    d_ko = float(np.linalg.norm(xy[jak2_idx] - xy[i], axis=1).min())
    d_alt = []
    if len(unpert_idx):
        d_alt.append(float(np.linalg.norm(xy[unpert_idx] - xy[i], axis=1).min()))
    if len(normal_idx):
        d_alt.append(float(np.linalg.norm(xy[normal_idx] - xy[i], axis=1).min()))
    if not d_alt:
        return True
    return d_ko <= min(d_alt)


def unmix_nnls_multi(y, sigs):
    A = np.stack([np.asarray(s, float) for s in sigs], axis=1)
    try:
        w, _ = nnls(A, y.astype(np.float64))
    except Exception:
        w, *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
        w = np.clip(w, 0, None)
    s = float(w.sum())
    return (w / s) if s > 0 else np.ones(len(sigs)) / len(sigs)


def assign_subspots_from_comp(parent_xy, pitch, comp, jak2_xy, normal_xy, n_dirs=6):
    """Center + ring typed by composition; KO toward Jak2, normal toward normal."""
    parent_xy = np.asarray(parent_xy, float)
    r = 0.36 * float(max(pitch, 1e-3))
    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False) + (np.pi / 6.0)
    keys = [k for k, v in comp.items() if float(v) > 1e-8]
    if not keys:
        keys, weights = ["unpert_tumor"], np.array([1.0])
    else:
        weights = np.array([float(comp[k]) for k in keys], float)
        weights = weights / weights.sum()
    n_tot = n_dirs + 1
    counts = np.floor(weights * n_tot + 1e-8).astype(int)
    # largest remainders
    rem = n_tot - int(counts.sum())
    order = np.argsort(-(weights * n_tot - counts))
    for j in order[:rem]:
        counts[j] += 1
    # center = majority
    maj_i = int(np.argmax(counts))
    maj = keys[maj_i]
    out = [(parent_xy.copy(), maj)]
    counts = counts.copy()
    counts[maj_i] -= 1
    positions = [
        parent_xy + r * np.array([np.cos(a), np.sin(a)], dtype=float) for a in angles
    ]
    remaining = []
    for k, c in zip(keys, counts):
        remaining.extend([k] * int(c))
    used = set()
    for typ in remaining:
        if typ == "KO_Jak2" and len(jak2_xy):
            ref = np.asarray(jak2_xy, float)
        elif typ == "normal" and len(normal_xy):
            ref = np.asarray(normal_xy, float)
        else:
            ref = None
        best_j, best_d = None, 1e18
        for j, pos in enumerate(positions):
            if j in used:
                continue
            d = float(np.linalg.norm(ref - pos, axis=1).min()) if ref is not None else float(j)
            if d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        out.append((positions[best_j], typ))
    return out


def draw_pie_comp(ax, x, y, comp, radius):
    """Multi-slice pie from composition dict {type: weight}."""
    items = [(k, float(v)) for k, v in comp.items() if float(v) > 1e-8]
    if not items:
        items = [("unpert_tumor", 1.0)]
    total = sum(v for _, v in items)
    items = [(k, v / total) for k, v in items]
    ang = 90.0
    for typ, w in items:
        sweep = 360.0 * w
        ax.add_patch(mpatches.Wedge(
            (x, y), radius, ang - sweep, ang,
            facecolor=TYPE_COLORS.get(typ, "#888888"),
            edgecolor="0.25", lw=0.35, zorder=5, transform=ax.transData,
        ))
        ang -= sweep


def local_type_ref(Y, type_idx, xy, x, radius=15.0):
    """Mean log-CPM of nearby pure spots of one tissue type."""
    if len(type_idx) == 0:
        return np.zeros(Y.shape[1], dtype=np.float32)
    d = np.linalg.norm(xy[type_idx] - x, axis=1)
    loc = type_idx[d <= radius] if (d <= radius).any() else type_idx
    return Y[loc].mean(0)


def local_jak2_ref(Y, jak2_idx, xy, x, radius=15.0):
    return local_type_ref(Y, jak2_idx, xy, x, radius=radius)


def face_type_non_ko(has_unpert_nbr, has_normal_nbr, xy_i, unpert_idx, normal_idx, xy):
    """Classify non-KO-facing peri by neighbor tissue (fallback: nearest pure)."""
    if has_normal_nbr and not has_unpert_nbr:
        return "normal"
    if has_unpert_nbr and not has_normal_nbr:
        return "unpert_tumor"
    d_u = float(np.linalg.norm(xy[unpert_idx] - xy_i, axis=1).min()) if len(unpert_idx) else 1e18
    d_n = float(np.linalg.norm(xy[normal_idx] - xy_i, axis=1).min()) if len(normal_idx) else 1e18
    if has_normal_nbr and has_unpert_nbr:
        return "normal" if d_n <= d_u else "unpert_tumor"
    return "normal" if d_n <= d_u else "unpert_tumor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--outdir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--ext", type=Path, default=EXT_DEFAULT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pad", type=float, default=3.0)
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    args.ext.mkdir(parents=True, exist_ok=True)

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
    hex_nbrs = hex_neighbors(rows, cols)
    layer = jak2_contiguous_patch(is_jak2, is_normal, hex_nbrs)
    in_patch = layer >= 0
    # Selection = all non-Jak2 in contiguous hex patch, plus every tissue-peri
    # that falls inside the plot frame (same crop as the slide: in_patch + pad).
    # Outside-frame peri are not "unselected niche peri" -- they are off-crop.
    xy_d0 = display_coords(xy)
    nn0 = cKDTree(xy_d0).query(xy_d0, k=2)[0][:, 1]
    pitch0 = float(np.median(nn0[nn0 > 0])) if np.any(nn0 > 0) else 1.0
    focus0 = np.where(in_patch)[0]
    fxy0 = xy_d0[focus0]
    span0 = max(float(fxy0[:, 0].ptp()), float(fxy0[:, 1].ptp()), 1.0)
    pad0 = 0.35 * span0  # must match slide crop pad
    in_view = (
        (xy_d0[:, 0] >= fxy0[:, 0].min() - pad0) & (xy_d0[:, 0] <= fxy0[:, 0].max() + pad0) &
        (xy_d0[:, 1] >= fxy0[:, 1].min() - pad0) & (xy_d0[:, 1] <= fxy0[:, 1].max() + pad0)
    )
    core = in_patch & (~is_jak2) & (~is_normal)
    peri_in_view = is_peri & (~is_jak2) & in_view
    peri_idx = np.where(core | peri_in_view)[0]
    jak2_idx = np.where(is_jak2)[0]
    labels = np.array([spot_label(pert[i], tissue[i], is_jak2[i]) for i in range(len(pert))], dtype=object)
    knn = build_spatial_knn(xy, k=K_NEIGHBORS)
    n_core = int(core.sum())
    n_extra = int((peri_in_view & ~core).sum())
    print(
        f"Jak2 niche: patch={in_patch.sum()} Jak2={jak2_idx.size} "
        f"deconv_targets={peri_idx.size} (patch_mixed={n_core}, peri_in_view_extra={n_extra})",
        flush=True,
    )

    z = np.load(outdir / "location_signatures.npz", allow_pickle=True)
    if "mu_normal" not in z.files:
        raise RuntimeError(
            f"{outdir / 'location_signatures.npz'} missing mu_normal. "
            "Run cache_task3_concert_state_signatures.py -- do NOT fall back to "
            "empirical panel means as CONCERT predictions."
        )
    spot_ids = z["spot_ids"].astype(int)
    mu_ko_raw = z["mu_ko"].astype(np.float32)
    mu_un_raw = z["mu_un"].astype(np.float32)
    mu_nm_raw = z["mu_normal"].astype(np.float32)
    concert_genes = z["genes"].astype(str)
    id_to_row = {int(s): i for i, s in enumerate(spot_ids)}
    mu_ko_all = np.log1p(np.clip(mu_ko_raw, 0, None)).astype(np.float32)
    mu_un_all = np.log1p(np.clip(mu_un_raw, 0, None)).astype(np.float32)
    mu_nm_all = np.log1p(np.clip(mu_nm_raw, 0, None)).astype(np.float32)

    g2i = {g: i for i, g in enumerate(np.asarray(meta.var_names).astype(str))}
    gene_cols, cidx = [], []
    for j, g in enumerate(concert_genes):
        if g in g2i:
            gene_cols.append(g2i[g]); cidx.append(j)
    gene_cols = np.asarray(gene_cols, int)
    cidx = np.asarray(cidx, int)
    Y_log = logcpm(_dense(meta.X).astype(np.float32)[:, gene_cols])
    mu_ko_all = mu_ko_all[:, cidx]
    mu_un_all = mu_un_all[:, cidx]
    mu_nm_all = mu_nm_all[:, cidx]
    hvg = np.argsort(Y_log.var(0))[-min(N_HVG, Y_log.shape[1]) :]
    Y_log = Y_log[:, hvg]
    mu_ko_all = mu_ko_all[:, hvg]
    mu_un_all = mu_un_all[:, hvg]
    mu_nm_all = mu_nm_all[:, hvg]

    missing = [int(i) for i in peri_idx if int(i) not in id_to_row]
    if missing:
        raise RuntimeError(
            f"{len(missing)} Task3 peri lack CONCERT signatures (e.g. idx={missing[:5]}). "
            "Run cache_task3_concert_state_signatures.py --force. "
            "Do not substitute empirical means."
        )

    pure_un = np.where(is_unpert & ~in_patch)[0]
    if pure_un.size < 10:
        pure_un = np.where(is_unpert)[0]
    normal_idx = np.where(is_normal)[0]
    unpert_idx = np.where(is_unpert)[0]
    # Empirical means only for the explicit NNLS baseline column (w_empirical), never as mu_imp.
    emp_ko = Y_log[jak2_idx].mean(0)
    emp_un = Y_log[pure_un].mean(0) if pure_un.size else Y_log[unpert_idx].mean(0)

    ckpt = torch.load(outdir / "amortized_head.pt", map_location="cpu")
    head = DeconvHead(int(ckpt["in_dim"]))
    head.load_state_dict(ckpt["state"])
    head.eval()

    # local spot pitch for regular sub-spot rings (in display coords)
    xy_d = display_coords(xy)
    nn = cKDTree(xy_d).query(xy_d, k=2)[0][:, 1]
    pitch = float(np.median(nn[nn > 0])) if np.any(nn > 0) else 1.0
    jak2_disp = xy_d[jak2_idx]
    normal_disp = xy_d[normal_idx] if normal_idx.size else np.zeros((0, 2))

    sub_rows, score_rows = [], []
    n_ko_face, n_norm_face = 0, 0
    for i in peri_idx:
        y = Y_log[i]
        r = id_to_row[int(i)]
        mu_k, mu_u, mu_n = mu_ko_all[r], mu_un_all[r], mu_nm_all[r]
        sig_src = "concert"
        nbr_labs = [labels[j] for j in hex_nbrs[i]]
        has_ko_nbr = any(l == "KO_Jak2" for l in nbr_labs)
        # IMPORTANT: do NOT treat neighbor label "peri" as unpert_tumor.
        # That was the regression: normal+peri neighborhoods got green tumor mass.
        has_unpert_nbr = any(l == "unpert_tumor" for l in nbr_labs)
        has_normal_nbr = any(l == "normal" for l in nbr_labs)

        if has_ko_nbr:
            # KO-facing: amortized head on CONCERT mu_KO vs mu_unpert.
            # Residual (1-w_hat) -> neighbor non-KO type(s), also CONCERT mu.
            n_ko_face += 1
            facing = True
            w_am = float(predict_w(head, make_features(y, mu_k, mu_u)[None, :], args.device)[0])
            w_am = float(np.clip(w_am, 0.0, 1.0))
            residual = 1.0 - w_am

            non_ko = []
            if has_unpert_nbr:
                non_ko.append("unpert_tumor")
            if has_normal_nbr:
                non_ko.append("normal")
            if not non_ko and any(l == "peri" for l in nbr_labs):
                non_ko.append("unpert_tumor")

            if not non_ko:
                comp = {"KO_Jak2": 1.0}
                w_am = 1.0
            elif len(non_ko) == 1:
                comp = {"KO_Jak2": w_am, non_ko[0]: residual}
            else:
                ww = unmix_nnls_multi(y, [mu_u, mu_n])
                comp = {
                    "KO_Jak2": w_am,
                    "unpert_tumor": residual * float(ww[0]),
                    "normal": residual * float(ww[1]),
                }

            w_emp = float(unmix_nnls(y, emp_ko, emp_un)[0])  # baseline only
            face = "jak2"
            ref = local_jak2_ref(Y_log, jak2_idx, xy, xy[i])
            rho_raw = safe_spearman(y, ref)
            rho_imp = safe_spearman(mu_k, ref)
        else:
            # Not KO-adjacent: unmix with CONCERT mu_un / mu_normal only.
            n_norm_face += 1
            facing = False
            panel_names, panel_sigs = [], []
            if has_unpert_nbr:
                panel_names.append("unpert_tumor")
                panel_sigs.append(mu_u)
            if has_normal_nbr:
                panel_names.append("normal")
                panel_sigs.append(mu_n)
            if not panel_names:
                d_u = float(np.linalg.norm(xy[unpert_idx] - xy[i], axis=1).min()) if len(unpert_idx) else 1e18
                d_n = float(np.linalg.norm(xy[normal_idx] - xy[i], axis=1).min()) if len(normal_idx) else 1e18
                if d_n <= d_u:
                    panel_names, panel_sigs = ["normal"], [mu_n]
                else:
                    panel_names, panel_sigs = ["unpert_tumor"], [mu_u]
            if len(panel_names) == 1:
                comp = {panel_names[0]: 1.0}
            else:
                ww = unmix_nnls_multi(y, panel_sigs)
                comp = {n: float(ww[k]) for k, n in enumerate(panel_names)}
            w_am = 0.0
            w_emp = 0.0
            face = face_type_non_ko(
                has_unpert_nbr, has_normal_nbr, xy[i], unpert_idx, normal_idx, xy
            )
            if face == "normal":
                ref = local_type_ref(Y_log, normal_idx, xy, xy[i])
                mu_imp = mu_n
            else:
                ref = local_type_ref(Y_log, pure_un if pure_un.size else unpert_idx, xy, xy[i])
                mu_imp = mu_u
            rho_raw = safe_spearman(y, ref)
            rho_imp = safe_spearman(mu_imp, ref)

        score_rows.append(dict(
            spot=str(meta.obs_names[i]), idx=int(i),
            ko_facing=bool(facing),
            face_type=face,
            w_empirical=w_emp, w_amortized=w_am,
            w_unpert=float(comp.get("unpert_tumor", 0.0)),
            w_normal=float(comp.get("normal", 0.0)),
            w_jak2=float(comp.get("KO_Jak2", 0.0)),
            rho_raw=rho_raw, rho_imputed_jak2=rho_imp,
            signature_source=sig_src,
            x=float(xy[i, 0]), y=float(xy[i, 1]),
            xd=float(xy_d[i, 0]), yd=float(xy_d[i, 1]),
        ))
        for x_sub, st in assign_subspots_from_comp(
            xy_d[i], pitch, comp, jak2_disp, normal_disp, n_dirs=6
        ):
            sub_rows.append(dict(
                parent=int(i), x=float(x_sub[0]), y=float(x_sub[1]), type=st,
                ko_facing=bool(facing), w_jak2=float(comp.get("KO_Jak2", 0.0)),
            ))
    print(
        f"facing: ko={n_ko_face} normal/tumor={n_norm_face}; "
        f"signatures=CONCERT-only (mu_ko/mu_un/mu_normal)",
        flush=True,
    )

    scores = pd.DataFrame(score_rows)
    subs = pd.DataFrame(sub_rows)
    scores.to_csv(outdir / "peri_deconv_impute_scores.csv", index=False)
    subs.to_csv(outdir / "jak2_niche_subspots.csv", index=False)
    print(
        f"peri scored={len(scores)} subspots={len(subs)} "
        f"KO-typed={(subs.type == 'KO_Jak2').sum() if len(subs) else 0} "
        f"normal-typed={(subs.type == 'normal').sum() if len(subs) else 0}",
        flush=True,
    )

    # ---- metric bars: raw peri vs deconv+impute for Jak2 / unpert / normal ----
    panels = [
        dict(
            key="jak2",
            title="Jak2-KO",
            ylabel="$\\rho$ vs true local Jak2 $\\uparrow$",
            mask=(scores.face_type == "jak2") & (scores.signature_source == "concert"),
            c_raw="#984ea3",
            c_imp=TYPE_COLORS["KO_Jak2"],
            label_imp="Deconv + impute",
            note="CONCERT $\\mu$",
        ),
        dict(
            key="unpert_tumor",
            title="Unpert tumor",
            ylabel="$\\rho$ vs true local unpert $\\uparrow$",
            mask=scores.face_type == "unpert_tumor",
            c_raw="#984ea3",
            c_imp=TYPE_COLORS["unpert_tumor"],
            label_imp="Deconv + impute",
            note="CONCERT $\\mu$",
        ),
        dict(
            key="normal",
            title="Normal",
            ylabel="$\\rho$ vs true local normal $\\uparrow$",
            mask=scores.face_type == "normal",
            c_raw="#984ea3",
            c_imp=TYPE_COLORS["normal"],
            label_imp="Deconv + impute",
            note="CONCERT $\\mu$",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharey=False)
    panel_means = {}
    for ax, p in zip(axes, panels):
        s_metric = scores.loc[p["mask"]].copy()
        m_raw = float(s_metric.rho_raw.mean()) if len(s_metric) else float("nan")
        m_imp = float(s_metric.rho_imputed_jak2.mean()) if len(s_metric) else float("nan")
        panel_means[p["key"]] = dict(n=int(len(s_metric)), m_raw=m_raw, m_imp=m_imp)
        vals = [m_raw, m_imp]
        ax.bar([0, 1], vals, color=[p["c_raw"], p["c_imp"]], width=0.42)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Raw peri", p["label_imp"]], fontsize=11)
        ax.set_ylabel(p["ylabel"], fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim(-0.55, 1.55)
        finite = [v for v in vals if np.isfinite(v)]
        ymax = (max(finite) * 1.35) if finite else 1.0
        ax.set_ylim(0, ymax)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v + 0.03 * ymax, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{p['title']} (n={len(s_metric)} {p['note']})", fontsize=12)
    fig.tight_layout()
    bar_path = outdir / "primary_raw_vs_deconv_impute.png"
    fig.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    m_raw = panel_means["jak2"]["m_raw"]
    m_imp = panel_means["jak2"]["m_imp"]

    # ---- 3-panel slide: raw (peri highlighted) | decon pies | regular sub-spots ----
    # Keep plot scope on the contiguous hex niche (same framing as before)
    focus = np.where(in_patch)[0]
    focus_xy = xy_d[focus]
    span = max(float(focus_xy[:, 0].ptp()), float(focus_xy[:, 1].ptp()), 1.0)
    pad = 0.55 * span  # wider context around niche (avoid edge-clipped spots)
    xmin, xmax = float(focus_xy[:, 0].min()) - pad, float(focus_xy[:, 0].max()) + pad
    ymin, ymaxv = float(focus_xy[:, 1].min()) - pad, float(focus_xy[:, 1].max()) + pad
    pie_r = 0.32 * float(pitch)  # slightly larger than spot markers
    comp_map = {}
    for r in scores.itertuples():
        comp_map[int(r.idx)] = {
            "KO_Jak2": float(r.w_jak2),
            "unpert_tumor": float(r.w_unpert),
            "normal": float(r.w_normal),
        }
    peri_mask = np.zeros(len(xy_d), dtype=bool)
    peri_mask[peri_idx] = True
    # Plotting labels: tissue-peri outside the deconv set -> "other" (not purple)
    labels_plot = labels.copy()
    labels_plot[is_peri & (~peri_mask) & (~is_jak2)] = "other"

    # Figure height matches data aspect; legend sits just under axes (not a tall empty band)
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymaxv - ymin, 1e-6)
    fig_w = 15.0
    panel_h = (fig_w / 3.08) * (span_y / span_x)
    fig_h = panel_h + 0.75
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    # Full-color typed spots (no washed-out underlay for real cells)
    s_other, s_normal, s_unpert, s_jak2, s_peri_ann, s_peri = 70, 110, 110, 110, 110, 155
    s_sub_ring, s_sub_ctr = 48, 85

    def draw_all_typed(ax, hide_eval_peri=False):
        """Draw typed spots; only deconv-set peri stay purple."""
        order = [
            ("other", s_other, TYPE_COLORS["other"]),
            ("normal", s_normal, TYPE_COLORS["normal"]),
            ("peri", s_peri_ann, TYPE_COLORS["peri"]),
            ("unpert_tumor", s_unpert, TYPE_COLORS["unpert_tumor"]),
            ("KO_Jak2", s_jak2, TYPE_COLORS["KO_Jak2"]),
        ]
        for lab, size, col in order:
            m = labels_plot == lab
            if hide_eval_peri:
                m = m & ~peri_mask
            if m.any():
                ax.scatter(xy_d[m, 0], xy_d[m, 1], s=size, c=col, linewidths=0, edgecolors="none", zorder=2, alpha=1.0)

    # Panel 1
    ax = axes[0]
    draw_all_typed(ax, hide_eval_peri=False)
    h1 = [
        mpatches.Patch(color=TYPE_COLORS["normal"], label="normal"),
        mpatches.Patch(color=TYPE_COLORS["unpert_tumor"], label="unpert tumor"),
        mpatches.Patch(color=TYPE_COLORS["KO_Jak2"], label="KO Jak2"),
    ]
    ax.scatter(
        xy_d[peri_idx, 0], xy_d[peri_idx, 1], s=s_peri, c=TYPE_COLORS["peri"],
        edgecolors="#ff9800", linewidths=2.4, zorder=4,
    )
    h1.append(mpatches.Patch(
        facecolor=TYPE_COLORS["peri"], edgecolor="#ff9800", linewidth=2,
        label=f"niche peri (n={peri_idx.size})",
    ))
    ax.set_title("Raw -- all niche peri", fontsize=15, pad=10)

    # Panel 2
    ax = axes[1]
    draw_all_typed(ax, hide_eval_peri=True)
    for i in peri_idx:
        if i in comp_map:
            draw_pie_comp(ax, float(xy_d[i, 0]), float(xy_d[i, 1]), comp_map[i], pie_r)
    ax.set_title("Deconv pies", fontsize=15, pad=10)

    # Panel 3
    ax = axes[2]
    draw_all_typed(ax, hide_eval_peri=True)
    if len(subs):
        for pid, g in subs.groupby("parent", sort=False):
            g = g.reset_index(drop=True)
            hub, ring = g.iloc[0], g.iloc[1:]
            if len(ring):
                ax.scatter(
                    ring["x"].to_numpy(), ring["y"].to_numpy(),
                    s=s_sub_ring,
                    c=[TYPE_COLORS[tt] for tt in ring["type"]],
                    edgecolors="0.2", linewidths=0.45, zorder=4,
                )
            # center hub -- fills the peri spot middle
            ax.scatter(
                [float(hub["x"])], [float(hub["y"])],
                s=s_sub_ctr, c=TYPE_COLORS[str(hub["type"])],
                edgecolors="0.15", linewidths=0.5, zorder=5,
            )
    ax.set_title("Deconv + impute sub-spots", fontsize=15, pad=10)

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymaxv)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)

    # Legend just under the middle panel (axes coords -> no orphan figure gap)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.03)
    axes[1].legend(
        handles=h1, loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4, fontsize=14, frameon=False,
        handlelength=1.4, columnspacing=1.4, labelspacing=0.2,
        borderaxespad=0.0,
    )

    slide_path = outdir / "slide_raw_vs_deconv_impute_subspots.png"
    fig.savefig(slide_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    summary = {
        "task": "deconv ALL non-Jak2 spots in contiguous Jak2 niche + impute sub-spots; compare imputed type to true local pure",
        "n_peri": int(len(scores)),
        "n_ko_facing": int(scores.ko_facing.sum()) if "ko_facing" in scores.columns else None,
        "n_normal_tumor_facing": int((~scores.ko_facing).sum()) if "ko_facing" in scores.columns else None,
        "n_face_jak2": int((scores.face_type == "jak2").sum()) if "face_type" in scores.columns else None,
        "n_face_unpert": int((scores.face_type == "unpert_tumor").sum()) if "face_type" in scores.columns else None,
        "n_face_normal": int((scores.face_type == "normal").sum()) if "face_type" in scores.columns else None,
        "n_peri_concert_mu": int((scores.signature_source == "concert").sum()) if "signature_source" in scores.columns else int(len(scores)),
        "n_peri_empirical_fallback": 0,
        "n_subspots": int(len(subs)),
        "n_imputed_jak2_subspots": int((subs.type == "KO_Jak2").sum()) if len(subs) else 0,
        "mean_rho_raw_vs_localJak2": m_raw,
        "mean_rho_imputedJak2_vs_localJak2": m_imp,
        "panel_means": panel_means,
        "metric_subset": "all face-matched peri; imputed = CONCERT mu_KO/mu_un/mu_normal only",
        "mean_w_jak2_ko_facing": float(scores.loc[scores.ko_facing, "w_jak2"].mean()) if scores.ko_facing.any() else None,
        "spot_pitch": pitch,
        "note": (
            "Sub-spots = center + hex ring at 0.36xpitch; "
            "all imputed signatures are location-matched CONCERT CFs (no empirical panel means)."
        ),
    }
    (outdir / "summary_deconv_impute.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    # single copies into ext_exp (no duplicates)
    bar_dst = args.ext / "task3_pmap_s0_jak2_raw_vs_deconv_impute.png"
    slide_dst = args.ext / "task3_pmap_s0_jak2_slide_raw_pies_subspots.png"
    bar_dst.write_bytes(bar_path.read_bytes())
    slide_dst.write_bytes(slide_path.read_bytes())
    (args.ext / "task3_pmap_s0_jak2_summary.json").write_text(json.dumps(summary, indent=2))
    scores.to_csv(args.ext / "task3_pmap_s0_jak2_peri_scores.csv", index=False)

    # remove old duplicated / superseded filenames
    for obsolete in [
        "task3_pmap_s0_jak2_deconv_vs_empirical_nnls.png",
        "task3_pmap_s0_jak2_iso_and_impute_vs_true_jak2.png",
        "task3_pmap_s0_jak2_slide_amortized_weights.png",
        "task3_pmap_s0_jak2_slide_cohort_deconv_impute.png",
        "task3_pmap_s0_jak2_real_mixed_spots.csv",
    ]:
        p = args.ext / obsolete
        if p.exists():
            p.unlink()
            print(f"removed duplicate/obsolete {obsolete}", flush=True)
    print(f"copied -> ext_exp/{bar_dst.name}", flush=True)
    print(f"copied -> ext_exp/{slide_dst.name}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
