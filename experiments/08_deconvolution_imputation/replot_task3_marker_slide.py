#!/usr/bin/env python3
"""Task-3 marker slide: CP10K->log1p(gene) raw vs deconv+impute sub-spots.

Left: observed expression on spots (full-transcriptome library).
Right: non-peri keep observed; peri replaced by typed sub-spots colored by
imputed expression from CONCERT mu_KO / mu_un / mu_normal (never empirical means).

Scale note: CONCERT mu is HVG-only NB mean rates (not log, not full transcriptome).
Do NOT CP10K with lib=sum(mu_HVG) -- that treats HVGs as the whole library and
inflates markers. Instead: mu composition x parent HVG depth -> counts, then
log1p(CP10K) with the parent spot's full library (same units as observed).
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/perturbmap_s0_spacfmt.h5ad")
OUT_DEFAULT = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/"
    "spac_seq/pmap_s0_jak2_concert_amortized"
)
EXT_DEFAULT = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/ext_exp"
)
KO_GENE = "Jak2"
DEFAULT_GENE = "Fn1"


def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def logcpm_gene(X_counts, gene_idx):
    """CP10K->log1p for one gene (library from all genes)."""
    X = np.asarray(_dense(X_counts), dtype=np.float64)
    lib = X.sum(1)
    lib[lib <= 0] = 1.0
    return np.log1p(X[:, gene_idx] / lib * 1e4).astype(np.float32)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--outdir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--ext", type=Path, default=EXT_DEFAULT)
    ap.add_argument("--gene", default=DEFAULT_GENE)
    args = ap.parse_args()
    outdir = args.outdir
    gene = str(args.gene)

    scores = pd.read_csv(outdir / "peri_deconv_impute_scores.csv")
    subs = pd.read_csv(outdir / "jak2_niche_subspots.csv")
    peri_idx = scores["idx"].astype(int).to_numpy()
    peri_set = set(map(int, peri_idx))

    meta = ad.read_h5ad(args.data)
    genes = np.asarray(meta.var_names).astype(str)
    if gene not in set(genes):
        raise SystemExit(f"gene {gene!r} not in adata.var_names")
    gidx = int(np.where(genes == gene)[0][0])

    pert = meta.obs["perturbation"].astype(str).to_numpy()
    tissue = meta.obs["tissue"].astype(str).to_numpy()
    xy = np.asarray(meta.obsm["spatial"], dtype=np.float64)
    rows = meta.obs["array_row"].astype(int).to_numpy()
    cols = meta.obs["array_col"].astype(int).to_numpy()
    is_jak2 = pert == KO_GENE
    is_normal = tissue == "normal"
    is_unpert = (pert == "unperturbed") & (tissue == "tumor")
    hex_nbrs = hex_neighbors(rows, cols)
    in_patch = jak2_contiguous_patch(is_jak2, is_normal, hex_nbrs) >= 0

    xy_d = display_coords(xy)
    focus = np.where(in_patch)[0]
    focus_xy = xy_d[focus]
    span = max(float(focus_xy[:, 0].ptp()), float(focus_xy[:, 1].ptp()), 1.0)
    pad = 0.35 * span
    xmin, xmax = float(focus_xy[:, 0].min()) - pad, float(focus_xy[:, 0].max()) + pad
    ymin, ymaxv = float(focus_xy[:, 1].min()) - pad, float(focus_xy[:, 1].max()) + pad
    in_view = (
        (xy_d[:, 0] >= xmin) & (xy_d[:, 0] <= xmax) &
        (xy_d[:, 1] >= ymin) & (xy_d[:, 1] <= ymaxv)
    )

    X = np.asarray(_dense(meta.X), dtype=np.float64)
    lib_full = X.sum(1)
    lib_full[lib_full <= 0] = 1.0
    obs = np.log1p(X[:, gidx] / lib_full * 1e4).astype(np.float32)

    z = np.load(outdir / "location_signatures.npz", allow_pickle=True)
    if "mu_normal" not in z.files:
        raise SystemExit(
            "location_signatures.npz missing mu_normal -- run "
            "cache_task3_concert_state_signatures.py (no empirical fallback)."
        )
    spot_ids = z["spot_ids"].astype(int)
    concert_genes = z["genes"].astype(str)
    id_to_row = {int(s): i for i, s in enumerate(spot_ids)}
    if gene not in set(concert_genes):
        raise SystemExit(f"gene {gene!r} not in location_signatures.npz")
    cg = int(np.where(concert_genes == gene)[0][0])
    # HVG indices in full adata (for parent HVG depth)
    gene_to_full = {g: i for i, g in enumerate(genes)}
    hvg_idx = np.array([gene_to_full[g] for g in concert_genes], dtype=int)
    lib_hvg = X[:, hvg_idx].sum(1)
    lib_hvg[lib_hvg <= 0] = 1.0

    def _gene_logcpm_matched(M, parent_ids):
        """mu (HVG rates) -> log1p(CP10K) on the parent spot's full library.

        Composition among HVGs x observed HVG depth -> gene count, then
        divide by full-spot library (same recipe as observed panel).
        """
        M = np.clip(np.asarray(M, dtype=np.float64), 0, None)
        row_sum = M.sum(1)
        row_sum[row_sum <= 0] = 1.0
        frac = M[:, cg] / row_sum
        pred_count = frac * lib_hvg[parent_ids]
        return np.log1p(pred_count / lib_full[parent_ids] * 1e4)

    mu_ko_g = _gene_logcpm_matched(z["mu_ko"], spot_ids)
    mu_un_g = _gene_logcpm_matched(z["mu_un"], spot_ids)
    mu_nm_g = _gene_logcpm_matched(z["mu_normal"], spot_ids)

    parent_imp = {}
    for r in scores.itertuples():
        i = int(r.idx)
        if i not in id_to_row:
            raise SystemExit(f"peri idx={i} missing CONCERT signatures")
        rr = id_to_row[i]
        parent_imp[i] = {
            "KO_Jak2": float(mu_ko_g[rr]),
            "unpert_tumor": float(mu_un_g[rr]),
            "normal": float(mu_nm_g[rr]),
        }

    sub_vals = []
    for r in subs.itertuples():
        pid = int(r.parent)
        typ = str(r.type)
        if pid not in parent_imp:
            raise SystemExit(f"subspot parent={pid} missing CONCERT signatures")
        sub_vals.append(float(parent_imp[pid][typ]))
    subs = subs.copy()
    subs["imp_log1p"] = sub_vals

    # shared color scale from observed in-view + imputed subspots
    view_vals = obs[in_view]
    all_for_scale = np.concatenate([view_vals, np.asarray(sub_vals, dtype=float)])
    vmin = 0.0
    vmax = float(np.percentile(all_for_scale, 99)) if all_for_scale.size else 1.0
    vmax = max(vmax, 1e-3)

    s_spot = 110
    s_sub_ring, s_sub_ctr = 48, 85

    TITLE_FS, CBAR_FS, CBAR_TICK_FS = 18, 16, 13
    # Two rows, shared colorbar (legend) on the right
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 11.6), sharex=True, sharey=True)

    # ---- Row 1: raw observed ----
    ax = axes[0]
    m = in_view
    sc0 = ax.scatter(
        xy_d[m, 0], xy_d[m, 1], c=obs[m], s=s_spot,
        cmap="magma", vmin=vmin, vmax=vmax,
        linewidths=0, edgecolors="face", zorder=2,
    )
    ax.set_title(f"Raw  $\\cdot$  {gene}", fontsize=TITLE_FS, pad=10)

    # ---- Row 2: deconv subspots with imputed gene (fill only, no type borders) ----
    ax = axes[1]
    keep = in_view & np.array([i not in peri_set for i in range(len(xy_d))])
    ax.scatter(
        xy_d[keep, 0], xy_d[keep, 1], c=obs[keep], s=s_spot,
        cmap="magma", vmin=vmin, vmax=vmax,
        linewidths=0, edgecolors="face", zorder=2,
    )
    if len(subs):
        for pid, g in subs.groupby("parent", sort=False):
            g = g.reset_index(drop=True)
            hub, ring = g.iloc[0], g.iloc[1:]
            if len(ring):
                ax.scatter(
                    ring["x"].to_numpy(), ring["y"].to_numpy(),
                    c=ring["imp_log1p"].to_numpy(),
                    s=s_sub_ring, cmap="magma", vmin=vmin, vmax=vmax,
                    linewidths=0, edgecolors="face", zorder=4,
                )
            ax.scatter(
                [float(hub["x"])], [float(hub["y"])],
                c=[float(hub["imp_log1p"])],
                s=s_sub_ctr, cmap="magma", vmin=vmin, vmax=vmax,
                linewidths=0, edgecolors="face", zorder=5,
            )
    ax.set_title(f"Deconv + impute  $\\cdot$  {gene}", fontsize=TITLE_FS, pad=10)

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymaxv)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.subplots_adjust(left=0.04, right=0.82, top=0.95, bottom=0.04, hspace=0.18)
    cax = fig.add_axes([0.86, 0.18, 0.035, 0.60])
    cb = fig.colorbar(sc0, cax=cax)
    cb.set_label("Log gene expression", fontsize=CBAR_FS)
    cb.ax.tick_params(labelsize=CBAR_TICK_FS)
    cb.ax.yaxis.set_label_position("right")
    cb.ax.yaxis.set_ticks_position("right")

    out_path = outdir / f"slide_marker_{gene}_raw_vs_deconv.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    dst = args.ext / f"task3_pmap_s0_jak2_slide_marker_{gene}.png"
    args.ext.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out_path.read_bytes())
    print(
        f"gene={gene} CONCERT-only imputed; vmax={vmax:.3f} n_sub={len(subs)}",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    print(f"copied -> {dst}", flush=True)


if __name__ == "__main__":
    main()
