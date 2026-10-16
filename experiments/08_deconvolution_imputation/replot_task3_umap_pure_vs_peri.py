#!/usr/bin/env python3
"""Task-3 UMAP: raw spots vs mixed peri vs decon+impute signatures.

Claim: imputed component profiles land with raw spots of the same state;
raw mixed peri do not form clean same-state clusters.

Imputed signatures: location-matched CONCERT mu_KO / mu_un / mu_normal only.
Never use empirical panel means as imputed points.

Scale: raw counts and CONCERT mu (non-negative rate-like) both use the same
CP10K -> log1p recipe before HVG / StandardScaler / PCA / UMAP.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.sparse as sp
import umap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATA = Path(f"{_rp.DATA}perturbmap/raw/perturbmap_s0_spacfmt.h5ad")
OUT_DEFAULT = Path(
    f"{_rp.WORKDIR}/paper/Outputs/"
    "spac_seq/pmap_s0_jak2_concert_amortized"
)
EXT_DEFAULT = Path(
    f"{_rp.WORKDIR}/paper/Outputs/ext_exp"
)
KO_GENE = "Jak2"
N_HVG = 1500
W_MIN = 0.15  # min composition weight to emit an imputed point
MAX_PURE = {"KO_Jak2": 80, "unpert_tumor": 120, "normal": 120}

COLORS = {
    "raw_KO_Jak2": "#e31a1c",
    "raw_unpert_tumor": "#33a02c",
    "raw_normal": "#9e9e9e",
    "peri_mixed": "#6a3d9a",
    "imp_KO_Jak2": "#fb9a99",
    "imp_unpert_tumor": "#b2df8a",
    "imp_normal": "#bdbdbd",
}


def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def logcpm(X):
    X = np.asarray(_dense(X), dtype=np.float32)
    lib = X.sum(1, keepdims=True)
    lib[lib <= 0] = 1.0
    return np.log1p(X / lib * 1e4).astype(np.float32)


def knn_pure_match_rate(emb, labels, query_idx, query_true, pure_labels, k=10):
    """Fraction of kNN among pure spots that match the query's true state."""
    pure_idx = np.where(np.isin(labels, list(pure_labels)))[0]
    if len(pure_idx) < k or len(query_idx) == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=min(k, len(pure_idx))).fit(emb[pure_idx])
    _, ind = nn.kneighbors(emb[query_idx])
    neigh_lab = labels[pure_idx][ind]
    hits = []
    for i, t in enumerate(query_true):
        hits.append(float(np.mean(neigh_lab[i] == t)))
    return float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--outdir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--ext", type=Path, default=EXT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    outdir = args.outdir
    rng = np.random.default_rng(args.seed)

    scores = pd.read_csv(outdir / "peri_deconv_impute_scores.csv")
    peri_idx = scores["idx"].astype(int).to_numpy()

    meta = ad.read_h5ad(args.data)
    pert = meta.obs["perturbation"].astype(str).to_numpy()
    tissue = meta.obs["tissue"].astype(str).to_numpy()
    is_jak2 = pert == KO_GENE
    is_unpert = (pert == "unperturbed") & (tissue == "tumor")
    is_normal = tissue == "normal"
    is_peri = tissue == "periphery"

    z = np.load(outdir / "location_signatures.npz", allow_pickle=True)
    if "mu_normal" not in z.files:
        raise RuntimeError(
            "location_signatures.npz missing mu_normal -- run "
            "cache_task3_concert_state_signatures.py. Do not use empirical means."
        )
    spot_ids = z["spot_ids"].astype(int)
    concert_genes = z["genes"].astype(str)
    id_to_row = {int(s): i for i, s in enumerate(spot_ids)}
    mu_ko_raw = z["mu_ko"].astype(np.float32)
    mu_un_raw = z["mu_un"].astype(np.float32)
    mu_nm_raw = z["mu_normal"].astype(np.float32)

    g2i = {g: i for i, g in enumerate(np.asarray(meta.var_names).astype(str))}
    gene_cols, cidx = [], []
    for j, g in enumerate(concert_genes):
        if g in g2i:
            gene_cols.append(g2i[g])
            cidx.append(j)
    gene_cols = np.asarray(gene_cols, int)
    cidx = np.asarray(cidx, int)

    # Same library-size recipe for raw and pred (mu is non-negative rate-like).
    Y = logcpm(_dense(meta.X)[:, gene_cols])
    mu_ko = logcpm(mu_ko_raw[:, cidx])
    mu_un = logcpm(mu_un_raw[:, cidx])
    mu_nm = logcpm(mu_nm_raw[:, cidx])
    hvg = np.argsort(Y.var(0))[-min(N_HVG, Y.shape[1]) :]
    Y = Y[:, hvg]
    mu_ko = mu_ko[:, hvg]
    mu_un = mu_un[:, hvg]
    mu_nm = mu_nm[:, hvg]

    pure_jak2 = np.where(is_jak2)[0]
    pure_un = np.where(is_unpert)[0]
    pure_n = np.where(is_normal)[0]

    def subsample(idx, nmax):
        idx = np.asarray(idx, int)
        if len(idx) <= nmax:
            return idx
        return rng.choice(idx, size=nmax, replace=False)

    pure_jak2 = subsample(pure_jak2, MAX_PURE["KO_Jak2"])
    pure_un = subsample(pure_un, MAX_PURE["unpert_tumor"])
    pure_n = subsample(pure_n, MAX_PURE["normal"])

    X_list, lab_list, kind_list, src_list = [], [], [], []

    def add_block(X, lab, kind, src):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        for i in range(len(X)):
            X_list.append(X[i])
            lab_list.append(lab)
            kind_list.append(kind)
            src_list.append(src)

    add_block(Y[pure_jak2], "KO_Jak2", "pure", "obs")
    add_block(Y[pure_un], "unpert_tumor", "pure", "obs")
    add_block(Y[pure_n], "normal", "pure", "obs")
    add_block(Y[peri_idx], "mixed", "peri", "obs")

    # Plot only deconv Jak2-KO (not imputed KP / normal)
    n_imp = {"KO_Jak2": 0, "unpert_tumor": 0, "normal": 0}
    for r in scores.itertuples():
        i = int(r.idx)
        if i not in id_to_row:
            raise RuntimeError(
                f"peri idx={i} missing CONCERT signatures; "
                "run cache_task3_concert_state_signatures.py --force"
            )
        if float(r.w_jak2) >= W_MIN:
            add_block(mu_ko[id_to_row[i]], "KO_Jak2", "imputed", "concert")
            n_imp["KO_Jak2"] += 1
    n_imp_src = {k: {"concert": int(v), "empirical": 0} for k, v in n_imp.items()}

    X = np.stack(X_list, axis=0)
    labels = np.asarray(lab_list, dtype=object)
    kinds = np.asarray(kind_list, dtype=object)

    pure_mask = kinds == "pure"
    scaler = StandardScaler()
    Xp = scaler.fit_transform(X[pure_mask])
    Xa = scaler.transform(X)

    n_comp = min(30, Xp.shape[0] - 1, Xp.shape[1])
    pca = PCA(n_components=n_comp, random_state=args.seed)
    Zp = pca.fit_transform(Xp)
    Za = pca.transform(Xa)

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.25, metric="euclidean",
        random_state=args.seed, n_components=2,
    )
    emb = np.zeros((len(X), 2), dtype=np.float64)
    emb[pure_mask] = reducer.fit_transform(Zp)
    emb[~pure_mask] = reducer.transform(Za[~pure_mask])

    # diagnostics (json only; not drawn on figure)
    pure_lab_names = {"KO_Jak2", "unpert_tumor", "normal"}
    peri_q = np.where(kinds == "peri")[0]
    face = scores.set_index("idx")["face_type"].to_dict()
    face_map = {"jak2": "KO_Jak2", "unpert_tumor": "unpert_tumor", "normal": "normal"}
    peri_true = np.array(
        [face_map.get(str(face.get(int(i), "")), "mixed") for i in peri_idx],
        dtype=object,
    )
    peri_match = knn_pure_match_rate(
        emb, labels, peri_q, peri_true, pure_lab_names, k=10,
    )
    imp_rates = {}
    q_jak2 = np.where((kinds == "imputed") & (labels == "KO_Jak2"))[0]
    imp_rates["KO_Jak2"] = knn_pure_match_rate(
        emb, labels, q_jak2, np.array(["KO_Jak2"] * len(q_jak2)), pure_lab_names, k=10,
    )
    # pairwise similarity among CONCERT imputed Jak2 (explains tight clusters)
    src_arr = np.asarray(src_list, dtype=object)
    concert_jak2 = np.where((kinds == "imputed") & (labels == "KO_Jak2") & (src_arr == "concert"))[0]
    pair_r = []
    if len(concert_jak2) >= 2:
        M = X[concert_jak2]
        for a in range(len(M)):
            for b in range(a + 1, len(M)):
                pair_r.append(float(np.corrcoef(M[a], M[b])[0, 1]))
    metrics = {
        "n_raw": int(pure_mask.sum()),
        "n_peri": int((kinds == "peri").sum()),
        "n_imputed_plotted": {k: int(((kinds == "imputed") & (labels == k)).sum()) for k in n_imp},
        "n_imputed_source_counts": n_imp_src,
        "knn10_raw_match_peri_by_face": peri_match,
        "knn10_raw_match_imputed": imp_rates,
        "concert_imputed_Jak2_pairwise_pearson_mean": float(np.mean(pair_r)) if pair_r else None,
        "w_min": W_MIN,
        "n_hvg": int(len(hvg)),
        "similarity_note": (
            "Figure shows only imputed Jak2-KO (CONCERT mu_KO) with raw spots; "
            "imputed KP/normal omitted. Raw and mu both CP10K->log1p."
        ),
        "normalization": "CP10K->log1p for raw counts and CONCERT mu; StandardScaler+PCA fit on raw",
    }

    def scatter_group(ax, mask, color, marker, s, alpha, z, lw=0.0, ec="none"):
        if not np.any(mask):
            return
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, marker=marker, s=s, alpha=alpha, zorder=z,
            linewidths=lw, edgecolors=ec,
        )

    TITLE_FS, LABEL_FS, LEGEND_FS = 18, 16, 15

    def style_ax(ax, title):
        ax.set_title(title, fontsize=TITLE_FS, pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.set_xlabel("UMAP1", fontsize=LABEL_FS)
        ax.set_ylabel("UMAP2", fontsize=LABEL_FS)

    def draw_raw(ax, alpha=0.65, z0=2):
        scatter_group(ax, (kinds == "pure") & (labels == "normal"), COLORS["raw_normal"], "o", 30, alpha, z0)
        scatter_group(ax, (kinds == "pure") & (labels == "unpert_tumor"), COLORS["raw_unpert_tumor"], "o", 30, alpha, z0 + 1)
        scatter_group(ax, (kinds == "pure") & (labels == "KO_Jak2"), COLORS["raw_KO_Jak2"], "o", 38, min(alpha + 0.1, 1.0), z0 + 2)

    def draw_imputed_jak2(ax):
        scatter_group(
            ax, (kinds == "imputed") & (labels == "KO_Jak2"),
            COLORS["imp_KO_Jak2"], "^", 70, 0.95, 1, lw=1.1, ec="#8B0000",
        )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["raw_KO_Jak2"],
                   markersize=10, label="raw Jak2-KO"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["raw_unpert_tumor"],
                   markersize=10, label="raw KP tumor"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["raw_normal"],
                   markersize=10, label="raw normal"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor=COLORS["peri_mixed"],
                   markeredgecolor="white", markersize=11, label="mixed peri"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=COLORS["imp_KO_Jak2"],
                   markeredgecolor="#8B0000", markersize=11, label="imputed Jak2-KO"),
    ]

    pad = 0.08
    xmin, xmax = float(emb[:, 0].min()), float(emb[:, 0].max())
    ymin, ymax = float(emb[:, 1].min()), float(emb[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin
    xlim = (xmin - pad * dx, xmax + pad * dx)
    ylim = (ymin - pad * dy, ymax + pad * dy)

    # 2-panel figure, shared legend on right
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    ax = axes[0]
    draw_raw(ax)
    scatter_group(ax, kinds == "peri", COLORS["peri_mixed"], "X", 55, 0.95, 5, lw=0.6, ec="white")
    style_ax(ax, "Raw spots + mixed peri")

    ax = axes[1]
    draw_imputed_jak2(ax)
    draw_raw(ax, alpha=0.7, z0=3)  # raw on top so normals remain visible
    style_ax(ax, "Raw spots + imputed Jak2-KO")

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    fig.subplots_adjust(left=0.06, right=0.78, top=0.90, bottom=0.10, wspace=0.14)
    fig.legend(
        handles=legend_handles, loc="center left",
        bbox_to_anchor=(0.80, 0.50), ncol=1, fontsize=LEGEND_FS, frameon=False,
        handletextpad=0.6, labelspacing=0.85,
    )
    out_path = outdir / "umap_pure_vs_peri_vs_impute.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # one-panel
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    draw_imputed_jak2(ax)
    draw_raw(ax, alpha=0.65, z0=3)
    scatter_group(ax, kinds == "peri", COLORS["peri_mixed"], "X", 58, 0.95, 6, lw=0.7, ec="white")
    style_ax(ax, "")
    ax.set_title("")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.subplots_adjust(left=0.08, right=0.72, top=0.95, bottom=0.08)
    fig.legend(
        handles=legend_handles, loc="center left",
        bbox_to_anchor=(0.74, 0.50), ncol=1, fontsize=LEGEND_FS, frameon=False,
        handletextpad=0.6, labelspacing=0.85,
    )
    out_one = outdir / "umap_pure_peri_impute_onepanel.png"
    fig.savefig(out_one, dpi=200, bbox_inches="tight")
    plt.close(fig)

    (outdir / "umap_pure_vs_peri_metrics.json").write_text(json.dumps(metrics, indent=2))
    args.ext.mkdir(parents=True, exist_ok=True)
    for src, name in [
        (out_path, "task3_pmap_s0_jak2_umap_pure_vs_peri.png"),
        (out_one, "task3_pmap_s0_jak2_umap_pure_peri_impute.png"),
    ]:
        dst = args.ext / name
        dst.write_bytes(src.read_bytes())
        print(f"copied -> {dst}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
