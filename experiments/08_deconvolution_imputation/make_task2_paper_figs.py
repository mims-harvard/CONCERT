#!/usr/bin/env python3
"""Paper figures for Task 2 (SPAC 2-2 eval5 / lfc10).

1) Patch-removal map (KO only, no unassigned panel) -> ext_exp
2) Metric bars without unassigned; KO colors match the map
3) One TRUE vs IMPUTED spatial gene panel per KO (most consistent gene)
4) Patch-level LFC scatter (pred vs true) on CP10K -> log1p (never raw counts)
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

# Same order + colors as spot_aggregate_plots.run_plot2 (tab10 linspace)
KO_ORDER = ["Iqgap1", "Ccn1", "Cttn", "Phgr1", "H2-Ea"]
KO_COLORS = {
    g: to_hex(c)
    for g, c in zip(KO_ORDER, plt.cm.tab10(np.linspace(0, 1, len(KO_ORDER))))
}

RAW = Path(
    f"{_rp.DATA}/spacseq/raw/raw_output_2-2.h5ad"
)
MASK = Path(
    f"{_rp.WORKDIR}/scoring/spot_plots/patch_removal_2-2_top5_cellindex.npz"
)
NPZ = Path(
    f"{_rp.WORKDIR}/scoring/task2_2-2_eval5_tune_lfc10/impute_results.npz"
)
CSV = Path(
    f"{_rp.WORKDIR}/scoring/task2_2-2_eval5_tune_lfc10/demo_figs/impute_per_group.csv"
)
OUT = Path(
    f"{_rp.WORKDIR}/paper/Outputs/ext_exp"
)


def _clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def make_patch_map():
    """Two panels: (A) raw KO patches, (B) all held-out masks overlaid."""
    m = np.load(MASK, allow_pickle=True)
    role = m["role"].astype(int)
    on = m["obs_name"].astype(str)
    grp = m["group"].astype(str)

    adata = ad.read_h5ad(RAW, backed="r")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    obs_pos = {o: i for i, o in enumerate(np.asarray(adata.obs_names).astype(str))}
    bg_step = max(1, coords.shape[0] // 60000)
    BG = "#eeeeee"
    LEGEND_FS = 12

    # pre-resolve per-KO indices
    ko_data = {}
    for g in KO_ORDER:
        sel = grp == g
        names = on[sel]
        roles = role[sel]
        name_ok = np.array([n in obs_pos for n in names])
        idx = np.array([obs_pos[n] for n in names[name_ok]], dtype=int)
        roles = roles[name_ok]
        ko_data[g] = dict(
            pts=coords[idx],
            kept=(roles == 0) | (roles == 2),  # kept patch + dispersed
            rem=roles == 1,
            col=KO_COLORS[g],
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    ax_a, ax_b = axes
    pad = 0.02 * max(coords[:, 0].ptp(), coords[:, 1].ptp())
    xlim = (coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ylim = (coords[:, 1].max() + pad, coords[:, 1].min() - pad)  # inverted

    for ax in (ax_a, ax_b):
        ax.scatter(
            coords[::bg_step, 0],
            coords[::bg_step, 1],
            c=BG,
            s=1,
            linewidths=0,
            rasterized=True,
        )

    # Panel A: raw -- all KO cells solid by perturbation (no mask distinction)
    for g in KO_ORDER:
        d = ko_data[g]
        ax_a.scatter(
            d["pts"][:, 0],
            d["pts"][:, 1],
            c=d["col"],
            s=6,
            linewidths=0,
            alpha=0.85,
            rasterized=True,
        )

    # Panel B: all masks together -- kept = KO color, removed = red x
    for g in KO_ORDER:
        d = ko_data[g]
        if d["kept"].sum():
            ax_b.scatter(
                d["pts"][d["kept"], 0],
                d["pts"][d["kept"], 1],
                c=d["col"],
                s=6,
                linewidths=0,
                alpha=0.85,
                rasterized=True,
            )
        if d["rem"].sum():
            ax_b.scatter(
                d["pts"][d["rem"], 0],
                d["pts"][d["rem"], 1],
                marker="x",
                c="red",
                s=12,
                linewidths=0.55,
                alpha=0.85,
                rasterized=True,
            )

    for ax in (ax_a, ax_b):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        _clean_axes(ax)

    # shared KO color legend (panel A, bottom-right inset)
    ko_handles = [
        Line2D([0], [0], marker="o", ls="", mfc=KO_COLORS[g], mec="none", ms=8, label=g)
        for g in KO_ORDER
    ]
    leg_a = ax_a.legend(
        handles=ko_handles,
        loc="lower right",
        fontsize=LEGEND_FS,
        framealpha=0.95,
        title="perturbation",
        borderpad=0.35,
        fancybox=False,
        edgecolor="0.35",
    )
    leg_a.get_title().set_color("black")
    leg_a.get_title().set_fontsize(LEGEND_FS)
    for t in leg_a.get_texts():
        t.set_color("black")

    # panel B: KO colors + kept/removed style
    mask_handles = ko_handles + [
        Line2D([0], [0], marker="o", ls="", mfc="0.35", mec="none", ms=8, label="kept"),
        Line2D([0], [0], marker="x", ls="", color="red", ms=8, label="removed (held-out)"),
    ]
    leg_b = ax_b.legend(
        handles=mask_handles,
        loc="lower right",
        fontsize=LEGEND_FS,
        framealpha=0.95,
        borderpad=0.35,
        fancybox=False,
        edgecolor="0.35",
    )
    for t in leg_b.get_texts():
        t.set_color("black")

    # small panel tags
    ax_a.text(0.02, 0.98, "A", transform=ax_a.transAxes, ha="left", va="top",
              fontsize=16, color="black", fontweight="bold")
    ax_b.text(0.02, 0.98, "B", transform=ax_b.transAxes, ha="left", va="top",
              fontsize=16, color="black", fontweight="bold")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.06)
    out = OUT / "task2_spac22_eval5_patch_removal_top5.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    adata.file.close()
    print("Wrote", out)
    return out


BASELINE_METRICS = Path(
    f"{_rp.WORKDIR}/spatial_baselines/concert_baselines"
    "/baseline_metrics_combined/all_baselines_metrics.csv"
)
# Display order for grouped bars (CONCERT first)
IMPUTE_METHODS = ["CONCERT", "Tangram", "SpatialScope", "SpaIM"]
IMPUTE_METHOD_COLORS = {
    "CONCERT": "#1f77b4",
    "Tangram": "#ff7f0e",
    "SpatialScope": "#2ca02c",
    "SpaIM": "#d62728",
}


def make_metric_bars(norm_meta: dict | None = None):
    """Grouped LFC Spearman bars: CONCERT + Tangram / SpatialScope / SpaIM.

    Genes sorted by CONCERT LFC Spearman (best -> left). Baseline `lfc_scc` comes
    from collaborator `baseline_metrics_combined` (same 5 KO patches).
    """
    if norm_meta is None:
        meta_path = OUT / "task2_spac22_eval5_lfc10_lfc_scatter_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing normalized metrics {meta_path}; run make_lfc_scatter() first"
            )
        norm_meta = json.loads(meta_path.read_text())
    df = pd.DataFrame(norm_meta["per_group"]).rename(
        columns={"lfc_spearman": "lfc_corr", "n_cells": "n"}
    )
    df = df[df["group"].isin(KO_ORDER)].copy()
    d = df.sort_values("lfc_corr", ascending=False).reset_index(drop=True)
    gene_order = list(d["group"])

    rows = [
        {"method": "CONCERT", "group": g, "lfc_corr": float(v)}
        for g, v in zip(d["group"], d["lfc_corr"])
    ]
    if BASELINE_METRICS.exists():
        b = pd.read_csv(BASELINE_METRICS)
        b = b[b["group"].isin(KO_ORDER) & b["method"].isin(IMPUTE_METHODS[1:])].copy()
        for _, r in b.iterrows():
            rows.append(
                {
                    "method": str(r["method"]),
                    "group": str(r["group"]),
                    "lfc_corr": float(r["lfc_scc"]),
                }
            )
    else:
        print(f"WARNING: missing baseline metrics at {BASELINE_METRICS}")

    wide = (
        pd.DataFrame(rows)
        .pivot(index="group", columns="method", values="lfc_corr")
        .reindex(index=gene_order, columns=IMPUTE_METHODS)
    )

    n_genes = len(gene_order)
    n_methods = len(IMPUTE_METHODS)
    x = np.arange(n_genes)
    width = 0.18
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, method in enumerate(IMPUTE_METHODS):
        vals = wide[method].to_numpy(dtype=float)
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.92,
            color=IMPUTE_METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            label=method,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(gene_order, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("LFC Spearman $\\uparrow$", fontsize=12)
    ax.set_title("LFC corr vs held-out GT (patch impute)", fontsize=13)
    ymax = float(np.nanmax(wide.to_numpy()))
    ax.set_ylim(0, min(1.05, ymax * 1.18 + 0.05))
    ax.axhline(0, color="0.5", lw=0.6, zorder=1)
    ax.legend(
        frameon=True,
        fontsize=9,
        loc="upper right",
        edgecolor="0.35",
        fancybox=False,
    )
    fig.suptitle(
        "SPAC 2-2 impute (lfc10): CONCERT vs Tangram / SpatialScope / SpaIM",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT / "task2_spac22_eval5_tune_lfc10_impute_edist_lfc_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)
    print("Wrote", out.with_suffix(".pdf"))

    csv_out = OUT / "task2_spac22_eval5_tune_lfc10_impute_per_group.csv"
    keep = [
        c
        for c in (
            "group",
            "n",
            "e_dist",
            "lfc_corr",
            "lfc_rmse",
            "impute_hvg_corr",
            "n_ref_una",
        )
        if c in d.columns
    ]
    d[keep].to_csv(csv_out, index=False)
    print("Wrote", csv_out)

    combined_out = OUT / "task2_spac22_eval5_tune_lfc10_impute_lfc_vs_baselines.csv"
    wide.reset_index().to_csv(combined_out, index=False)
    print("Wrote", combined_out)

    metrics = {
        "impute_method": "counterfactualPrediction",
        "baselines": IMPUTE_METHODS[1:],
        "baseline_metrics": str(BASELINE_METRICS),
        "normalization": "CP10K->log1p (never raw counts)",
        "lfc_order": gene_order,
        "e_dist_mean": float(df["e_dist"].mean()) if "e_dist" in df.columns else None,
        "lfc_corr_mean": float(np.nanmean(df["lfc_corr"])),
        "lfc_corr_mean_by_method": {
            m: float(np.nanmean(wide[m].to_numpy(dtype=float)))
            for m in IMPUTE_METHODS
            if m in wide.columns
        },
        "lfc_rmse_mean": float(np.nanmean(df["lfc_rmse"]))
        if "lfc_rmse" in df.columns
        else None,
        "impute_hvg_corr_mean": float(np.nanmean(df["impute_hvg_corr"]))
        if "impute_hvg_corr" in df.columns
        else None,
    }
    metrics_path = OUT / "task2_spac22_eval5_tune_lfc10_impute_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("Wrote", metrics_path)
    return out


def _rank_consistent_genes(
    gt_g: np.ndarray, imp_g: np.ndarray, genes: np.ndarray, top_var: int = 120
) -> list[tuple[str, float]]:
    """Rank genes by Spearman on CP10K->log1p (never raw counts)."""
    gt_n = _cp10k_log1p(gt_g)
    imp_n = _cp10k_log1p(imp_g)
    var = gt_n.var(0)
    cand = np.argsort(-var)[:top_var]
    scored: list[tuple[str, float]] = []
    for j in cand:
        t, i = gt_n[:, j], imp_n[:, j]
        if np.std(t) < 1e-6 or np.std(i) < 1e-6:
            continue
        r = spearmanr(t, i).correlation
        if np.isfinite(r):
            scored.append((str(genes[j]), float(r)))
    scored.sort(key=lambda x: -x[1])
    return scored


def make_gene_panels():
    d = np.load(NPZ, allow_pickle=True)
    gt = d["gt_counts"].astype(np.float64)
    imp = d["imputed"].astype(np.float64)
    grp = d["group"].astype(str)
    coord = d["coord"].astype(np.float64)
    genes = d["genes"].astype(str)

    # Prefer unique markers across KOs (avoid e.g. S100a10 twice).
    used: set[str] = set()
    picks = {}
    for g in KO_ORDER:
        sel = grp == g
        ranked = _rank_consistent_genes(gt[sel], imp[sel], genes)
        gene, rho = None, float("nan")
        for cand_g, cand_r in ranked:
            if cand_g not in used:
                gene, rho = cand_g, cand_r
                break
        if gene is None and ranked:
            gene, rho = ranked[0]
        if gene is None:
            gene = str(genes[int(np.argmax(gt[sel].var(0)))])
        used.add(gene)
        picks[g] = {"gene": gene, "spearman_cell": rho, "n": int(sel.sum())}

    def _style_cbar(cbar):
        cbar.ax.tick_params(colors="black", labelsize=9)
        for t in cbar.ax.get_yticklabels():
            t.set_color("black")
        cbar.set_label("CP10K$\\rightarrow$log1p", color="black", fontsize=9)

    def _norm_pair(tv: np.ndarray, iv: np.ndarray, lib_t: np.ndarray, lib_i: np.ndarray):
        """CP10K->log1p for the gene (using per-cell total library); shared vmax."""
        t = np.log1p(np.clip(tv, 0, None) / np.maximum(lib_t, 1e-8) * 1e4)
        i = np.log1p(np.clip(iv, 0, None) / np.maximum(lib_i, 1e-8) * 1e4)
        vmax = max(float(np.percentile(t, 99)), float(np.percentile(i, 99)), 1e-6)
        return t, i, vmax

    # individual panels -- CP10K->log1p + shared scale within the TRUE/IMPUTED pair
    for g in KO_ORDER:
        sel = grp == g
        gene = picks[g]["gene"]
        rho = picks[g]["spearman_cell"]
        gj = int(np.where(genes == gene)[0][0])
        xy = coord[sel]
        lib_t = np.clip(gt[sel], 0, None).sum(1)
        lib_i = np.clip(imp[sel], 0, None).sum(1)
        tv, iv, vmax = _norm_pair(gt[sel, gj], imp[sel, gj], lib_t, lib_i)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
        for a, val, ttl in [
            (ax[0], tv, f"TRUE {gene}"),
            (ax[1], iv, f"IMPUTED {gene}"),
        ]:
            sc = a.scatter(
                xy[:, 0],
                xy[:, 1],
                c=val,
                s=10,
                cmap="magma",
                vmin=0,
                vmax=vmax,
                rasterized=True,
            )
            a.set_title(ttl, fontsize=12, color="black")
            a.set_aspect("equal")
            a.invert_yaxis()
            a.axis("off")
            _style_cbar(plt.colorbar(sc, ax=a, fraction=0.046))
        fig.suptitle(
            f"Task 2 CF impute -- {g}  |  marker {gene}  (CP10K$\\rightarrow$log1p; cell Spearman $\\rho$={rho:.2f})",
            y=1.02,
            fontsize=12,
            color="black",
        )
        fig.tight_layout()
        out = OUT / f"task2_spac22_eval5_lfc10_headline_{g}_{gene}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", out)

    # combined 5-row strip: CP10K->log1p + shared scale per row
    fig, axes = plt.subplots(len(KO_ORDER), 2, figsize=(9.5, 3.2 * len(KO_ORDER)))
    for row, g in enumerate(KO_ORDER):
        sel = grp == g
        gene = picks[g]["gene"]
        rho = picks[g]["spearman_cell"]
        gj = int(np.where(genes == gene)[0][0])
        xy = coord[sel]
        lib_t = np.clip(gt[sel], 0, None).sum(1)
        lib_i = np.clip(imp[sel], 0, None).sum(1)
        tv, iv, vmax = _norm_pair(gt[sel, gj], imp[sel, gj], lib_t, lib_i)
        for col, (val, ttl) in enumerate(
            [(tv, f"TRUE {gene}"), (iv, f"IMPUTED {gene}")]
        ):
            ax = axes[row, col]
            sc = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=val,
                s=6,
                cmap="magma",
                vmin=0,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_title(
                f"{g}: {ttl}  ($\\rho$={rho:.2f})", fontsize=11, color="black"
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")
            _style_cbar(plt.colorbar(sc, ax=ax, fraction=0.046))
    fig.suptitle(
        "Task 2 CONCERT (lfc10) -- CP10K$\\rightarrow$log1p expression, shared scale per KO row",
        y=0.995,
        fontsize=13,
        color="black",
    )
    fig.tight_layout()
    comb = OUT / "task2_spac22_eval5_lfc10_headline_all_kos.png"
    fig.savefig(comb, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", comb)

    meta = OUT / "task2_spac22_eval5_lfc10_headline_genes.json"
    meta.write_text(json.dumps({"ko_colors": KO_COLORS, "picks": picks}, indent=2))
    print("Wrote", meta)
    return picks


def _cp10k_log1p(X: np.ndarray, target: float = 1e4) -> np.ndarray:
    """Library-size normalize to CP10K then log1p. Never analyze raw counts."""
    X = np.asarray(X, dtype=np.float64)
    X = np.clip(X, 0, None)
    lib = X.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1e-8)
    return np.log1p(X / lib * target)


def _pb_lfc_norm(pert: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Patch-level LFC on CP10K->log1p: mean(log1p_cp10k_pert) - mean(log1p_cp10k_ref)."""
    return _cp10k_log1p(pert).mean(axis=0) - _cp10k_log1p(ref).mean(axis=0)


def _energy_distance_cp10k(
    X: np.ndarray,
    Y: np.ndarray,
    gene_mask: np.ndarray,
    subsample: int = 500,
    n_pcs: int = 50,
    random_state: int = 0,
) -> float:
    """Energy distance after CP10K->log1p (do not pass raw counts / do not double-log)."""
    X = _cp10k_log1p(X)[:, gene_mask]
    Y = _cp10k_log1p(Y)[:, gene_mask]
    rng = np.random.default_rng(random_state)
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]
    n_comp = min(n_pcs, X.shape[1], X.shape[0] + Y.shape[0] - 1)
    proj = PCA(n_components=n_comp, random_state=random_state).fit_transform(
        np.vstack([X, Y])
    )
    Xp, Yp = proj[: len(X)], proj[len(X) :]
    xy = cdist(Xp, Yp).mean()
    n, m = len(Xp), len(Yp)
    xx = cdist(Xp, Xp)[np.triu_indices(n, k=1)].mean() if n > 1 else 0.0
    yy = cdist(Yp, Yp)[np.triu_indices(m, k=1)].mean() if m > 1 else 0.0
    return float(2 * xy - xx - yy)


def _percell_hvg_corr(pred_n: np.ndarray, gt_n: np.ndarray, hv: np.ndarray) -> float:
    """Mean per-cell Spearman on HVGs in already-normalized space."""
    vals = []
    for k in range(gt_n.shape[0]):
        a, b = pred_n[k, hv], gt_n[k, hv]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        r = spearmanr(a, b).correlation
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def make_lfc_scatter(n_hvg: int = 200):
    """Full Task-2 re-eval + LFC scatters in CP10K->log1p space (never raw counts).

    Recomputes e-distance, LFC Spearman/RMSE, and per-cell HVG corr with the
    same LFC definition as eval_impute (obs: GT vs nearby kept UNA; model:
    imputed KO vs imputed basal), after CP10K->log1p.
    """
    _src = (
        f"{_rp.WORKDIR}/spaVAE/"
        "CONCERT-main/src/CONCERT-spac-rffbasal"
    )
    if _src not in sys.path:
        sys.path.insert(0, _src)

    d = np.load(NPZ, allow_pickle=True)
    imp = d["imputed"].astype(np.float64)
    gt = d["gt_counts"].astype(np.float64)
    basal = d["imputed_basal"].astype(np.float64)
    grp = np.asarray(d["group"]).astype(str)
    coord = np.asarray(d["coord"], dtype=np.float64)
    genes = np.asarray(d["genes"]).astype(str)

    # Nearby kept-unassigned reference (same construction as eval_impute).
    # Backed read + row/col subset to avoid loading the full slide into RAM.
    raw = ad.read_h5ad(RAW, backed="r")
    var_names = np.asarray(raw.var_names).astype(str)
    # handle non-unique var names without mutating backed object
    seen, uniq = {}, []
    for v in var_names:
        if v in seen:
            seen[v] += 1
            uniq.append(f"{v}-{seen[v]}")
        else:
            seen[v] = 0
            uniq.append(v)
    rgi = {g: i for i, g in enumerate(uniq)}
    cols = np.array([rgi[g] for g in genes if g in rgi], dtype=int)
    gk = np.array([j for j, g in enumerate(genes) if g in rgi], dtype=int)
    obs_names = np.asarray(raw.obs_names).astype(str)
    pert_all = raw.obs["perturbation"].astype(str).values
    coords_all = np.asarray(raw.obsm["spatial"], dtype=np.float64)
    m = np.load(MASK, allow_pickle=True)
    masked_names = set(
        np.asarray(m["obs_name"])[np.asarray(m["role"]) == 1].astype(str)
    )
    kept = np.array([nm not in masked_names for nm in obs_names])
    una_kept = kept & np.array(
        [
            ("unassign" in p.lower()) or (p.lower() in {"no_guide", "unknown"})
            for p in pert_all
        ]
    )
    una_idx = np.where(una_kept)[0]
    coords_una = coords_all[una_idx]
    tree_una = cKDTree(coords_una)

    G = gt.shape[1]
    # HVGs from variance of CP10K->log1p GT (not raw counts)
    gt_norm = _cp10k_log1p(gt)
    hv = np.zeros(G, dtype=bool)
    hv[np.argsort(gt_norm.var(axis=0))[-n_hvg:]] = True

    # First pass: collect nearby UNA neighbor indices per KO (coords only)
    nn_by_ko = {}
    needed_local = set()
    for g in KO_ORDER:
        sel = grp == g
        n = int(sel.sum())
        if n == 0:
            continue
        _, una_nn = tree_una.query(coord[sel], k=min(50, len(coords_una)))
        una_nn = np.atleast_2d(una_nn)
        if una_nn.shape[0] == 1 and n > 1:
            una_nn = una_nn.T
        ref_local = np.unique(una_nn.ravel())
        nn_by_ko[g] = ref_local
        needed_local.update(ref_local.tolist())

    # Load only the UNA neighbors we need (<< all ~324k UNA cells)
    needed_local = np.array(sorted(needed_local), dtype=int)
    global_rows = una_idx[needed_local]
    local_to_row = {int(i): j for j, i in enumerate(needed_local)}
    X_needed = np.zeros((len(global_rows), len(cols)), dtype=np.float32)
    chunk = 2000
    for s in range(0, len(global_rows), chunk):
        rows = global_rows[s : s + chunk]
        block = raw.X[rows][:, cols]
        if hasattr(block, "toarray"):
            block = block.toarray()
        X_needed[s : s + len(rows)] = np.asarray(block, dtype=np.float32)
    raw.file.close()
    del raw

    rows = []
    lfc_by_ko = {}
    for g in KO_ORDER:
        sel = grp == g
        n = int(sel.sum())
        if n == 0:
            continue
        ref_local = nn_by_ko[g]
        take = np.array([local_to_row[int(i)] for i in ref_local], dtype=int)
        ref = np.zeros((len(take), G), dtype=np.float64)
        ref[:, gk] = X_needed[take]

        obs_lfc = _pb_lfc_norm(gt[sel], ref)
        model_lfc = _pb_lfc_norm(imp[sel], basal[sel])
        o_h, m_h = obs_lfc[hv], model_lfc[hv]
        rs = float(spearmanr(o_h, m_h).correlation)
        rp = float(pearsonr(o_h, m_h).correlation)
        rmse = float(np.sqrt(np.mean((m_h - o_h) ** 2)))
        e_dist = _energy_distance_cp10k(imp[sel], gt[sel], hv)
        gt_n = _cp10k_log1p(gt[sel])
        imp_n = _cp10k_log1p(imp[sel])
        hvg_corr = _percell_hvg_corr(imp_n, gt_n, hv)
        rows.append(
            {
                "group": g,
                "n_cells": n,
                "n_ref_una": int(len(take)),
                "n_hvg": int(hv.sum()),
                "e_dist": e_dist,
                "lfc_spearman": rs,
                "lfc_pearson": rp,
                "lfc_rmse": rmse,
                "impute_hvg_corr": hvg_corr,
                "mean_abs_true_lfc": float(np.mean(np.abs(o_h))),
                "mean_abs_pred_lfc": float(np.mean(np.abs(m_h))),
            }
        )
        lfc_by_ko[g] = (o_h, m_h, genes[hv])
        print(
            f"{g}: e_dist={e_dist:.3f}  LFC rho={rs:.3f}  LFC RMSE={rmse:.3f}  "
            f"HVG corr={hvg_corr:.3f}"
        )

    # per-KO facet scatter (same order as LFC Spearman bar: best -> left)
    panel_order = [
        r["group"]
        for r in sorted(rows, key=lambda r: -r["lfc_spearman"])
    ]
    fig, axes = plt.subplots(1, len(panel_order), figsize=(3.4 * len(panel_order), 3.6))
    for ax, g in zip(axes, panel_order):
        o_h, m_h, gnames = lfc_by_ko[g]
        rs = next(r["lfc_spearman"] for r in rows if r["group"] == g)
        ax.scatter(
            o_h,
            m_h,
            s=12,
            alpha=0.55,
            c=KO_COLORS[g],
            edgecolors="none",
            rasterized=True,
        )
        lim = float(max(np.percentile(np.abs(o_h), 99), np.percentile(np.abs(m_h), 99), 0.05))
        lim *= 1.15
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.55)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(0, color="0.7", lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{g}\n$\\rho$={rs:.2f}", fontsize=11, color="black")
        ax.set_xlabel("true LFC", fontsize=10, color="black")
        ax.tick_params(colors="black")
        for sp in ax.spines.values():
            sp.set_color("black")
    axes[0].set_ylabel("pred LFC", fontsize=10, color="black")
    fig.suptitle(
        "Task 2 patch LFC (CP10K$\\rightarrow$log1p) -- true vs CONCERT pred, top-200 HVG",
        y=1.02,
        fontsize=12,
        color="black",
    )
    fig.tight_layout()
    out_facet = OUT / "task2_spac22_eval5_lfc10_lfc_scatter_per_ko.png"
    fig.savefig(out_facet, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_facet)

    # pooled scatter, colored by KO
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    all_o, all_m = [], []
    for g in KO_ORDER:
        o_h, m_h, _ = lfc_by_ko[g]
        ax.scatter(
            o_h,
            m_h,
            s=10,
            alpha=0.45,
            c=KO_COLORS[g],
            edgecolors="none",
            label=g,
            rasterized=True,
        )
        all_o.append(o_h)
        all_m.append(m_h)
    all_o = np.concatenate(all_o)
    all_m = np.concatenate(all_m)
    rs_all = float(spearmanr(all_o, all_m).correlation)
    rp_all = float(pearsonr(all_o, all_m).correlation)
    lim = float(
        max(np.percentile(np.abs(all_o), 99), np.percentile(np.abs(all_m), 99), 0.05)
    ) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.9, alpha=0.55)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(0, color="0.7", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("true patch LFC (CP10K$\\rightarrow$log1p)", fontsize=11, color="black")
    ax.set_ylabel("pred patch LFC (CP10K$\\rightarrow$log1p)", fontsize=11, color="black")
    ax.set_title(
        f"Task 2 lfc10 -- pooled HVG LFC\nSpearman $\\rho$={rs_all:.2f}  Pearson r={rp_all:.2f}",
        fontsize=12,
        color="black",
    )
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        sp.set_color("black")
    ax.legend(
        fontsize=9,
        frameon=False,
        loc="upper left",
        title="KO",
        title_fontsize=9,
        labelcolor="black",
    )
    fig.tight_layout()
    out_pool = OUT / "task2_spac22_eval5_lfc10_lfc_scatter_pooled.png"
    fig.savefig(out_pool, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_pool)

    meta = {
        "normalization": (
            "ALL metrics: CP10K (target_sum=1e4) then log1p. "
            "LFC = mean(log1p_cp10k_KO) - mean(log1p_cp10k_ref). "
            "e-distance: CP10K->log1p -> HVG -> PCA-50. "
            "Never analyze raw counts."
        ),
        "true_ref": "nearby kept unassigned (k=50 NN pool)",
        "pred_ref": "imputed_basal (same held-out cells)",
        "hvg": f"top-{n_hvg} by variance of CP10K->log1p GT across all held-out cells",
        "pooled_spearman": rs_all,
        "pooled_pearson": rp_all,
        "e_dist_mean": float(np.mean([r["e_dist"] for r in rows])),
        "lfc_corr_mean": float(np.nanmean([r["lfc_spearman"] for r in rows])),
        "lfc_rmse_mean": float(np.nanmean([r["lfc_rmse"] for r in rows])),
        "impute_hvg_corr_mean": float(np.nanmean([r["impute_hvg_corr"] for r in rows])),
        "per_group": rows,
        "figures": {"per_ko": str(out_facet), "pooled": str(out_pool)},
    }
    meta_path = OUT / "task2_spac22_eval5_lfc10_lfc_scatter_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Wrote", meta_path)
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("KO colors:", KO_COLORS)
    make_patch_map()
    make_gene_panels()
    lfc_meta = make_lfc_scatter()
    make_metric_bars(lfc_meta)


if __name__ == "__main__":
    main()
