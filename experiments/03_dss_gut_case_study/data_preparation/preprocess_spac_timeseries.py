#!/usr/bin/env python3
"""
preprocess_spac_timeseries.py — Convert Day*/rep* GEX + guide h5 pairs into h5ad.

Third SPAC-seq dataset: spatiotemporal CD8+ T cell screen (Fig 3, Zhang et al. 2026 Cell).
  - Mini-pool of 33 top-hit genes + sgnon-targeting (34 guides total)
  - Harvested at Day 4, Day 7, Day 10 (2 replicates each → 6 slides)
  - Stereo-seq poly(dT) platform

Guide barcode format quirk:
  GEX barcodes:   "304204"        (coordinate-based DNB ID)
  Guide barcodes: "304204-5"      (same ID + UMI-count suffix 1-5)
  → strip everything after the last '-' to align barcodes.

Perturbation assignment (per spot):
  guide_max = max UMI across 34 guides
  if guide_max == 0              → "unassigned"   (no guide captured)
  elif ties at guide_max > 0    → "multiple"
  elif top guide == sgnon-targeting → "control"
  else                           → gene name (strip "sg" prefix)

Usage (single sample):
    python preprocess_spac_timeseries.py \\
        --gex   /path/to/Day4_rep1.h5 \\
        --guide /path/to/Day4_rep1.guide.h5 \\
        --out   /path/to/raw_output_3-1.h5ad \\
        --day   Day4  --rep rep1 --sample_id 3-1

Usage (all 6 via SLURM array, see sub_preprocess_timeseries.sh):
    See sub_preprocess_timeseries.sh
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── guide-name utilities ────────────────────────────────────────────────────

NON_TARGETING_RAW = "sgnon-targeting"


def guide_name_to_perturbation(raw: str) -> str:
    """'sgCd44' → 'Cd44', 'sgnon-targeting' → 'control'."""
    if raw == NON_TARGETING_RAW:
        return "control"
    if raw.startswith("sg"):
        return raw[2:]
    return raw


# ── I/O helpers ─────────────────────────────────────────────────────────────

def load_gex(gex_path: Path) -> ad.AnnData:
    """Load GEX h5 (Stereo-seq format) into AnnData with sparse X."""
    with h5py.File(gex_path, "r") as f:
        barcodes = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in f["obs/_index"][:]])
        genes = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in f["var/_index"][:]])
        data    = f["X/data"][:]
        indices = f["X/indices"][:]
        indptr  = f["X/indptr"][:]
        shape   = (len(barcodes), len(genes))
        X = sp.csr_matrix((data, indices, indptr), shape=shape)

        # extra var metadata
        var_dict = {}
        for k in f["var"].keys():
            if k == "_index":
                continue
            v = f["var"][k][:]
            var_dict[k] = v

        # spatial coords and marker
        spatial = f["obsm/spatial"][:].astype(np.float32)
        marker_cats  = [x.decode() if isinstance(x, bytes) else x
                        for x in f["obs/marker/categories"][:]]
        marker_codes = f["obs/marker/codes"][:].astype(int)
        marker_vals  = np.array([marker_cats[c] for c in marker_codes])

    var_df = pd.DataFrame(var_dict, index=genes)
    obs_df = pd.DataFrame({"marker": marker_vals}, index=barcodes)

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.obsm["spatial"] = spatial
    return adata


def load_guide_matrix(guide_path: Path, gex_barcodes: np.ndarray):
    """
    Load guide UMI matrix aligned to gex_barcodes.

    Guide barcodes have a '-N' suffix (N=1-5).  Strip it, then reindex to
    match GEX barcodes (inner join; unmatched guide spots are dropped).

    Returns
    -------
    guide_mat : (n_gex, n_guides) int32 ndarray
    guide_names_raw : list[str]   e.g. ['sgAdrb2', ..., 'sgnon-targeting']
    """
    with h5py.File(guide_path, "r") as f:
        guide_bc_raw = [x.decode() if isinstance(x, bytes) else x
                        for x in f["obs/_index"][:]]
        guide_names  = [x.decode() if isinstance(x, bytes) else x
                        for x in f["var/_index"][:]]
        guide_mat    = f["X"][:].astype(np.int32)      # dense (n_guide_bc, n_guides)

    # Strip suffix '-N' from guide barcodes
    guide_bc_stripped = np.array([b.rsplit("-", 1)[0] for b in guide_bc_raw])

    # Build mapping: stripped barcode → row in guide matrix
    guide_bc_to_row = {bc: i for i, bc in enumerate(guide_bc_stripped)}

    n_gex = len(gex_barcodes)
    n_guides = len(guide_names)
    out_mat = np.zeros((n_gex, n_guides), dtype=np.int32)

    matched = 0
    for gex_idx, bc in enumerate(gex_barcodes):
        row = guide_bc_to_row.get(bc)
        if row is not None:
            out_mat[gex_idx] = guide_mat[row]
            matched += 1

    print(f"  Guide barcode match: {matched}/{n_gex} GEX spots ({100*matched/n_gex:.1f}%)")
    return out_mat, guide_names


def assign_perturbation(guide_mat: np.ndarray, guide_names: list):
    """
    Assign perturbation label per spot.

    Returns
    -------
    perturbation : (N,) str array — gene name / "control" / "unassigned" / "multiple"
    n_guide_umi  : (N,) int32 — total guide UMI per spot
    """
    n_guide_umi = guide_mat.sum(axis=1).astype(np.int32)
    row_max     = guide_mat.max(axis=1)

    pert_labels = [guide_name_to_perturbation(g) for g in guide_names]

    N = guide_mat.shape[0]
    perturbation = np.full(N, "unassigned", dtype=object)

    for i in range(N):
        mx = row_max[i]
        if mx == 0:
            perturbation[i] = "unassigned"
            continue
        top_indices = np.where(guide_mat[i] == mx)[0]
        if len(top_indices) > 1:
            perturbation[i] = "multiple"
        else:
            perturbation[i] = pert_labels[top_indices[0]]

    return perturbation.astype(str), n_guide_umi


# ── spatial plots ─────────────────────────────────────────────────────────────

SPECIAL_PERT_COLORS = {
    "unassigned": "#CCCCCC",
    "multiple":   "#FF8C00",
    "control":    "#A8D8A8",
}
# Ambiguous / non-gene labels: background only, excluded from top-K ranking.
PLOT_BACKGROUND_LABELS = frozenset({"unassigned", "multiple"})


def _subsample_coords_pert(
    coords: np.ndarray,
    perturbation: np.ndarray,
    max_spots_plot: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(perturbation)
    if n <= max_spots_plot:
        return coords, perturbation
    rng = np.random.default_rng(42)
    idx = rng.choice(n, max_spots_plot, replace=False)
    return coords[idx], perturbation[idx]


def make_spatial_plot_clean(
    coords: np.ndarray,
    perturbation: np.ndarray,
    out_path: Path,
    title: str = "",
    top_k: int = 20,
    max_spots_plot: int = 300_000,
) -> None:
    """Single-panel spatial map: top-K gene perturbations, no legend."""
    coords_plot, pert_plot = _subsample_coords_pert(coords, perturbation, max_spots_plot)

    counts = pd.Series(pert_plot).value_counts()
    eligible = counts.drop(
        labels=[l for l in PLOT_BACKGROUND_LABELS if l in counts.index],
        errors="ignore",
    )
    top_labels = eligible.head(top_k).index.tolist()
    top_set = set(top_labels)

    gene_in_top = [l for l in top_labels if l not in SPECIAL_PERT_COLORS]
    cmap_base = plt.cm.get_cmap("tab20", max(len(gene_in_top), 1))
    label_to_color = dict(SPECIAL_PERT_COLORS)
    for i, gene in enumerate(gene_in_top):
        label_to_color[gene] = mcolors.to_hex(cmap_base(i % 20))

    fig, ax = plt.subplots(figsize=(9, 9))
    bg_mask = ~np.isin(pert_plot, list(top_set))
    if bg_mask.any():
        ax.scatter(
            coords_plot[bg_mask, 0], coords_plot[bg_mask, 1],
            s=0.04, c="#ECECEC", alpha=0.35, rasterized=True, zorder=1,
        )

    for lbl in reversed(top_labels):
        mask = pert_plot == lbl
        if not mask.any():
            continue
        color = label_to_color.get(lbl, "#888888")
        if lbl == "unassigned":
            size, alpha = 0.05, 0.25
        elif lbl == "multiple":
            size, alpha = 0.35, 0.75
        elif lbl == "control":
            size, alpha = 0.5, 0.8
        else:
            size, alpha = 0.7, 0.85
        ax.scatter(
            coords_plot[mask, 0], coords_plot[mask, 1],
            s=size, c=color, alpha=alpha, rasterized=True, zorder=2,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Spatial plot → {out_path}")


def make_spatial_plot(
    adata: ad.AnnData,
    perturbation: np.ndarray,
    outdir: Path,
    sample_id: str,
    day: str,
    rep: str,
    max_spots_plot: int = 300_000,
    top_k: int = 20,
):
    """Plot clean spatial map (top-K labels, no legend) plus perturbation count bar chart."""
    coords = adata.obsm["spatial"]
    title = (
        f"SPAC-seq — {day} {rep} ({sample_id})\n"
        f"Top {top_k} gene perturbations by abundance"
    )
    out_path = outdir / f"{sample_id}_spatial_perturbation.png"
    make_spatial_plot_clean(
        coords, perturbation, out_path,
        title=title, top_k=top_k, max_spots_plot=max_spots_plot,
    )

    unique_labels = sorted(set(perturbation.tolist()))
    gene_labels = [l for l in unique_labels if l not in SPECIAL_PERT_COLORS]
    cmap_base = plt.cm.get_cmap("tab20", max(len(gene_labels), 1))
    gene_colors = {g: mcolors.to_hex(cmap_base(i % 20)) for i, g in enumerate(gene_labels)}
    label_to_color = {**SPECIAL_PERT_COLORS, **gene_colors}

    # ── bar chart: spot counts per perturbation ─────────────────────────────
    counts = pd.Series(perturbation).value_counts().sort_values(ascending=False)
    n_show = min(len(counts), 40)
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    bars = counts.iloc[:n_show]
    bar_colors = [label_to_color.get(l, "#888888") for l in bars.index]
    bars.plot.bar(ax=ax2, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax2.set_title(f"Spot counts per perturbation — {day} {rep}", fontsize=11)
    ax2.set_xlabel("Perturbation", fontsize=9)
    ax2.set_ylabel("# spots", fontsize=9)
    ax2.tick_params(axis="x", labelsize=6, rotation=60)
    plt.tight_layout()
    bar_path = outdir / f"{sample_id}_perturbation_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar plot     → {bar_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Day*/rep* GEX+guide h5 pairs → h5ad (SPAC-seq time-series).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gex",       required=True, type=Path, help="GEX h5 file")
    p.add_argument("--guide",     required=True, type=Path, help="Guide h5 file")
    p.add_argument("--out",       required=True, type=Path, help="Output h5ad path")
    p.add_argument("--day",       required=True, help="Timepoint label, e.g. Day4")
    p.add_argument("--rep",       required=True, help="Replicate label, e.g. rep1")
    p.add_argument("--sample_id", required=True, help="Sample ID, e.g. 3-1")
    p.add_argument("--plot_dir",  type=Path, default=None,
                   help="Directory for spatial plots (default: sibling 'analysis/<sample_id>')")
    return p.parse_args()


def main():
    args = parse_args()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = args.plot_dir or (out_path.parent / f"analysis/{args.sample_id}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    sample_id = args.sample_id
    day       = args.day
    rep       = args.rep

    print(f"\n{'='*60}")
    print(f"  SPAC-seq time-series  |  {day} {rep}  (sample {sample_id})")
    print(f"{'='*60}")

    # 1. GEX
    print("\n[1] Loading GEX...")
    adata = load_gex(args.gex)
    print(f"  Shape: {adata.shape}")

    # 2. Guide matrix
    print("\n[2] Loading guide matrix...")
    guide_mat, guide_names_raw = load_guide_matrix(
        args.guide, adata.obs_names.to_numpy()
    )
    print(f"  Guides ({len(guide_names_raw)}): {guide_names_raw}")

    # 3. Perturbation assignment
    print("\n[3] Assigning perturbation labels...")
    perturbation, n_guide_umi = assign_perturbation(guide_mat, guide_names_raw)

    n_unassigned = int((perturbation == "unassigned").sum())
    n_control    = int((perturbation == "control").sum())
    n_multiple   = int((perturbation == "multiple").sum())
    n_perturbed  = int((~np.isin(perturbation, ["unassigned", "multiple", "control"])).sum())
    n_targets    = len(set(perturbation) - {"unassigned", "multiple", "control"})
    print(f"  unassigned : {n_unassigned:>8,}")
    print(f"  control    : {n_control:>8,}")
    print(f"  gene-KO    : {n_perturbed:>8,}  ({n_targets} unique targets)")
    print(f"  multiple   : {n_multiple:>8,}")

    # 4. Build obs
    print("\n[4] Building obs...")
    N = adata.n_obs
    n_counts = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.int64)
    n_genes  = np.diff(adata.X.indptr).astype(np.int32)

    adata.obs = pd.DataFrame({
        "perturbation":        pd.Categorical(perturbation),
        "perturbation_source": pd.Categorical(["guide_capture_umi"] * N),
        "sample":              pd.Categorical([sample_id] * N),
        "group":               pd.Categorical([day] * N),
        "replicate":           pd.Categorical([rep] * N),
        "dataset":             pd.Categorical(["spac_seq"] * N),
        "species":             pd.Categorical(["mouse"] * N),
        "platform":            pd.Categorical(["Stereo_seq"] * N),
        "n_counts":            n_counts,
        "n_genes":             n_genes,
        "n_guide_umi":         n_guide_umi,
    }, index=adata.obs_names)

    # 5. uns
    adata.uns.update({
        "dataset":     "spac_seq",
        "group":       day,
        "sample":      sample_id,
        "n_targets":   n_targets,
        "perturbations": sorted(set(perturbation) - {"unassigned", "multiple", "control"}),
        "platform":    "Stereo_seq",
        "species":     "mouse",
        "probe_set":   "Stereo-seq poly(dT) — mini-pool T cell screen (Day4/7/10)",
        "scalefactors": {},
    })

    # 6. array_coords alias
    adata.obsm["array_coords"] = adata.obsm["spatial"].copy()

    # 7. Save (atomic write)
    print(f"\n[5] Saving → {out_path}")
    tmp = out_path.with_suffix(".tmp.h5ad")
    adata.write_h5ad(tmp, compression="gzip")
    tmp.rename(out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"  Written: {out_path}  ({size_gb:.2f} GB)")

    # 8. Spatial plots
    print("\n[6] Plotting...")
    make_spatial_plot(adata, perturbation, plot_dir, sample_id, day, rep)

    print(f"\n{'='*60}")
    print(f"  Done  {day} {rep}  |  {adata.shape}  |  {n_targets} gene targets")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
