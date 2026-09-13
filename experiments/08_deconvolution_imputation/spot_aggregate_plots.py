#!/usr/bin/env python3
"""Spatial pseudo-bulk spot aggregation + perturbation-patch removal plots.

Plot set 1 (slide 2-1): aggregate ~9 spatially-proximal cells into pseudo-bulk
  spots (greedy non-overlapping kNN partition). Spot coord = centroid; each spot
  keeps the perturbation-identity composition of its member cells. Two variants:
    (1a) whole slide   (all cells)
    (1b) top-N perturbation cells only
  Spatial scatter colors each spot by its DOMINANT perturbation; full composition
  fractions are saved to CSV, and a pseudo-bulk h5ad (mean expression) is written.

Plot set 2 (slide 2-2): for the top-5 KO perturbations, treat each perturbation's
  cells as a spatial patch, remove a spatially-contiguous half (cells on one side
  of the patch's principal axis), and plot the kept cells with the missing region
  highlighted (removed cells + convex-hull shading).

"Cell type" == perturbation identity (no cell-type annotation exists in the data).
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import anndata as ad
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq")
UNASSIGNED = frozenset(["unassigned", "Unassigned", "No_guide", "no_guide", "unknown"])
CONTROL = frozenset(["Non_targeting", "non_targeting", "NonTargeting", "control", "Control", "CTRL", "ctrl", "NT"])
EXCLUDED = frozenset(["multiple", "Multiple"])


def classify(p):
    if p in UNASSIGNED:
        return "unassigned"
    if p in CONTROL:
        return "control"
    if p in EXCLUDED:
        return "multiple"
    return "gene_KO"


def top_ko_genes(pert, n):
    ko = [p for p in pert if classify(p) == "gene_KO"]
    return [g for g, _ in Counter(ko).most_common(n)]


# ---
# Greedy non-overlapping spatial kNN partition (~target_size cells per spot)
# ---
def knn_partition(coords, target_size=9, seed=42):
    """Assign each cell to a non-overlapping spot of ~target_size nearest cells.
    Returns spot_id per cell (int array)."""
    n = coords.shape[0]
    tree = cKDTree(coords)
    assigned = np.full(n, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    k_query = min(n, max(target_size * 4, 32))
    spot = 0
    for seed_i in order:
        if assigned[seed_i] != -1:
            continue
        _, idx = tree.query(coords[seed_i], k=k_query)
        idx = np.atleast_1d(idx)
        free = idx[assigned[idx] == -1]
        take = free[:target_size]
        assigned[take] = spot
        spot += 1
    # any stragglers (rare; if k_query too small) -> own spots
    leftover = np.where(assigned == -1)[0]
    for i in leftover:
        assigned[i] = spot; spot += 1
    log.info("kNN partition: %d cells -> %d spots (target=%d, mean=%.1f)",
             n, spot, target_size, n / spot)
    return assigned


def aggregate_spots(adata, coords, spot_id, pert):
    """Build pseudo-bulk per spot: mean expression, centroid, dominant pert + composition."""
    n_spots = spot_id.max() + 1
    G = sp.csr_matrix((np.ones(len(spot_id)), (spot_id, np.arange(len(spot_id)))),
                      shape=(n_spots, adata.n_obs))
    counts = np.asarray(G.sum(1)).ravel()
    pb = G @ adata.X                       # summed counts per spot (sparse)
    pb = sp.csr_matrix(pb / counts[:, None])  # mean expression
    cx = (G @ coords[:, 0]) / counts
    cy = (G @ coords[:, 1]) / counts
    centroid = np.c_[cx, cy]
    # dominant pert + composition
    dom = np.empty(n_spots, dtype=object)
    comp_rows = []
    perts_in_spot = [[] for _ in range(n_spots)]
    for cell, s in enumerate(spot_id):
        perts_in_spot[s].append(pert[cell])
    for s in range(n_spots):
        c = Counter(perts_in_spot[s])
        dom[s] = c.most_common(1)[0][0]
        comp_rows.append({"spot": s, "n_cells": int(counts[s]),
                          "x": float(centroid[s, 0]), "y": float(centroid[s, 1]),
                          "dominant": dom[s],
                          **{f"frac_{g}": v / counts[s] for g, v in c.items()}})
    pb_ad = ad.AnnData(pb, var=adata.var.copy())
    pb_ad.obs["dominant_pert"] = dom
    pb_ad.obs["n_cells"] = counts.astype(int)
    pb_ad.obsm["spatial"] = centroid
    return pb_ad, centroid, dom, comp_rows


# ---
# Coloring
# ---
def pert_color_map(top_genes):
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(top_genes), 1)))
    m = {g: to_hex(cmap[i]) for i, g in enumerate(top_genes)}
    m["unassigned"] = "#dddddd"
    m["control"] = "#1f77b4"
    m["multiple"] = "#eeccee"
    m["__otherKO__"] = "#777777"
    return m


def color_for(label, cmap, top_set):
    if label in top_set:
        return cmap[label]
    if label in UNASSIGNED:
        return cmap["unassigned"]
    if label in CONTROL:
        return cmap["control"]
    if label in EXCLUDED:
        return cmap["multiple"]
    return cmap["__otherKO__"]


def _clean_axes(ax):
    """Hide x/y axis labels, titles, and tick text per request."""
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])


def plot_spots(centroid, dom, top_genes, title, outpath, spot_size=6,
               bg_coords=None, bg_label="unassigned (raw cells)"):
    cmap = pert_color_map(top_genes)
    top_set = set(top_genes)
    cols = np.array([color_for(d, cmap, top_set) for d in dom])
    fig, ax = plt.subplots(figsize=(11, 10))
    # optional: raw (un-aggregated) unassigned cells at their original coords
    if bg_coords is not None and len(bg_coords):
        ax.scatter(bg_coords[:, 0], bg_coords[:, 1], c="#dddddd",
                   s=1.0, linewidths=0, alpha=0.45, rasterized=True, zorder=0)
    # draw background (non-top) first, then top on top
    is_top = np.array([d in top_set for d in dom])
    ax.scatter(centroid[~is_top, 0], centroid[~is_top, 1], c=cols[~is_top],
               s=spot_size * 0.7, linewidths=0, alpha=0.55, rasterized=True)
    ax.scatter(centroid[is_top, 0], centroid[is_top, 1], c=cols[is_top],
               s=spot_size, linewidths=0, alpha=0.95, rasterized=True)
    handles = [Line2D([0], [0], marker="o", ls="", mfc=cmap[g], mec="none", ms=7, label=g)
               for g in top_genes]
    handles += [Line2D([0], [0], marker="o", ls="", mfc=cmap[k], mec="none", ms=7, label=lab)
                for k, lab in [("__otherKO__", "other KO"), ("control", "control"),
                               ("unassigned", "unassigned")]]
    if bg_coords is not None and len(bg_coords):
        handles += [Line2D([0], [0], marker="o", ls="", mfc="#dddddd", mec="none", ms=7, label=bg_label)]
    ax.legend(handles=handles, fontsize=7, ncol=1, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), framealpha=0.9, title="dominant perturbation")
    ax.set_title(title, fontsize=11)
    _clean_axes(ax)
    ax.set_aspect("equal"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Wrote %s", outpath)


# ---
# Plot set 1
# ---
def run_plot1(args):
    f = DATA_DIR / f"raw_output_{args.slide1}.h5ad"
    log.info("Loading %s ...", f)
    adata = ad.read_h5ad(f)
    coords = np.asarray(adata.obsm[args.coord_key], dtype=np.float64)
    pert = adata.obs["perturbation"].astype(str).to_numpy()
    top_genes = top_ko_genes(pert, args.top_n)
    log.info("Top-%d KO on %s: %s", args.top_n, args.slide1, top_genes)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    import csv
    # (1a) whole slide
    if not args.skip_1a:
        sid = knn_partition(coords, args.spot_size_cells, args.seed)
        pb, cen, dom, comp = aggregate_spots(adata, coords, sid, pert)
        pb.write_h5ad(outdir / f"pseudobulk_{args.slide1}_wholeslide.h5ad")
        with (outdir / f"spot_composition_{args.slide1}_wholeslide.csv").open("w", newline="") as fh:
            keys = sorted({k for r in comp for k in r})
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(comp)
        plot_spots(cen, dom, top_genes,
                   f"Slide {args.slide1} -- pseudo-bulk spots (~{args.spot_size_cells} cells)\n"
                   f"whole slide, {len(dom):,} spots, colored by dominant perturbation",
                   outdir / f"spots_{args.slide1}_wholeslide.png")

    # (1b) top-perturbation cells only
    keep = np.array([p in set(top_genes) for p in pert])
    log.info("(1b) top-pert cells: %d / %d", keep.sum(), len(pert))
    sub = adata[keep].copy()
    sub_coords = coords[keep]
    sub_pert = pert[keep]
    sid2 = knn_partition(sub_coords, args.spot_size_cells, args.seed)
    pb2, cen2, dom2, comp2 = aggregate_spots(sub, sub_coords, sid2, sub_pert)
    pb2.write_h5ad(outdir / f"pseudobulk_{args.slide1}_toppert.h5ad")
    with (outdir / f"spot_composition_{args.slide1}_toppert.csv").open("w", newline="") as fh:
        keys = sorted({k for r in comp2 for k in r})
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(comp2)
    # raw (un-aggregated) unassigned cells drawn at their original coordinates
    una_mask = np.array([classify(p) == "unassigned" for p in pert])
    plot_spots(cen2, dom2, top_genes,
               f"Slide {args.slide1} -- pseudo-bulk spots (~{args.spot_size_cells} cells)\n"
               f"top-{args.top_n} perturbation cells only, {len(dom2):,} spots "
               f"(+ {int(una_mask.sum()):,} raw unassigned cells)",
               outdir / f"spots_{args.slide1}_toppert.png",
               bg_coords=coords[una_mask])


# ---
# Plot set 2 -- contiguous-half patch removal
# ---
def _principal_half(pts):
    """Boolean mask of the negative-median side of pts' first principal axis."""
    X = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ vt[0]
    return proj < np.median(proj)


def contiguous_half_mask(pts, seed=42, eps_factor=3.0, min_samples=5):
    """Remove a spatially-contiguous half of the perturbation's MAIN patch only.

    The perturbation's cells often form one dense patch plus scattered/isolated
    cells elsewhere. We first isolate the main patch as the largest dense cluster
    (DBSCAN, eps = eps_factor x median nearest-neighbor spacing); dispersed/isolated
    cells (DBSCAN noise or minor clusters) are NEVER removed. Within that main
    patch we then discard the contiguous half on the negative side of its first
    principal axis. Returns (remove_mask, patch_mask) over all rows of pts."""
    n = pts.shape[0]
    remove = np.zeros(n, dtype=bool)
    if n < max(2 * min_samples, 4) or n > 40000:
        # too few cells to cluster, or a dense whole-tissue population (e.g.
        # unassigned): the population is itself one contiguous region -> take the
        # principal-axis half directly.
        half = _principal_half(pts)
        remove[half] = True
        return remove, np.ones(n, dtype=bool)
    # characteristic spacing = median nearest-neighbor distance
    tree = cKDTree(pts)
    dd, _ = tree.query(pts, k=2)
    eps = float(np.median(dd[:, 1])) * eps_factor
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    except Exception:
        labels = np.zeros(n, dtype=int)
    valid = labels[labels >= 0]
    if valid.size == 0:
        patch_mask = np.ones(n, dtype=bool)               # all noise -> treat all as patch
    else:
        vals, cnts = np.unique(valid, return_counts=True)
        patch_mask = labels == vals[np.argmax(cnts)]      # largest dense cluster
    idx_patch = np.where(patch_mask)[0]
    half = _principal_half(pts[patch_mask])
    remove[idx_patch[half]] = True
    return remove, patch_mask


def sample_small_patch(pts, size=800, seed=42):
    """Pick a random seed cell and remove its `size` nearest neighbors -- a small,
    spatially-contiguous patch (used for unassigned cells)."""
    n = pts.shape[0]
    size = min(size, n)
    tree = cKDTree(pts)
    s = int(np.random.default_rng(seed).integers(n))
    _, idx = tree.query(pts[s], k=size)
    rem = np.zeros(n, dtype=bool)
    rem[np.atleast_1d(idx)] = True
    return rem


def sample_dense_patch(pts, size=800, k_density=50):
    """Contiguous patch in the MIDDLE of a large/dense region.

    Seed = the cell with the highest local density (smallest distance to its
    k-th nearest neighbor), which lies in the interior of the largest contiguous
    region; remove that seed's `size` nearest neighbors. Deterministic."""
    n = pts.shape[0]
    size = min(size, n)
    tree = cKDTree(pts)
    kq = min(n, k_density + 1)
    dd, _ = tree.query(pts, k=kq)
    seed_i = int(np.argmin(dd[:, -1]))      # densest cell -> interior of a large region
    _, idx = tree.query(pts[seed_i], k=size)
    rem = np.zeros(n, dtype=bool)
    rem[np.atleast_1d(idx)] = True
    return rem


def run_plot2(args):
    f = DATA_DIR / f"raw_output_{args.slide2}.h5ad"
    log.info("Loading %s ...", f)
    adata = ad.read_h5ad(f, backed="r")
    coords = np.asarray(adata.obsm[args.coord_key], dtype=np.float64)
    pert = adata.obs["perturbation"].astype(str).to_numpy()
    top5 = top_ko_genes(pert, args.top5)
    log.info("Top-%d KO on %s: %s", args.top5, args.slide2, top5)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(top5)))

    # per-gene panels + overview + 1 unassigned-patch panel
    n_panels = len(top5) + 2
    ncol = 3
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 5.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    bg_step = max(1, len(pert) // 60000)  # thin background for speed
    BG_GREY = "#eeeeee"
    PANEL_TITLE_FS = 24
    PANEL_TITLE_PAD = 18
    LEGEND_FS = 20
    LEGEND_KW = dict(loc="center left", bbox_to_anchor=(1.02, 0.5),
                     framealpha=0.9, borderaxespad=0)

    removed_records = []
    obs_names = np.asarray(adata.obs_names)
    cell_rows = []  # (group, global_cell_index, obs_name, role) role: 1=removed,0=kept-patch,2=dispersed
    ax_ov = axes[0]
    ax_ov.scatter(coords[::bg_step, 0], coords[::bg_step, 1], c=BG_GREY, s=1,
                  linewidths=0, rasterized=True)
    for gi, g in enumerate(top5):
        m = pert == g
        pts = coords[m]
        rem, patch = contiguous_half_mask(pts, args.seed)
        kept = ~rem
        disp = ~patch                      # dispersed/isolated cells: ignored (kept)
        kept_patch = kept & patch          # retained half of the main patch
        col = to_hex(cmap[gi])
        # overview: kept solid, removed hollow
        ax_ov.scatter(pts[kept, 0], pts[kept, 1], c=col, s=6, linewidths=0, alpha=0.8, rasterized=True)
        ax_ov.scatter(pts[rem, 0], pts[rem, 1], facecolors="none", edgecolors=col,
                      s=10, linewidths=0.4, alpha=0.7, rasterized=True)
        removed_records.append({"gene": g, "n_total": int(m.sum()),
                                "n_patch": int(patch.sum()), "n_dispersed": int(disp.sum()),
                                "n_kept": int(kept.sum()), "n_removed": int(rem.sum())})
        # persist per-cell removal mask (global indices) for this gene's patch
        gidx = np.where(m)[0]
        roles_g = np.zeros(gidx.size, dtype=int)  # 0 = kept in-patch
        roles_g[rem] = 1                          # 1 = removed / masked
        roles_g[disp] = 2                         # 2 = dispersed (kept, outside main patch)
        for j, ci in enumerate(gidx):
            cell_rows.append((g, int(ci), obs_names[ci], int(roles_g[j])))
        # per-gene panel
        ax = axes[gi + 1]
        ax.scatter(coords[::bg_step, 0], coords[::bg_step, 1], c=BG_GREY, s=1,
                   linewidths=0, rasterized=True)
        # shade missing region (convex hull of removed cells)
        try:
            from scipy.spatial import ConvexHull
            if rem.sum() >= 3:
                h = ConvexHull(pts[rem])
                poly = pts[rem][h.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color="red", alpha=0.10, zorder=1)
        except Exception:
            pass
        ax.scatter(pts[kept_patch, 0], pts[kept_patch, 1], c=col, s=10, linewidths=0, alpha=0.9,
                   label=f"kept patch ({kept_patch.sum()})", rasterized=True)
        if disp.sum():
            ax.scatter(pts[disp, 0], pts[disp, 1], c="#bbbbbb", s=6, linewidths=0, alpha=0.7,
                       rasterized=True)  # shown but omitted from legend
        ax.scatter(pts[rem, 0], pts[rem, 1], marker="x", c="red", s=14, linewidths=0.6,
                   alpha=0.8, label=f"removed ({rem.sum()})", rasterized=True)
        ax.set_title(f"{g}\ncontiguous half removed", fontsize=PANEL_TITLE_FS,
                     pad=PANEL_TITLE_PAD, linespacing=1.15)
        ax.legend(fontsize=LEGEND_FS, markerscale=2, **LEGEND_KW)
        ax.set_aspect("equal"); ax.invert_yaxis()
        _clean_axes(ax)

    ax_ov.set_title(f"Slide {args.slide2} -- top-{args.top5} perturbations\n"
                    "filled = kept, hollow = removed (missing) half",
                    fontsize=PANEL_TITLE_FS, pad=PANEL_TITLE_PAD, linespacing=1.15)
    ax_ov.set_aspect("equal"); ax_ov.invert_yaxis()
    _clean_axes(ax_ov)
    handles = [Line2D([0], [0], marker="o", ls="", mfc=to_hex(cmap[i]), mec="none", ms=10, label=g)
               for i, g in enumerate(top5)]
    leg = ax_ov.legend(handles=handles, fontsize=LEGEND_FS, title="perturbation",
                       markerscale=1.5, **LEGEND_KW)
    leg.get_title().set_fontsize(LEGEND_FS)

    # --- unassigned-cell patch removal: a random small contiguous patch (<1000) ---
    una_mask = np.array([classify(p) == "unassigned" for p in pert])
    pts_u = coords[una_mask]
    rem_u = sample_dense_patch(pts_u, size=args.una_patch_size)  # middle of a large unassigned region
    kept_u = ~rem_u
    removed_records.append({"gene": "unassigned", "n_total": int(una_mask.sum()),
                            "n_patch": int(rem_u.sum()), "n_dispersed": 0,
                            "n_kept": int(kept_u.sum()), "n_removed": int(rem_u.sum())})
    # persist the removed unassigned patch (global indices)
    uidx = np.where(una_mask)[0]
    for ci in uidx[rem_u]:
        cell_rows.append(("unassigned", int(ci), obs_names[ci], 1))
    ax_u = axes[len(top5) + 1]
    # same tissue background as other panels (do not re-draw all kept unassigned cells)
    ax_u.scatter(coords[::bg_step, 0], coords[::bg_step, 1], c=BG_GREY, s=1,
                 linewidths=0, rasterized=True)
    try:
        from scipy.spatial import ConvexHull
        if rem_u.sum() >= 3:
            h = ConvexHull(pts_u[rem_u])
            poly = pts_u[rem_u][h.vertices]
            ax_u.fill(poly[:, 0], poly[:, 1], color="red", alpha=0.15, zorder=1)
    except Exception:
        pass
    ax_u.scatter(pts_u[rem_u, 0], pts_u[rem_u, 1], c="red", s=8, linewidths=0,
                 alpha=0.85, label=f"removed patch ({int(rem_u.sum()):,})", rasterized=True)
    ax_u.set_title(f"unassigned\npatch in middle of large region removed (n={int(rem_u.sum())})",
                   fontsize=PANEL_TITLE_FS, pad=PANEL_TITLE_PAD, linespacing=1.15)
    ax_u.legend(fontsize=LEGEND_FS, markerscale=3, **LEGEND_KW)
    ax_u.set_aspect("equal"); ax_u.invert_yaxis()
    _clean_axes(ax_u)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Slide {args.slide2}: top-{args.top5} perturbation patches with contiguous half removed",
                 fontsize=26, y=0.99)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.04, hspace=0.22, wspace=0.35)
    out = outdir / f"patch_removal_{args.slide2}_top{args.top5}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.12); plt.close(fig)
    log.info("Wrote %s", out)

    import csv
    with (outdir / f"patch_removal_{args.slide2}_top{args.top5}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene", "n_total", "n_patch", "n_dispersed",
                                           "n_kept", "n_removed"])
        w.writeheader(); w.writerows(removed_records)

    # --- per-cell removal index (shared, fixed indices for all methods) ---
    groups = np.array([r[0] for r in cell_rows], dtype=object)
    cidx = np.array([r[1] for r in cell_rows], dtype=np.int64)
    onames = np.array([r[2] for r in cell_rows], dtype=object)
    roles = np.array([r[3] for r in cell_rows], dtype=np.int64)
    idx_npz = outdir / f"patch_removal_{args.slide2}_top{args.top5}_cellindex.npz"
    np.savez(idx_npz, cell_index=cidx, obs_name=onames, group=groups, role=roles)
    log.info("Wrote %s (%d experiment cells; %d removed/masked)",
             idx_npz, cidx.size, int((roles == 1).sum()))
    with (outdir / f"patch_removal_{args.slide2}_top{args.top5}_removed_cells.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["group", "cell_index", "obs_name"])
        for gg, ci, on, rr in cell_rows:
            if rr == 1:
                w.writerow([gg, ci, on])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["plot1", "plot2", "both"], default="both")
    p.add_argument("--slide1", default="2-1")
    p.add_argument("--slide2", default="2-2")
    p.add_argument("--coord_key", default="spatial", choices=["spatial", "array_coords"])
    p.add_argument("--spot_size_cells", type=int, default=9)
    p.add_argument("--top_n", type=int, default=10, help="plot1b: top-N perturbations")
    p.add_argument("--top5", type=int, default=5, help="plot2: top-N perturbations")
    p.add_argument("--una_patch_size", type=int, default=800,
                   help="plot2: # cells in the random unassigned patch removed (<1000)")
    p.add_argument("--skip_1a", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir",
                   default="/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/spot_plots")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode in ("plot1", "both"):
        run_plot1(args)
    if args.mode in ("plot2", "both"):
        run_plot2(args)
    log.info("DONE")


if __name__ == "__main__":
    main()
