#!/usr/bin/env python
"""Per-GENE field figure: the same gene on the observed and the predicted row.

A lesion marker carries its own spatial profile -- a core gene (Gm42418, Scgb1a1) peaks in the
infarct core and falls off at the rim, a rim gene (Vim, Lgals1, Timp1) does the opposite. Pooling
a mixed panel hides that: the earlier figure averaged five genes of which three peak at the rim,
so the observed row showed a rim and the prediction a core, and the two rows were never showing
the same quantity. Here ONE gene is plotted on both rows, and the question is whether the
prediction reproduces THAT GENE's own core-vs-rim profile.

Four rows per figure:
  1. observed gene on the stroke mouse            3. observed lesion regions (core / rim)
  2. predicted gene on the sham mouse             4. predicted lesion region (core / rim)

The predicted region is defined ONCE from the pooled injury score (so it does not move from gene
to gene), thresholded at the rate that matches the real lesion's prevalence. Its core/rim split
uses distance to the region boundary, with the cut placed so the rim:core spot ratio matches the
observed PIA:ICA ratio -- matching the effective rate rather than a nominal percentile, so
"core" and "rim" mean the same thing on both sides.

ICA/PIA labels come from the delivery h5 (`Celltype_coarse`); prep rewrote perturbed spots'
tissue to CTX, so they are not in the slide files.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")
POOLED = ("Lcn2", "Gfap", "Vim", "Timp1", "Cd68")
CORE_C, RIM_C, OTHER_C = "#b2182b", "#ef8a62", "#dcdcdc"
LABEL_FS, TICK_FS, CBAR_FS = 17, 14, 13


def _obs_str(f_obs, col):
    o = f_obs[col]
    if isinstance(o, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]])
        return cats[o["codes"][:]]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in o[:]])


def log1p_cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def stack_norm(xy):
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / np.maximum(hi - lo, 1e-8)


def load_stack(slide_dir: Path, prefix: str, stack: str):
    paths = sorted(slide_dir.glob(f"{prefix}_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in paths], label="sf", keys=[p.stem for p in paths],
                  index_unique=None)
    sec = np.unique(np.asarray(A.obsm["section_index"]).reshape(-1), return_inverse=True)[1]
    return A, np.asarray(A.obsm["array_coords"], np.float32), sec


def boundary_distance(xy, inside):
    """Distance from each inside spot to the nearest outside spot -- depth within the region."""
    from scipy.spatial import cKDTree
    if (~inside).sum() == 0 or inside.sum() == 0:
        return np.zeros(len(xy))
    d = np.zeros(len(xy))
    d[inside] = cKDTree(xy[~inside]).query(xy[inside])[0]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--arm", default="3d")
    ap.add_argument("--prefix", default="stroke_paste2")
    ap.add_argument("--shape_dir", default="",
                    help="use an IMPOSED lesion (shape_{lesion,core,rim}_local.txt) as the "
                         "predicted region instead of thresholding the model's own field")
    ap.add_argument("--on_glob", default="",
                    help="override which counterfactual is the ON arm (e.g. shape_on)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sections", default="",
                    help="comma list of 1-based sections to draw (default all). The 2D arm sees\n"
                         "each slide independently, so drawing four sections there invites the\n"
                         "reader to compare across z that the 2D model never had access to.")
    ap.add_argument("--genes", default="Gm42418,Vim",
                    help="one figure per gene; pick a core-peaking and a rim-peaking one")
    a = ap.parse_args()
    out = Path(a.outdir); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    paper_style.use()

    # predictions (stroke ON vs matched control) over the Sham spots
    on_pat = a.on_glob or f"stroke_{a.arm}_on_*"
    P_on = ad.read_h5ad(next(out.glob(f"{on_pat}_perturbed_counts.h5ad")))
    P_off = ad.read_h5ad(next(out.glob(f"stroke_{a.arm}_off_*_perturbed_counts.h5ad")))
    genes_pred = np.loadtxt(out / "hvg_gene_names.txt", dtype=str)
    side = json.loads((out / "sham_cells.json").read_text())
    if P_on.n_obs != side["n_total_expected"]:
        raise SystemExit(f"row guard failed: {P_on.n_obs} vs {side['n_total_expected']}")
    sham_rows = np.concatenate([np.arange(o, o + c) for o, c, t in zip(
        np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
    Lon = log1p_cpm(np.asarray(P_on.X[sham_rows], np.float32))
    Loff = log1p_cpm(np.asarray(P_off.X[sham_rows], np.float32))

    A_sh, xy_sh, sec_sh = load_stack(Path(a.slides), a.prefix, "Sham")
    A_pt, xy_pt, sec_pt = load_stack(Path(a.slides), a.prefix, "PT")
    n_sh, n_pt_norm = stack_norm(xy_sh), stack_norm(xy_pt)
    Lpt = log1p_cpm(np.asarray(A_pt.X.todense(), np.float32))
    genes_slide = np.asarray(A_pt.var_names)

    # observed ICA / PIA labels, recovered from the delivery h5
    with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
        bc = np.array([b.decode() for b in f["Barcode"][:]])
        region_all = np.array([b.decode() for b in f["Celltype_coarse"][:]])
    lut = pd.Series(region_all, index=bc)
    reg_pt = lut.loc[np.asarray(A_pt.obs_names)].to_numpy()
    obs_core, obs_rim = reg_pt == "ICA", reg_pt == "PIA"
    keep_sec = {int(v) - 1 for v in a.sections.split(",") if v.strip()}
    if keep_sec:
        # Restrict the OBSERVED side to the same sections that are imposed and drawn, so the two
        # sides of the comparison cover the same tissue.
        in_sec = np.isin(sec_pt, sorted(keep_sec))
        obs_core, obs_rim = obs_core & in_sec, obs_rim & in_sec
    rim_core_ratio = obs_rim.sum() / max(obs_core.sum(), 1)

    # predicted region: fixed across genes, from the pooled injury score
    pidx = [int(np.where(genes_pred == g)[0][0]) for g in POOLED if g in set(genes_pred.tolist())]
    pooled_delta = (Lon[:, pidx] - Loff[:, pidx]).mean(1)
    lesion_rate = float((obs_core | obs_rim).mean())
    if a.shape_dir:
        # IMPOSED lesion: the perturbed spots ARE the region, and its core/rim come from the real
        # ICA/PIA masks transferred onto this brain -- no threshold, no geometric proxy.
        sd = Path(a.shape_dir)
        mk = lambda f: np.isin(np.arange(len(xy_sh)), np.loadtxt(sd / f, dtype=np.int64))
        pred_in, pred_core, pred_rim = mk("shape_lesion_local.txt"), mk("shape_core_local.txt"), mk("shape_rim_local.txt")
        pred_core, pred_rim = pred_core & pred_in, pred_rim & pred_in
    else:
        pred_in = pooled_delta > np.quantile(pooled_delta, 1.0 - lesion_rate)
        # core = deepest spots of the predicted region; rim:core ratio matched to observed PIA:ICA
        depth = boundary_distance(xy_sh, pred_in)
        n_core = int(round(pred_in.sum() / (1.0 + rim_core_ratio)))
        thr = np.sort(depth[pred_in])[::-1][max(n_core - 1, 0)]
        pred_core = pred_in & (depth >= thr)
        pred_rim = pred_in & ~pred_core

    stats = dict(arm=a.arm, sections=sorted(x + 1 for x in keep_sec) or "all",
                 observed_ICA=int(obs_core.sum()), observed_PIA=int(obs_rim.sum()),
                 rim_core_ratio_observed=float(rim_core_ratio),
                 predicted_region=int(pred_in.sum()), predicted_core=int(pred_core.sum()),
                 predicted_rim=int(pred_rim.sum()),
                 rim_core_ratio_predicted=float(pred_rim.sum() / max(pred_core.sum(), 1)),
                 lesion_rate=lesion_rate, genes={})

    nsec_all = int(sec_pt.max()) + 1
    keep = sorted(keep_sec) if keep_sec else list(range(nsec_all))
    nsec = len(keep)
    for gene in [g.strip() for g in a.genes.split(",") if g.strip()]:
        gi_s = np.where(genes_slide == gene)[0]
        gi_p = np.where(genes_pred == gene)[0]
        if not len(gi_s) or not len(gi_p):
            print(f"[skip] {gene}: slide={len(gi_s)} model={len(gi_p)}")
            continue
        obs_v = Lpt[:, int(gi_s[0])]
        pred_v = (Lon[:, int(gi_p[0])] - Loff[:, int(gi_p[0])])

        g_stat = dict(
            observed_core=float(obs_v[obs_core].mean()), observed_rim=float(obs_v[obs_rim].mean()),
            observed_elsewhere=float(obs_v[~(obs_core | obs_rim)].mean()),
            predicted_core=float(pred_v[pred_core].mean()), predicted_rim=float(pred_v[pred_rim].mean()),
            predicted_elsewhere=float(pred_v[~pred_in].mean()))
        g_stat["observed_core_minus_rim"] = g_stat["observed_core"] - g_stat["observed_rim"]
        g_stat["predicted_core_minus_rim"] = g_stat["predicted_core"] - g_stat["predicted_rim"]
        g_stat["profile_sign_agrees"] = bool(
            np.sign(g_stat["observed_core_minus_rim"]) == np.sign(g_stat["predicted_core_minus_rim"]))
        stats["genes"][gene] = g_stat

        fig, axes = plt.subplots(4, nsec, figsize=(2.45 * nsec + 1.9, 10.4), squeeze=False)
        vo = np.quantile(obs_v, [0.03, 0.97]); vp = np.quantile(pred_v, [0.03, 0.97])
        reg_cmap = ListedColormap([OTHER_C, RIM_C, CORE_C])
        obs_reg = np.where(obs_core, 2, np.where(obs_rim, 1, 0))
        pred_reg = np.where(pred_core, 2, np.where(pred_rim, 1, 0))
        for ci, s in enumerate(keep):
            mp, ms = sec_pt == s, sec_sh == s
            h0 = axes[0, ci].scatter(n_pt_norm[mp, 0], n_pt_norm[mp, 1], c=obs_v[mp], s=2.6, lw=0,
                                    cmap=paper_style.FIELD_CMAP, vmin=vo[0], vmax=vo[1], rasterized=True)
            h1 = axes[1, ci].scatter(n_sh[ms, 0], n_sh[ms, 1], c=pred_v[ms], s=2.6, lw=0,
                                    cmap=paper_style.FIELD_CMAP, vmin=vp[0], vmax=vp[1], rasterized=True)
            axes[2, ci].scatter(n_pt_norm[mp, 0], n_pt_norm[mp, 1], c=obs_reg[mp], s=2.6, lw=0,
                               cmap=reg_cmap, vmin=0, vmax=2, rasterized=True)
            axes[3, ci].scatter(n_sh[ms, 0], n_sh[ms, 1], c=pred_reg[ms], s=2.6, lw=0,
                               cmap=reg_cmap, vmin=0, vmax=2, rasterized=True)
            for r in range(4):
                ax = axes[r, ci]
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
            axes[3, ci].set_xlabel(f"Section {s + 1}", fontsize=LABEL_FS)
        # Short row labels: a two-line label at LABEL_FS is TALLER than one row of this grid and
        # spills into the row above. The gene name is carried by the colorbar labels instead.
        for r, lab in enumerate(("Observed\n(stroke)", "Predicted\n(sham)",
                                 "Lesion\n(stroke)", "Lesion\n(sham)")):
            axes[r, 0].set_ylabel(lab, fontsize=LABEL_FS - 2, labelpad=6)
        fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.10, right=0.87, top=0.995, bottom=0.075)
        for h, row, lab in ((h0, 0, f"Observed {gene}"), (h1, 1, f"Predicted {gene} induction")):
            pos = axes[row, -1].get_position(); pad = 0.12 * pos.height
            cax = fig.add_axes([0.885, pos.y0 + pad, 0.014, pos.height - 2 * pad])
            cb = fig.colorbar(h, cax=cax); cb.set_label(lab, fontsize=CBAR_FS, labelpad=2)
            cb.ax.tick_params(labelsize=TICK_FS - 3); cb.set_ticks([cb.vmin, cb.vmax])
        pos = axes[3, -1].get_position()
        fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, mfc=c, mec="none", label=l)
                            for c, l in ((CORE_C, "Core"), (RIM_C, "Peri-lesion"), (OTHER_C, "Unaffected"))],
                   loc="center left", bbox_to_anchor=(0.883, (pos.y0 + pos.y1) / 2),
                   fontsize=CBAR_FS, frameon=False, handletextpad=0.4, labelspacing=0.5)
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"fig_stroke_gene_{gene}_{a.arm}{a.tag}.{ext}", dpi=300)
        plt.close(fig)
        print(f"[{gene}] observed core {g_stat['observed_core']:.3f} rim {g_stat['observed_rim']:.3f} "
              f"(core-rim {g_stat['observed_core_minus_rim']:+.3f}) | predicted core "
              f"{g_stat['predicted_core']:.3f} rim {g_stat['predicted_rim']:.3f} "
              f"(core-rim {g_stat['predicted_core_minus_rim']:+.3f}) | sign agrees: "
              f"{g_stat['profile_sign_agrees']}")

    (out / f"figure_stats_genes_{a.arm}{a.tag}.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nobserved ICA {stats['observed_ICA']} / PIA {stats['observed_PIA']} "
          f"(rim:core {rim_core_ratio:.2f}) | predicted region {stats['predicted_region']} "
          f"= core {stats['predicted_core']} + rim {stats['predicted_rim']} "
          f"(rim:core {stats['rim_core_ratio_predicted']:.2f})")
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
