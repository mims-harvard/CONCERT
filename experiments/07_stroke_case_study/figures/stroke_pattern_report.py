#!/usr/bin/env python
"""Does the stroke predicted onto the Sham brain land where the real lesion is?

That is the whole claim, and it is made three ways over the four sections:

  1. the field itself      -- observed PT injury-marker score beside the predicted Sham response;
  2. where it lands        -- spatial agreement of the two fields, binned into the shared
                              stack-normalised frame the model itself trains in;
  3. how much of it        -- perturbed fraction per section, predicted vs the real lesion.

The two predictions differ ONLY in the perturbation attribute (stroke vs unperturbed) through the
same decode path, so their difference is the model's stroke effect and not a reconstruction offset.

PT and Sham are different brains, so the fields are compared in the per-stack MinMax frame the
runner builds (`tile_multislide_coords`), which is the only frame in which "the same place" is
defined for them -- and it is the frame the footprint was learned in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Shared house style (repo copy is the source of truth) so this figure reads as one system with
# the rest of the CONCERT figure set.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

# Reactive-injury panel verified on this data in Stage 0c (PIA/ICA score +1.69 / +0.45 in PT).
# The registration benchmark's own marker list is [Lcn2, Gm42418]; Gm42418 is an rRNA-contamination
# pseudogene rather than an injury gene, so Lcn2 is also reported alone for comparability.
MARKERS = ("Lcn2", "Gfap", "Vim", "Timp1", "Cd68")
FIELD_CMAP = paper_style.FIELD_CMAP
LABEL_FS, TICK_FS, CBAR_FS = 18, 15, 14


def _obs_str(f_obs, col):
    o = f_obs[col]
    if isinstance(o, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]])
        return cats[o["codes"][:]]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in o[:]])


def load_slides(slide_dir: Path, arm_prefix: str, stack: str):
    """Per-cell counts, coords, section index and lesion label for one mouse, sections in order."""
    paths = sorted(slide_dir.glob(f"{arm_prefix}_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in paths], label="slide_file",
                  keys=[p.stem for p in paths], index_unique=None)
    xy = np.asarray(A.obsm["array_coords"], dtype=np.float32)
    sec = np.asarray(A.obsm["section_index"], dtype=np.float32).reshape(-1)
    sec = np.unique(sec, return_inverse=True)[1]          # -> 0..3 whatever the stored codes are
    return A, xy, sec


def stack_norm(xy: np.ndarray) -> np.ndarray:
    """MinMax the whole stack jointly to [0,1]^2 -- the runner's per-stack tiling frame."""
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / np.maximum(hi - lo, 1e-8)


def log1p_cpm(X: np.ndarray) -> np.ndarray:
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def marker_score(X: np.ndarray, genes: np.ndarray, markers) -> np.ndarray:
    """Mean z-scored log1p-CP10K over the marker panel (one value per spot)."""
    idx = [int(np.where(genes == m)[0][0]) for m in markers if m in set(genes.tolist())]
    if not idx:
        raise ValueError(f"none of {markers} present among {len(genes)} genes")
    L = log1p_cpm(X)[:, idx]
    return ((L - L.mean(0)) / np.maximum(L.std(0), 1e-8)).mean(1)


def binned_agreement(xy_a, val_a, xy_b, val_b, nbins=24, min_n=5):
    """Mean of each field per shared-frame bin, then correlate over bins occupied by both."""
    edges = np.linspace(0, 1, nbins + 1)
    def grid(xy, v):
        ix = np.clip(np.digitize(xy[:, 0], edges) - 1, 0, nbins - 1)
        iy = np.clip(np.digitize(xy[:, 1], edges) - 1, 0, nbins - 1)
        k = ix * nbins + iy
        s = np.bincount(k, weights=v, minlength=nbins * nbins)
        n = np.bincount(k, minlength=nbins * nbins)
        return np.where(n >= min_n, s / np.maximum(n, 1), np.nan)
    ga, gb = grid(xy_a, val_a), grid(xy_b, val_b)
    m = ~np.isnan(ga) & ~np.isnan(gb)
    if m.sum() < 10:
        return np.nan, np.nan, int(m.sum())
    return float(pearsonr(ga[m], gb[m])[0]), float(spearmanr(ga[m], gb[m])[0]), int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="model outdir, e.g. .../outputs/stroke_3d")
    ap.add_argument("--slides", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--arm_prefix", default="stroke_paste2")
    ap.add_argument("--nbins", type=int, default=24)
    a = ap.parse_args()

    paper_style.use()
    out = Path(a.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # predictions: stroke ON vs OFF over the same Sham cells
    on = next(out.glob(f"stroke_{a.arm}_on_*_perturbed_counts.h5ad"))
    off = next(out.glob(f"stroke_{a.arm}_off_*_perturbed_counts.h5ad"))
    P_on, P_off = ad.read_h5ad(on), ad.read_h5ad(off)
    genes_pred = np.loadtxt(out / "hvg_gene_names.txt", dtype=str)

    side = json.loads((out / "sham_cells.json").read_text())
    if P_on.n_obs != side["n_total_expected"]:
        raise SystemExit(
            f"row-order guard FAILED: prediction has {P_on.n_obs} rows but the slides hold "
            f"{side['n_total_expected']}; the runner dropped cells, so the Sham indices in "
            f"sham_cells.txt no longer address the intended spots. Fix before reading anything.")
    sham_rows = np.concatenate([np.arange(o, o + c) for o, c, t in zip(
        np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])

    Xon = np.asarray(P_on.X[sham_rows], dtype=np.float32)
    Xoff = np.asarray(P_off.X[sham_rows], dtype=np.float32)

    def pred_delta(markers):
        idx = [int(np.where(genes_pred == m)[0][0]) for m in markers if m in set(genes_pred.tolist())]
        if not idx:
            raise ValueError(f"none of {markers} among the {len(genes_pred)} modelled genes")
        return (log1p_cpm(Xon)[:, idx] - log1p_cpm(Xoff)[:, idx]).mean(1), [genes_pred[i] for i in idx]

    delta, used = pred_delta(MARKERS)
    delta_lcn2, _ = pred_delta(("Lcn2",))

    # observed sides
    A_sham, xy_sham, sec_sham = load_slides(Path(a.slides), a.arm_prefix, "Sham")
    A_pt, xy_pt, sec_pt = load_slides(Path(a.slides), a.arm_prefix, "PT")
    if A_sham.n_obs != len(delta):
        raise SystemExit(f"Sham slides hold {A_sham.n_obs} spots but {len(delta)} were predicted")

    genes_slide = np.asarray(A_pt.var_names)
    pt_score = marker_score(np.asarray(A_pt.X.todense(), dtype=np.float32), genes_slide, MARKERS)
    pt_lesion = (np.asarray(A_pt.obs["perturbation"]) == "stroke").astype(float)

    n_sham = stack_norm(xy_sham)
    n_pt = stack_norm(xy_pt)

    # perturbed fraction per section, on matched effective rates
    # The observed side is an annotation, so its rate is fixed; the predicted side is a continuous
    # response with no natural cutoff. Setting the cutoff so the POOLED predicted rate equals the
    # pooled real lesion rate makes the two profiles comparable in shape, which is the claim --
    # any other cutoff would just rescale every section by the same factor.
    pooled_rate = float(pt_lesion.mean())
    cutoff = float(np.quantile(delta, 1.0 - pooled_rate))
    pred_pos = delta > cutoff

    rows = []
    for s in range(int(sec_sham.max()) + 1):
        ms, mp = sec_sham == s, sec_pt == s
        r_p, r_s, nb = binned_agreement(n_pt[mp], pt_score[mp], n_sham[ms], delta[ms], a.nbins)
        r_pl, r_sl, _ = binned_agreement(n_pt[mp], pt_lesion[mp], n_sham[ms], delta[ms], a.nbins)
        rows.append(dict(
            section=s + 1,
            n_pt=int(mp.sum()), n_sham=int(ms.sum()),
            observed_lesion_frac=float(pt_lesion[mp].mean()),
            predicted_pert_frac=float(pred_pos[ms].mean()),
            predicted_delta_mean=float(delta[ms].mean()),
            predicted_delta_mean_Lcn2=float(delta_lcn2[ms].mean()),
            observed_marker_mean=float(pt_score[mp].mean()),
            r_pearson_field=r_p, r_spearman_field=r_s,
            r_pearson_lesion=r_pl, r_spearman_lesion=r_sl, n_bins=nb))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"stroke_pattern_{a.arm}.csv", index=False)

    prof_r = float(pearsonr(tab.observed_lesion_frac, tab.predicted_pert_frac)[0])
    r_all_p, r_all_s, nb_all = binned_agreement(n_pt, pt_score, n_sham, delta, a.nbins)

    # --- figure: observed PT field (top) beside predicted Sham field (bottom) -
    nsec = len(tab)
    fig, axes = plt.subplots(2, nsec, figsize=(2.6 * nsec + 1.4, 5.6))
    v_pt = np.quantile(pt_score, [0.03, 0.97])
    v_sh = np.quantile(delta, [0.03, 0.97])
    for s in range(nsec):
        mp, ms = sec_pt == s, sec_sham == s
        h0 = axes[0, s].scatter(n_pt[mp, 0], n_pt[mp, 1], c=pt_score[mp], s=3, lw=0,
                                cmap=FIELD_CMAP, vmin=v_pt[0], vmax=v_pt[1], rasterized=True)
        h1 = axes[1, s].scatter(n_sham[ms, 0], n_sham[ms, 1], c=delta[ms], s=3, lw=0,
                                cmap=FIELD_CMAP, vmin=v_sh[0], vmax=v_sh[1], rasterized=True)
        for r in (0, 1):
            # Not paper_style.bare(): it calls axis("off"), which would also hide the x-axis
            # label -- and the x-axis label IS the panel identity in this house style.
            ax = axes[r, s]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[1, s].set_xlabel(f"Section {s + 1}", fontsize=LABEL_FS)
    axes[0, 0].set_ylabel("Observed\n(stroke mouse)", fontsize=LABEL_FS)
    axes[1, 0].set_ylabel("Predicted\n(sham mouse)", fontsize=LABEL_FS)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.09, right=0.88, top=0.99, bottom=0.11)
    for h, ax_row, lab in ((h0, axes[0, -1], "Injury-marker score"),
                           (h1, axes[1, -1], "Predicted marker induction")):
        # Inset each bar inside its row so the upper bar's bottom tick cannot collide with the
        # lower bar's rotated axis label.
        pos = ax_row.get_position()
        pad = 0.12 * pos.height
        cax = fig.add_axes([0.895, pos.y0 + pad, 0.016, pos.height - 2 * pad])
        cb = fig.colorbar(h, cax=cax)
        cb.set_label(lab, fontsize=CBAR_FS, labelpad=2)
        cb.ax.tick_params(labelsize=TICK_FS - 3)
        cb.set_ticks([cb.vmin, cb.vmax])
    fig.savefig(figdir / f"fig_stroke_field_{a.arm}.png", dpi=300)
    fig.savefig(figdir / f"fig_stroke_field_{a.arm}.pdf")
    plt.close(fig)

    # figure: perturbed fraction per section
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    w, x = 0.38, np.arange(nsec)
    ax.bar(x - w / 2, 100 * tab.observed_lesion_frac, w, color=paper_style.MUTED, label="Observed")
    ax.bar(x + w / 2, 100 * tab.predicted_pert_frac, w, color=paper_style.ACCENT, label="Predicted")
    ax.set_xticks(x); ax.set_xticklabels([f"Section {s}" for s in tab.section],
                                         fontsize=TICK_FS, rotation=30, ha="right")
    ax.set_ylabel("Perturbed spots (%)", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(fontsize=14, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(figdir / f"fig_stroke_proportion_{a.arm}.png", dpi=300)
    fig.savefig(figdir / f"fig_stroke_proportion_{a.arm}.pdf")
    plt.close(fig)

    stats = dict(arm=a.arm, markers_used=[str(g) for g in used], nbins=a.nbins,
                 pooled_lesion_rate=pooled_rate, predicted_cutoff=cutoff,
                 profile_pearson_across_sections=prof_r,
                 field_pearson_all_sections=r_all_p, field_spearman_all_sections=r_all_s,
                 n_bins_all_sections=nb_all, per_section=rows)
    (out / f"figure_stats_stroke_{a.arm}.json").write_text(json.dumps(stats, indent=2) + "\n")

    with pd.option_context("display.width", 200):
        print(tab.to_string(index=False))
    print(f"\nper-section proportion profile r = {prof_r:.3f}   "
          f"field agreement (all sections) r = {r_all_p:.3f} (rho = {r_all_s:.3f}, {nb_all} bins)")
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
