#!/usr/bin/env python
"""Spatial view of the FOUR single-region counterfactual conditions.

report_demos.py summarises these four as a bar chart (inside vs remote), which shows the
contrast but not the pattern -- you cannot see WHERE the model put the lesion or how the induced
marker field falls off around it. This is the missing spatial slide: one column per condition,
the imposed lesion on top and the marker response CONCERT predicts for it underneath.

Prediction-only by design. These are lesions that were never delivered -- there is no stroke in
thalamus, striatum or hypothalamus in this cohort -- so no observed comparator exists for three
of the four columns. The house rule about always showing ground truth beside a prediction does
not apply to a counterfactual with no counterpart; that is stated in the emitted caption.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

MARKERS = ("Lcn2", "Gfap", "Vim", "Timp1", "Cd68")
ORDER = ["region_CTX", "region_TH", "region_STR", "region_HY"]
PRETTY = {"region_CTX": "Cortex", "region_TH": "Thalamus",
          "region_STR": "Striatum", "region_HY": "Hypothalamus"}
OTHER_C, LESION_C = "#dcdcdc", "#b2182b"
LABEL_FS, CBAR_FS = 19, 15


def log1p_cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arm", default="2dpara")
    a = ap.parse_args()
    out = Path(a.outdir); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    paper_style.use()

    man = json.loads((out / "pert_sets" / "manifest.json").read_text())
    off, n_slide = man["offset"], man["n_slide"]
    rows = slice(off, off + n_slide)
    genes = np.loadtxt(out / "hvg_gene_names.txt", dtype=str)
    gidx = [int(np.where(genes == m)[0][0]) for m in MARKERS if m in set(genes.tolist())]

    def load(sample):
        hits = sorted(out.glob(f"{sample}_stroke_*_perturbed_counts.h5ad"))
        if not hits:
            raise SystemExit(f"missing counterfactual for {sample}")
        return log1p_cpm(np.asarray(ad.read_h5ad(hits[-1]).X[rows], np.float32))[:, gidx].mean(1)

    base = load("demo_CONTROL")
    tgt = next(p for p in man["files"] if man["target_slide"] in Path(p).stem)
    with h5py.File(tgt, "r") as f:
        xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)
    lo, hi = xy.min(0), xy.max(0)
    xyn = (xy - lo) / np.maximum(hi - lo, 1e-8)

    recs = {}
    for name in ORDER:
        d = load(f"demo_{name}") - base
        idx = np.loadtxt(out / "pert_sets" / f"{name}.txt", dtype=np.int64) - 1 - off
        m = np.zeros(n_slide, dtype=bool); m[idx] = True
        recs[name] = dict(delta=d, mask=m, inside=float(d[m].mean()),
                          remote=float(d[~m].mean()), n_spots=int(m.sum()))

    # one shared colour scale across the four columns: the comparison between conditions IS the
    # point, so per-panel rescaling would erase it. Robust 97th percentile, not the max.
    alld = np.concatenate([recs[n]["delta"] for n in ORDER])
    vlo, vhi = np.quantile(alld, [0.03, 0.97])
    sat = float(np.mean((alld < vlo) | (alld > vhi)))

    fig, axes = plt.subplots(3, len(ORDER), figsize=(2.6 * len(ORDER) + 1.5, 8.2),
                             gridspec_kw={"height_ratios": [1, 1, 0.72]})
    reg_cmap = ListedColormap([OTHER_C, LESION_C])
    for c, name in enumerate(ORDER):
        r = recs[name]
        axes[0, c].scatter(xyn[:, 0], xyn[:, 1], c=r["mask"].astype(int), s=3.4, lw=0,
                           cmap=reg_cmap, vmin=0, vmax=1, rasterized=True)
        h = axes[1, c].scatter(xyn[:, 0], xyn[:, 1], c=r["delta"], s=3.4, lw=0,
                               cmap=paper_style.FIELD_CMAP, vmin=vlo, vmax=vhi, rasterized=True)
        for rr in (0, 1):
            ax = axes[rr, c]
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        w = 0.5; xb = np.arange(2)
        axes[2, c].bar(xb, [r["inside"], r["remote"]], w,
                       color=[LESION_C, OTHER_C])
        axes[2, c].set_xticks(xb)
        axes[2, c].set_xticklabels(["Lesion", "Rest"], fontsize=LABEL_FS - 4)
        axes[2, c].tick_params(labelsize=LABEL_FS - 6)
        axes[2, c].set_ylim(0, max(r2["inside"] for r2 in recs.values()) * 1.15)
        axes[2, c].grid(False)
        for sp in ("top", "right"):
            axes[2, c].spines[sp].set_visible(False)
        if c:
            axes[2, c].set_yticklabels([])
        axes[2, c].set_xlabel(PRETTY[name], fontsize=LABEL_FS)
    axes[0, 0].set_ylabel("Imposed lesion", fontsize=LABEL_FS)
    axes[1, 0].set_ylabel("Predicted marker\ninduction", fontsize=LABEL_FS)
    fig.subplots_adjust(wspace=0.06, hspace=0.16, left=0.10, right=0.88, top=0.99, bottom=0.09)
    pos = axes[1, -1].get_position()
    pad = 0.10 * pos.height
    cax = fig.add_axes([0.895, pos.y0 + pad, 0.017, pos.height - 2 * pad])
    cb = fig.colorbar(h, cax=cax)
    cb.set_label("Predicted marker induction", fontsize=CBAR_FS, labelpad=2)
    cb.ax.tick_params(labelsize=CBAR_FS - 3)
    cb.set_ticks([cb.vmin, cb.vmax])
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig_demo_region_spatial_{a.arm}.{ext}", dpi=300)
    plt.close(fig)

    stats = {"arm": a.arm, "markers": [str(genes[i]) for i in gidx],
             "target_slide": man["target_slide"], "no_ground_truth": True,
             "shared_colour_scale": [float(vlo), float(vhi)],
             "fraction_saturated": sat,
             "conditions": {PRETTY[n]: {k: recs[n][k] for k in ("inside", "remote", "n_spots")}
                            for n in ORDER}}
    (out / f"figure_stats_region_spatial_{a.arm}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
