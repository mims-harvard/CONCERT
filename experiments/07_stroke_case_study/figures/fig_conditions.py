#!/usr/bin/env python
"""One counterfactual scenario per figure pair: what we struck, and how the slide responds.

  <scen>_slides   row 1 = which cells were perturbed; row 2 = predicted expression of an ICA
                  marker over the WHOLE slide
  <scen>_bars     evidence: induction inside the lesion, conditions against each other

WHY EXPRESSION, NOT DELTA. Earlier versions plotted the perturbed-minus-control difference, which
is exactly zero for any cell outside the effect's range -- so two thirds of the slide rendered as
one flat tone and looked like missing data. Unperturbed cells of course HAVE expression; they just
have no CHANGE. Row 2 therefore shows the predicted expression level itself, so every cell carries
a real value and the lesion reads as elevation above a living background.

MARKER. A single ICA (core) marker, not the pooled injury panel: the pooled panel is PIA-weighted
(Vim and Timp1 are rim genes, delta_PIA - delta_ICA = +1.10 and +1.02), so scoring with it reports
the rim compartment even when the question is about the core.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

LABEL_FS, TICK_FS, CBAR_FS = 19, 15, 14
PERT_C, REST_C, SAT_Q = "#b2182b", "#d9d9d9", 0.98
EXPR_CMAP = LinearSegmentedColormap.from_list(
    "expr", ["#f7f7f7", "#fdd0a2", "#fc8d59", "#d7301f", "#7f0000"])

SCEN = {
    "size":     dict(names=[f"size_{s:04d}" for s in (25, 50, 100, 200, 400)],
                     lab=lambda n: str(int(n.split("_")[1])), bx="Lesion size (spots)"),
    # The old "depth" pair was CTX vs TH -- two REGIONS, not two depths. Kept, relabelled, because
    # it answers a different and real question (does the injury program transfer to thalamus?).
    "region":   dict(names=["depth_surface_CTX", "depth_deep_TH"],
                     lab=lambda n: "Cortex" if "surface" in n else "Thalamus",
                     bx="Lesion region"),
}


def log1p_cpm(X, size=None):
    """Normalise to a FIXED library.

    Passing `size` from the control pass keeps the denominator identical across conditions. Using
    each pass's own library instead makes the perturbation's +20.1% count increase inside the
    lesion depress every gene that did not itself respond -- which is what made Gm42418 print
    negative (raw delta -0.02, CPM delta -0.23) and drove the negative inductions.
    """
    s = X.sum(1, keepdims=True) if size is None else size
    s = np.where(s == 0, 1.0, s)
    return np.log1p(X / s * 1e4)


def two_dec(cb):
    cb.ax.set_yticklabels([f"{t:.2f}" for t in cb.get_ticks()])


ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True); ap.add_argument("--arm", default="2dpara")
ap.add_argument("--gene", default="Gm42418", help="a single ICA-specific marker")
a = ap.parse_args()
out = Path(a.outdir); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
paper_style.use()

man = json.loads((out / "pert_sets" / "manifest.json").read_text())
off, n_slide = man["offset"], man["n_slide"]
rows = slice(off, off + n_slide)
genes = np.loadtxt(out / "hvg_gene_names.txt", dtype=str)
gi = int(np.where(genes == a.gene)[0][0])


CTRL_SIZE = None


def load(sample):
    hits = sorted(out.glob(f"{sample}_stroke_*_perturbed_counts.h5ad"))
    if not hits:
        return None
    X = np.asarray(ad.read_h5ad(hits[-1]).X[rows], np.float32)
    return log1p_cpm(X, CTRL_SIZE)[:, gi]


_h = sorted(out.glob("demo_CONTROL_stroke_*_perturbed_counts.h5ad"))
CTRL_SIZE = np.asarray(ad.read_h5ad(_h[-1]).X[rows], np.float32).sum(1, keepdims=True)
base = load("demo_CONTROL")
tgt = next(p for p in man["files"] if man["target_slide"] in Path(p).stem)
with h5py.File(tgt, "r") as f:
    xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)
lo, hi = xy.min(0), xy.max(0)
xyn = (xy - lo) / np.maximum(hi - lo, 1e-8)
stats = {"arm": a.arm, "gene": a.gene, "target_slide": man["target_slide"],
         "no_ground_truth": True, "scenarios": {}}

for scen, cfg in SCEN.items():
    recs = {}
    for n in cfg["names"]:
        e = load(f"demo_{n}")
        f_ = out / "pert_sets" / f"{n}.txt"
        if e is None or not f_.exists():
            continue
        idx = np.loadtxt(f_, dtype=np.int64) - 1 - off
        m = np.zeros(n_slide, bool); m[idx] = True
        recs[n] = dict(expr=e, delta=e - base, mask=m,
                       induction=float((e - base)[m].mean()), n_spots=int(m.sum()))
    names = [n for n in cfg["names"] if n in recs]
    if not names:
        print(f"[{scen}] no data, skipped"); continue

    alld = np.concatenate([recs[n]["delta"] for n in names])
    vlo, vhi = 0.0, float(np.quantile(alld, SAT_Q))
    ncol = len(names)
    fig, ax = plt.subplots(2, ncol, figsize=(2.5 * ncol + 1.8, 5.5), squeeze=False)
    pcmap = ListedColormap([REST_C, PERT_C])
    for c, n in enumerate(names):
        r = recs[n]
        ax[0, c].scatter(xyn[:, 0], xyn[:, 1], c=r["mask"].astype(int), s=3.4, lw=0,
                         cmap=pcmap, vmin=0, vmax=1, rasterized=True)
        h = ax[1, c].scatter(xyn[:, 0], xyn[:, 1], c=r["delta"], s=3.4, lw=0,
                             cmap=EXPR_CMAP, vmin=vlo, vmax=vhi, rasterized=True)
        # no outline overlay here: the rings were being read as a ring of expression
        for rr in (0, 1):
            ax[rr, c].set_xticks([]); ax[rr, c].set_yticks([]); ax[rr, c].set_aspect("equal")
            ax[rr, c].grid(False)
            for sp in ax[rr, c].spines.values():
                sp.set_visible(False)
        ax[1, c].set_xlabel(cfg["lab"](n), fontsize=LABEL_FS)
    ax[0, 0].set_ylabel("Perturbed cells", fontsize=LABEL_FS)
    ax[1, 0].set_ylabel(f"{a.gene} induction", fontsize=LABEL_FS)
    fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.115, right=0.875, top=0.99, bottom=0.11)
    po = ax[1, -1].get_position(); pad = 0.10 * po.height
    cax = fig.add_axes([0.888, po.y0 + pad, 0.013, po.height - 2 * pad])
    cb = fig.colorbar(h, cax=cax)
    cb.set_label(f"{a.gene} induction", fontsize=CBAR_FS - 2, labelpad=3)
    cb.set_ticks([vlo, vhi]); two_dec(cb); cb.ax.tick_params(labelsize=CBAR_FS - 4)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig_demo_{scen}_slides_{a.arm}.{ext}", dpi=300)
    plt.close(fig)

    vals = [recs[n]["induction"] for n in names]
    figb, axb = plt.subplots(figsize=(1.05 * len(names) + 2.4, 3.9))
    x = np.arange(len(names))
    axb.bar(x, vals, 0.52, color=paper_style.ACCENT)
    axb.set_xticks(x); axb.set_xticklabels([cfg["lab"](n) for n in names], fontsize=LABEL_FS - 3)
    span = max(vals) - min(vals)
    y0 = min(vals) - (0.30 * span if span > 0 else 0.1 * abs(min(vals)) + 0.01)
    y1 = max(vals) + (0.12 * span if span > 0 else 0.1 * abs(max(vals)) + 0.01)
    axb.set_ylim(y0, y1)
    axb.set_yticks([round(y0, 2), round((y0 + y1) / 2, 2), round(y1, 2)])
    axb.set_yticklabels([f"{t:.2f}" for t in axb.get_yticks()])
    axb.tick_params(labelsize=TICK_FS)
    axb.set_ylabel(f"Mean {a.gene} induction", fontsize=LABEL_FS - 3)
    axb.set_xlabel(cfg["bx"], fontsize=LABEL_FS)
    axb.grid(False)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    figb.subplots_adjust(left=0.22, right=0.98, top=0.97, bottom=0.22)
    for ext in ("png", "pdf"):
        figb.savefig(figdir / f"fig_demo_{scen}_bars_{a.arm}.{ext}", dpi=300)
    plt.close(figb)

    stats["scenarios"][scen] = dict(
        delta_colour_range=[vlo, vhi], bar_axis_truncated=True, bar_axis_range=[y0, y1],
        conditions={cfg["lab"](n): dict(induction=recs[n]["induction"],
                                        n_spots=recs[n]["n_spots"]) for n in names})
    print(f"[{scen}] " + "  ".join(f"{cfg['lab'](n)}={recs[n]['induction']:.3f}" for n in names))

(out / f"figure_stats_conditions_{a.arm}.json").write_text(json.dumps(stats, indent=2))
print(f"figures -> {figdir}")
