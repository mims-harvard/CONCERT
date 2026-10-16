#!/usr/bin/env python
"""Does the 3D kernel carry a lesion ACROSS sections, or does each section answer for itself?

The existing across-sections figure perturbed the cortex of all four sham sections, so the
response on section k is driven by section k's own perturbation. That cannot distinguish a 3D
kernel from four independent 2D ones. Here the lesion is imposed on SECTION 1 ONLY and the
response is read on sections 2-4, where not a single spot was perturbed:

  2D model (fpert_n_z_levels=1) -- no z coordinate in the footprint field. Sections 2-4 must be
      flat zero. That is the negative control, and it is built in, not fitted.
  3D model (fpert_n_z_levels=4) -- the footprint field spans z. A non-zero, decaying response on
      sections 2-4 is propagation through the kernel and nothing else.

Rows 1/2 of the map figure are the two models on the same spots and the same colour scale.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import anndata as ad, h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

PANEL = ("Lcn2", "Gfap", "Vim", "Timp1", "Cd68")
LABEL_FS, TICK_FS, CBAR_FS = 18, 14, 13
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "d", ["#f7f7f7", "#fdd0a2", "#fc8d59", "#d7301f", "#7f0000"])


def log1p_cpm(X, size=None):
    s = X.sum(1, keepdims=True) if size is None else size
    return np.log1p(X / np.where(s == 0, 1.0, s) * 1e4)


def arm_delta(out: Path, gi, sham_rows, ctrl_size):
    on = ad.read_h5ad(next(out.glob("zprop_on_*_perturbed_counts.h5ad")))
    off = ad.read_h5ad(next(out.glob("zprop_off_*_perturbed_counts.h5ad")))
    Xon = np.asarray(on.X[sham_rows], np.float32)
    Xoff = np.asarray(off.X[sham_rows], np.float32)
    size = Xoff.sum(1, keepdims=True) if ctrl_size is None else ctrl_size
    return (log1p_cpm(Xon, size)[:, gi] - log1p_cpm(Xoff, size)[:, gi]).mean(1), size


ap = argparse.ArgumentParser()
ap.add_argument("--out2d", required=True); ap.add_argument("--out3d", required=True)
ap.add_argument("--figdir", required=True)
ap.add_argument("--files", required=True)
ap.add_argument("--pert_set", required=True)
a = ap.parse_args()
figdir = Path(a.figdir); figdir.mkdir(parents=True, exist_ok=True)
paper_style.use()

paths = [p.strip() for p in a.files.split(",") if p.strip()]
counts, xy_l, sec_l, is_sham, off_sham = [], [], [], [], None
run = 0
for p in paths:
    with h5py.File(p, "r") as f:
        n = f["obs"]["_index"].shape[0]
        m = f["obs"]["mouse"]
        m = m["categories"][:][m["codes"][0]] if isinstance(m, h5py.Group) else m[0]
        m = m.decode() if isinstance(m, bytes) else str(m)
        if m == "Sham":
            if off_sham is None:
                off_sham = run
            xy_l.append(np.asarray(f["obsm"]["array_coords"][:], np.float32))
            sec_l.append(np.full(n, len(sec_l)))
    counts.append(n); is_sham.append(m == "Sham"); run += n
offsets = np.concatenate([[0], np.cumsum(counts)])
sham_rows = np.concatenate([np.arange(offsets[i], offsets[i + 1])
                            for i, t in enumerate(is_sham) if t])
xy = np.vstack(xy_l); sec = np.concatenate(sec_l)
lo, hi = xy.min(0), xy.max(0)
xyn = (xy - lo) / np.maximum(hi - lo, 1e-8)

pert_local = np.loadtxt(a.pert_set, dtype=np.int64) - 1 - off_sham
pmask = np.zeros(len(sham_rows), bool); pmask[pert_local] = True
genes = np.loadtxt(Path(a.out3d) / "hvg_gene_names.txt", dtype=str)
gi = [int(np.where(genes == g)[0][0]) for g in PANEL if g in set(genes.tolist())]

d2, ctrl = arm_delta(Path(a.out2d), gi, sham_rows, None)
d3, _ = arm_delta(Path(a.out3d), gi, sham_rows, ctrl)

nsec = int(sec.max()) + 1
stats = {"panel": PANEL, "n_perturbed_section1": int(pmask.sum()),
         "perturbed_sections": [1], "per_section": {}}
for s in range(nsec):
    m = (sec == s) & ~pmask                      # unperturbed spots only, every section
    stats["per_section"][f"section_{s+1}"] = dict(
        n_unperturbed=int(m.sum()), model_2d=float(d2[m].mean()), model_3d=float(d3[m].mean()))
off_only = [s for s in range(nsec) if s > 0]
stats["off_section_mean"] = dict(
    model_2d=float(np.mean([stats["per_section"][f"section_{s+1}"]["model_2d"] for s in off_only])),
    model_3d=float(np.mean([stats["per_section"][f"section_{s+1}"]["model_3d"] for s in off_only])))

# maps: both models, same scale
vhi = float(np.quantile(np.concatenate([d2, d3]), 0.995)); vlo = 0.0
fig, ax = plt.subplots(2, nsec, figsize=(2.5 * nsec + 1.8, 5.6), squeeze=False)
for c in range(nsec):
    m = sec == c
    for r, d in ((0, d2), (1, d3)):
        h = ax[r, c].scatter(xyn[m, 0], xyn[m, 1], c=d[m], s=3.2, lw=0, cmap=DELTA_CMAP,
                             vmin=vlo, vmax=vhi, rasterized=True)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([]); ax[r, c].set_aspect("equal")
        ax[r, c].grid(False)
        for sp in ax[r, c].spines.values():
            sp.set_visible(False)
    ax[1, c].set_xlabel(f"Section {c + 1}" + (" (lesion)" if c == 0 else ""), fontsize=LABEL_FS - 3)
ax[0, 0].set_ylabel("2D kernel", fontsize=LABEL_FS)
ax[1, 0].set_ylabel("3D kernel", fontsize=LABEL_FS)
fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.105, right=0.875, top=0.99, bottom=0.11)
po = ax[1, -1].get_position(); pad = 0.10 * po.height
cax = fig.add_axes([0.888, po.y0 + pad, 0.013, po.height - 2 * pad])
cb = fig.colorbar(h, cax=cax); cb.set_label("Predicted injury response", fontsize=CBAR_FS - 2)
cb.set_ticks([vlo, vhi]); cb.ax.set_yticklabels([f"{vlo:.2f}", f"{vhi:.2f}"])
cb.ax.tick_params(labelsize=CBAR_FS - 4)
for ext in ("png", "pdf"):
    fig.savefig(figdir / f"fig_zprop_maps.{ext}", dpi=300)
plt.close(fig)

# bars: response per section, unperturbed spots only
figb, axb = plt.subplots(figsize=(1.25 * nsec + 2.6, 3.9))
x = np.arange(nsec); w = 0.38
v_2d = [stats["per_section"][f"section_{s+1}"]["model_2d"] for s in range(nsec)]
v_3d = [stats["per_section"][f"section_{s+1}"]["model_3d"] for s in range(nsec)]
axb.bar(x - w / 2, v_2d, w, color="#bdbdbd", label="2D kernel")
axb.bar(x + w / 2, v_3d, w, color=paper_style.ACCENT, label="3D kernel")
axb.set_xticks(x); axb.set_xticklabels([f"{s+1}" for s in range(nsec)], fontsize=TICK_FS)
axb.set_xlabel("Section (lesion on section 1 only)", fontsize=LABEL_FS - 3)
axb.set_ylabel("Injury response,\nunperturbed spots", fontsize=LABEL_FS - 4)
axb.tick_params(labelsize=TICK_FS); axb.grid(False)
axb.axhline(0, color="#333333", lw=0.8)
axb.legend(fontsize=TICK_FS - 1, frameon=False)
for sp in ("top", "right"):
    axb.spines[sp].set_visible(False)
figb.subplots_adjust(left=0.24, right=0.98, top=0.97, bottom=0.20)
for ext in ("png", "pdf"):
    figb.savefig(figdir / f"fig_zprop_bars.{ext}", dpi=300)
plt.close(figb)

(Path(a.figdir).parent / "figure_stats_zprop.json").write_text(json.dumps(stats, indent=2) + "\n")
for s in range(nsec):
    d = stats["per_section"][f"section_{s+1}"]
    print(f"section {s+1}  n_unpert={d['n_unperturbed']:5d}  2D={d['model_2d']:+.4f}  "
          f"3D={d['model_3d']:+.4f}")
print(f"off-section mean  2D={stats['off_section_mean']['model_2d']:+.4f}  "
      f"3D={stats['off_section_mean']['model_3d']:+.4f}")
print(f"figures -> {figdir}")
