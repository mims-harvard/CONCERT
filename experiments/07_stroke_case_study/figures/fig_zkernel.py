#!/usr/bin/env python
"""2D kernel vs 3D kernel: the same lesion imposed at each z, read on the perturbed spots.

Response is the mean induction of the injury panel INSIDE the lesion, one value per section, once
per model. The 2D model has no z input, so its profile is the control: whatever it varies by comes
from that section's tissue and niche. The 3D model sees the identical footprint plus a z
coordinate, so the difference between the two profiles is the z term.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import anndata as ad, h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/CONCERT/codebase/"
                   "experiments/03_dss_gut_case_study/figures")
import paper_style

PANEL = ("Lcn2", "Gfap", "Vim", "Timp1", "Cd68")
LABEL_FS, TICK_FS = 18, 14


def log1p_cpm(X, size):
    return np.log1p(X / np.where(size == 0, 1.0, size) * 1e4)


ap = argparse.ArgumentParser()
ap.add_argument("--out2d", required=True); ap.add_argument("--out3d", required=True)
ap.add_argument("--sets", required=True); ap.add_argument("--figdir", required=True)
ap.add_argument("--files", required=True)
a = ap.parse_args()
figdir = Path(a.figdir); figdir.mkdir(parents=True, exist_ok=True)
paper_style.use()

paths = [p.strip() for p in a.files.split(",") if p.strip()]
counts, off_sham, run = [], None, 0
for p in paths:
    with h5py.File(p, "r") as f:
        n = f["obs"]["_index"].shape[0]
        m = f["obs"]["mouse"]
        m = m["categories"][:][m["codes"][0]] if isinstance(m, h5py.Group) else m[0]
        m = m.decode() if isinstance(m, bytes) else str(m)
    if m == "Sham" and off_sham is None:
        off_sham = run
    counts.append(n); run += n
offsets = np.concatenate([[0], np.cumsum(counts)])

man = json.loads((Path(a.sets) / "manifest_zslice.json").read_text())
nsec = len(man["sets"])
genes = np.loadtxt(Path(a.out3d) / "hvg_gene_names.txt", dtype=str)
gi = [int(np.where(genes == g)[0][0]) for g in PANEL if g in set(genes.tolist())]

res = {}
for arm, out in (("2d", Path(a.out2d)), ("3d", Path(a.out3d))):
    off = ad.read_h5ad(next(out.glob("zk_CONTROL_*_perturbed_counts.h5ad")))
    Xoff = np.asarray(off.X, np.float32)
    size = Xoff.sum(1, keepdims=True)
    Loff = log1p_cpm(Xoff, size)[:, gi]
    vals = []
    for s in range(1, nsec + 1):
        on = ad.read_h5ad(next(out.glob(f"zk_zslice_s{s}_*_perturbed_counts.h5ad")))
        Lon = log1p_cpm(np.asarray(on.X, np.float32), size)[:, gi]
        idx = np.loadtxt(Path(a.sets) / f"zslice_s{s}.txt", dtype=np.int64) - 1
        vals.append(float((Lon - Loff)[idx].mean()))
    res[arm] = vals

stats = dict(panel=list(PANEL), source_section=man["source_section"],
             n_footprint=man["n_footprint"],
             per_section={f"section_{s+1}": dict(model_2d=res["2d"][s], model_3d=res["3d"][s],
                                                 n_spots=man["sets"][s]["n_spots"])
                          for s in range(nsec)},
             spread_2d=float(np.ptp(res["2d"])), spread_3d=float(np.ptp(res["3d"])))

fig, ax = plt.subplots(figsize=(1.3 * nsec + 2.8, 3.9))
x = np.arange(nsec); w = 0.38
ax.bar(x - w / 2, res["2d"], w, color="#bdbdbd", label="2D kernel")
ax.bar(x + w / 2, res["3d"], w, color=paper_style.ACCENT, label="3D kernel")
ax.set_xticks(x); ax.set_xticklabels([str(s + 1) for s in range(nsec)], fontsize=TICK_FS)
ax.set_xlabel("Section the lesion was imposed on", fontsize=LABEL_FS - 3)
ax.set_ylabel("Injury response\ninside the lesion", fontsize=LABEL_FS - 4)
ax.tick_params(labelsize=TICK_FS); ax.grid(False); ax.axhline(0, color="#333333", lw=0.8)
ax.legend(fontsize=TICK_FS - 1, frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.24, right=0.98, top=0.97, bottom=0.20)
for ext in ("png", "pdf"):
    fig.savefig(figdir / f"fig_zkernel_bars.{ext}", dpi=300)
plt.close(fig)

(figdir.parent / "figure_stats_zkernel.json").write_text(json.dumps(stats, indent=2) + "\n")
for s in range(nsec):
    print(f"section {s+1}  2D={res['2d'][s]:+.4f}  3D={res['3d'][s]:+.4f}")
print(f"across-section spread  2D={stats['spread_2d']:.4f}  3D={stats['spread_3d']:.4f}")
print(f"figures -> {figdir}")
