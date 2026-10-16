#!/usr/bin/env python
"""Decompose the rim: control field vs delta vs perturbed field, on the figure's own colour scale.

Row per condition, columns:
  1 mask        2 CONTROL predicted   3 delta (pert-ctrl)   4 PERTURBED predicted (figure panel)
Plus a fine radial profile in units of the lesion radius, and the in-mask relation between
delta and the control baseline (tests whether the effect is multiplicative on the baseline).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json
from pathlib import Path
import numpy as np, anndata as ad, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

OUT = Path(f"{_rp.WORKDIR}/outputs/stroke_2dpara")
GENE = "Scgb1a1"
EXPR = LinearSegmentedColormap.from_list("expr", ["#f7f7f7", "#fdd0a2", "#fc8d59", "#d7301f", "#7f0000"])

man = json.loads((OUT / "pert_sets" / "manifest.json").read_text())
off, n_slide = man["offset"], man["n_slide"]
rows = slice(off, off + n_slide)
genes = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
gi = int(np.where(genes == GENE)[0][0])
tgt = next(p for p in man["files"] if man["target_slide"] in Path(p).stem)
with h5py.File(tgt, "r") as f:
    xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)

raw = lambda s: np.asarray(ad.read_h5ad(sorted(OUT.glob(f"{s}_stroke_*_perturbed_counts.h5ad"))[-1]).X[rows], np.float32)
Xc = raw("demo_CONTROL"); CS = Xc.sum(1, keepdims=True)
cpm = lambda X: np.log1p(X / np.where(CS == 0, 1.0, CS) * 1e4)
base = cpm(Xc)[:, gi]

SIZES = (25, 50, 100, 200, 400, 800)
E = {s: cpm(raw(f"demo_size_{s:04d}"))[:, gi] for s in SIZES}
M = {}
for s in SIZES:
    idx = np.loadtxt(OUT / "pert_sets" / f"size_{s:04d}.txt", dtype=np.int64) - 1 - off
    m = np.zeros(n_slide, bool); m[idx] = True; M[s] = m

alle = np.concatenate([E[s] for s in SIZES])
vlo, vhi = float(np.quantile(alle, 0.02)), float(np.quantile(alle, 0.98))
print(f"figure colour range vlo={vlo:.4f} vhi={vhi:.4f}")

# fine radial profile in units of lesion radius
print("\nRADIAL PROFILE, r/R bins, MASKED spots only (so mask leakage cannot fake a ring)")
for s in SIZES:
    m = M[s]; cen = xy[m].mean(0)
    r = np.linalg.norm(xy - cen, axis=1); R = r[m].max()
    rr = r / R
    print(f"\n size {s}  R={R:.3f}  n={m.sum()}")
    print(f"   {'r/R':>10} {'n':>5} {'base':>8} {'delta':>8} {'pred':>8}")
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)]:
        sel = m & (rr >= lo) & (rr < hi)
        if sel.sum() < 3: continue
        print(f"   [{lo:.1f},{hi:.1f}) {sel.sum():5d} {base[sel].mean():8.4f} "
              f"{(E[s]-base)[sel].mean():8.4f} {E[s][sel].mean():8.4f}")
    for lo, hi in [(1.0, 1.3), (1.3, 1.8), (1.8, 3.0)]:
        sel = (~m) & (rr >= lo) & (rr < hi)
        if sel.sum() < 3: continue
        print(f"   [{lo:.1f},{hi:.1f}) {sel.sum():5d} {base[sel].mean():8.4f} "
              f"{(E[s]-base)[sel].mean():8.4f} {E[s][sel].mean():8.4f}   (unmasked)")
    d_out = (E[s] - base)[~m]
    print(f"   unmasked spots: max|delta|={np.abs(d_out).max():.2e}  n_nonzero={int((d_out!=0).sum())}")

# is the in-mask effect multiplicative on the baseline?
print("\nIN-MASK delta vs control baseline (multiplicative test)")
for s in SIZES:
    m = M[s]; b = base[m]; d = (E[s] - base)[m]
    q = np.quantile(b, [0, .25, .5, .75, 1.0])
    line = []
    for i in range(4):
        sel = (b >= q[i]) & (b <= q[i + 1])
        line.append(f"base~{b[sel].mean():.3f}->d={d[sel].mean():+.3f}")
    print(f" size {s:4d}  r(base,delta)={np.corrcoef(b, d)[0,1]:+.3f}   " + "  ".join(line))

# diagnostic figure
lo_, hi_ = xy.min(0), xy.max(0); xyn = (xy - lo_) / np.maximum(hi_ - lo_, 1e-8)
show = (100, 200, 400, 800)
fig, ax = plt.subplots(len(show), 4, figsize=(14, 3.2 * len(show)), squeeze=False)
for ri, s in enumerate(show):
    m = M[s]; d = E[s] - base
    dmax = float(np.quantile(np.abs(d), 0.999)) or 1.0
    panels = [(m.astype(float), ListedColormap(["#d9d9d9", "#b2182b"]), 0, 1, f"mask n={m.sum()}"),
              (base, EXPR, vlo, vhi, "CONTROL predicted"),
              (d, EXPR, 0, dmax, f"delta (0..{dmax:.2f})"),
              (E[s], EXPR, vlo, vhi, "PERTURBED predicted (figure)")]
    for ci, (v, cm, a_, b_, t) in enumerate(panels):
        ax[ri, ci].scatter(xyn[:, 0], xyn[:, 1], c=v, s=2.2, lw=0, cmap=cm, vmin=a_, vmax=b_, rasterized=True)
        ax[ri, ci].set_aspect("equal"); ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
        if ri == 0: ax[ri, ci].set_title(t, fontsize=11)
    ax[ri, 0].set_ylabel(f"size {s}", fontsize=12)
fig.tight_layout()
p = OUT / "figures" / "diag_rim_decomposition.png"
fig.savefig(p, dpi=170); print("\n->", p)
