#!/usr/bin/env python
"""Phase-1 evidence for the rim/ring seen around the perturbed disc in fig_demo_size_slides.

Question: is the bright ring (a) a real boundary effect the model produces, (b) baseline
anatomy revealed by the colour scale, or (c) a plotting artefact?

Measures, for each size condition, the radial profile of
   predicted expression   (what the figure paints)
   delta = pert - control (what the perturbation actually did)
binned on signed distance to the lesion boundary, plus the same for the raw model mean mu
(no CPM) so we can tell the normalisation apart from the model.
"""
import json, sys
from pathlib import Path
import numpy as np, anndata as ad, h5py

OUT = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/stroke_2dpara")
GENE = "Scgb1a1"

man = json.loads((OUT / "pert_sets" / "manifest.json").read_text())
off, n_slide = man["offset"], man["n_slide"]
rows = slice(off, off + n_slide)
genes = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
gi = int(np.where(genes == GENE)[0][0])

tgt = next(p for p in man["files"] if man["target_slide"] in Path(p).stem)
with h5py.File(tgt, "r") as f:
    xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)
print("slide spots", xy.shape, "n_slide", n_slide, "offset", off)


def raw(sample):
    hits = sorted(OUT.glob(f"{sample}_stroke_*_perturbed_counts.h5ad"))
    if not hits:
        return None
    return np.asarray(ad.read_h5ad(hits[-1]).X[rows], np.float32)


def cpm(X, size):
    s = np.where(size == 0, 1.0, size)
    return np.log1p(X / s * 1e4)


Xc = raw("demo_CONTROL")
CS = Xc.sum(1, keepdims=True)
base_cpm = cpm(Xc, CS)[:, gi]
base_raw = Xc[:, gi]

res = {"gene": GENE, "conditions": {}}
EDGES = np.array([-1e9, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 1e9], float)

for s in (25, 50, 100, 200, 400, 800):
    name = f"size_{s:04d}"
    Xp = raw(f"demo_{name}")
    f_ = OUT / "pert_sets" / f"{name}.txt"
    if Xp is None or not f_.exists():
        continue
    idx = np.loadtxt(f_, dtype=np.int64) - 1 - off
    m = np.zeros(n_slide, bool); m[idx] = True

    e_cpm = cpm(Xp, CS)[:, gi]
    e_raw = Xp[:, gi]

    cen = xy[m].mean(0)
    r = np.linalg.norm(xy - cen, axis=1)
    R = np.quantile(r[m], 0.95)              # lesion radius
    d = r - R                                 # signed distance to boundary, spot units

    prof = []
    for i in range(len(EDGES) - 1):
        sel = (d >= EDGES[i]) & (d < EDGES[i + 1])
        if sel.sum() < 3:
            continue
        prof.append(dict(
            lo=float(EDGES[i]), hi=float(EDGES[i + 1]), n=int(sel.sum()),
            frac_masked=float(m[sel].mean()),
            pred_cpm=float(e_cpm[sel].mean()),
            base_cpm=float(base_cpm[sel].mean()),
            delta_cpm=float((e_cpm - base_cpm)[sel].mean()),
            pred_raw=float(e_raw[sel].mean()),
            base_raw=float(base_raw[sel].mean()),
            delta_raw=float((e_raw - base_raw)[sel].mean()),
        ))
    # library-size check: does the perturbed pass change the row sum?
    ls_ratio = float((Xp.sum(1) / np.maximum(Xc.sum(1), 1e-8))[m].mean())
    res["conditions"][str(s)] = dict(
        n_spots=int(m.sum()), radius=float(R), centroid=[float(cen[0]), float(cen[1])],
        ls_ratio_in_mask=ls_ratio,
        in_mask_delta_cpm=float((e_cpm - base_cpm)[m].mean()),
        out_mask_delta_cpm=float((e_cpm - base_cpm)[~m].mean()),
        profile=prof)
    print(f"\n=== size {s}  n={m.sum()}  R={R:.2f}  ls_ratio={ls_ratio:.4f}")
    print(f"{'bin':>12} {'n':>5} {'fmask':>6} {'pred':>8} {'base':>8} {'delta':>9} {'d_raw':>9}")
    for p in prof:
        lab = f"[{p['lo']:.0f},{p['hi']:.0f})".replace("-1000000000", "-inf").replace("1000000000", "inf")
        print(f"{lab:>12} {p['n']:5d} {p['frac_masked']:6.2f} {p['pred_cpm']:8.4f} "
              f"{p['base_cpm']:8.4f} {p['delta_cpm']:9.4f} {p['delta_raw']:9.4f}")

# colour-scale context: what the figure's vmin/vmax were
alle = np.concatenate([cpm(raw(f"demo_size_{s:04d}"), CS)[:, gi]
                       for s in (25, 50, 100, 200, 400, 800)])
res["colour"] = dict(vlo=float(np.quantile(alle, 0.02)), vhi=float(np.quantile(alle, 0.98)),
                     frac_saturated=float((alle > np.quantile(alle, 0.98)).mean()))
print("\ncolour", res["colour"])

(OUT / "diag_rim.json").write_text(json.dumps(res, indent=2))
print("->", OUT / "diag_rim.json")
