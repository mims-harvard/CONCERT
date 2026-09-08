#!/usr/bin/env python
"""Is the negative Gm42418 prediction biological, or a compositional artefact?

Everything is compared in log1p-CPM, which renormalises each spot to a fixed library. Gm42418 is
one of the most abundant transcripts in the data, so when the model induces OTHER genes inside the
lesion, Gm42418's SHARE of the library falls even if its absolute expression does not -- and the
delta prints negative. Test: compare the delta in CPM space against the delta in RAW count space
for the same spots, and report Gm42418's share of the library.
"""
import json, numpy as np, anndata as ad
from pathlib import Path
OUT = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/stroke_2d")
G = ["Gm42418", "Vim", "Timp1"]

def cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)

side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o+c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
on = ad.read_h5ad(next(OUT.glob("shape_on_*_perturbed_counts.h5ad")))
off = ad.read_h5ad(next(OUT.glob("stroke_2d_off_*_perturbed_counts.h5ad")))
Xon = np.asarray(on.X[rows], np.float32); Xoff = np.asarray(off.X[rows], np.float32)
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
les = np.loadtxt(OUT / "shape_sets/shape_lesion_local.txt", dtype=np.int64)
m = np.zeros(Xon.shape[0], bool); m[les] = True

Lon, Loff = cpm(Xon), cpm(Xoff)
print(f"library size in lesion: OFF {Xoff[m].sum(1).mean():.1f} -> ON {Xon[m].sum(1).mean():.1f} "
      f"({100*(Xon[m].sum(1).mean()/Xoff[m].sum(1).mean()-1):+.1f}%)\n")
for g in G:
    i = int(np.where(gp == g)[0][0])
    d_cpm = float((Lon[m, i] - Loff[m, i]).mean())
    d_raw = float((np.log1p(Xon[m, i]) - np.log1p(Xoff[m, i])).mean())
    share = float((Xoff[m, i] / np.maximum(Xoff[m].sum(1), 1)).mean())
    print(f"{g:9s} delta_CPM={d_cpm:+.3f}  delta_RAW={d_raw:+.3f}  library_share={100*share:.2f}%")
