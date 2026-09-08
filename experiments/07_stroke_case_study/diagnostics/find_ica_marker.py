#!/usr/bin/env python
"""Is there a clean ICA marker at all? Search every modelled gene, not the 12-gene shortlist.

A usable display marker has to satisfy three things at once, and they trade off:
  strength     high in ICA relative to healthy cortex
  specificity  high in ICA relative to PIA (else it reports the rim)
  learnability the model actually reproduces it

region_markers.json holds only the top 12 per region, which is why the choice kept coming down to
Gm42418 (most specific, but an rRNA-contamination gene the model drives negative) versus Scgb1a1
(specific but weak). This ranks all genes so the trade-off is visible rather than assumed.
"""
from __future__ import annotations
import json
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd

NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
DEL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/bof695/CONCERT/delivery/stroke-registration-benchmark")


def cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


A = ad.concat([ad.read_h5ad(p) for p in sorted((NS/"stroke3d/slides").glob("stroke_paste2_PT*.h5ad"))],
              index_unique=None)
L = cpm(np.asarray(A.X.todense(), np.float32)); gs = np.asarray(A.var_names)
with h5py.File(DEL/"benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]]); rg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
rp = pd.Series(rg, index=bc).loc[np.asarray(A.obs_names)].to_numpy()
ica, pia, ctx = rp == "ICA", rp == "PIA", rp == "CTX"

strength = L[ica].mean(0) - L[ctx].mean(0)        # ICA vs healthy cortex
spec     = L[ica].mean(0) - L[pia].mean(0)        # ICA vs rim
det      = (np.asarray(A.X.todense())[ica] > 0).mean(0)

OUT = NS/"outputs/stroke_3d"
side = json.loads((OUT/"sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o+c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
Xon = np.asarray(ad.read_h5ad(next(OUT.glob("stroke_3d_on_*_perturbed_counts.h5ad"))).X[rows], np.float32)
Xoff = np.asarray(ad.read_h5ad(next(OUT.glob("stroke_3d_off_*_perturbed_counts.h5ad"))).X[rows], np.float32)
sz = Xoff.sum(1, keepdims=True); sz[sz == 0] = 1.0
pred = (np.log1p(Xon/sz*1e4) - np.log1p(Xoff/sz*1e4)).mean(0)     # fixed-library induction
gp = np.loadtxt(OUT/"hvg_gene_names.txt", dtype=str)

common = [g for g in gp if g in set(gs.tolist())]
oi = np.array([int(np.where(gs == g)[0][0]) for g in common])
pi = np.array([int(np.where(gp == g)[0][0]) for g in common])
df = pd.DataFrame(dict(gene=common, strength=strength[oi], specificity=spec[oi],
                       detected=det[oi], predicted=pred[pi]))
# a usable marker: detected in most core spots, clearly up vs healthy, not a rim gene
ok = df[(df.detected > 0.5) & (df.strength > 0.5) & (df.specificity > 0.2)]
print(f"genes screened: {len(df)}   passing (detected>50%, strength>0.5, ICA>PIA by 0.2): {len(ok)}\n")
print("TOP BY SPECIFICITY")
print(ok.sort_values("specificity", ascending=False).head(12).to_string(index=False,
      float_format=lambda v: f"{v:6.3f}"))
print("\nTOP BY WHAT THE MODEL REPRODUCES (predicted induction)")
print(ok.sort_values("predicted", ascending=False).head(12).to_string(index=False,
      float_format=lambda v: f"{v:6.3f}"))
df.to_csv(NS/"outputs/ica_marker_screen.csv", index=False)
print(f"\nfull table -> {NS/'outputs/ica_marker_screen.csv'}")
