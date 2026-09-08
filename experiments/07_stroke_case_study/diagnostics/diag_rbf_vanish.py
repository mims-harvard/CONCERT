#!/usr/bin/env python
"""Do the RBF features vanish inside the lesion? Direct test of the coarse-grid hypothesis.

feat_m(x) = exp(-||x - Z_m||^2 / 2 ls^2). If the inducing grid is too coarse for the fitted
lengthscale, every feature is ~0 for a lesion spot, the MLP sees an all-zero spatial block, and
its output collapses to the bias -- a flat field, regardless of how short ls is.
"""
import json
from pathlib import Path
import anndata as ad, numpy as np, torch

NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
SETS = NS / "outputs/thick_rim_sets"
EK = "lord_encoder.s_encoder.perturbation.weight"
A = ad.concat([ad.read_h5ad(p) for p in
               sorted((NS / "stroke3d/slides_cr").glob("stroke_paste2_Sham*.h5ad"))])
xy = np.asarray(A.obsm["array_coords"], np.float32)
c = torch.tensor((xy - 20.0 * np.floor(xy / 20.0)) / 20.0, dtype=torch.float32)
core = np.zeros(len(xy), bool); core[np.loadtxt(SETS/"thick_core_local.txt", dtype=np.int64)] = True
rim = np.zeros(len(xy), bool); rim[np.loadtxt(SETS/"thick_rim_obs_local.txt", dtype=np.int64)] = True
les = torch.tensor(core | (rim & ~core))

for arm in ("stroke_2d_cr", "stroke_3d_cr"):
    sd = torch.load(NS/"outputs"/arm/"model.pt", map_location="cpu")
    sd = sd.get("state_dict", sd.get("model_state_dict", sd))
    E = sd[EK].float(); Z = sd["fpert_field.Z"].float()
    Wl, bl = sd["fpert_field.ls_head.weight"].float(), sd["fpert_field.ls_head.bias"].float()
    pm = json.loads((NS/"outputs"/arm/"pert_map.json").read_text())["pert_map"]
    cc = torch.cat([c, torch.zeros(len(c), 1)], 1) if Z.shape[1] == 3 else c
    zxy = torch.unique(Z[:, :2], dim=0)
    lo, hi = cc[les][:, :2].min(0).values, cc[les][:, :2].max(0).values
    inside = ((zxy >= lo) & (zxy <= hi)).all(1).sum()
    print(f"### {arm}: {Z.shape[0]} inducing ({zxy.shape[0]} distinct xy); "
          f"{int(inside)} of them inside the lesion bounding box")
    for name, code in pm.items():
        if name == "__basal__":
            continue
        ls = 0.05 + 0.55 * torch.sigmoid(E[code] @ Wl.T + bl)
        feat = torch.exp(-torch.cdist(cc[les], Z) ** 2 / (2.0 * ls * ls))
        mx = feat.max(1).values
        print(f"  {name:5s} ls={float(ls):.4f}  max feature per lesion spot: "
              f"mean {float(mx.mean()):.4f}, median {float(mx.median()):.4f}, "
              f"min {float(mx.min()):.4f}  |  frac of spots with max<0.01: "
              f"{float((mx < 0.01).float().mean()):.3f}")
    print()
