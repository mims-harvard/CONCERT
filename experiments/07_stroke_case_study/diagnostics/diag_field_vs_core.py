#!/usr/bin/env python
"""How much of the perturbation shift is SPATIAL, and can it vary inside one lesion?

shift(p, x) = f_pert(p)                      constant core, one vector per perturbation
            + fpert_field(p, x)              coordinate-conditioned residual
(then niche-FiLM gamma/beta, then the gate; those are measured separately.)

Reported per perturbation:
  |core|          RMS of the constant term over genes
  |field|         RMS of the residual over genes and over lesion spots
  field/core      what fraction of the shift is carried by position at all
  field spatial   std ACROSS lesion spots of the residual, i.e. how much it varies WITHIN the
                  lesion -- this is the quantity a core-vs-rim contrast needs
  core vs rim     the residual averaged over imposed core spots vs imposed rim spots
"""
import json
from pathlib import Path
import anndata as ad, numpy as np, torch

NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
COORD_SCALE, TILE = 20.0, 20.0
EK = "lord_encoder.s_encoder.perturbation.weight"
SETS = NS / "outputs/thick_rim_sets"

A = ad.concat([ad.read_h5ad(p) for p in
               sorted((NS / "stroke3d/slides_cr").glob("stroke_paste2_Sham*.h5ad"))])
xy = np.asarray(A.obsm["array_coords"], np.float32)
loc = xy - TILE * np.floor(xy / TILE)
c = torch.tensor(loc / COORD_SCALE, dtype=torch.float32)

core_m = np.zeros(len(xy), bool)
core_m[np.loadtxt(SETS / "thick_core_local.txt", dtype=np.int64)] = True
rim_m = np.zeros(len(xy), bool)
rim_m[np.loadtxt(SETS / "thick_rim_obs_local.txt", dtype=np.int64)] = True
rim_m &= ~core_m
les = core_m | rim_m
print(f"lesion spots: core {core_m.sum()}, rim {rim_m.sum()}\n")

import sys
ARMS = sys.argv[1:] or ["stroke_2d_cr", "stroke_3d_cr"]
for arm, ls_min, ls_max in [(a, 0.05, 0.60) for a in ARMS]:
    p = NS / "outputs" / arm / "model.pt"
    if not p.exists():
        continue
    sd = torch.load(p, map_location="cpu")
    sd = sd.get("state_dict", sd.get("model_state_dict", sd))
    E = sd[EK].float()
    Wc = sd["f_pert.weight"].float()                      # (G, encoder_dim), no bias
    Z = sd["fpert_field.Z"].float()
    W0, b0 = sd["fpert_field.mlp.0.weight"].float(), sd["fpert_field.mlp.0.bias"].float()
    W2, b2 = sd["fpert_field.mlp.2.weight"].float(), sd["fpert_field.mlp.2.bias"].float()
    Wl, bl = sd["fpert_field.ls_head.weight"].float(), sd["fpert_field.ls_head.bias"].float()
    pm = json.loads((NS / "outputs" / arm / "pert_map.json").read_text())["pert_map"]
    print(f"### {arm}   inducing points: {Z.shape[0]} (dim {Z.shape[1]})")
    cc = c
    if Z.shape[1] == 3:                                   # 3D: append the z level of section 1
        cc = torch.cat([c, torch.zeros(len(c), 1)], dim=1)
    for name, code in pm.items():
        if name == "__basal__":
            continue
        emb = E[code]
        ls = ls_min + (ls_max - ls_min) * torch.sigmoid(emb @ Wl.T + bl)
        d2 = torch.cdist(cc, Z) ** 2
        feat = torch.exp(-d2 / (2.0 * ls * ls))
        h = torch.relu(torch.cat([emb.expand(len(cc), -1), feat], dim=1) @ W0.T + b0)
        field = h @ W2.T + b2                             # (N, G)
        core_vec = emb @ Wc.T                             # (G,)
        cmag = float(core_vec.pow(2).mean().sqrt())
        fles = field[torch.tensor(les)]
        fmag = float(fles.pow(2).mean().sqrt())
        spatial = float(fles.std(dim=0).mean())           # per-gene std across lesion spots
        fc = float(field[torch.tensor(core_m)].mean())
        fr = float(field[torch.tensor(rim_m)].mean())
        print(f"  {name:8s} ls={float(ls):.4f}  |core|={cmag:.4f}  |field|={fmag:.4f}  "
              f"field/core={fmag/max(cmag,1e-9):.3f}")
        print(f"  {'':8s} field spatial std within lesion = {spatial:.5f} "
              f"({100*spatial/max(fmag,1e-9):.1f}% of |field|)   "
              f"field mean core {fc:+.4f} vs rim {fr:+.4f}")
    print()
