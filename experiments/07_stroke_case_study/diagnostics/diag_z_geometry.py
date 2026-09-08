#!/usr/bin/env python
"""Why does the 3D arm blob where the 2D arm can ring?

f_pert's spatial basis is an RBF against inducing points Z with a SINGLE isotropic lengthscale.
In 3D, Z is z-augmented: the same XY grid replicated across n_z_levels, and the third coordinate
carries z_gain. One isotropic ls must then serve BOTH axes -- so if the model needs to correlate
across sections it has to grow ls, and that same ls over-smooths in-plane, which would wipe out
any ring. This dumps the numbers that decide it:

  * in-plane inducing spacing (the resolution ceiling)
  * the learned per-perturbation lengthscale
  * ls / spacing -- >>1 means the field cannot resolve structure finer than the patch
  * z separation between adjacent sections, for scale
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, torch

NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs")
ARMS = ["stroke_2d", "stroke_3d", "stroke_2d_cr", "stroke_3d_cr"]
LS_MIN, LS_MAX = 0.05, 0.60
Z_GAIN = 0.25

res = {}
for arm in ARMS:
    p = NS / arm / "model.pt"
    if not p.exists():
        continue
    sd = torch.load(p, map_location="cpu")
    sd = sd.get("state_dict", sd.get("model_state_dict", sd))
    Z = sd["fpert_field.Z"].float()
    d = {"Z_shape": list(Z.shape)}
    xy = Z[:, :2]
    # distinct in-plane sites: cluster XY that coincide to 3 decimals (z-replicas collapse)
    uxy = torch.unique(torch.round(xy * 1000) / 1000, dim=0)
    d["n_inducing_xy_distinct"] = int(uxy.shape[0])
    if uxy.shape[0] > 1:
        D = torch.cdist(uxy, uxy); D.fill_diagonal_(float("inf"))
        nn = D.min(dim=1).values
        d["xy_nn_spacing_median"] = float(nn.median())
    if Z.shape[1] == 3:
        uz = torch.unique(torch.round(Z[:, 2] * 1000) / 1000)
        d["n_z_levels_distinct"] = int(uz.numel())
        d["z_values"] = [round(float(v), 4) for v in uz]
        if uz.numel() > 1:
            d["z_adjacent_gap"] = float(torch.diff(torch.sort(uz).values).median())
    # learned lengthscale per perturbation
    W, b = sd["fpert_field.ls_head.weight"].float(), sd["fpert_field.ls_head.bias"].float()
    pm = json.loads((NS / arm / "pert_map.json").read_text())["pert_map"]
    ekey = next((k for k in sd if "pert" in k.lower() and "emb" in k.lower()
                 and sd[k].dim() == 2 and sd[k].shape[1] == W.shape[1]), None)
    d["emb_key"] = ekey
    if ekey is not None:
        E = sd[ekey].float()
        ls = LS_MIN + (LS_MAX - LS_MIN) * torch.sigmoid(E @ W.T + b)
        d["lengthscale"] = {name: round(float(ls[code]), 4)
                            for name, code in sorted(pm.items(), key=lambda kv: kv[1])
                            if code < ls.shape[0]}
        if "xy_nn_spacing_median" in d:
            d["ls_over_xy_spacing"] = {k: round(v / d["xy_nn_spacing_median"], 2)
                                       for k, v in d["lengthscale"].items()}
    res[arm] = d

print(json.dumps(res, indent=2))
(NS / "diag_z_geometry.json").write_text(json.dumps(res, indent=2))
