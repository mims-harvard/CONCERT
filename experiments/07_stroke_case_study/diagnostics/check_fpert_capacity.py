#!/usr/bin/env python
"""Capacity of the f_pert head on the stroke 3D checkpoint: how fine a spatial pattern can it draw?

f_pert(p, x) = MLP([pert_emb , RBF(x, Z)]). With ONE perturbation the embedding is a constant,
so every gene-to-gene difference in the predicted map must be carried by the RBF basis. The
question is whether that basis can express an ANNULUS (rim high, core low) at lesion scale.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json, torch, numpy as np
from pathlib import Path
OUT = Path(f"{_rp.WORKDIR}/outputs/stroke_3d")
sd = torch.load(OUT / "model.pt", map_location="cpu")
sd = sd.get("state_dict", sd.get("model_state_dict", sd))
d = {k: list(v.shape) for k, v in sd.items() if k.startswith("fpert_field.")}
Z = sd["fpert_field.Z"].float()
res = {"shapes": d, "n_inducing_total": int(Z.shape[0]), "Z_dim": int(Z.shape[1])}
# distinct XY inducing sites and their spacing (the resolution ceiling of the field)
xy = torch.unique(Z[:, :2], dim=0)
res["n_inducing_xy"] = int(xy.shape[0])
D = torch.cdist(xy, xy); D[D == 0] = float("inf")
res["min_xy_spacing_normalised"] = float(D.min())
if "fpert_field.ls_head.weight" in sd:
    res["ls_band"] = "sigmoid-bounded; see train flags fpert_ls_min/max = 0.05 / 0.60"
res["hidden_dim"] = d["fpert_field.mlp.0.weight"][0]
res["num_genes_out"] = d["fpert_field.mlp.2.weight"][0]
# effective rank of the gene readout: how many independent spatial shapes can reach the genes?
W2 = sd["fpert_field.mlp.2.weight"].float()
s = torch.linalg.svdvals(W2)
res["readout_pc1_frac"] = float((s[0] ** 2 / (s ** 2).sum()))
res["readout_participation_ratio"] = float((s ** 2).sum() ** 2 / (s ** 4).sum())
print(json.dumps(res, indent=2))
(OUT / "check_fpert_capacity.json").write_text(json.dumps(res, indent=2))
