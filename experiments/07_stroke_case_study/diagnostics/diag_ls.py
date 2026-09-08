import torch, json, numpy as np, anndata as ad
from pathlib import Path
NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
OUT = NS / "outputs"
import os
LS_MIN_D = {"stroke_3d_crC": 0.02}
COORD_SCALE = 20.0
CAP = {"stroke_3d_crA": 0.20, "stroke_3d_crC": 0.35}
EK = "lord_encoder.s_encoder.perturbation.weight"

# how big is the imposed lesion in the SAME normalised units the RBF works in?
sd_sets = OUT / "stroke_3d/shape_sets"
A = ad.concat([ad.read_h5ad(p) for p in sorted((NS/"stroke3d/slides").glob("stroke_paste2_Sham*.h5ad"))])
xy = np.asarray(A.obsm["array_coords"], np.float32)
S = 20.0
loc = xy - S*np.floor(xy/S)          # per-slide local, as tile_by_stack does
c = loc / COORD_SCALE
les = np.loadtxt(sd_sets/"shape_lesion_local.txt", dtype=np.int64)
ext = c[les].max(0) - c[les].min(0)
print(f"imposed lesion extent in RBF units: x={ext[0]:.3f} y={ext[1]:.3f}")
print(f"whole sham section extent:          x={(c.max(0)-c.min(0))[0]:.3f} y={(c.max(0)-c.min(0))[1]:.3f}\n")

for arm in ["stroke_3d_crC","stroke_3d_crA","stroke_2d_cr"]:
    p = OUT/arm/"model.pt"
    if not p.exists(): continue
    sd = torch.load(p, map_location="cpu"); sd = sd.get("state_dict", sd.get("model_state_dict", sd))
    W,b = sd["fpert_field.ls_head.weight"].float(), sd["fpert_field.ls_head.bias"].float()
    E = sd[EK].float()
    LS_MAX = CAP.get(arm, 0.60); LS_MIN = LS_MIN_D.get(arm, 0.05)
    ls = LS_MIN + (LS_MAX-LS_MIN)*torch.sigmoid(E @ W.T + b)
    pm = json.loads((OUT/arm/"pert_map.json").read_text())["pert_map"]
    names = {v:k for k,v in pm.items()}
    Z = sd["fpert_field.Z"].float()
    uxy = torch.unique(torch.round(Z[:,:2]*1000)/1000, dim=0)
    D = torch.cdist(uxy,uxy); D.fill_diagonal_(float("inf")); sp = float(D.min(1).values.median())
    out = {names.get(i,f"code{i}"): round(float(ls[i]),4) for i in range(ls.shape[0])}
    hit = {k:("AT MAX" if v>=LS_MAX-1e-3 else "at min" if v<=LS_MIN+1e-3 else "") for k,v in out.items()}
    print(f"{arm:15s} ls={out}  {hit}")
    print(f"{'':15s} xy_spacing={sp:.4f}  ls/spacing={ {k:round(v/sp,1) for k,v in out.items()} }")
    print(f"{'':15s} ls/lesion_extent={ {k:round(v/float(ext.max()),2) for k,v in out.items()} }\n")
