#!/usr/bin/env python
"""Did f_pert learn anything, and does the basal coord field own the lesion?

A) weight check: fpert output layers vs their init scale (last layer init N(0,1e-3), bias 0).
B) leak test : predicted injury genes on the REAL PT1 lesion with the lesion ON vs OFF,
   paracrine disabled, against what is actually observed there.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
from pathlib import Path
import numpy as np, anndata as ad, h5py, torch

NS = Path(f"{_rp.WORKDIR}")
import sys
WOUT = Path(sys.argv[1])
GENES = ["Gm42418", "Scgb1a1", "Lcn2", "Vim", "Gfap", "Timp1"]

print("=" * 78); print("A) f_pert weight magnitudes vs init (last layer init std = 1e-3)")
sd = torch.load(NS / f"outputs/stroke_{sys.argv[2]}/model.pt", map_location="cpu")
sd = sd.get("state_dict", sd.get("model_state_dict", sd))
for k, v in sd.items():
    if any(t in k for t in ("f_pert", "fpert_field", "g_scale_head", "gamma_head",
                            "beta_out", "pert_gate", "basal_rff", "s_signature")):
        if v.ndim >= 1 and v.numel() > 1:
            print(f"  {k:46s} shape={tuple(v.shape)!s:16s} |w|_rms={v.float().pow(2).mean().sqrt():.5f} "
                  f"max={v.float().abs().max():.5f}")

print()
print("=" * 78); print("B) leak test on the REAL PT1 lesion (paracrine OFF)")
gn = np.loadtxt(NS / "outputs/stroke_2dpara/hvg_gene_names.txt", dtype=str)
gi = {g: int(np.where(gn == g)[0][0]) for g in GENES if (gn == g).any()}
idx = np.loadtxt(NS / "outputs/stroke_2dpara/pert_sets/real_lesion_PT1.txt", dtype=np.int64) - 1

def load(tag):
    h = sorted(WOUT.glob(f"{tag}_stroke_*_perturbed_counts.h5ad"))
    return np.asarray(ad.read_h5ad(h[-1]).X, np.float32) if h else None

ON, OFF = load("leak_ON"), load("leak_OFF")
if ON is None or OFF is None:
    raise SystemExit("leak_ON / leak_OFF not found yet")

# observed expression of the same spots, same HVG order
A = ad.read_h5ad(NS / "stroke3d/slides/stroke_paste2_PT1.h5ad")
Xo = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
Xo = np.asarray(Xo, np.float32)
ov = np.asarray(A.var_names)
lab = A.obs["perturbation"].astype(str).to_numpy()
tis = A.obs["tissue"].astype(str).to_numpy()
les = lab == "stroke"
cpmo = np.log1p(Xo / np.maximum(Xo.sum(1, keepdims=True), 1) * 1e4)

def cpm(X, s):
    return np.log1p(X / np.where(s == 0, 1.0, s) * 1e4)

S = ON.sum(1, keepdims=True)                      # fixed library, the ON pass
on_c, off_c = cpm(ON, S)[idx], cpm(OFF, S)[idx]

print(f"  lesion spots n={len(idx)}   tissues: "
      f"{dict(zip(*[x.tolist() for x in np.unique(tis[les], return_counts=True)]))}")
print(f"  {'gene':10s} {'pred ON':>9} {'pred OFF':>9} {'ON-OFF':>9} {'f_pert share':>13} "
      f"| {'obs lesion':>10} {'obs CTX':>9} {'obs LFC':>9}")
for g in gi:
    d = float((on_c - off_c)[:, gi[g]].mean())
    a, b = float(on_c[:, gi[g]].mean()), float(off_c[:, gi[g]].mean())
    share = d / a if abs(a) > 1e-9 else float("nan")
    if (ov == g).any():
        oi = int(np.where(ov == g)[0][0])
        ol, oc = float(cpmo[les, oi].mean()), float(cpmo[tis == "CTX", oi].mean())
        obs = f"| {ol:10.3f} {oc:9.3f} {ol-oc:+9.3f}"
    else:
        obs = "| " + " " * 30
    print(f"  {g:10s} {a:9.3f} {b:9.3f} {d:+9.3f} {share:12.1%} {obs}")
