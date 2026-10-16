"""Early-stage sanity: load mid-training checkpoints and confirm the f_pert field
is LEARNING -- i.e. the zero-init residual MLP departed from init (std 1e-3) and,
for svgp, the lengthscale/inducing points moved. No data load; weights only."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import torch, glob, os, math
BASE = f"{_rp.WORKDIR}/scratch/CONCERT/outputs"

def softplus(x): return math.log1p(math.exp(-abs(x))) + max(x, 0.0)

def inspect(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    keys = list(sd.keys())
    out = {}
    # core constant f_pert (present in all variants)
    core = [k for k in keys if k.startswith("f_pert.") and k.endswith("weight")]
    if core:
        out["core_w_absmean"] = float(torch.cat([sd[k].reshape(-1) for k in core]).abs().mean())
    # residual field MLP last layer (zero-init std=1e-3); large => learned spatial residual
    last = [k for k in keys if k.startswith("fpert_field.mlp") and k.endswith("weight")]
    if last:
        lw = sd[sorted(last)[-1]]
        out["resid_last_w_absmean"] = float(lw.abs().mean())
        out["resid_last_w_max"] = float(lw.abs().max())
    # svgp lengthscale + inducing spread
    if "fpert_field._raw_ls" in sd:
        out["svgp_lengthscale"] = round(softplus(float(sd["fpert_field._raw_ls"])), 4)
    if "fpert_field.Z" in sd:
        Z = sd["fpert_field.Z"]; out["svgp_Z_std"] = round(float(Z.std()), 4)
    return out

for v in ["constant", "rff", "svgp", "hybrid"]:
    for s in ["1-1", "2-2"]:
        ck = f"{BASE}/fpert_{s}_{v}/model_spac_{s}_fpert_{v}.pt"
        if not os.path.exists(ck):
            print(f"[{s} {v}] MISSING"); continue
        try:
            info = inspect(ck)
            print(f"[{s} {v}] " + "  ".join(f"{k}={val}" for k, val in info.items()))
        except Exception as e:
            print(f"[{s} {v}] ERR {e}")
print("\nGate: resid_last_w_absmean >> 1e-3 (init) => spatial residual is LEARNING.")
print("      constant has no fpert_field (resid keys absent) -- expected.")
