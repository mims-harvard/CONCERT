import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import torch, json
from pathlib import Path
NS = Path(f"{_rp.WORKDIR}/outputs")
for arm in ["stroke_2d", "stroke_3d"]:
    sd = torch.load(NS / arm / "model.pt", map_location="cpu")
    sd = sd.get("state_dict", sd.get("model_state_dict", sd))
    W = sd["fpert_field.ls_head.weight"]
    print(f"=== {arm} ===  ls_head.weight {list(W.shape)}  (needs emb dim {W.shape[1]})")
    for k, v in sd.items():
        if v.dim() >= 1 and (v.shape[-1] == W.shape[1] or (v.dim() == 2 and v.shape[0] <= 8)):
            print(f"   {k:52s} {list(v.shape)}")
