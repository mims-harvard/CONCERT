import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json, numpy as np, anndata as ad
from pathlib import Path
NS = Path(f"{_rp.WORKDIR}")
OUT = NS/"outputs/stroke_2dpara"
rm = json.loads((NS/"stroke3d/region_markers.json").read_text())
allm = {e["gene"]: e for e in rm["ICA"]+rm["PIA"]}
man = json.loads((OUT/"pert_sets"/"manifest.json").read_text())
off, n = man["offset"], man["n_slide"]; rows = slice(off, off+n)
g = np.loadtxt(OUT/"hvg_gene_names.txt", dtype=str); gset = set(g.tolist())
def cpm(X):
    s = X.sum(1, keepdims=True); s[s==0]=1.0
    return np.log1p(X/s*1e4)
def load(s):
    h = sorted(OUT.glob(f"{s}_stroke_*_perturbed_counts.h5ad"))
    return cpm(np.asarray(ad.read_h5ad(h[-1]).X[rows], np.float32))
B, E = load("demo_CONTROL"), load("demo_size_0800")
idx = np.loadtxt(OUT/"pert_sets"/"size_0800.txt", dtype=np.int64)-1-off
m = np.zeros(n, bool); m[idx]=True
# rank ICA-leaning genes by the induction the model actually produces
cand = sorted(allm.items(), key=lambda kv: -(kv[1]["delta_ICA"]-kv[1]["delta_PIA"]))
print(f"{'gene':10s} {'ICA-PIA':>8s} {'dICA':>6s} {'induction(800)':>15s}")
for name, e in cand:
    if name not in gset: continue
    i = int(np.where(g==name)[0][0])
    ind = float((E[:,i]-B[:,i])[m].mean())
    print(f"{name:10s} {e['delta_ICA']-e['delta_PIA']:8.2f} {e['delta_ICA']:6.2f} {ind:15.3f}")
