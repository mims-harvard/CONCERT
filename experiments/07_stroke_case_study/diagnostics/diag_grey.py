import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json, numpy as np, anndata as ad
from pathlib import Path
OUT = Path(f"{_rp.WORKDIR}/outputs/stroke_2dpara")
M = ("Lcn2","Gfap","Vim","Timp1","Cd68")
man = json.loads((OUT/"pert_sets"/"manifest.json").read_text())
off, n = man["offset"], man["n_slide"]; rows = slice(off, off+n)
g = np.loadtxt(OUT/"hvg_gene_names.txt", dtype=str)
gi = [int(np.where(g==m)[0][0]) for m in M if m in set(g.tolist())]
def cpm(X):
    s = X.sum(1, keepdims=True); s[s==0]=1.0
    return np.log1p(X/s*1e4)
def load(s):
    h = sorted(OUT.glob(f"{s}_stroke_*_perturbed_counts.h5ad"))
    return cpm(np.asarray(ad.read_h5ad(h[-1]).X[rows], np.float32))[:, gi].mean(1)
base = load("demo_CONTROL")
for cond in ["size_0800", "size_0100", "region_CTX"]:
    d = load(f"demo_{cond}") - base
    idx = np.loadtxt(OUT/"pert_sets"/f"{cond}.txt", dtype=np.int64)-1-off
    m = np.zeros(n, bool); m[idx]=True
    o = d[~m]
    print(f"\n=== {cond}: {m.sum()} lesion / {(~m).sum()} outside ===")
    print(f"  outside: exact zero      = {100*np.mean(o==0):.1f}%")
    print(f"           |delta| < 1e-4  = {100*np.mean(np.abs(o)<1e-4):.1f}%")
    print(f"           |delta| < 0.01  = {100*np.mean(np.abs(o)<0.01):.1f}%")
    print(f"  outside quantiles 50/90/99/max: {np.quantile(o,.5):.4f} {np.quantile(o,.9):.4f} "
          f"{np.quantile(o,.99):.4f} {o.max():.4f}")
    print(f"  inside  mean = {d[m].mean():.4f}")
