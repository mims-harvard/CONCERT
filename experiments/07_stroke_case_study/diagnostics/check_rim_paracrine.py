#!/usr/bin/env python
"""Answers three questions with one table, WITH vs WITHOUT paracrine spillover.

  Q1 depth  : is the deep lesion's induction still negative?
  Q2 multi  : are the small foci still negative?
  Q3 marker : does Gm42418 (the real ICA marker) work, or only Scgb1a1?

induction   = mean over PERTURBED spots of (pert - control), fixed-library log1p-CPM
spill       = mean over UNPERTURBED spots of the same  (0 if the effect is confined to the lesion)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json
from pathlib import Path
import numpy as np, anndata as ad, h5py

WITH = Path(f"{_rp.WORKDIR}/outputs/stroke_2dpara")
WOUT = Path(f"{_rp.WORKDIR}/scratch/CONCERT_rimtest/stroke_2dpara_nopara")
GENES = ["Gm42418", "Scgb1a1", "Lcn2", "Vim", "Gfap", "Timp1"]
NAMES = [f"size_{s:04d}" for s in (25, 50, 100, 200, 400, 800)] + \
        ["depth_surface_CTX", "depth_deep_TH"] + [f"multidot_n{n:02d}" for n in (1, 2, 4, 8, 16)]

man = json.loads((WITH / "pert_sets" / "manifest.json").read_text())
off, n_slide = man["offset"], man["n_slide"]
rows = slice(off, off + n_slide)
gnames = np.loadtxt(WITH / "hvg_gene_names.txt", dtype=str)
gidx = {g: int(np.where(gnames == g)[0][0]) for g in GENES if (gnames == g).any()}
print("genes found:", list(gidx))

print("\n=== slide coordinate frames (does the r_max=5 paracrine graph reach across slides?) ===")
for p in man["files"]:
    with h5py.File(p, "r") as f:
        c = np.asarray(f["obsm"]["array_coords"][:], np.float32)
    print(f"  {Path(p).stem:26s} n={c.shape[0]:6d} x[{c[:,0].min():6.2f},{c[:,0].max():6.2f}] "
          f"y[{c[:,1].min():6.2f},{c[:,1].max():6.2f}]")


def table(root, tag):
    g = lambda s: sorted(root.glob(f"{s}_stroke_*_perturbed_counts.h5ad"))
    if not g("demo_CONTROL"):
        print(f"[{tag}] no control in {root}"); return
    Xc = np.asarray(ad.read_h5ad(g("demo_CONTROL")[-1]).X[rows], np.float32)
    CS = np.where(Xc.sum(1, keepdims=True) == 0, 1.0, Xc.sum(1, keepdims=True))
    cpm = lambda X: np.log1p(X / CS * 1e4)
    B = cpm(Xc)
    hdr = "  ".join(f"{g_:>9}" for g_ in gidx)
    print(f"\n--- {tag} ---")
    print(f"{'condition':20s} {'n':>5}  " + hdr + "   | spill(unperturbed) " + hdr)
    for n in NAMES:
        hits = g(f"demo_{n}")
        f_ = WITH / "pert_sets" / f"{n}.txt"
        if not hits or not f_.exists():
            continue
        E = cpm(np.asarray(ad.read_h5ad(hits[-1]).X[rows], np.float32))
        idx = np.loadtxt(f_, dtype=np.int64) - 1 - off
        m = np.zeros(n_slide, bool); m[idx] = True
        D = E - B
        ind = "  ".join(f"{D[m, gidx[g_]].mean():+9.3f}" for g_ in gidx)
        spl = "  ".join(f"{D[~m, gidx[g_]].mean():+9.3f}" for g_ in gidx)
        print(f"{n:20s} {m.sum():5d}  {ind}   | {' ':18s} {spl}")


table(WITH, "WITH paracrine  (the published figures)")
table(WOUT, "WITHOUT paracrine  (rerun, same model weights)")
