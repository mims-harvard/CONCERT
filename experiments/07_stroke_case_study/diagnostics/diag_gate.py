#!/usr/bin/env python
"""Transplant / position gate: ON-OFF per lesion location, as trained vs with gamma disabled.

Columns are the mean (perturbed - control) over the imposed lesion's own spots, fixed library.
PASS for a location = the injury genes (Lcn2, Vim, Timp1) move POSITIVE, i.e. the model predicts
an injury response wherever the lesion is placed. gcap=0.0 clamps the niche-FiLM scalar to 1.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import sys
from pathlib import Path
import numpy as np, anndata as ad

B = Path(f"{_rp.WORKDIR}/scratch/CONCERT_rimtest")
NS = Path(f"{_rp.WORKDIR}")
SETS = NS / "outputs/stroke_2dpara/pert_sets"
GENES = ["Gm42418", "Scgb1a1", "Lcn2", "Vim", "Gfap", "Timp1"]
KEY = ["Lcn2", "Vim", "Timp1"]                       # the genes the gate is decided on
SETNAMES = ["transplant_PT1", "region_CTX", "region_TH", "region_STR", "region_HY"]

for arm in ["2d", "3d"]:
    gn = np.loadtxt(NS / f"outputs/stroke_{arm}/hvg_gene_names.txt", dtype=str)
    gi = {g: int(np.where(gn == g)[0][0]) for g in GENES if (gn == g).any()}
    print(f"\n{'='*104}\nARM stroke_{arm}")
    for gcap, tag in [("3.0", "g30"), ("0.0", "g00")]:
        d = B / f"gate_{arm}_{tag}"
        print(f"\n  --- niche_film_gcap={gcap} "
              f"({'as trained' if gcap=='3.0' else 'gamma forced to 1'}) ---")
        print(f"    {'location':17s} {'n':>5} " + " ".join(f"{g:>9}" for g in gi) + "   verdict")
        for s in SETNAMES:
            on = sorted(d.glob(f"g_{s}_ON_stroke_*_perturbed_counts.h5ad"))
            off = sorted(d.glob(f"g_{s}_OFF_stroke_*_perturbed_counts.h5ad"))
            if not on or not off:
                continue
            A = np.asarray(ad.read_h5ad(on[-1]).X, np.float32)
            C = np.asarray(ad.read_h5ad(off[-1]).X, np.float32)
            S = np.where(A.sum(1, keepdims=True) == 0, 1.0, A.sum(1, keepdims=True))
            cpm = lambda X: np.log1p(X / S * 1e4)
            idx = np.loadtxt(SETS / f"{s}.txt", dtype=np.int64) - 1
            D = (cpm(A) - cpm(C))[idx]
            vals = {g: float(D[:, gi[g]].mean()) for g in gi}
            ok = all(vals[g] > 0 for g in KEY if g in vals)
            print(f"    {s:17s} {len(idx):5d} " +
                  " ".join(f"{vals[g]:+9.3f}" for g in gi) +
                  f"   {'PASS' if ok else 'FAIL'}")
