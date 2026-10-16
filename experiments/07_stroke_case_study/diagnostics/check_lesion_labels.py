#!/usr/bin/env python
"""Which spots carry the 'stroke' perturbation label, and which tissues do they sit in?"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
from pathlib import Path
import numpy as np, h5py
SL = Path(f"{_rp.WORKDIR}/stroke3d/slides")
def cat(f, col):
    o = f["obs"][col]
    c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in o["categories"][:]])
    return c[o["codes"][:]]
for s in ["PT1", "PT2", "PT3", "PT4", "Sham3"]:
    with h5py.File(SL / f"stroke_paste2_{s}.h5ad", "r") as f:
        p, t = cat(f, "perturbation"), cat(f, "tissue")
    print(f"\n== {s}  n={len(p)}")
    for lab in np.unique(p):
        m = p == lab
        u, c = np.unique(t[m], return_counts=True)
        top = ", ".join(f"{a}:{b}" for a, b in sorted(zip(u, c), key=lambda z: -z[1])[:6])
        print(f"   {lab:12s} n={m.sum():5d}  tissues: {top}")
    for reg in ("ICA", "PIA"):
        m = t == reg
        if m.sum():
            u, c = np.unique(p[m], return_counts=True)
            print(f"   [{reg}] n={m.sum():5d} -> labels: " + ", ".join(f"{a}:{b}" for a, b in zip(u, c)))
        else:
            print(f"   [{reg}] absent")
