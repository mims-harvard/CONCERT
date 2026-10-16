#!/usr/bin/env python
"""Impose the REAL lesion's shape on the sham brain, instead of perturbing every sham spot.

The 3D counterfactual switched all 9194 sham spots to `stroke` and let the footprint field decide
where the effect landed. That answers "where does the model put a stroke?" -- but it means the
perturbed region is whatever the field carved out (a thin surface arc), so it can never be
compared shape-for-shape with the real lesion, which is a filled wedge with a core and a rim.

Here the lesion mask is transferred from the stroke mouse to the sham mouse through the shared
per-stack min-max frame -- the same frame the model tiles in, and the only one in which "the same
place" is defined for two different brains. A sham spot joins the lesion when its nearest stroke-
mouse spot IN THE SAME SECTION is a lesion spot, so the imposed lesion keeps the real one's shape,
size AND depth profile (it shrinks section by section exactly as the real one does).

Emits three sets: the whole lesion (ICA u PIA), the core alone (ICA) and the rim alone (PIA), so
the core/rim response can be read out on a region that actually has a core and a rim.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")


def stack_norm(xy):
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / np.maximum(hi - lo, 1e-8)


def load_stack(slide_dir: Path, prefix: str, stack: str):
    paths = sorted(slide_dir.glob(f"{prefix}_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in paths], label="sf", keys=[p.stem for p in paths],
                  index_unique=None)
    sec = np.unique(np.asarray(A.obsm["section_index"]).reshape(-1), return_inverse=True)[1]
    return A, np.asarray(A.obsm["array_coords"], np.float32), sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="comma list in --multi_sample_files order")
    ap.add_argument("--slides", required=True)
    ap.add_argument("--prefix", default="stroke_paste2")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sections", default="",
                    help="comma list of 1-based sections to impose the lesion on (default all). "
                         "The 2D model sees each section independently, so a 2D figure must "
                         "impose and score on ONE section -- four sections invite a z reading "
                         "the 2D model cannot support.")
    a = ap.parse_args()
    keep_sec = {int(v) - 1 for v in a.sections.split(",") if v.strip()}

    paths = [p.strip() for p in a.files.split(",") if p.strip()]
    counts, off_sham = [], None
    running = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            n = f["obs"]["_index"].shape[0]
            mouse = f["obs"]["mouse"]
            mouse = (mouse["categories"][:][mouse["codes"][0]] if isinstance(mouse, h5py.Group)
                     else mouse[0])
            mouse = mouse.decode() if isinstance(mouse, bytes) else str(mouse)
        if mouse == "Sham" and off_sham is None:
            off_sham = running
        counts.append(n); running += n
    n_total = running

    A_pt, xy_pt, sec_pt = load_stack(Path(a.slides), a.prefix, "PT")
    A_sh, xy_sh, sec_sh = load_stack(Path(a.slides), a.prefix, "Sham")
    n_pt, n_sh = stack_norm(xy_pt), stack_norm(xy_sh)

    with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
        bc = np.array([b.decode() for b in f["Barcode"][:]])
        reg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
    reg_pt = pd.Series(reg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()

    masks = {"lesion": np.isin(reg_pt, ("ICA", "PIA")), "core": reg_pt == "ICA",
             "rim": reg_pt == "PIA"}
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"target": "Sham", "sections": sorted(x + 1 for x in keep_sec) or "all",
                "offset": int(off_sham), "n_slide": int(A_sh.n_obs),
                "n_total": int(n_total), "counts": [int(c) for c in counts],
                "is_target": [bool(off_sham is not None and i >= 4) for i in range(len(counts))],
                "sets": {}}

    for name, m_pt in masks.items():
        sel = np.zeros(A_sh.n_obs, dtype=bool)
        per_sec = []
        for s in range(int(sec_pt.max()) + 1):
            if keep_sec and s not in keep_sec:
                continue
            mp, ms = sec_pt == s, sec_sh == s
            if not mp.sum() or not ms.sum():
                continue
            # nearest stroke-mouse spot of the SAME section, in the shared frame
            j = cKDTree(n_pt[mp]).query(n_sh[ms])[1]
            hit = m_pt[mp][j]
            sel[np.where(ms)[0][hit]] = True
            per_sec.append(dict(section=s + 1, observed=int(m_pt[mp].sum()),
                                observed_frac=float(m_pt[mp].mean()),
                                imposed=int(hit.sum()), imposed_frac=float(hit.mean())))
        idx = np.where(sel)[0]
        np.savetxt(outdir / f"shape_{name}.txt", idx + off_sham + 1, fmt="%d")
        np.savetxt(outdir / f"shape_{name}_local.txt", idx, fmt="%d")
        manifest["sets"][name] = dict(n_spots=int(len(idx)), per_section=per_sec)
        print(f"[shape] {name:7s} imposed {len(idx):5d} sham spots "
              f"(observed {int(m_pt.sum()):5d} in the stroke mouse)")
        for d in per_sec:
            print(f"          section {d['section']}: observed {100*d['observed_frac']:5.1f}%  "
                  f"imposed {100*d['imposed_frac']:5.1f}%")

    (outdir / "shape_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[shape] -> {outdir}")


if __name__ == "__main__":
    main()
