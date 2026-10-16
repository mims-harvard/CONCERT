#!/usr/bin/env python
"""Dilate the imposed peri-infarct rim and see whether a ring-shaped response appears.

The transplanted PIA rim is a thin shell around the transplanted ICA core. If the rim response is
weak because too few spots carry the PIA program -- the niche descriptor of a rim spot is then
dominated by unperturbed neighbours -- then thickening the rim should strengthen it, and the
predicted marker field should develop the ring the observed data shows.

The CORE is held fixed (the transplanted ICA mask). Only the rim grows: rim(t) = every sham spot
outside the core whose distance to the nearest core spot is <= t. t is swept in multiples of the
median nearest-neighbour spacing, so the sweep is in units of spot rings, not raw coordinates.
The observed PIA rim is measured in the same units and emitted as the reference point.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import argparse, json
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd
from scipy.spatial import cKDTree

DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")


def load_stack(slide_dir: Path, prefix: str, stack: str):
    ps = sorted(slide_dir.glob(f"{prefix}_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in ps], label="sf", keys=[p.stem for p in ps],
                  index_unique=None)
    sec = np.unique(np.asarray(A.obsm["section_index"]).reshape(-1), return_inverse=True)[1]
    return A, np.asarray(A.obsm["array_coords"], np.float32), sec


def norm(xy):
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / np.maximum(hi - lo, 1e-8)


ap = argparse.ArgumentParser()
ap.add_argument("--files", required=True)
ap.add_argument("--slides", required=True)
ap.add_argument("--prefix", default="stroke_paste2")
ap.add_argument("--outdir", required=True)
ap.add_argument("--thickness", default="1,2,3,5",
                help="rim thickness in spot rings (multiples of median NN spacing)")
a = ap.parse_args()

paths = [p.strip() for p in a.files.split(",") if p.strip()]
counts, off_sham, run = [], None, 0
for p in paths:
    with h5py.File(p, "r") as f:
        n = f["obs"]["_index"].shape[0]
        m = f["obs"]["mouse"]
        m = m["categories"][:][m["codes"][0]] if isinstance(m, h5py.Group) else m[0]
        m = m.decode() if isinstance(m, bytes) else str(m)
    if m == "Sham" and off_sham is None:
        off_sham = run
    counts.append(n); run += n

A_pt, xy_pt, sec_pt = load_stack(Path(a.slides), a.prefix, "PT")
A_sh, xy_sh, sec_sh = load_stack(Path(a.slides), a.prefix, "Sham")
n_pt, n_sh = norm(xy_pt), norm(xy_sh)

with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]])
    reg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
rp = pd.Series(reg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
obs_core, obs_rim = rp == "ICA", rp == "PIA"

# transplant the core and the observed rim, section by section, through the shared frame
def transplant(mask_pt):
    sel = np.zeros(len(xy_sh), bool)
    for s in range(int(sec_pt.max()) + 1):
        mp, ms = sec_pt == s, sec_sh == s
        if not mp.sum() or not ms.sum():
            continue
        j = cKDTree(n_pt[mp]).query(n_sh[ms])[1]
        sel[np.where(ms)[0][mask_pt[mp][j]]] = True
    return sel

core = transplant(obs_core)
rim_obs = transplant(obs_rim) & ~core
spacing = float(np.median(cKDTree(xy_sh).query(xy_sh, k=2)[0][:, 1]))

# how thick is each rim, in spot rings? mean distance of a rim spot to the nearest core spot
tree_core = cKDTree(xy_sh[core])
d_all = tree_core.query(xy_sh)[0]
print(f"[rim] sham spot spacing {spacing:.4f}; transplanted core {core.sum()} spots")
print(f"[rim] OBSERVED transplanted rim: {rim_obs.sum()} spots, mean depth "
      f"{d_all[rim_obs].mean()/spacing:.2f} rings, max {d_all[rim_obs].max()/spacing:.2f}, "
      f"rim:core {rim_obs.sum()/max(core.sum(),1):.2f}")
tree_pt = cKDTree(xy_pt[obs_core])
d_pt = tree_pt.query(xy_pt)[0]
sp_pt = float(np.median(cKDTree(xy_pt).query(xy_pt, k=2)[0][:, 1]))
print(f"[rim] the REAL PIA in the stroke mouse: {obs_rim.sum()} spots, mean depth "
      f"{d_pt[obs_rim].mean()/sp_pt:.2f} rings, rim:core {obs_rim.sum()/obs_core.sum():.2f}")

outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
np.savetxt(outdir / "thick_core.txt", np.where(core)[0] + off_sham + 1, fmt="%d")
np.savetxt(outdir / "thick_core_local.txt", np.where(core)[0], fmt="%d")

man = {"spacing": spacing, "offset": int(off_sham), "n_sham": int(len(xy_sh)),
       "counts": [int(c) for c in counts], "n_core": int(core.sum()),
       "observed_rim": {"n_spots": int(rim_obs.sum()),
                        "mean_rings": float(d_all[rim_obs].mean() / spacing),
                        "rim_core_ratio": float(rim_obs.sum() / max(core.sum(), 1))},
       "rims": []}

# the transplanted PIA itself, as the t=obs reference
np.savetxt(outdir / "thick_rim_obs.txt", np.where(rim_obs)[0] + off_sham + 1, fmt="%d")
np.savetxt(outdir / "thick_rim_obs_local.txt", np.where(rim_obs)[0], fmt="%d")
man["rims"].append(dict(tag="obs", thickness_rings=None, n_spots=int(rim_obs.sum()),
                        rim_core_ratio=float(rim_obs.sum() / max(core.sum(), 1))))

for t in [float(v) for v in a.thickness.split(",") if v.strip()]:
    rim = (~core) & (d_all > 0) & (d_all <= t * spacing)
    tag = f"t{int(t)}"
    np.savetxt(outdir / f"thick_rim_{tag}.txt", np.where(rim)[0] + off_sham + 1, fmt="%d")
    np.savetxt(outdir / f"thick_rim_{tag}_local.txt", np.where(rim)[0], fmt="%d")
    man["rims"].append(dict(tag=tag, thickness_rings=t, n_spots=int(rim.sum()),
                            rim_core_ratio=float(rim.sum() / max(core.sum(), 1))))
    print(f"  rim {tag}: <= {t:.0f} rings -> {rim.sum():5d} spots, "
          f"rim:core {rim.sum()/max(core.sum(),1):.2f}")

(outdir / "thick_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
print(f"[rim] -> {outdir}")
