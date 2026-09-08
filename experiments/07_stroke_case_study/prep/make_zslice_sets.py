#!/usr/bin/env python
"""The SAME cortical lesion placed on each section in turn, for the 2D-vs-3D kernel comparison.

WHY NOT "PROPAGATION". f_pert is gated by is_real_KO (`concert_spac.py:1223`,
`is_real_KO.unsqueeze(1) * shift`), so with paracrine off an unperturbed spot receives exactly
zero no matter which model runs. A perturbation cannot reach an unperturbed cell, in-plane or
across z, and asking whether it does can only ever return 0.0000 for both arms.

WHAT THE 3D KERNEL ACTUALLY IS. `pos_dim = 2 + int(fpert_n_z_levels > 1)` (`concert_spac.py:730`):
z enters the footprint field as a THIRD COORDINATE, so f_pert is z-dependent AT THE PERTURBED SPOT.
The test that separates the two models is therefore: impose one fixed lesion footprint at each z
in turn and read the response on the perturbed spots.

  2D model (n_z_levels=1) -- z is not an input. Whatever differences appear across sections come
      from that section's own tissue and niche, nothing else. This is the control profile.
  3D model (n_z_levels=4) -- same footprint, same niche inputs, plus a z coordinate. Any
      DIFFERENCE from the 2D profile is the z term and nothing else.

The footprint is transferred between sections through the shared per-stack min-max frame, the same
frame `make_lesion_shape_set.py` uses to move the real lesion onto the sham brain -- the only one
in which "the same place" is defined across sections.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import anndata as ad, h5py, numpy as np
from scipy.spatial import cKDTree


def load_stack(slide_dir: Path, prefix: str, stack: str):
    paths = sorted(slide_dir.glob(f"{prefix}_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in paths], label="sf", keys=[p.stem for p in paths],
                  index_unique=None)
    sec = np.unique(np.asarray(A.obsm["section_index"]).reshape(-1), return_inverse=True)[1]
    return A, np.asarray(A.obsm["array_coords"], np.float32), sec


def stack_norm(xy):
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / np.maximum(hi - lo, 1e-8)


ap = argparse.ArgumentParser()
ap.add_argument("--files", required=True)
ap.add_argument("--slides", required=True)
ap.add_argument("--prefix", default="stroke_paste2")
ap.add_argument("--seed_set", required=True, help="local indices of the footprint on section 1")
ap.add_argument("--outdir", required=True)
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

A_sh, xy_sh, sec_sh = load_stack(Path(a.slides), a.prefix, "Sham")
n_sh = stack_norm(xy_sh)

seed_global = np.loadtxt(a.seed_set, dtype=np.int64)
seed_local = seed_global - 1 - off_sham
if seed_local.min() < 0 or seed_local.max() >= len(xy_sh):
    raise SystemExit(f"seed set out of range for the sham stack ({len(xy_sh)} spots)")
src_sec = np.unique(sec_sh[seed_local])
if len(src_sec) != 1:
    raise SystemExit(f"seed footprint spans sections {src_sec}; it must sit on exactly one")
src = int(src_sec[0])
foot = n_sh[seed_local]                       # the footprint in the shared normalised frame

outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
man = []
for s in range(int(sec_sh.max()) + 1):
    ms = np.where(sec_sh == s)[0]
    # every section spot whose nearest footprint point is close enough to be inside the footprint
    j, d = cKDTree(foot).query(n_sh[ms])[1], cKDTree(foot).query(n_sh[ms])[0]
    # match the source patch's spot count so lesion SIZE is held fixed across sections
    take = ms[np.argsort(d)[:len(seed_local)]]
    np.savetxt(outdir / f"zslice_s{s+1}.txt", np.sort(take) + off_sham + 1, fmt="%d")
    man.append(dict(name=f"zslice_s{s+1}", section=s + 1, n_spots=int(len(take)),
                    mean_frame_dist=float(d[np.argsort(d)[:len(seed_local)]].mean())))
    print(f"  zslice_s{s+1}  n={len(take):4d}  mean frame distance to the section-{src+1} "
          f"footprint {man[-1]['mean_frame_dist']:.4f}")

(outdir / "manifest_zslice.json").write_text(json.dumps(dict(
    source_section=src + 1, n_footprint=int(len(seed_local)), offset=int(off_sham),
    n_sham=int(len(xy_sh)), counts=[int(c) for c in counts], sets=man), indent=2) + "\n")
print(f"[zslice] source section {src+1}, {len(seed_local)} spots -> {outdir}")
