#!/usr/bin/env python
"""Build the counterfactual perturbation sets for the 2D capability demos.

These are lesions no real experiment can deliver: several regions struck at once, a lesion whose
size is dialled up while everything else is held fixed, and the same lesion placed at the brain
surface versus deep inside it. Each set is just a list of spots to switch to `stroke`, so they all
run through the tested `--stage eval` path; the only new thing here is WHICH spots.

Three scenarios, one target section (default Sham1 -- the healthy brain, so nothing is overwritten):

  A. multi_region  -- an equal-sized patch in each of several regions, plus the UNION of them.
     The union is the point: comparing it against the sum of the single-region responses is what
     shows a systemic effect rather than independent local ones.
  B. size          -- concentric patches of growing spot count around ONE fixed centre, so the
     only thing that varies is lesion size and the response can be read as a dose-response.
  C. depth         -- the same patch size centred in a surface region vs a deep one.

Indices are 1-based into the loader's concatenation order (files in --files order), which is what
`run_concert_spac.py --pert_cells` expects. A sidecar manifest records what each set is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SURFACE_REGIONS = ("CTX", "PirCTX")
DEEP_REGIONS = ("TH", "STR", "HY", "PAL")


def _obs_str(f_obs, col):
    o = f_obs[col]
    if isinstance(o, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]])
        return cats[o["codes"][:]]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in o[:]])


def patch_around(xy, pool, centre, n):
    """The n spots of `pool` nearest `centre` -- a contiguous lesion, not a scatter of spots."""
    d = np.linalg.norm(xy[pool] - centre, axis=1)
    return pool[np.argsort(d)[:n]]


def region_centroid_spot(xy, pool):
    """Spot of `pool` closest to the pool centroid, so the patch starts inside the region."""
    c = xy[pool].mean(0)
    return xy[pool][np.argmin(np.linalg.norm(xy[pool] - c, axis=1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="comma list in --multi_sample_files order")
    ap.add_argument("--target_slide", default="Sham1")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sizes", default="25,50,100,200,400,800")
    ap.add_argument("--region_patch_n", type=int, default=200)
    ap.add_argument("--min_region_spots", type=int, default=300)
    ap.add_argument("--max_regions", type=int, default=4)
    a = ap.parse_args()

    paths = [p.strip() for p in a.files.split(",") if p.strip()]
    counts, offset, target = [], 0, None
    for p in paths:
        with h5py.File(p, "r") as f:
            n = f["obs"]["_index"].shape[0]
            slide = _obs_str(f["obs"], "slide")[0]
            if slide == a.target_slide:
                target = dict(offset=offset, n=n,
                              xy=np.asarray(f["obsm"]["array_coords"][:], dtype=np.float32),
                              tissue=_obs_str(f["obs"], "tissue"))
        counts.append(n)
        offset += n
    if target is None:
        raise SystemExit(f"target slide {a.target_slide} not among {paths}")

    xy, tissue, off = target["xy"], target["tissue"], target["offset"]
    regions, rcounts = np.unique(tissue, return_counts=True)
    big = [r for r, c in sorted(zip(regions, rcounts), key=lambda t: -t[1])
           if c >= a.min_region_spots]
    print(f"[sets] {a.target_slide}: {target['n']} spots; regions >= {a.min_region_spots}: "
          + ", ".join(f"{r}({c})" for r, c in sorted(zip(regions, rcounts), key=lambda t: -t[1])
                      if c >= a.min_region_spots))

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(name, scenario, idx, **meta):
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        np.savetxt(outdir / f"{name}.txt", idx + off + 1, fmt="%d")   # runner subtracts 1
        manifest.append(dict(name=name, scenario=scenario, n_spots=int(len(idx)),
                             slide=a.target_slide, **meta))
        print(f"  {scenario:12s} {name:28s} n={len(idx):5d} {meta}")

    # A. multi-region, and their union
    chosen = big[:a.max_regions]
    per_region = {}
    for r in chosen:
        pool = np.where(tissue == r)[0]
        idx = patch_around(xy, pool, region_centroid_spot(xy, pool), a.region_patch_n)
        per_region[r] = idx
        emit(f"region_{r}", "multi_region", idx, region=r, patch_n=a.region_patch_n)
    if len(per_region) > 1:
        emit("region_ALL", "multi_region", np.concatenate(list(per_region.values())),
             region="+".join(per_region), patch_n=a.region_patch_n,
             components=[f"region_{r}" for r in per_region])

    # B. lesion size sweep, one fixed centre
    surf = next((r for r in SURFACE_REGIONS if r in set(regions.tolist())), chosen[0])
    pool_s = np.where(tissue == surf)[0]
    centre = region_centroid_spot(xy, pool_s)
    all_spots = np.arange(target["n"])
    for n in [int(v) for v in a.sizes.split(",") if v.strip()]:
        # grown over ALL spots, not just the region: a big lesion spills past a region border,
        # exactly as a real infarct does.
        emit(f"size_{n:04d}", "size", patch_around(xy, all_spots, centre, n),
             requested_n=n, centre_region=surf)

    # C. surface vs deep, same size
    deep = next((r for r in DEEP_REGIONS if r in set(big)), None)
    if deep is None:
        print("[sets] no deep region above the size floor; skipping the depth scenario")
    else:
        for tag, r in (("surface", surf), ("deep", deep)):
            pool = np.where(tissue == r)[0]
            emit(f"depth_{tag}_{r}", "depth",
                 patch_around(xy, pool, region_centroid_spot(xy, pool), a.region_patch_n),
                 region=r, placement=tag, patch_n=a.region_patch_n)

    (outdir / "manifest.json").write_text(json.dumps(
        dict(target_slide=a.target_slide, offset=int(off), n_slide=int(target["n"]),
             n_total=int(sum(counts)), files=paths, sets=manifest), indent=2) + "\n")
    print(f"[sets] {len(manifest)} sets -> {outdir}")


if __name__ == "__main__":
    main()
