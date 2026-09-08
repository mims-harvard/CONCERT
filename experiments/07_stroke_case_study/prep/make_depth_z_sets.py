#!/usr/bin/env python
"""Two perturbation sets the previous demos did not have.

A. TRUE CORTICAL DEPTH.  `make_pert_sets.py` built its "depth" contrast as CTX (surface) versus
   TH (thalamus) -- two different ANATOMICAL REGIONS, not two depths.  A gene whose injury
   program does not transfer to thalamus (Timp1: +0.47 in CTX, +0.05 in TH) then reads as
   "no response in deep tissue", which is a region effect wearing a depth label.  Here both
   patches stay inside CTX and differ only in distance from the pial surface.

   Surface distance is measured from the tissue perimeter: a spot is on the perimeter when it has
   few neighbours inside `--perim_r` (the cloud thins out at the edge), and every spot's depth is
   its distance to the nearest perimeter spot.  Superficial patch = grown from the shallowest CTX
   spot, deep patch = grown from the deepest CTX spot, same spot count for both.

B. SINGLE-SECTION LESION for the z-propagation test.  A cortical patch on section 1 ONLY.
   Evaluated under the 2D model (one z level) and the 3D model (four z levels), the response on
   sections 2-4 -- where nothing was perturbed -- is what separates a 3D kernel from four
   independent 2D ones.  The existing across-sections figure perturbed the cortex of ALL FOUR
   sections, so its per-section response is driven by that section's own perturbation and says
   nothing about propagation through z.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np
from scipy.spatial import cKDTree


def _obs_str(f_obs, col):
    o = f_obs[col]
    if isinstance(o, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]])
        return cats[o["codes"][:]]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in o[:]])


def perimeter_depth(xy, perim_r, perim_q):
    """Depth of every spot below the tissue surface, in coordinate units.

    `perim_r` <= 0 means auto: 3x the median nearest-neighbour spacing, so the perimeter test does
    not depend on what units this slide's coordinates happen to be in.
    """
    tree = cKDTree(xy)
    if perim_r <= 0:
        nn = tree.query(xy, k=2)[0][:, 1]
        perim_r = 3.0 * float(np.median(nn))
        print(f"[depth] auto perimeter radius {perim_r:.3f} (median NN {np.median(nn):.3f})")
    deg = np.array([len(tree.query_ball_point(p, perim_r)) for p in xy])
    perim = deg <= np.quantile(deg, perim_q)
    return cKDTree(xy[perim]).query(xy)[0], perim


def grow(xy, pool, seed_local, n):
    """The n spots of `pool` nearest the seed -- contiguous, and confined to the pool."""
    d = np.linalg.norm(xy[pool] - xy[pool][seed_local], axis=1)
    return pool[np.argsort(d)[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--target_slide", default="Sham1")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--patch_n", type=int, default=200)
    ap.add_argument("--zpatch_n", type=int, default=400)
    ap.add_argument("--cortex", default="CTX")
    ap.add_argument("--perim_r", type=float, default=-1.0, help="<=0 = auto")
    ap.add_argument("--perim_q", type=float, default=0.15)
    ap.add_argument("--emit", default="all", choices=("all", "depth", "z"),
                    help="depth sets and the z set live on DIFFERENT slides: the depth pair must "
                         "sit on whatever slide the other condition demos used (Sham3), while the "
                         "z test needs the FIRST section so sections 2-4 are downstream of it.")
    a = ap.parse_args()

    paths = [p.strip() for p in a.files.split(",") if p.strip()]
    offset, target, counts = 0, None, []
    for p in paths:
        with h5py.File(p, "r") as f:
            n = f["obs"]["_index"].shape[0]
            if _obs_str(f["obs"], "slide")[0] == a.target_slide:
                target = dict(offset=offset, n=n,
                              xy=np.asarray(f["obsm"]["array_coords"][:], np.float32),
                              tissue=_obs_str(f["obs"], "tissue"))
        counts.append(n); offset += n
    if target is None:
        raise SystemExit(f"{a.target_slide} not among {paths}")

    xy, tissue, off = target["xy"], target["tissue"], target["offset"]
    depth, perim = perimeter_depth(xy, a.perim_r, a.perim_q)
    print(f"[depth] {perim.sum()} perimeter spots of {len(xy)}; "
          f"depth {depth.min():.3f}..{depth.max():.3f}")
    ctx = np.where(tissue == a.cortex)[0]
    if len(ctx) < 2 * a.patch_n:
        raise SystemExit(f"{a.cortex} has {len(ctx)} spots, need >= {2 * a.patch_n}")

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    man = []

    def emit(name, scenario, idx, **meta):
        idx = np.unique(np.asarray(idx, np.int64))
        np.savetxt(outdir / f"{name}.txt", idx + off + 1, fmt="%d")
        man.append(dict(name=name, scenario=scenario, n_spots=int(len(idx)),
                        slide=a.target_slide, mean_depth=float(depth[idx].mean()), **meta))
        print(f"  {scenario:10s} {name:26s} n={len(idx):4d} depth={depth[idx].mean():.2f} {meta}")

    # --- A. cortical depth: both patches inside CTX ---
    d_ctx = depth[ctx]
    for tag, seed in [] if a.emit == "z" else (("superficial", int(np.argmin(d_ctx))), ("deep", int(np.argmax(d_ctx)))):
        emit(f"ctxdepth_{tag}", "ctx_depth", grow(xy, ctx, seed, a.patch_n),
             region=a.cortex, placement=tag)

    # --- B. single-section lesion for the z test ---
    # Seeded at the shallowest cortical spot, i.e. where a photothrombotic lesion is made.
    if a.emit != "depth":
        emit("zsec1_ctx", "z_propagation", grow(xy, ctx, int(np.argmin(d_ctx)), a.zpatch_n),
             region=a.cortex, note=f"perturbed on {a.target_slide} only")

    (outdir / f"manifest_{a.emit}.json").write_text(json.dumps(dict(
        target_slide=a.target_slide, offset=int(off), n_slide=int(target["n"]),
        n_total=int(sum(counts)), files=paths, counts=[int(c) for c in counts],
        perimeter_spots=int(perim.sum()), cortex_spots=int(len(ctx)),
        depth_range_cortex=[float(d_ctx.min()), float(d_ctx.max())], sets=man), indent=2) + "\n")
    print(f"[depth/z] cortex {len(ctx)} spots, depth {d_ctx.min():.2f}..{d_ctx.max():.2f}; "
          f"{len(man)} sets -> {outdir}")


if __name__ == "__main__":
    main()
