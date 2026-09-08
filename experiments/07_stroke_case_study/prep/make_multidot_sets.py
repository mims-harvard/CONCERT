#!/usr/bin/env python
"""Multi-dot perturbation sets: n small lesions scattered over ONE slide, n small -> large.

This is what "multiple regions" should have meant. The earlier region_CTX/TH/STR/HY sets each
struck a SINGLE patch in one anatomy and were four separate counterfactuals -- that tests where
you put one lesion, not what happens when the slide carries many.

Here each set puts n foci of a fixed size on the same slide, with the centres chosen by
farthest-point sampling so the dots are spread out rather than clumped. n doubles each step.

NOTE ON THE CONFOUND: with patch size fixed, total struck tissue grows with n, so the response is
expected to rise for that reason alone. The `size` scenario is what isolates total area; read this
one for whether many small foci behave like one big lesion of the same total, not as an
independent dose axis. Both totals are recorded in the manifest so the comparison is checkable.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np

PATCH = 30                      # spots per focus
NS_LIST = [1, 2, 4, 8, 16]


def _obs_str(f_obs, col):
    o = f_obs[col]
    if isinstance(o, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]])
        return cats[o["codes"][:]]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in o[:]])


def interior_pool(xy, pool, min_nb=18):
    """Spots well inside the tissue.

    Plain farthest-point sampling maximises spread, which drives every centre onto the slide
    PERIMETER -- foci there sit on sparse edge tissue and the model returns almost no response
    (a 30-spot edge focus scored 0.024 against 0.300 for a 25-spot interior one). Requiring a
    dense neighbourhood keeps centres in real tissue.
    """
    from scipy.spatial import cKDTree
    t = cKDTree(xy[pool])
    r = 2.5 * np.median(t.query(xy[pool], k=2)[0][:, 1])
    return pool[np.array([len(x) for x in t.query_ball_point(xy[pool], r)]) >= min_nb]


def farthest_points(xy, pool, k, seed=42):
    """Spread n centres over the INTERIOR, seeded at the pool centroid for determinism."""
    inner = interior_pool(xy, pool)
    if len(inner) < k * 2:
        inner = pool
    c = xy[inner].mean(0)
    chosen = [inner[int(np.argmin(np.linalg.norm(xy[inner] - c, axis=1)))]]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(xy[inner][:, None, :] - xy[chosen][None, :, :], axis=2), axis=1)
        chosen.append(inner[int(np.argmax(d))])
    return np.array(chosen)


ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--files", required=True)
a = ap.parse_args()
out = Path(a.outdir); sets = out / "pert_sets"; sets.mkdir(parents=True, exist_ok=True)
man = json.loads((sets / "manifest.json").read_text())
files = [f.strip() for f in a.files.split(",")]
tgt = next(p for p in man["files"] if man["target_slide"] in Path(p).stem)
off, n_slide = man["offset"], man["n_slide"]
with h5py.File(tgt, "r") as f:
    xy = np.asarray(f["obsm"]["array_coords"][:], np.float32)
    tissue = _obs_str(f["obs"], "tissue")
assert len(xy) == n_slide, f"{len(xy)} vs {n_slide}"

# keep foci out of ventricles / white matter so every dot lands in real tissue
pool = np.where(~np.isin(tissue, ["VL", "WM"]))[0]
added = []
for n in NS_LIST:
    centres = farthest_points(xy, pool, n)
    idx = set()
    for c in centres:
        d = np.linalg.norm(xy[pool] - xy[c], axis=1)
        idx.update(pool[np.argsort(d)[:PATCH]].tolist())
    idx = np.array(sorted(idx))
    name = f"multidot_n{n:02d}"
    np.savetxt(sets / f"{name}.txt", (idx + off + 1).astype(np.int64), fmt="%d")
    added.append(dict(name=name, scenario="multidot", n_spots=int(len(idx)), n_foci=int(n),
                      patch_size=PATCH))
    print(f"  {name}: {n} foci x {PATCH} -> {len(idx)} spots")

man["sets"] = [s for s in man["sets"] if s.get("scenario") != "multidot"] + added
(sets / "manifest.json").write_text(json.dumps(man, indent=2))
print(f"[multidot] wrote {len(added)} sets -> {sets}")
