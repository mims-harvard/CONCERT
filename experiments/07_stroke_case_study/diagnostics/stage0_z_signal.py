#!/usr/bin/env python
"""Stage 0 gate: does z carry expression signal in the stroke slides, beyond in-plane distance?

The point of a 3D kernel is that a spot is informed by spots in OTHER slices. That is only true
if, at matched in-plane distance, expression similarity still depends on the z gap. With z gaps of
0.4-1.2 mm against 0.06-0.09 mm in-plane spacing, it may not be, and then 3D buys nothing over
"one model across slices with slice as an attribute".

Two tests, run per mouse. PT (stroke) is the case; Sham is the no-perturbation control, so a
z-effect present in BOTH is anatomy or batch, and only a PT-specific excess is stroke-related.

  A. Delta-xy-matched decay. Bin spot pairs by in-plane distance; within each bin compare
     similarity at each z gap. A pure per-slice batch offset shifts all cross-slice bins down by a
     constant; real 3D structure makes them ORDER by z gap.
  B. Correspondence test (the decisive one). For each spot, compare its expression to the
     nearest-in-xy spot in another slice vs a RANDOM spot in that slice. If aligned coordinates
     carry real 3D correspondence, nearest beats random. Run on the PASTE2 coordinates and on the
     old naive 3D coordinates, so the registration itself is also tested.

Similarity is computed in PC space (sparse per-spot counts make raw gene correlation noise), and
every test is also run slice-mean-centred, which removes the additive per-slice batch offset.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")
H5 = DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5"
OUT = Path(f"{_rp.WORKDIR}/stroke3d")
RNG = np.random.default_rng(42)
N_PC = 30


def load():
    with h5py.File(H5, "r") as f:
        X = np.asarray(f["X"][:], dtype=np.float32)
        bc = np.array([b.decode() for b in f["Barcode"][:]])
        batch = np.array([b.decode() for b in f["Batch"][:]])
        pos3d_pt = np.asarray(f["3D_pos_PT"][:]).T
        pos3d_sh = np.asarray(f["3D_pos_sham"][:]).T
    # counts -> log1p CP10K only if this still looks like raw counts
    if X.min() >= 0 and X.max() > 50:
        s = X.sum(1, keepdims=True); s[s == 0] = 1
        X = np.log1p(X / s * 1e4)
    Z = PCA(n_components=N_PC, random_state=42).fit_transform(
        (X - X.mean(0)) / (X.std(0) + 1e-8))
    return Z, bc, batch, pos3d_pt, pos3d_sh


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(1)


def test_A(Z, xy, z, slide, centred, n_pairs=400_000):
    """Similarity vs (in-plane distance bin, z gap)."""
    E = Z.copy()
    if centred:
        for s in np.unique(slide):
            m = slide == s
            E[m] -= E[m].mean(0)
    n = len(E)
    i = RNG.integers(0, n, n_pairs); j = RNG.integers(0, n, n_pairs)
    keep = i != j; i, j = i[keep], j[keep]
    dxy = np.linalg.norm(xy[i] - xy[j], axis=1)
    dz = np.abs(z[i] - z[j])
    sim = cosine(E[i], E[j])
    bins = np.array([0, 0.25, 0.5, 1.0, 2.0, 4.0])
    out = {}
    for b in range(len(bins) - 1):
        m = (dxy >= bins[b]) & (dxy < bins[b + 1])
        lab = f"{bins[b]:g}-{bins[b+1]:g}mm"
        out[lab] = {f"{g:g}": (round(float(np.mean(sim[m & (dz == g)])), 4)
                               if (m & (dz == g)).sum() >= 200 else None)
                    for g in np.unique(dz)}
    return out


def test_B(Z, xy, slide, centred, n_rand=20):
    """Nearest-in-xy in another slice vs random in that slice."""
    E = Z.copy()
    if centred:
        for s in np.unique(slide):
            m = slide == s
            E[m] -= E[m].mean(0)
    slides = list(dict.fromkeys(slide))
    res = {}
    for a in range(len(slides)):
        for b in range(a + 1, len(slides)):
            ia = np.where(slide == slides[a])[0]
            ib = np.where(slide == slides[b])[0]
            nn = ib[cKDTree(xy[ib]).query(xy[ia], k=1)[1]]
            s_nn = cosine(E[ia], E[nn])
            s_rand = np.stack([cosine(E[ia], E[RNG.choice(ib, len(ia))])
                               for _ in range(n_rand)]).mean(0)
            d = s_nn - s_rand
            res[f"{slides[a]}--{slides[b]}"] = {
                "nearest": round(float(s_nn.mean()), 4),
                "random": round(float(s_rand.mean()), 4),
                "gain": round(float(d.mean()), 4),
                # paired across the same spots, so a paired t is the right scale
                "t": round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)), 1),
                "n": int(len(ia)),
            }
    return res


def main():
    Z, bc, batch, pos3d_pt, pos3d_sh = load()
    coords = pd.read_parquet(DEL / "benchmark_artifacts/arms/paste2/coordinates.parquet")
    idx = pd.Series(np.arange(len(bc)), index=bc)
    assert coords.barcode.isin(idx.index).all(), "PASTE2 barcodes do not match the h5"
    order = idx.loc[coords.barcode].to_numpy()

    naive = np.zeros((len(bc), 3), dtype=np.float64)
    naive[batch_is := np.array([b.startswith("PT") for b in batch])] = pos3d_pt
    naive[~batch_is] = pos3d_sh

    report = {"n_spots": int(len(bc)), "n_pcs": N_PC}
    for mouse in ["PT", "Sham"]:
        m = (coords.mouse == mouse).to_numpy()
        sub, o = coords[m], order[m]
        xy = sub[["x", "y"]].to_numpy(); z = sub["z_mm"].to_numpy()
        sl = sub["slide"].to_numpy(); E = Z[o]
        nv = naive[o]
        report[mouse] = {
            "n": int(m.sum()),
            "A_decay_raw": test_A(E, xy, z, sl, centred=False),
            "A_decay_slice_centred": test_A(E, xy, z, sl, centred=True),
            "B_paste2_raw": test_B(E, xy, sl, centred=False),
            "B_paste2_slice_centred": test_B(E, xy, sl, centred=True),
            "B_naive_slice_centred": test_B(E, nv[:, :2], sl, centred=True),
        }
    (OUT / "stage0_z_signal.json").write_text(json.dumps(report, indent=1))

    print("\n===== TEST B: nearest-in-xy vs random, in ANOTHER slice (slice-centred) =====")
    for mouse in ["PT", "Sham"]:
        print(f"\n--- {mouse} ---   PASTE2                       | naive 3D coords")
        p = report[mouse]["B_paste2_slice_centred"]; q = report[mouse]["B_naive_slice_centred"]
        for k in p:
            a, b = p[k], q.get(k, {})
            print(f"  {k:16s} gain {a['gain']:+.4f} (t={a['t']:>7.1f}) | "
                  f"gain {b.get('gain', float('nan')):+.4f} (t={b.get('t', float('nan')):>7.1f})")
    print("\n===== TEST A: similarity by in-plane bin x z gap (slice-centred) =====")
    for mouse in ["PT", "Sham"]:
        print(f"\n--- {mouse} ---")
        d = report[mouse]["A_decay_slice_centred"]
        gaps = sorted({g for v in d.values() for g in v})
        print("   in-plane bin   " + "".join(f"{('dz='+g):>10s}" for g in gaps))
        for b, v in d.items():
            print(f"   {b:14s} " + "".join(
                f"{(f'{v[g]:.4f}' if v.get(g) is not None else '-'):>10s}" for g in gaps))
    print(f"\n[ok] {OUT/'stage0_z_signal.json'}")


if __name__ == "__main__":
    main()
