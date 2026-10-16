#!/usr/bin/env python
"""Stage 0b: do CROSS-SLICE neighbours actually inform a spot, and how should the niche be built?

Stage 0/Test B showed cross-slice neighbours are SIMILAR. That is not the same as informative:
within-slice neighbours may already carry everything they would add. Here each spot is held out
and reconstructed from a neighbour set; the set that reconstructs it best is the neighbourhood the
niche descriptor should use.

The control that makes this a real test is `w50`. Adding cross-slice neighbours also adds
neighbours, and more neighbours help on their own, so a cross-slice set is only worth anything if
it beats a within-slice set of the SAME SIZE.

Everything is slice-mean-centred: in the model the slide attribute absorbs the additive per-slice
batch offset, so the centred comparison is the honest analogue. Raw is reported alongside.
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
OUT = Path(f"{_rp.WORKDIR}/stroke3d")
N_PC = 30

# (label, k within own slice, k per slice at offset +-1, k per slice at offset +-2)
CONFIGS = [
    ("w25            (current model)", 25, 0, 0),
    ("w50            (size control)",  50, 0, 0),
    ("w25 + c1x5",                     25, 5, 0),
    ("w25 + c1x10",                    25, 10, 0),
    ("w25 + c1x25",                    25, 25, 0),
    ("c1x25          (cross only)",     0, 25, 0),
    ("w25 + c1x10 + c2x10",            25, 10, 10),
]


def load():
    with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
        X = np.asarray(f["X"][:], dtype=np.float32)
        bc = np.array([b.decode() for b in f["Barcode"][:]])
    if X.min() >= 0 and X.max() > 50:
        s = X.sum(1, keepdims=True); s[s == 0] = 1
        X = np.log1p(X / s * 1e4)
    Z = PCA(n_components=N_PC, random_state=42).fit_transform(
        (X - X.mean(0)) / (X.std(0) + 1e-8))
    return Z, bc


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(1)


def run(E, xy, order, kw, kc1, kc2):
    """Reconstruct each spot from its neighbour set. `order` = slice index per spot."""
    trees = {o: cKDTree(xy[order == o]) for o in np.unique(order)}
    ids = {o: np.where(order == o)[0] for o in np.unique(order)}
    pred = np.zeros_like(E); got = np.zeros(len(E), dtype=bool)
    for o in np.unique(order):
        me = ids[o]
        parts = []
        if kw:
            # k+1 then drop self: the query point is its own nearest neighbour
            nb = ids[o][trees[o].query(xy[me], k=min(kw + 1, len(me)))[1][:, 1:]]
            parts.append(E[nb])
        for off, k in ((1, kc1), (2, kc2)):
            if not k:
                continue
            for d in (-off, off):
                t = o + d
                if t not in trees:
                    continue
                nb = ids[t][trees[t].query(xy[me], k=min(k, len(ids[t])))[1]]
                # query(k=1) returns (n,) not (n,1), so E[nb] loses the neighbour axis
                parts.append(E[nb][:, None, :] if nb.ndim == 1 else E[nb])
        if not parts:
            continue
        pred[me] = np.concatenate(parts, axis=1).mean(axis=1)
        got[me] = True
    return float(np.mean(cos(pred[got], E[got]))), int(got.sum())


def main():
    Z, bc = load()
    c = pd.read_parquet(DEL / "benchmark_artifacts/arms/paste2/coordinates.parquet")
    idx = pd.Series(np.arange(len(bc)), index=bc)
    c = c.assign(row=idx.loc[c.barcode].to_numpy())

    report = {}
    for mouse in ["PT", "Sham"]:
        s = c[c.mouse == mouse].copy()
        # slide names sort by depth (PT1..PT4), so rank gives the slice index the decay follows
        s["order"] = s.slide.rank(method="dense").astype(int) - 1
        xy = s[["x", "y"]].to_numpy(); order = s["order"].to_numpy()
        E_raw = Z[s.row.to_numpy()]
        E_ctr = E_raw.copy()
        for o in np.unique(order):
            E_ctr[order == o] -= E_ctr[order == o].mean(0)
        report[mouse] = {}
        for lab, kw, kc1, kc2 in CONFIGS:
            r, n = run(E_raw, xy, order, kw, kc1, kc2)
            v, _ = run(E_ctr, xy, order, kw, kc1, kc2)
            report[mouse][f"w{kw}_c1x{kc1}_c2x{kc2}"] = {
                "label": lab.strip(), "raw": round(r, 4), "slice_centred": round(v, 4), "n": n}
    (OUT / "stage0b_neighbourhood.json").write_text(json.dumps(report, indent=1))

    print("\n===== Stage 0b: reconstruction of a held-out spot from its neighbours =====")
    print("(cosine to the true expression; higher is better)\n")
    print(f"  {'neighbourhood':34s}{'PT centred':>12s}{'Sham centred':>14s}{'PT raw':>10s}{'Sham raw':>10s}")
    for lab, kw, kc1, kc2 in CONFIGS:
        k = f"w{kw}_c1x{kc1}_c2x{kc2}"
        p, q = report["PT"][k], report["Sham"][k]
        print(f"  {lab.strip():34s}{p['slice_centred']:>12.4f}{q['slice_centred']:>14.4f}"
              f"{p['raw']:>10.4f}{q['raw']:>10.4f}")
    base = report["PT"]["w25_c1x0_c2x0"]["slice_centred"]
    ctrl = report["PT"]["w50_c1x0_c2x0"]["slice_centred"]
    key = lambda c: f"w{c[1]}_c1x{c[2]}_c2x{c[3]}"
    best = max(CONFIGS, key=lambda c: report["PT"][key(c)]["slice_centred"])
    bv = report["PT"][key(best)]["slice_centred"]
    print(f"\n  PT: within-25 {base:.4f} | within-50 (size control) {ctrl:.4f} | best {best[0].strip()} {bv:.4f}")
    print(f"  cross-slice gain OVER THE SIZE CONTROL: {bv - ctrl:+.4f}")
    print(f"\n[ok] {OUT/'stage0b_neighbourhood.json'}")


if __name__ == "__main__":
    main()
