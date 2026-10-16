#!/usr/bin/env python
"""Stage 0c: does the STROKE RESPONSE have 3D extent? (Stage 0b asked the wrong question.)

Stage 0b measured whether neighbours reconstruct a spot's BASAL expression. That is dominated by
local cell type, so in-plane neighbours at 0.06-0.09 mm necessarily win over anything 1.2 mm away
in z, and the answer says little about 3D. What the 3D kernel is actually for here is letting the
perturbation propagate as a volume: a spot's counterfactual response should depend on its 3D
position relative to the lesion. So test the lesion field directly.

  C. Region coherence. `Celltype_coarse` carries the ICA (infarct) label -- the attribute the
     counterfactual switches. If the lesion is a 3D volume, an ICA spot's nearest-xy neighbour in
     an ADJACENT slice should also be ICA, well above that slice's ICA base rate.
  D. Response-field coherence. Score each spot on injury markers (Lcn2, Gfap, Vim, Timp1, Cd68).
     Does that score at matched xy correlate across slices, beyond a random spot in the same
     slice? Sham is the control: no lesion, so no coherent field to find.
  E. Residual test (the Stage 0b follow-up). Predict basal expression from 25 within-slice
     neighbours, take the residual, and ask whether cross-slice neighbours predict THAT. This is
     the orthogonal-information question a uniform mean could not answer.
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
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")
OUT = Path(f"{_rp.WORKDIR}/stroke3d")
MARKERS = ["Lcn2", "Gfap", "Vim", "Timp1", "Cd68"]
RNG = np.random.default_rng(42)


def main():
    with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
        X = np.asarray(f["X"][:], dtype=np.float32)
        bc = np.array([b.decode() for b in f["Barcode"][:]])
        genes = np.array([b.decode() for b in f["Gene"][:]])
        ct = np.array([b.decode() for b in f["Celltype_coarse"][:]])
    if X.min() >= 0 and X.max() > 50:
        s = X.sum(1, keepdims=True); s[s == 0] = 1
        X = np.log1p(X / s * 1e4)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    mk = np.array([np.where(genes == m)[0][0] for m in MARKERS])
    score_all = Xz[:, mk].mean(1)               # injury-marker score per spot
    Z = PCA(n_components=30, random_state=42).fit_transform(Xz)

    c = pd.read_parquet(DEL / "benchmark_artifacts/arms/paste2/coordinates.parquet")
    idx = pd.Series(np.arange(len(bc)), index=bc)
    c = c.assign(row=idx.loc[c.barcode].to_numpy())

    rep = {"markers": MARKERS, "note": "Distance field is z depth, not lesion distance"}
    for mouse in ["PT", "Sham"]:
        s = c[c.mouse == mouse].copy()
        s["order"] = s.slide.rank(method="dense").astype(int) - 1
        rows = s.row.to_numpy(); xy = s[["x", "y"]].to_numpy(); o = s["order"].to_numpy()
        lab = ct[rows]; sc = score_all[rows]; E = Z[rows]
        Ec = E.copy()
        for k in np.unique(o):
            Ec[o == k] -= Ec[o == k].mean(0)

        ids = {k: np.where(o == k)[0] for k in np.unique(o)}
        trees = {k: cKDTree(xy[ids[k]]) for k in ids}
        ica_rate = {int(k): round(float((lab[ids[k]] == "ICA").mean()), 4) for k in ids}

        C, Dd = {}, {}
        for a in sorted(ids):
            for b in sorted(ids):
                if b <= a:
                    continue
                ia, ib = ids[a], ids[b]
                nn = ib[trees[b].query(xy[ia], k=1)[1]]
                # C: of ICA spots in slice a, what fraction map onto ICA in slice b?
                m = lab[ia] == "ICA"
                if m.sum() >= 20:
                    C[f"{a}->{b}"] = {
                        "hit": round(float((lab[nn[m]] == "ICA").mean()), 4),
                        "base_rate": ica_rate[int(b)], "n_ica": int(m.sum()),
                        "lift": round(float((lab[nn[m]] == "ICA").mean()) / (ica_rate[int(b)] + 1e-9), 2)}
                # D: does the marker score track across slices at matched xy?
                r_nn = pearsonr(sc[ia], sc[nn])[0]
                r_rd = np.mean([pearsonr(sc[ia], sc[RNG.choice(ib, len(ia))])[0] for _ in range(10)])
                Dd[f"{a}--{b}"] = {"r_matched": round(float(r_nn), 4),
                                   "r_random": round(float(r_rd), 4),
                                   "gain": round(float(r_nn - r_rd), 4)}

        # E: residual of a within-slice fit, then ask cross-slice to explain it
        pred = np.zeros_like(Ec)
        for k in ids:
            me = ids[k]
            nb = me[trees[k].query(xy[me], k=min(26, len(me)))[1][:, 1:]]
            pred[me] = Ec[nb].mean(1)
        resid = Ec - pred
        cross = np.zeros_like(Ec); has = np.zeros(len(Ec), bool)
        for k in ids:
            me = ids[k]; parts = []
            for d in (-1, 1):
                if k + d in ids:
                    t = ids[k + d]
                    parts.append(Ec[t[trees[k + d].query(xy[me], k=min(25, len(t)))[1]]])
            if parts:
                cross[me] = np.concatenate(parts, 1).mean(1); has[me] = True
        # per-PC correlation between the within-slice residual and the cross-slice signal
        rs = [pearsonr(resid[has][:, j], cross[has][:, j])[0] for j in range(Ec.shape[1])]
        rep[mouse] = {"ica_rate_per_slice": ica_rate, "C_region_coherence": C,
                      "D_response_field": Dd,
                      "E_residual_vs_cross": {"mean_abs_r": round(float(np.mean(np.abs(rs))), 4),
                                              "max_abs_r": round(float(np.max(np.abs(rs))), 4),
                                              "n": int(has.sum())}}
    (OUT / "check_3d_response.json").write_text(json.dumps(rep, indent=1))

    for mouse in ["PT", "Sham"]:
        r = rep[mouse]
        print(f"\n===== {mouse} =====")
        print(f"  ICA fraction per slice: {r['ica_rate_per_slice']}")
        if r["C_region_coherence"]:
            print("  C. lesion-region coherence (ICA spot -> nearest xy in other slice):")
            for k, v in r["C_region_coherence"].items():
                print(f"     {k}: {100*v['hit']:5.1f}% land on ICA vs {100*v['base_rate']:5.1f}% base "
                      f"(lift {v['lift']:.1f}x, n={v['n_ica']})")
        else:
            print("  C. no slice has >=20 ICA spots")
        print("  D. injury-marker field across slices:")
        for k, v in r["D_response_field"].items():
            print(f"     {k}: matched r={v['r_matched']:+.3f} random r={v['r_random']:+.3f} gain {v['gain']:+.3f}")
        e = r["E_residual_vs_cross"]
        print(f"  E. within-slice residual vs cross-slice signal: mean|r|={e['mean_abs_r']:.4f} "
              f"max|r|={e['max_abs_r']:.4f}")
    print(f"\n[ok] {OUT/'check_3d_response.json'}")


if __name__ == "__main__":
    main()
