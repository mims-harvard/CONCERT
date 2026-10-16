#!/usr/bin/env python
"""Build CONCERT-format .h5ad slides for the stroke dataset (PASTE2-aligned coordinates).

One .h5ad per section (PT1-4, Sham1-4) in the layout run_concert_spac.py::_load_h5ad_dataset
expects: raw counts in X, obs/perturbation, obs/tissue, obsm/array_coords (N,2), var/hvg_rank.

Two decisions encoded here, both consequential:

1. PERTURBED REGION = ICA u PIA. PIA is the peri-infarct area, not pia mater: it is 5-6x enriched
   in PT over Sham and carries the HIGHEST injury-marker score of any region (+1.69, above ICA's
   +0.452) because the reactive penumbra is more active than the dead core. Including it also
   extends the lesion field across all four sections instead of two.

2. TISSUE OF A PERTURBED SPOT = CTX, not its own label. If tissue carried ICA/PIA the lesion would
   appear in BOTH the perturbation and the tissue attribute and the counterfactual degenerates to
   "what if this stroke spot had a stroke". The lesion is cortical, so its underlying anatomy is
   cortex; switching perturbation on a Sham CTX spot is then exactly the intended counterfactual.

There is NO non-targeting control in this dataset, so unperturbed spots are the `unassigned`
class (true basal) and no control class exists -- train with --use_action_correction False and
--control_label __NONE__.

z is written as the SECTION INDEX (0-3), not z_mm: Stage 0 showed similarity decays by slice
index and ignores the 3x difference between the 0.4 and 1.2 mm gaps, so mm is the wrong metric.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")
PERT_REGIONS = ("ICA", "PIA")
PERT_LABEL, BASAL_LABEL, LESION_ANATOMY = "stroke", "unperturbed", "CTX"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="paste2", choices=["paste2", "stalign", "current"])
    ap.add_argument("--outdir", type=Path,
                    default=Path(f"{_rp.WORKDIR}/stroke3d/slides"))
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--pert_by_region", action="store_true",
                    help="label perturbed spots by COMPARTMENT (ICA/PIA) instead of one pooled\n"
                         "`stroke` class. With one class pert_emb is constant across every lesion\n"
                         "spot, so f_pert can only draw a smooth blob that is monotone in depth\n"
                         "and EVERY gene comes out core-peaked. Two classes give the core and the\n"
                         "rim their own embedding, lengthscale and gene loadings.")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
        X = np.asarray(f["X"][:], dtype=np.float32)
        bc = np.array([b.decode() for b in f["Barcode"][:]])
        genes = np.array([b.decode() for b in f["Gene"][:]])
        region = np.array([b.decode() for b in f["Celltype_coarse"][:]])

    coords = pd.read_parquet(DEL / f"benchmark_artifacts/arms/{a.arm}/coordinates.parquet")
    row = pd.Series(np.arange(len(bc)), index=bc).loc[coords.barcode].to_numpy()
    assert len(row) == len(coords), "barcode join lost rows"

    # HVG rank on log-normalised data, computed ONCE over all slides so every slide shares the
    # same gene order (the multi-sample loader requires it).
    s = X.sum(1, keepdims=True); s[s == 0] = 1
    Xl = np.log1p(X / s * 1e4)
    hvg_rank = (-Xl.var(0)).argsort().argsort() + 1        # 1-based, lower = more variable

    # The SAME region label means opposite things in the two mice: PIA scores +1.690 on injury
    # markers in PT (peri-infarct) but -0.123 in Sham (ordinary pia mater), a gap of 1.81. So the
    # perturbation is defined by MOUSE x REGION -- Sham has no stroke by construction and must
    # contribute basal spots only, or the model learns "stroke" from healthy meninges.
    is_pt = (coords.mouse == "PT").to_numpy()
    is_pert_region = np.isin(region, PERT_REGIONS)
    is_pert_row = np.zeros(len(region), dtype=bool)
    is_pert_row[row[is_pt]] = is_pert_region[row[is_pt]]
    is_pert = is_pert_row
    print(f"[prep] arm={a.arm}  perturbed spots (ICA u PIA): {is_pert.sum()} / {len(region)}")

    for mouse, sl in coords.groupby("mouse").slide.unique().items():
        for slide in sorted(sl):
            m = (coords.slide == slide).to_numpy()
            r = row[m]
            pert = np.where(is_pert[r], region[r] if a.pert_by_region else PERT_LABEL, BASAL_LABEL)
            # decision 2: a perturbed spot's tissue is the anatomy the lesion replaced
            tissue = np.where(is_pert[r], LESION_ANATOMY, region[r])
            obs = pd.DataFrame(
                {"perturbation": pert, "tissue": tissue,
                 "slide": slide, "mouse": mouse,
                 "section_index": coords.slide.rank(method="dense").astype(int).to_numpy()[m] - 1,
                 "filter_pass": "1"},
                index=coords.barcode.to_numpy()[m])
            A = ad.AnnData(X=sp.csr_matrix(X[r]), obs=obs,
                           var=pd.DataFrame({"hvg_rank": hvg_rank}, index=genes))
            A.obsm["array_coords"] = coords[["x", "y"]].to_numpy()[m].astype(np.float32)
            # z kept beside the 2D coords: the fpert field consumes it, basal/niche do not
            A.obsm["section_index"] = obs["section_index"].to_numpy().astype(np.float32)[:, None]
            out = a.outdir / f"stroke_{a.arm}_{slide}.h5ad"
            A.write_h5ad(out)
            print(f"  {slide:6s} n={m.sum():5d} perturbed={is_pert[r].sum():5d} "
                  f"({100*is_pert[r].mean():4.1f}%)  -> {out.name}")
    print(f"[prep] done -> {a.outdir}")


if __name__ == "__main__":
    main()
