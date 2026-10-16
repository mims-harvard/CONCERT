#!/usr/bin/env python
"""Which genes actually peak in the infarct CORE (ICA) and which in the peri-infarct rim (PIA)?

The field figure used one pooled injury panel, which mixes the two: the observed panel then lights
up the rim while the prediction lights up the core, so the two rows are not showing the same
quantity. This ranks genes separately for each region so a single, region-matched marker can be
used on BOTH rows.

ICA/PIA are NOT recoverable from the slide .h5ad files -- prep rewrote every perturbed spot's
tissue to CTX (so the lesion would not appear in both the perturbation and the tissue attribute).
The labels come back from the delivery h5 (`Celltype_coarse`), joined by barcode.

Score = mean log1p-CP10K in the region minus the mean over all OTHER PT spots, restricted to genes
the model actually carries (hvg_gene_names.txt), so anything picked here can be plotted on the
predicted row too.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="paste2")
    ap.add_argument("--model_genes", required=True, help="hvg_gene_names.txt from a trained run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min_frac", type=float, default=0.10,
                    help="drop genes detected in fewer than this fraction of the region's spots")
    a = ap.parse_args()

    with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
        X = np.asarray(f["X"][:], dtype=np.float32)
        bc = np.array([b.decode() for b in f["Barcode"][:]])
        genes = np.array([b.decode() for b in f["Gene"][:]])
        region = np.array([b.decode() for b in f["Celltype_coarse"][:]])

    coords = pd.read_parquet(DEL / f"benchmark_artifacts/arms/{a.arm}/coordinates.parquet")
    row = pd.Series(np.arange(len(bc)), index=bc).loc[coords.barcode].to_numpy()
    is_pt = (coords.mouse == "PT").to_numpy()

    pt_rows = row[is_pt]
    reg_pt = region[pt_rows]
    L = X[pt_rows]
    s = L.sum(1, keepdims=True); s[s == 0] = 1.0
    L = np.log1p(L / s * 1e4)

    keep = np.isin(genes, np.loadtxt(a.model_genes, dtype=str))
    print(f"[markers] PT spots {len(pt_rows)}; regions "
          + ", ".join(f"{r}:{c}" for r, c in zip(*np.unique(reg_pt, return_counts=True))))
    print(f"[markers] genes in model: {keep.sum()} / {len(genes)}")

    out = {}
    for target in ("ICA", "PIA"):
        m = reg_pt == target
        if m.sum() < 30:
            print(f"[markers] {target}: only {m.sum()} spots, skipping")
            continue
        inside, outside = L[m], L[~m]
        frac = (inside > 0).mean(0)
        score = inside.mean(0) - outside.mean(0)
        ok = keep & (frac >= a.min_frac)
        order = np.argsort(-np.where(ok, score, -np.inf))[:a.top]
        out[target] = [dict(gene=str(genes[i]), delta=float(score[i]),
                            mean_in=float(inside[:, i].mean()), mean_out=float(outside[:, i].mean()),
                            detected_frac=float(frac[i])) for i in order]
        print(f"\n[markers] {target} (n={m.sum()}) top {a.top} vs all other PT spots:")
        for d in out[target]:
            print(f"   {d['gene']:12s} delta={d['delta']:+.3f}  in={d['mean_in']:.3f} "
                  f"out={d['mean_out']:.3f}  detected={100*d['detected_frac']:.0f}%")

    # How the two regions rank each other's winners -- shows whether a marker is region-SPECIFIC
    # or just generally injury-related.
    if "ICA" in out and "PIA" in out:
        print("\n[markers] cross-check (delta in the other region):")
        d_ica = {g: s for g, s in zip(genes, L[reg_pt == "ICA"].mean(0) - L[reg_pt != "ICA"].mean(0))}
        d_pia = {g: s for g, s in zip(genes, L[reg_pt == "PIA"].mean(0) - L[reg_pt != "PIA"].mean(0))}
        for tag, lst in (("ICA", out["ICA"]), ("PIA", out["PIA"])):
            for d in lst[:5]:
                g = d["gene"]
                print(f"   {tag}-top {g:12s} ICA{d_ica[g]:+.3f}  PIA{d_pia[g]:+.3f}")
        for tag, lst in (("ICA", out["ICA"]), ("PIA", out["PIA"])):
            for d in lst:
                d["delta_ICA"], d["delta_PIA"] = float(d_ica[d["gene"]]), float(d_pia[d["gene"]])

    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[markers] -> {a.out}")


if __name__ == "__main__":
    main()
