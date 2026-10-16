"""Inspect obs structure of the 3 datasets to design the CRISPR-noise analysis.
Lazy (backed) read of obs only -- does not load X."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import anndata as ad, numpy as np, collections, h5py

FILES = {
    "spac_1-1":   f"{_rp.DATA}/spacseq/raw/raw_output_1-1.h5ad",
    "spac_2-2":   f"{_rp.DATA}/spacseq/raw/raw_output_2-2.h5ad",
    "perturbmap": f"{_rp.DATA}perturbmap/raw/perturb_map_data_all_fullgene.h5ad",
}

def h5_obs_keys(path):
    with h5py.File(path, "r") as f:
        return list(f["obs"].keys())

for name, path in FILES.items():
    print("\n" + "=" * 70)
    print(f"{name}  {path}")
    print("=" * 70)
    a = ad.read_h5ad(path, backed="r")
    print("shape:", a.shape)
    print("obs columns:", list(a.obs.columns))
    # show value_counts for any column that looks perturbation/guide/control related
    for col in a.obs.columns:
        lc = col.lower()
        if any(t in lc for t in ["pert", "guide", "sgrna", "grna", "target", "ko",
                                 "control", "ntc", "assign", "tissue", "slide", "sample", "barcode", "gene"]):
            vc = a.obs[col].astype(str).value_counts()
            print(f"\n  [{col}] n_unique={vc.shape[0]}")
            print("   top:", dict(list(vc.items())[:12]))
            # flag control/ntc/unassigned-like tokens present
            toks = [v for v in vc.index if any(t in str(v).lower()
                    for t in ["control", "ntc", "non_target", "nontarget", "non-target",
                              "unassign", "unpert", "no_guide", "scramble", "safe"])]
            if toks:
                print("   control/NTC-like labels:", {t: int(vc[t]) for t in toks})
    a.file.close()
print("\nDONE")
