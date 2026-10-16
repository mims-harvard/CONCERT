#!/usr/bin/env python3
"""SPAC-seq SEEN reference strategies under the shared gene rule (reference_strategies.py).

  * master (gene, slide, niche) cells = the CONCERT seen records (<outdir>/spac_seen_lfc_dist_per_record.csv)
  * scored genes: build_gene_mask is ANDed with CONCERT's model gene list (shared rule). The kNN / PCA feature
    columns are the slide HVG columns.
STATE is not rescored: its seen predictions (bof695 scratch) are not readable (permission denied).
Usage: reference_strategies_seen.py --outdir DIR [--sampled_output True]   (PYTHONHASHSEED=0)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_strategies as nb  # noqa: E402
from shared_gene_rule import load_concert_genes  # noqa: E402

GENES = set(np.asarray(load_concert_genes()).astype(str).tolist())
_orig_mask = nb.build_gene_mask


def build_gene_mask_shared(gene_names, ko_gene, hvg_indices=None):
    keep = _orig_mask(gene_names, ko_gene, hvg_indices)
    names = np.asarray(gene_names).astype(str)
    return keep & np.fromiter((g in GENES for g in names), dtype=bool, count=len(names))


nb.build_gene_mask = build_gene_mask_shared

if __name__ == "__main__":
    outdir = Path(sys.argv[sys.argv.index("--outdir") + 1])
    if "--sampled_output" in sys.argv:
        nb.SAMPLED = sys.argv[sys.argv.index("--sampled_output") + 1].lower() == "true"
    print(f"[seen reference strategies] sampled_output={nb.SAMPLED}", flush=True)
    nb.MASTER_SPAC = outdir / "spac_seen_lfc_dist_per_record.csv"
    assert nb.MASTER_SPAC.is_file(), nb.MASTER_SPAC
    sys.argv = [sys.argv[0], "--datasets", "spac", "--methods", "mean_target,mean_niche,local_knn,gex_knn",
                "--outdir", str(outdir)]
    nb.main()
