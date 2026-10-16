#!/usr/bin/env python3
"""Genefix run of the SPAC-seq SEEN reference strategies (rescore_collab_naive_baselines.py, unchanged code).

Two settings differ from the lsfix run that produced Outputs/spac_seq_lsfix/spac_seen_naive_baselines_our_rule.csv:
  * master (gene, slide, niche) cells = the genefix CONCERT seen records (v4_aggregate_genefix/spac_seen_lfc_dist_per_record.csv)
  * scored genes: build_gene_mask is ANDed with CONCERT's model gene list (shared rule). The kNN / PCA feature
    columns (slide HVG columns) are unchanged.
STATE is not rescored: its seen predictions (bof695 scratch) are not readable (permission denied).
Usage: rescore_collab_naive_baselines_seen_genefix.py --outdir DIR [--sampled_output True]   (PYTHONHASHSEED=0)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_collab_naive_baselines as nb  # noqa: E402
from genefix_wrap import load_concert_genes  # noqa: E402

GENES = set(np.asarray(load_concert_genes()).astype(str).tolist())
_orig_mask = nb.build_gene_mask


def build_gene_mask_genefix(gene_names, ko_gene, hvg_indices=None):
    keep = _orig_mask(gene_names, ko_gene, hvg_indices)
    names = np.asarray(gene_names).astype(str)
    return keep & np.fromiter((g in GENES for g in names), dtype=bool, count=len(names))


nb.build_gene_mask = build_gene_mask_genefix

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
