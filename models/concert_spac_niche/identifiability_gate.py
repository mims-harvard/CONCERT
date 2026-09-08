"""Pre-training identifiability gate for the paracrine / cell-autonomous split.

Question: are guides spatially clustered? If a spot's OWN guide is strongly
correlated with its NEIGHBORS' guide composition, then the cell-autonomous
f_pert and the paracrine s_signature become partially confounded (both driven
by the same spatial guide structure). This script quantifies that.

Metric: neighbor-guide concordance = fraction of (spot, kNN-neighbor) pairs
whose neighbor carries the SAME perturbation code as the spot, compared to the
random-mixing baseline sum_k p_k^2 (concordance expected if guides were placed
spatially at random). ratio = concordance / baseline.
  ratio ~ 1   -> well mixed; paracrine vs cell-autonomous are separable.
  ratio >> 1  -> clonally clustered; report a confidence caveat on the split.

Reads the precomputed graph (knn_idx) and graph inputs (pert_idx) from A1/A2.
CPU-only, seconds to run.
"""
from __future__ import annotations
import argparse, logging
import numpy as np


def neighbor_guide_concordance(knn_idx: np.ndarray, pert_idx: np.ndarray):
    """Returns (concordance, baseline, ratio)."""
    same = (pert_idx[knn_idx] == pert_idx[:, None]).mean()
    _, counts = np.unique(pert_idx, return_counts=True)
    p = counts / counts.sum()
    baseline = float((p ** 2).sum())
    return float(same), baseline, float(same / max(baseline, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_npz", required=True, help="npz with knn_idx (N,k)")
    ap.add_argument("--graph_inputs_npz", required=True, help="npz with pert_idx (N,)")
    ap.add_argument("--clustered_ratio", type=float, default=2.0,
                    help="ratio above which guides are deemed clonally clustered")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    knn_idx  = np.load(a.graph_npz)["knn_idx"]
    pert_idx = np.load(a.graph_inputs_npz)["pert_idx"]
    assert knn_idx.shape[0] == pert_idx.shape[0], "graph / inputs row mismatch"

    conc, base, ratio = neighbor_guide_concordance(knn_idx, pert_idx)
    logging.info("N=%d  k=%d  n_pert=%d", knn_idx.shape[0], knn_idx.shape[1],
                 int(pert_idx.max()) + 1)
    logging.info("neighbor-guide concordance = %.4f", conc)
    logging.info("random-mixing baseline     = %.4f", base)
    logging.info("ratio (conc / baseline)    = %.2f", ratio)
    if ratio > a.clustered_ratio:
        logging.info("VERDICT: CLUSTERED (ratio>%.1f) -> f_pert / s_signature may partially "
                     "confound; report as a confidence caveat on the separation.", a.clustered_ratio)
    else:
        logging.info("VERDICT: WELL-MIXED -> cell-autonomous vs paracrine split is identifiable.")


if __name__ == "__main__":
    main()
