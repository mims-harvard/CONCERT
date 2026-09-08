"""Dump (pos, pert_idx, is_real_KO) in EXACT training (post-filter+normalize) order.

Reuses make_umap.load_data so the cell count (507237 for sample 1-1) and ordering
match training exactly, avoiding the LORD index-alignment gotcha. Also computes a
data-driven r_max (radius giving a target average KO-neighbor degree) so the
paracrine graph stays tractable.

load_data returns a TUPLE:
  (adata, pos_batched, cell_atts, batch, pert_name, cutoff_t,
   perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx)
where pos_batched = [pos_scaled(N,2), batch_onehot(N,n_batch)] and pos_scaled is
MinMax-scaled to LOC_RANGE (=20). The spatial coords are pos_batched[:, :2].
"""
from __future__ import annotations
import argparse, logging
import numpy as np
from sklearn.neighbors import NearestNeighbors

from make_umap import load_data


def suggest_r_max(pos: np.ndarray, is_real_KO: np.ndarray, target_degree: int = 50,
                  sample: int = 5000, seed: int = 0) -> float:
    """Radius s.t. an average receiver sees ~target_degree KO emitters.

    Estimated from the distance to the target_degree-th nearest KO emitter,
    median over a random sample of receiver spots.
    """
    emit = pos[is_real_KO.astype(bool)]
    if emit.shape[0] <= target_degree:
        # too few emitters -- fall back to a generous fraction of the field
        return float(0.5)
    nn = NearestNeighbors(n_neighbors=target_degree).fit(emit)
    rng = np.random.default_rng(seed)
    idx = rng.choice(pos.shape[0], size=min(sample, pos.shape[0]), replace=False)
    d, _ = nn.kneighbors(pos[idx])
    return float(np.median(d[:, -1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--target_degree", type=int, default=50,
                   help="used only to suggest r_max; printed for the build step")
    p.add_argument("--hvg_max_rank", type=int, default=10000)
    p.add_argument("--min_cells_per_pert", type=int, default=25)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    (adata, pos_batched, cell_atts, batch, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(
         a.data_file,
         hvg_max_rank=a.hvg_max_rank,
         min_cells_per_pert=a.min_cells_per_pert,
     )

    pos        = np.ascontiguousarray(pos_batched[:, :2], dtype=np.float32)  # scaled spatial coords
    pert_idx   = np.asarray(perturb_idx, dtype=np.int64)
    is_real_KO = np.asarray(is_real_KO, dtype=np.int8)
    assert pos.shape[0] == pert_idx.shape[0] == is_real_KO.shape[0], "row mismatch"

    r_max = suggest_r_max(pos, is_real_KO, target_degree=a.target_degree)

    np.savez(a.out_npz, pos=pos, pert_idx=pert_idx, is_real_KO=is_real_KO,
             n_batch=np.int64(n_batch), suggested_r_max=np.float32(r_max))
    logging.info("Dumped %d spots | %d KO emitters | n_pert=%d | field=[%.2f,%.2f]^2",
                 pos.shape[0], int(is_real_KO.sum()), n_batch,
                 float(pos.min()), float(pos.max()))
    logging.info("Suggested r_max for ~%d KO-neighbors/spot = %.4f", a.target_degree, r_max)


if __name__ == "__main__":
    main()
