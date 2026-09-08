# build_spatial_graph.py
"""Precompute spatial neighbor graphs for CONCERT-spac Phase 2.8.

Two products, both keyed by GLOBAL spot index (post-normalize, must match the
LORD basal table / sample_index = arange(N)):
  - knn graph (all-neighbor)  -> NNGP local field
  - KO-neighbor list per spot -> paracrine exposure convolution

CPU-only; no torch. Run once, save .npz, load at train time.
"""
from __future__ import annotations
import argparse, logging
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_graph(pos: np.ndarray, pert_idx: np.ndarray, is_real_KO: np.ndarray,
                r_max: float, knn_k: int) -> dict:
    """Build kNN + KO-neighbor structures.

    pos        : (N, 2) float spatial coords (scaled the SAME way as training input x).
    pert_idx   : (N,) int per-spot perturbation code.
    is_real_KO : (N,) {0,1} -- emitter mask (only real KO spots emit spillover).
    r_max      : radius cutoff for KO-neighbor inclusion (paracrine).
    knn_k      : k for the all-neighbor NNGP graph.

    Returns dict with:
      knn_idx  : (N, knn_k) int   -- NNGP neighbor indices (excl. self)
      knn_dist : (N, knn_k) float
      ko_neighbors : list[N] of list[(idx_j:int, d_ij:float, p_j:int)] within r_max
    """
    N = pos.shape[0]

    # All-neighbor kNN (for NNGP). +1 then drop self.
    nn_all = NearestNeighbors(n_neighbors=min(knn_k + 1, N)).fit(pos)
    dist_all, idx_all = nn_all.kneighbors(pos)
    knn_idx, knn_dist = idx_all[:, 1:], dist_all[:, 1:]   # drop self (col 0)

    # KO-neighbor radius graph: only emitters are candidates.
    emitter_mask = is_real_KO.astype(bool)
    emit_pos = pos[emitter_mask]
    emit_global = np.nonzero(emitter_mask)[0]
    emit_pert = pert_idx[emitter_mask]

    ko_neighbors: list = [[] for _ in range(N)]
    if emit_pos.shape[0] > 0:
        nn_ko = NearestNeighbors(radius=r_max).fit(emit_pos)
        dists, locs = nn_ko.radius_neighbors(pos)   # ragged
        for i in range(N):
            for loc, d in zip(locs[i], dists[i]):
                j = int(emit_global[loc])
                if j == i:
                    continue   # a KO spot does not self-emit
                ko_neighbors[i].append((j, float(d), int(emit_pert[loc])))

    return {"knn_idx": knn_idx.astype(np.int64),
            "knn_dist": knn_dist.astype(np.float32),
            "ko_neighbors": ko_neighbors}


def to_csr_arrays(ko_neighbors: list) -> dict:
    """Flatten ragged KO-neighbor lists into CSR arrays for compact storage / fast gather."""
    indptr = np.zeros(len(ko_neighbors) + 1, dtype=np.int64)
    j_list, d_list, p_list = [], [], []
    for i, nbrs in enumerate(ko_neighbors):
        indptr[i + 1] = indptr[i] + len(nbrs)
        for j, d, p in nbrs:
            j_list.append(j); d_list.append(d); p_list.append(p)
    return {"ko_indptr": indptr,
            "ko_j": np.asarray(j_list, dtype=np.int64),
            "ko_d": np.asarray(d_list, dtype=np.float32),
            "ko_p": np.asarray(p_list, dtype=np.int64)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_in_npz", required=True,
                   help="npz with pos (N,2), pert_idx (N,), is_real_KO (N,) -- post-normalize order")
    p.add_argument("--out_npz", required=True)
    p.add_argument("--r_max", type=float, required=True)
    p.add_argument("--knn_k", type=int, default=15)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    d = np.load(a.graph_in_npz)
    g = build_graph(d["pos"], d["pert_idx"], d["is_real_KO"], r_max=a.r_max, knn_k=a.knn_k)
    csr = to_csr_arrays(g["ko_neighbors"])
    np.savez(a.out_npz, knn_idx=g["knn_idx"], knn_dist=g["knn_dist"], **csr)
    n_edges = int(csr["ko_indptr"][-1])
    logging.info("Saved %s | N=%d | knn_k=%d | KO-edges=%d (avg %.1f/spot)",
                 a.out_npz, g["knn_idx"].shape[0], g["knn_idx"].shape[1],
                 n_edges, n_edges / max(1, g["knn_idx"].shape[0]))


if __name__ == "__main__":
    main()
