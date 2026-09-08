# spatial_graph.py
from __future__ import annotations
import numpy as np, torch

class SpatialGraph:
    """Holds precomputed neighbor structures as device tensors, keyed by GLOBAL spot idx."""
    def __init__(self, knn_idx, knn_dist, ko_indptr, ko_j, ko_d, ko_p):
        self.knn_idx = knn_idx; self.knn_dist = knn_dist
        self.ko_indptr = ko_indptr; self.ko_j = ko_j; self.ko_d = ko_d; self.ko_p = ko_p

    @classmethod
    def load(cls, path: str, device: str) -> "SpatialGraph":
        d = np.load(path)
        t = lambda a, dt: torch.as_tensor(d[a], dtype=dt, device=device)
        return cls(t("knn_idx", torch.long), t("knn_dist", torch.float32),
                   t("ko_indptr", torch.long), t("ko_j", torch.long),
                   t("ko_d", torch.float32), t("ko_p", torch.long))


def paracrine_log_shift(graph: "SpatialGraph", global_idx: torch.Tensor,
                        radius_per_pert: torch.Tensor, s_signature: torch.nn.Module,
                        pert_emb_table: torch.Tensor, clamp: float = 5.0) -> torch.Tensor:
    """Sum over KO-neighbors(i) of K_{r_{p_j}}(d_ij) * s_signature(pert_emb[p_j]) -> (B, G).

    global_idx       : (B,) long -- receiver spots' GLOBAL indices (== forward's sample_index).
    radius_per_pert  : (n_pert,) positive -- learnable Cauchy radius r_p.
    s_signature      : nn.Linear(encoder_dim -> G).
    pert_emb_table   : (n_pert, encoder_dim) -- precomputed LORD pert embeddings this step.
    """
    B = global_idx.shape[0]
    G = s_signature.out_features
    out = global_idx.new_zeros((B, G), dtype=pert_emb_table.dtype)
    # Per-pert spillover gene signature for all perts at once: (n_pert, G)
    sig_all = torch.clamp(s_signature(pert_emb_table), -clamp, clamp)
    for b in range(B):
        i = int(global_idx[b].item())
        s, e = int(graph.ko_indptr[i].item()), int(graph.ko_indptr[i + 1].item())
        if e == s:
            continue
        d = graph.ko_d[s:e]                       # (m,)
        p = graph.ko_p[s:e]                       # (m,)
        r = radius_per_pert[p]                    # (m,)
        w = 1.0 / (1.0 + (d / r) ** 2)            # Cauchy weights (m,)
        out[b] = (w.unsqueeze(1) * sig_all[p]).sum(0)   # (G,)
    return out
