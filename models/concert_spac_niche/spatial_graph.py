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


    def with_ko_from(self, pos: "torch.Tensor", pert_idx: "torch.Tensor",
                     is_emitter: "torch.Tensor", r_max: float) -> "SpatialGraph":
        """Copy of this graph whose KO-neighbour lists are rebuilt from GIVEN labels.

        The stored KO lists describe the OBSERVED lesion. A counterfactual moves the lesion
        somewhere else, so reusing them would spill the real lesion's signal into a brain that,
        in the counterfactual, was never struck there. Rebuilding from the counterfactual labels
        is what lets spillover follow the imposed perturbation. The kNN graph is geometry only,
        so it is shared unchanged.
        """
        dev = pos.device
        emit = torch.nonzero(is_emitter.reshape(-1).bool(), as_tuple=False).reshape(-1)
        N = int(pos.shape[0])
        if emit.numel() == 0:
            z = torch.zeros(0, dtype=torch.long, device=dev)
            return SpatialGraph(self.knn_idx, self.knn_dist,
                                torch.zeros(N + 1, dtype=torch.long, device=dev),
                                z, torch.zeros(0, dtype=torch.float32, device=dev), z)
        ep = pos[emit]
        counts = torch.zeros(N, dtype=torch.long, device=dev)
        js, ds, ps, rows = [], [], [], []
        step = 4096                                    # chunked: N x n_emitter can be large
        for a in range(0, N, step):
            b = min(a + step, N)
            d = torch.cdist(pos[a:b], ep)              # (chunk, n_emit)
            hit = (d <= r_max)
            hit &= (torch.arange(a, b, device=dev).unsqueeze(1) != emit.unsqueeze(0))  # no self-emit
            ri, ci = torch.nonzero(hit, as_tuple=True)
            if ri.numel():
                rows.append(ri + a); js.append(emit[ci]); ds.append(d[ri, ci])
                ps.append(pert_idx.reshape(-1)[emit[ci]])
        if not rows:
            z = torch.zeros(0, dtype=torch.long, device=dev)
            return SpatialGraph(self.knn_idx, self.knn_dist,
                                torch.zeros(N + 1, dtype=torch.long, device=dev),
                                z, torch.zeros(0, dtype=torch.float32, device=dev), z)
        rows = torch.cat(rows); js = torch.cat(js); ds = torch.cat(ds); ps = torch.cat(ps)
        order = torch.argsort(rows, stable=True)       # CSR needs rows grouped and ordered
        rows, js, ds, ps = rows[order], js[order], ds[order], ps[order]
        counts = torch.bincount(rows, minlength=N)
        indptr = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
        return SpatialGraph(self.knn_idx, self.knn_dist, indptr, js, ds.float(), ps.long())


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
        # NORMALISED. This used to be a raw .sum(0) over every emitter within r_max, so the shift a
        # spot received grew with the NUMBER of perturbed neighbours rather than with proximity to
        # the lesion. With r_max=5 on a slide spanning ~4.5 x 6.6 that gave the term effectively
        # unbounded capacity: it absorbed the entire stroke effect during training and left f_pert
        # to learn a NEGATIVE correction (real PT1 lesion, f_pert alone: Lcn2 -0.574, Gfap -1.248,
        # Vim -0.460, against positive observed LFCs). A concentration field must saturate, not
        # accumulate. Divide by (1 + total weight), NOT by the total weight: a plain mean is
        # degenerate when there is only one emitter type (every spot with any emitter inside
        # r_max would receive the identical sig, a binary indicator). This form decays with
        # distance when exposure is low and saturates at sig when the spot is surrounded.
        out[b] = (w.unsqueeze(1) * sig_all[p]).sum(0) / (1.0 + w.sum())   # (G,)
    return out
