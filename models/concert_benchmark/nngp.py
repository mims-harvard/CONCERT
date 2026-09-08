# nngp.py
"""Variational nearest-neighbor GP local field (pert-agnostic) over a fixed kNN graph.

A deliberately simple "graph-smoothing local field with a shrinkage KL": per spot,
a kernel-weighted average of neighbor GP means plus a soft N(0,1) penalty. The call
site is fixed so this can be upgraded to a full Vecchia GP later without changes.
"""
from __future__ import annotations
import torch, torch.nn as nn

class NNGPLocal(nn.Module):
    def __init__(self, gp_dim: int, init_lengthscale: float = 1.0):
        super().__init__()
        # one positive lengthscale per GP dim (softplus-parameterized)
        self._raw_ls = nn.Parameter(torch.full((gp_dim,), float(init_lengthscale)))
        self.gp_dim = gp_dim

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._raw_ls) + 1e-4

    def forward(self, knn_dist_b: torch.Tensor, neighbor_vals: torch.Tensor):
        """knn_dist_b   : (B, k) distances to each spot's kNN (precomputed, static).
           neighbor_vals: (B, k, gp_dim) encoder GP means at the neighbors.
        Returns local mean (B, gp_dim) = kernel-weighted neighbor average,
        and a KL-style penalty pulling the local field toward 0 (regularizer).
        """
        ls = self.lengthscale.view(1, 1, -1)                     # (1,1,gp_dim)
        w = 1.0 / (1.0 + (knn_dist_b.unsqueeze(-1) / ls) ** 2)   # (B,k,gp_dim) Cauchy
        w = w / (w.sum(1, keepdim=True) + 1e-8)
        local_mean = (w * neighbor_vals).sum(1)                  # (B, gp_dim)
        kl = 0.5 * (local_mean ** 2).sum(-1).mean()              # soft N(0,1) prior
        return local_mean, kl
