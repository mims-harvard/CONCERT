"""
hier_pert_emb.py — Hierarchical Perturbation Embedding (Phase 2.3).

Design
------
Perturbation embeddings are factored into three additive levels:

    pert_emb_i^(s) = pathway_emb[g(i)] + gene_delta_i + sample_delta_i^(s)

Where:
  • pathway_emb[g(i)] : shared pathway-level embedding (all genes in same KEGG
                         pathway share a base vector). Promotes sharing across
                         related knockouts (e.g., all TGF-β pathway genes).
  • gene_delta_i      : gene-specific residual (learned independently per KO).
  • sample_delta_i^(s): sample×gene interaction (captures replicate-specific KO
                        effects). Set to zero for single-sample training (Phase 2).

Without pathway annotations, pathway_emb degenerates to a gene-level embedding
(equivalent to LORD's existing perturbation embedding). The key value-add is:
  (a) Explicit regularization structure: pathway_emb has a shared L2 penalty
      over genes in the same pathway.
  (b) Sample axis: sample_delta captures batch × perturbation interactions for
      multi-sample joint training (Phase 2.2).
  (c) The two-level decomposition encourages shared representations across
      biologically related knockouts, improving generalization to unseen targets.

Usage
-----
    from hier_pert_emb import HierarchicalPertEmbedding

    # At model init:
    hier_emb = HierarchicalPertEmbedding(
        n_genes=n_unique_perts,       # number of unique perturbation labels
        emb_dim=encoder_dim,          # embedding dimension
        n_samples=1,                  # 1 for single-sample, 8 for multi-sample
        pathway_map=None,             # optional: dict{pert_idx → pathway_idx}
        n_pathways=None,              # number of unique pathways
        dropout=0.1,
    )

    # In forward:
    pert_idx = cell_atts[:, 1]        # (B,) int — perturbation codes
    sample_idx = ...                   # (B,) int — sample/replicate index
    emb = hier_emb(pert_idx, sample_idx)  # (B, emb_dim)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPertEmbedding(nn.Module):
    """Three-level hierarchical perturbation embedding.

    Level 1: Pathway embedding   — shared across genes in same pathway.
    Level 2: Gene-specific delta — independent residual per KO gene.
    Level 3: Sample-gene delta   — replicate × gene interaction (for multi-sample).

    Parameters
    ----------
    n_genes     : int  — number of unique perturbation labels (including
                         "unassigned", "control", and all gene KO targets)
    emb_dim     : int  — embedding dimension (= LORD encoder_dim)
    n_samples   : int  — number of samples/replicates (1 for single-sample)
    pathway_map : Optional[np.ndarray] — (n_genes,) int mapping pert_idx → pathway_idx.
                  None means each gene is its own pathway (no sharing).
    n_pathways  : Optional[int] — number of unique pathways. Required if pathway_map != None.
    dropout     : float — dropout on output embedding
    sigma_noise : float — noise for regularization during training (à la LORD)
    """

    def __init__(
        self,
        n_genes: int,
        emb_dim: int,
        n_samples: int = 1,
        pathway_map: Optional[np.ndarray] = None,
        n_pathways: Optional[int] = None,
        dropout: float = 0.1,
        sigma_noise: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_genes   = n_genes
        self.emb_dim   = emb_dim
        self.n_samples = n_samples
        self.sigma_noise = sigma_noise

        # ── Level 1: Pathway embedding ─────────────────────────────────────
        if pathway_map is not None and n_pathways is not None:
            self.pathway_map = torch.tensor(pathway_map, dtype=torch.long)
            self.pathway_emb = nn.Embedding(n_pathways, emb_dim)
            nn.init.xavier_uniform_(self.pathway_emb.weight)
            self._has_pathways = True
            logging.info("[HierPertEmb] Using %d pathways for %d genes", n_pathways, n_genes)
        else:
            self.pathway_map = None
            # Without pathway annotations: pathway_emb is per-gene (= standard embedding)
            self.pathway_emb = nn.Embedding(n_genes, emb_dim)
            nn.init.xavier_uniform_(self.pathway_emb.weight)
            self._has_pathways = False
            logging.info("[HierPertEmb] No pathway map — gene-level pathway embedding (%d genes)", n_genes)

        # ── Level 2: Gene-specific delta ─────────────────────────────────
        # Initialized to zero; learns the residual not captured by pathway emb
        self.gene_delta = nn.Embedding(n_genes, emb_dim)
        nn.init.zeros_(self.gene_delta.weight)

        # ── Level 3: Sample × Gene delta (multi-sample) ──────────────────
        # Zero for single-sample; non-trivial for multi-sample Phase 2.2
        if n_samples > 1:
            self.sample_gene_delta = nn.Embedding(n_samples * n_genes, emb_dim)
            nn.init.zeros_(self.sample_gene_delta.weight)
        else:
            self.sample_gene_delta = None

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        pert_idx: torch.Tensor,    # (B,) int — perturbation code [0, n_genes)
        sample_idx: Optional[torch.Tensor] = None,  # (B,) int — sample index [0, n_samples)
    ) -> torch.Tensor:
        """Return (B, emb_dim) hierarchical perturbation embedding."""
        # Level 1: pathway embedding
        if self._has_pathways and self.pathway_map is not None:
            pw_idx = self.pathway_map.to(pert_idx.device)[pert_idx]  # (B,) pathway codes
            emb = self.pathway_emb(pw_idx)                           # (B, D)
        else:
            emb = self.pathway_emb(pert_idx)                         # (B, D)

        # Level 2: gene-specific delta
        emb = emb + self.gene_delta(pert_idx)                        # (B, D)

        # Level 3: sample × gene interaction (multi-sample only)
        if self.sample_gene_delta is not None and sample_idx is not None:
            # Flatten (sample_idx, pert_idx) → single index
            flat_idx = sample_idx * self.n_genes + pert_idx          # (B,)
            emb = emb + self.sample_gene_delta(flat_idx)             # (B, D)

        # Gaussian noise regularization during training (à la LORD)
        if self.training and self.sigma_noise > 0.0:
            emb = emb + torch.randn_like(emb) * self.sigma_noise

        return self.dropout(emb)                                      # (B, D)

    def pathway_regularization_loss(self, lambda_reg: float = 1e-4) -> torch.Tensor:
        """L2 penalty encouraging genes in the same pathway to share embeddings.
        Only meaningful when pathway_map is provided.
        Returns scalar tensor (0.0 if no pathway map).
        """
        if not self._has_pathways or self.pathway_map is None:
            return torch.tensor(0.0, device=self.pathway_emb.weight.device)

        # gene_delta should be small (encouraged by L2 penalty)
        return lambda_reg * (self.gene_delta.weight ** 2).sum()
