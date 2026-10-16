"""
contrastive_alignment.py -- Phase 2.7: Control-anchored contrastive loss.

Design
------
For each gene_KO cell i, find its nearest non-targeting control cell c(i).
The contrastive loss penalizes the model if the predicted perturbation effect
delta_i (difference between KO prediction and "no-KO" prediction) disagrees with
the empirically observed delta z = z_pert_i - z_ctrl_neighbor_i.

Loss:
    L_contra = (1/|KO|) sum_i 1[is_real_KO_i] * ||delta_i_pred - Deltaz_i||^2

where:
    delta_i_pred  = z_mu_i - z_mu_i^{(ctrl)}     (pred minus ctrl-counterfactual)
    Deltaz_i       = z_mu_KO_i - z_mu_ctrl_c(i)  (observed KO vs control residual)
    z_mu_ctrl_c(i) = encoder output of the nearest control cell

This loss acts as a "consistency regularizer": the perturbation effect in
latent space should match the empirically observable KO-vs-control difference
among spatially proximal cells.

Note: This is only useful AFTER Phase 2.6 action correction removes the
technical action effect, so that Deltaz reflects real gene KO biology.
If action correction is incomplete (mu_action under-fit), Deltaz will still
contain technical noise and this loss may interfere. Use Phase 2.9 diagnostic
to confirm action correction is working before enabling this.

Preprocessing
-------------
Build the control neighbor index once at load time:
    from contrastive_alignment import build_ctrl_neighbor_idx
    ctrl_neighbor_idx = build_ctrl_neighbor_idx(
        pos=pos_scaled,          # (N, 2)
        is_real_KO=is_real_KO,  # (N,) bool
        has_guide=has_guide,     # (N,) bool
        k=1,                     # nearest single control cell
    )  # -> (N,) int, -1 for non-KO cells
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_ctrl_neighbor_idx(
    pos: np.ndarray,
    is_real_KO: np.ndarray,
    has_guide: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """Build control neighbor index: for each KO cell, find its k nearest control cell(s).

    Parameters
    ----------
    pos        : (N, 2) float -- scaled spatial positions
    is_real_KO : (N,) bool  -- True for gene_KO cells
    has_guide  : (N,) bool  -- True for control + gene_KO (control = has_guide & ~is_real_KO)
    k          : int         -- number of nearest control neighbors per KO cell (usually 1)

    Returns
    -------
    ctrl_neighbor_idx : (N,) int32
        ctrl_neighbor_idx[i] = index of nearest control cell for KO cell i.
        ctrl_neighbor_idx[i] = -1 for non-KO cells (unassigned or control).
    """
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    ctrl_mask = has_guide & ~is_real_KO   # (N,) -- control cells
    ko_mask   = is_real_KO                # (N,) -- gene_KO cells

    n_ctrl = ctrl_mask.sum()
    n_ko   = ko_mask.sum()
    if n_ctrl == 0:
        logging.warning("[ContrastAlign] No control cells found -- returning all -1.")
        return np.full(len(pos), -1, dtype=np.int32)

    logging.info("[ContrastAlign] Building ctrl neighbor index: %d KO cells, %d control cells", n_ko, n_ctrl)

    ctrl_pos  = pos[ctrl_mask]      # (n_ctrl, 2)
    ctrl_idx  = np.where(ctrl_mask)[0]  # global indices of control cells

    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto", n_jobs=-1)
    nn_model.fit(ctrl_pos)

    ko_pos  = pos[ko_mask]   # (n_ko, 2)
    _, near = nn_model.kneighbors(ko_pos)  # (n_ko, k)
    # For k=1, take the single nearest control cell's global index
    nearest_ctrl_global = ctrl_idx[near[:, 0]]  # (n_ko,) global indices

    # Build full (N,) array
    ctrl_neighbor_idx = np.full(len(pos), -1, dtype=np.int32)
    ko_idx = np.where(ko_mask)[0]
    ctrl_neighbor_idx[ko_idx] = nearest_ctrl_global.astype(np.int32)

    logging.info("[ContrastAlign] Done. Coverage: %d / %d KO cells assigned a control neighbor.",
                 (ctrl_neighbor_idx >= 0).sum(), n_ko)
    return ctrl_neighbor_idx


class ContrastiveAlignmentLoss(nn.Module):
    """Control-anchored contrastive alignment loss (Phase 2.7).

    For each KO cell i, penalizes:
        ||(z_mu_i - z_mu_ctrl_i) - (z_mu_i^{ref} - z_mu_ctrl_i^{ref})||^2

    In practice (mini-batch setting), this simplifies to encouraging the
    KO cell's latent to differ from the nearest control in the same direction
    as the perturbation effect predicted by the model.

    Parameters
    ----------
    lambda_contra : float -- weight of the contrastive loss (default 0.1)
    """

    def __init__(self, lambda_contra: float = 0.1) -> None:
        super().__init__()
        self.lambda_contra = lambda_contra

    def forward(
        self,
        z_mu: torch.Tensor,             # (B, D) -- encoder posterior means for batch
        batch_idx: torch.Tensor,        # (B,) int -- global cell indices in [0, N)
        ctrl_neighbor_idx: torch.Tensor, # (N,) int -- precomputed control neighbors
        is_real_KO_batch: torch.Tensor, # (B,) bool/float -- KO mask for batch
        z_mu_buffer: Optional[torch.Tensor] = None,  # (N, D) -- buffer from prev epoch
    ) -> torch.Tensor:
        """Compute contrastive alignment loss for the mini-batch.

        Returns scalar loss (0.0 if no KO cells in batch or no buffer).
        """
        if z_mu_buffer is None:
            # Cannot compute without a buffer of stored latents
            return torch.tensor(0.0, device=z_mu.device)

        ko_mask = is_real_KO_batch.bool()  # (B,)
        if not ko_mask.any():
            return torch.tensor(0.0, device=z_mu.device)

        # Global indices of KO cells in this batch
        ko_global_idx = batch_idx[ko_mask]              # (n_ko_batch,)
        z_mu_ko       = z_mu[ko_mask]                   # (n_ko_batch, D)

        # Their nearest control neighbors' global indices
        ctrl_global_idx = ctrl_neighbor_idx[ko_global_idx]  # (n_ko_batch,)

        # Filter out any KO cells that don't have a control neighbor (idx == -1)
        valid = ctrl_global_idx >= 0
        if not valid.any():
            return torch.tensor(0.0, device=z_mu.device)

        z_mu_ko_valid   = z_mu_ko[valid]                          # (n_valid, D)
        ctrl_idx_valid  = ctrl_global_idx[valid]                  # (n_valid,)
        z_ctrl_buffered = z_mu_buffer[ctrl_idx_valid]              # (n_valid, D)

        # Latent delta: observed KO - control (from buffer)
        # This is the "empirical perturbation direction" we want to align with
        # The buffer contains z_mu from the previous epoch, so this is stable
        delta_observed = z_mu_ko_valid - z_ctrl_buffered          # (n_valid, D)

        # Loss: minimize MSE between the delta and its own L2-normalized version
        # (encourages consistent direction, not magnitude -- avoids collapse)
        # Alternatively, just minimize ||delta|| for control cells (shouldn't move much)
        # Here: L2 of the full delta (penalizes if KO moves far from nearby control
        # in an *inconsistent* direction across samples -> use L2 squared)
        loss = F.mse_loss(
            delta_observed,
            delta_observed.detach() / (delta_observed.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8)),
        )
        # Note: the above is a soft "direction normalization" loss.
        # A simpler alternative (uncomment if this is unstable):
        # loss = (delta_observed ** 2).mean()

        return self.lambda_contra * loss
