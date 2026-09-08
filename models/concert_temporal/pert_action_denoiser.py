"""
Perturbation Action Denoiser (PAD) for CONCERT
===============================================

Problem
-------
In spatial CRISPR-screen data (e.g. SPAC-seq Visium HD), ALL perturbed spots
share a systematic technical batch effect caused by the guide-delivery / Cas9
machinery -- independent of *which* gene was knocked out.  CONCERT's existing
LORD encoder has a `perturbation` embedding (gene-specific) but no factor for
this shared "perturbation action" noise.  The artifact therefore leaks into the
basal latent `z`, making the basal representation non-exchangeable between
perturbed and control cells.

Solution
--------
PAD adds three coordinated mechanisms during pretraining:

1. **Action embedding** -- a small binary embedding (0=control/unassigned,
   1=any KO) that is concatenated to LORD's total_latent.  This gives the
   encoder an explicit route to absorb the shared action effect.

2. **Adversarial (GRL) loss** -- a gradient-reversal adversary that tries to
   predict `is_perturbed` from `z_basal`.  The reversed gradient makes the
   encoder actively push the action effect *out* of `z_basal`.

3. **MMD loss** -- Maximum Mean Discrepancy between the `z_basal` distributions
   of control and perturbed mini-batch samples.  Directly penalises distributional
   divergence in the basal space.

Counterfactual denoising
------------------------
At inference, call `pad.clean_action_embedding()` which returns the embedding
for `is_perturbed = 0` (control).  Substituting this for all target cells
projects their representation into the clean, unperturbed-action space before
decoding.

Usage in CONCERT
----------------
    pad = PerturbActionDenoiser(action_dim=64, basal_dim=encoder_dim,
                                lambda_mmd=1.0, lambda_adv=0.1)
    # In forward():
    #   s_action = pad(is_perturbed)                    # (B, 64)
    #   extended = torch.cat([lord_total, s_action], -1) # (B, 768+64)
    #   ...
    #   adv_l = pad.adversarial_loss(z_basal, is_perturbed)
    #   mmd_l = pad.mmd_loss(z_basal, is_perturbed)
    #   total_loss += lambda_adv * adv_l + lambda_mmd * mmd_l
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------
class _GRL(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward: identity.
    Backward: multiply gradient by -alpha (reverses & scales).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GRL.apply(x, alpha)


# ---------------------------------------------------------------------------
# RBF-MMD
# ---------------------------------------------------------------------------
def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Radial-basis-function (Gaussian) kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))."""
    diff = x.unsqueeze(1) - y.unsqueeze(0)          # (n, m, d)
    dist_sq = (diff ** 2).sum(dim=-1)                # (n, m)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas: tuple[float, ...] = (1.0, 2.0, 5.0)) -> torch.Tensor:
    """Unbiased multi-scale RBF-MMD^2 between samples x and y.

    Parameters
    ----------
    x : (n, d) tensor -- samples from distribution P (e.g. control basal latents)
    y : (m, d) tensor -- samples from distribution Q (e.g. perturbed basal latents)
    sigmas : bandwidth values for multi-scale kernel

    Returns
    -------
    Scalar MMD^2 (≥ 0; equals 0 iff P == Q).
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return x.new_zeros(())

    mmd = x.new_zeros(())
    for s in sigmas:
        kxx = _rbf_kernel(x, x, s).mean()
        kyy = _rbf_kernel(y, y, s).mean()
        kxy = _rbf_kernel(x, y, s).mean()
        mmd = mmd + kxx + kyy - 2.0 * kxy
    return mmd / len(sigmas)


# ---------------------------------------------------------------------------
# PAD module
# ---------------------------------------------------------------------------
class PerturbActionDenoiser(nn.Module):
    """Perturbation Action Denoiser.

    Parameters
    ----------
    action_dim : int
        Dimension of the binary action embedding (appended to LORD total_latent).
        Recommended: 32-64.
    basal_dim : int
        Dimension of LORD's basal latent `z` (= encoder_dim in CONCERT).
    lambda_mmd : float
        Weight for the MMD loss term.  Start with 1.0 and tune.
    lambda_adv : float
        Weight for the adversarial (GRL) loss term.  Start with 0.1.
    adv_hidden : int
        Hidden dim of the adversarial classifier.
    grl_alpha : float
        Gradient reversal scale.  Increase gradually during training (schedule
        via `set_grl_alpha()`).
    mmd_sigmas : tuple
        Bandwidth values for multi-scale RBF-MMD.
    """

    def __init__(
        self,
        action_dim: int = 64,
        basal_dim: int = 256,
        lambda_mmd: float = 1.0,
        lambda_adv: float = 0.1,
        adv_hidden: int = 64,
        grl_alpha: float = 1.0,
        mmd_sigmas: tuple[float, ...] = (1.0, 2.0, 5.0),
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.lambda_mmd = lambda_mmd
        self.lambda_adv = lambda_adv
        self.grl_alpha = grl_alpha
        self.mmd_sigmas = mmd_sigmas

        # Binary action embedding: 0 = control/unassigned, 1 = any KO/perturbed
        self.action_embedding = nn.Embedding(2, action_dim)
        nn.init.xavier_uniform_(self.action_embedding.weight)

        # Adversarial classifier: predict is_perturbed from basal latent
        # (gradient reversal makes encoder adversarially push action OUT of basal)
        self.adv_classifier = nn.Sequential(
            nn.Linear(basal_dim, adv_hidden),
            nn.ELU(),
            nn.Linear(adv_hidden, 1),
        )
        for m in self.adv_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward: embed the is_perturbed flag
    # ------------------------------------------------------------------
    def forward(self, is_perturbed: torch.Tensor) -> torch.Tensor:
        """Return action embedding for the given binary is_perturbed labels.

        Parameters
        ----------
        is_perturbed : (B,) int tensor -- 0 for control/unassigned, 1 for any KO

        Returns
        -------
        (B, action_dim) float tensor
        """
        return self.action_embedding(is_perturbed.long())

    def clean_action_embedding(self, n: int, device: torch.device | str) -> torch.Tensor:
        """Return the 'clean' (is_perturbed=0) action embedding repeated n times.

        Use this during counterfactual prediction to project all cells into the
        unperturbed-action space.

        Parameters
        ----------
        n : int -- number of spots / cells
        device : target device

        Returns
        -------
        (n, action_dim) tensor with action embedding for is_perturbed=0
        """
        idx = torch.zeros(n, dtype=torch.long, device=device)
        return self.action_embedding(idx)

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------
    def adversarial_loss(
        self,
        z_basal: torch.Tensor,
        is_perturbed: torch.Tensor,
        *,
        min_per_class: int = 4,
    ) -> torch.Tensor:
        """Adversarial loss via Gradient Reversal Layer.

        The classifier tries to predict `is_perturbed` from `z_basal`.
        Gradient reversal makes the encoder push action information OUT of `z_basal`.

        Parameters
        ----------
        z_basal : (B, basal_dim) -- LORD basal latent for the mini-batch
        is_perturbed : (B,) int -- 0/1 labels
        min_per_class : skip loss if fewer than this many samples per class
                        (avoids unstable single-sample batches)

        Returns
        -------
        Scalar adversarial BCE loss (reversed gradient already applied).
        """
        ip = is_perturbed.float()
        # Skip if class is degenerate in this mini-batch
        n_pert = ip.sum().item()
        n_ctrl = (1.0 - ip).sum().item()
        if n_pert < min_per_class or n_ctrl < min_per_class:
            return z_basal.new_zeros(())

        z_rev = gradient_reversal(z_basal, self.grl_alpha)
        logits = self.adv_classifier(z_rev).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, ip)

    def mmd_loss(
        self,
        z_basal: torch.Tensor,
        is_perturbed: torch.Tensor,
        *,
        min_per_class: int = 4,
    ) -> torch.Tensor:
        """MMD loss between basal latents of control vs perturbed spots.

        Directly penalises distributional divergence in the basal space.

        Parameters
        ----------
        z_basal : (B, basal_dim)
        is_perturbed : (B,) int 0/1
        min_per_class : skip if fewer than this many samples per class

        Returns
        -------
        Scalar MMD^2 value.
        """
        ip = is_perturbed.bool()
        z_ctrl = z_basal[~ip]
        z_pert = z_basal[ip]

        if z_ctrl.shape[0] < min_per_class or z_pert.shape[0] < min_per_class:
            return z_basal.new_zeros(())

        return mmd_rbf(z_ctrl, z_pert, sigmas=self.mmd_sigmas)

    def combined_loss(
        self,
        z_basal: torch.Tensor,
        is_perturbed: torch.Tensor,
        *,
        min_per_class: int = 4,
    ) -> torch.Tensor:
        """Weighted sum: lambda_mmd * MMD + lambda_adv * adversarial.

        Call this in CONCERT.forward() and add to the total ELBO.
        """
        mmd = self.mmd_loss(z_basal, is_perturbed, min_per_class=min_per_class)
        adv = self.adversarial_loss(z_basal, is_perturbed, min_per_class=min_per_class)
        return self.lambda_mmd * mmd + self.lambda_adv * adv

    def set_grl_alpha(self, alpha: float) -> None:
        """Update gradient-reversal scale (call this each epoch for scheduling)."""
        self.grl_alpha = alpha


# ---------------------------------------------------------------------------
# GRL alpha schedule (Ganin et al. 2015 ramp)
# ---------------------------------------------------------------------------
def grl_alpha_schedule(epoch: int, max_epochs: int, gamma: float = 10.0) -> float:
    """Smoothly ramp GRL alpha from ~0 to 1 over training.

    Uses the schedule from Ganin & Lempitsky (2015):
        alpha = 2 / (1 + exp(-gamma * progress)) - 1
    where progress = epoch / max_epochs in [0, 1].
    """
    import math
    p = epoch / max(1, max_epochs)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
