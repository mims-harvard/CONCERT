# ==========================
# CONCERT for SPAC-seq
# (Visium HD spatial CRISPR screen)
# Extends concert_map with the Perturbation Action Denoiser (PAD)
# to remove the shared guide-delivery / Cas9 batch effect.
# ==========================
from __future__ import annotations

import logging
import math
import os
from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler

# Optional wandb import (safe if not installed)
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from I_PID import PIDControl
from SVGP_Batch import SVGP
from VAE_utils import MeanAct, NBLoss, buildNetwork, DenseEncoder
from lord_batch import LordEncoder
from pert_action_denoiser import PerturbActionDenoiser, grl_alpha_schedule
from spatial_graph import paracrine_log_shift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unknown_attribute_penalty(latent_unknown: torch.Tensor) -> torch.Tensor:
    """Quadratic penalty on the unknown/basal attributes (encourages small magnitude)."""
    return torch.sum(latent_unknown ** 2, dim=1).mean()


def _hsic_biased(z, c):
    """Biased HSIC between z (B,dz) and c (B,dc) with RBF kernels + median-heuristic
    bandwidths. Differentiable; 0 iff z, c independent in the RKHS. Used to make the
    non-spatial latent gs independent of spatial coordinates."""
    B = z.shape[0]
    def _rbf(x):
        d2 = torch.cdist(x, x) ** 2
        med = torch.median(d2[d2 > 0]).detach()
        sigma2 = torch.clamp(med, min=1e-8)
        return torch.exp(-d2 / sigma2)
    K = _rbf(z)
    L = _rbf(c)
    H = torch.eye(B, device=z.device, dtype=z.dtype) - 1.0 / B
    Kc = H @ K @ H
    Lc = H @ L @ H
    return (Kc * Lc).sum() / float((B - 1) ** 2)


def _moran_i(z, coords, k=6):
    """Sum over latent dims of squared Moran's I of z (B,dz) over a batch-local kNN graph
    built from coords (B,2). Penalizes spatial autocorrelation of a latent: ~0 when z is
    spatially white, large when z is a smooth function of location. Differentiable; >= 0.
    Used to force spatially-predictable variation out of the basal latent and into the GP."""
    B = z.shape[0]
    kk = int(min(k, B - 1))
    if kk < 1:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    d2 = torch.cdist(coords, coords) ** 2                      # (B,B)
    d2 = d2 + torch.eye(B, device=z.device, dtype=z.dtype) * 1e12   # mask self
    nn_idx = torch.topk(d2, kk, dim=1, largest=False).indices  # (B,kk) nearest neighbors
    W = torch.zeros(B, B, device=z.device, dtype=z.dtype)
    W.scatter_(1, nn_idx, 1.0 / kk)                            # row-normalized weights
    zc = z - z.mean(dim=0, keepdim=True)                       # (B,dz)
    num = (zc * (W @ zc)).sum(dim=0)                           # (dz,) sum_i zc_i * sum_j w_ij zc_j
    den = (zc ** 2).sum(dim=0) + 1e-8                          # (dz,)
    moran = num / den                                         # (dz,) Moran's I per dim (N/sumW = 1)
    return (moran ** 2).sum()


def _moran_i_scale(z, coords, radius):
    """Gaussian-weighted Moran's I at a chosen spatial SCALE (footprint radius).

    Unlike _moran_i (k=6 nearest neighbours, which removes *local* structure), this
    weights pairs by exp(-d^2 / 2 radius^2), so only autocorrelation at ~`radius`
    (the perturbation footprint scale) is penalised; finer cell-type structure
    survives. Returns sum over latent dims of squared Moran's I (scalar).
    """
    coords = coords[:, :2]
    d2 = torch.cdist(coords, coords) ** 2                  # (N, N)
    W = torch.exp(-d2 / (2.0 * radius * radius))
    W = W - torch.diag(torch.diagonal(W))                  # zero self-weight
    Wsum = W.sum().clamp(min=1e-8)
    zc = z - z.mean(dim=0, keepdim=True)                   # (N, D)
    var = (zc * zc).sum(dim=0).clamp(min=1e-8)             # (D,)
    # numerator per dim: sum_ij W_ij zc_i zc_j  == zc^T W zc (diagonal over dims)
    num = (zc * (W @ zc)).sum(dim=0)                       # (D,)
    N = z.shape[0]
    moran = (N / Wsum) * (num / var)                       # (D,)
    return (moran ** 2).sum()


def _knn_smooth(y, coords, k=6):
    """Row-normalized batch-local kNN neighbor-average of y (B,G) over coords (B,2) -> (B,G).
    Extracts the spatially-coherent component of expression (averages out per-cell high-freq
    noise + cell-state idiosyncrasy). Used as the supervision target for the coord-driven GP
    field so z_gp is forced to carry the smooth spatial expression structure (Route 2)."""
    B = y.shape[0]
    kk = int(min(k, B - 1))
    if kk < 1:
        return y
    d2 = torch.cdist(coords, coords) ** 2                      # (B,B)
    d2 = d2 + torch.eye(B, device=y.device, dtype=y.dtype) * 1e12   # mask self
    nn_idx = torch.topk(d2, kk, dim=1, largest=False).indices  # (B,kk)
    return y[nn_idx].mean(dim=1)                               # (B,G) neighbor mean


class FpertField(nn.Module):
    """Coordinate-conditioned per-perturbation gene-space log-fold RESIDUAL field.

    Maps (pert_emb (B,E), coords (B,2)) -> (B, num_genes) spatial residual that is
    ADDED on top of the spatially-constant core ``self.f_pert(pert_emb)``. Initialized
    near zero so training starts from the constant-f_pert(p) baseline and only departs
    from it if a spatial footprint genuinely improves the likelihood / LFC supervision.

    Inductive by construction: a pure function of (pert_emb, coords), no per-cell params.

    Variants (``field``):
      - 'rff'    : MLP([pert_emb ; RFF(coords)]). RFF = fixed Gaussian random Fourier
                   features (cos/sin), bandwidth set by ``rff_bandwidth`` (~1/lengthscale).
      - 'svgp'   : learnable inducing points Z (M,2) + learnable lengthscale; RBF kernel
                   features k(coords, Z) feed the MLP. Explicit, readable lengthscale.
      - 'hybrid' : 'rff' field for the shift + post-hoc ``estimate_lengthscale`` readout.
    """

    def __init__(self, encoder_dim, num_genes, field="rff", rff_features=128,
                 rff_bandwidth=0.15, hidden=256, svgp_inducing=32, svgp_ls_init=5.0,
                 coord_scale=1.0, mb_bandwidths=None, tile_spacing=0.0,
                 ls_min=None, ls_max=None, time_emb_dim=0, time_conditioned_ls=False,
                 time_in_mlp=False):
        super().__init__()
        self.field = field
        self.num_genes = num_genes
        self.coord_scale = float(coord_scale)
        # Timekernel: make the gp_pp lengthscale ℓ a function of time by feeding the
        # (per-cell) time embedding into ls_head alongside the perturbation embedding.
        # Still squashed into [ls_min, ls_max] so ℓ(t) cannot escape the collapse guard.
        self.time_conditioned_ls = bool(time_conditioned_ls)
        # Width of the time embedding fed to ls_head; used to zero-pad the ls_head
        # input when time_conditioned_ls=True but a call site omits time_emb (keeps
        # every call site — diagnostics, CF eval — non-crashing at time-part=0).
        self.time_emb_dim = int(time_emb_dim)
        # Bounded per-perturbation lengthscale band (gp_pp). When both set, ls is
        # squashed into (ls_min, ls_max) via sigmoid so it CANNOT collapse below the
        # floor — fixes the multimodal-perturbation lengthscale-collapse failure where
        # a scattered-multi-patch KO (e.g. Tgfbr2) drives ls->0 and overfits, since the
        # soft ls_prior penalty alone was too weak to hold the band.
        self.ls_min = None if ls_min is None else float(ls_min)
        self.ls_max = None if ls_max is None else float(ls_max)
        # C1 (multi-slide): map tiled coords back to each slide's own [0, loc_range] frame
        # so gp_pp/svgp Z-grid and rff bandwidth are consistent across slides. 0 = disabled
        # (single-slide). Buffer so it round-trips through save/load.
        self.register_buffer("tile_spacing", torch.tensor(float(tile_spacing)))
        if field in ("rff", "hybrid"):
            W = torch.randn(2, rff_features) * rff_bandwidth
            b = torch.rand(rff_features) * 2.0 * math.pi
            self.register_buffer("rff_W", W)
            self.register_buffer("rff_b", b)
            feat_dim = 2 * rff_features
        elif field == "mb_rff":
            # M3: multi-bandwidth RFF — K fixed-frequency blocks concatenated; the per-pert
            # MLP weights pick the effective scale. Collapse-proof (frequencies fixed).
            bws = list(mb_bandwidths) if mb_bandwidths else [0.05, 0.15, 0.5]
            Ws = [torch.randn(2, rff_features) * float(bw) for bw in bws]
            bs = [torch.rand(rff_features) * 2.0 * math.pi for _ in bws]
            self.register_buffer("rff_W", torch.cat(Ws, dim=1))    # (2, K*F)
            self.register_buffer("rff_b", torch.cat(bs, dim=0))    # (K*F,)
            feat_dim = 2 * rff_features * len(bws)
        elif field == "svgp":
            # inducing points on a coarse grid in the (normalized) coord box, learnable
            m = int(round(svgp_inducing ** 0.5))
            gx, gy = torch.meshgrid(torch.linspace(0, 1, m), torch.linspace(0, 1, m),
                                    indexing="ij")
            self.Z = nn.Parameter(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1))
            # one softplus-parameterized lengthscale (per-pert readout via the MLP weights)
            self._raw_ls = nn.Parameter(torch.tensor(float(svgp_ls_init)))
            feat_dim = self.Z.shape[0]
        elif field == "gp_pp":
            # M1: per-PERTURBATION GP footprint — RBF kernel features to a learnable grid Z,
            # with a per-pert lengthscale ls(pert_emb) = softplus(linear(emb)). Guarded by the
            # ls_prior band in the model elbo (interpretable, but can drift -> guard required).
            m = int(round(svgp_inducing ** 0.5))
            gx, gy = torch.meshgrid(torch.linspace(0, 1, m), torch.linspace(0, 1, m),
                                    indexing="ij")
            self.Z = nn.Parameter(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1))
            # time-conditioned: ls_head sees [pert_emb ; time_emb] so ℓ can vary with t.
            _ls_in = encoder_dim + int(time_emb_dim) if self.time_conditioned_ls else encoder_dim
            self.ls_head = nn.Linear(_ls_in, 1)
            nn.init.zeros_(self.ls_head.weight)
            if ls_min is not None and ls_max is not None:
                # bounded: ls = ls_min + (ls_max-ls_min)*sigmoid(raw); init at svgp_ls_init
                frac = min(max((float(svgp_ls_init) - float(ls_min)) /
                               max(float(ls_max) - float(ls_min), 1e-6), 1e-3), 1.0 - 1e-3)
                nn.init.constant_(self.ls_head.bias, math.log(frac / (1.0 - frac)))
            else:
                inv_sp = math.log(math.expm1(max(float(svgp_ls_init), 1e-3)))   # softplus^{-1}
                nn.init.constant_(self.ls_head.bias, inv_sp)                    # initial ls ~ ls_init
            self._last_ls = None
            feat_dim = self.Z.shape[0]
        else:
            raise ValueError(f"unknown fpert_field={field!r}")

        self.time_in_mlp = bool(time_in_mlp)
        _mlp_in = encoder_dim + feat_dim + (self.time_emb_dim if self.time_in_mlp else 0)
        self.mlp = nn.Sequential(
            nn.Linear(_mlp_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_genes),
        )
        # near-zero init on the last layer -> residual starts at ~0
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    @property
    def lengthscale(self):
        """Explicit lengthscale (svgp only); None otherwise."""
        if self.field == "svgp":
            return torch.nn.functional.softplus(self._raw_ls)
        return None

    def _gp_pp_ls(self, pert_emb, time_emb=None):
        """Per-perturbation lengthscale for gp_pp. Bounded to (ls_min, ls_max) via a
        sigmoid squash when the band is set (collapse-proof); else softplus (legacy).
        When time-conditioned, ℓ additionally depends on the time embedding ℓ(t)."""
        # Guard: if ls_head was widened for time-conditioning but this call site
        # omitted time_emb, zero-pad so the matmul width matches (ℓ at time-part=0).
        if self.time_conditioned_ls and time_emb is None:
            time_emb = pert_emb.new_zeros(pert_emb.shape[0], self.time_emb_dim)
        raw_in = (torch.cat([pert_emb, time_emb], dim=-1)
                  if self.time_conditioned_ls else pert_emb)
        raw = self.ls_head(raw_in)
        if self.ls_min is not None and self.ls_max is not None:
            return self.ls_min + (self.ls_max - self.ls_min) * torch.sigmoid(raw)
        return torch.nn.functional.softplus(raw).clamp(min=1e-2)

    def _spatial_features(self, coords, pert_emb=None, time_emb=None):
        if float(self.tile_spacing) > 0:
            S = self.tile_spacing
            coords = coords - S * torch.floor(coords / S)   # -> per-slide local [0, loc_range]
        c = coords / self.coord_scale
        if self.field in ("rff", "hybrid", "mb_rff"):
            proj = c @ self.rff_W + self.rff_b               # (B, F) or (B, K*F)
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (B, 2F)
        if self.field == "gp_pp":
            # per-perturbation lengthscale from the embedding -> per-cell RBF features
            ls = self._gp_pp_ls(pert_emb, time_emb)           # (B,1) bounded/collapse-proof, ℓ(t)
            self._last_ls = ls
            d2 = torch.cdist(c, self.Z) ** 2                  # (B, M)
            return torch.exp(-d2 / (2.0 * ls * ls))          # (B, M)
        # svgp: RBF kernel features to learnable inducing points (single global lengthscale)
        ls = torch.nn.functional.softplus(self._raw_ls).clamp(min=1e-3)
        d2 = torch.cdist(c, self.Z) ** 2                      # (B, M)
        return torch.exp(-d2 / (2.0 * ls * ls))              # (B, M)

    @torch.no_grad()
    def per_pert_lengthscale(self, pert_emb):
        """Per-perturbation lengthscale (gp_pp only); shape (B,). None otherwise."""
        if self.field != "gp_pp":
            return None
        if pert_emb.dim() == 1:
            pert_emb = pert_emb.unsqueeze(0)
        return self._gp_pp_ls(pert_emb).squeeze(-1)

    def forward(self, pert_emb, coords, time_emb=None):
        parts = [pert_emb, self._spatial_features(coords, pert_emb, time_emb)]
        if self.time_in_mlp:
            # Same zero-pad contract as _gp_pp_ls: call sites that omit time_emb
            # (diagnostics, some CF paths) evaluate the field at time-part = 0
            # rather than crashing on a width mismatch.
            parts.append(time_emb if time_emb is not None
                         else pert_emb.new_zeros(pert_emb.shape[0], self.time_emb_dim))
        return self.mlp(torch.cat(parts, dim=-1))             # (B, num_genes) residual

    @torch.no_grad()
    def estimate_lengthscale(self, pert_emb, coord_grid):
        """Post-hoc lengthscale (hybrid/rff): half-decay distance of the field's spatial
        autocovariance evaluated on a grid for a single perturbation embedding.
        pert_emb: (E,) one perturbation; coord_grid: (P,2)."""
        P = coord_grid.shape[0]
        emb = pert_emb.unsqueeze(0).expand(P, -1)
        vals = self(emb, coord_grid).mean(dim=1)              # (P,) avg over genes
        vals = vals - vals.mean()
        d = torch.cdist(coord_grid, coord_grid)               # (P,P)
        cov0 = (vals * vals).mean()
        if cov0 <= 1e-12:
            return torch.tensor(0.0)
        # binned autocovariance C(r); lengthscale = smallest r with C(r) < 0.5*C(0)
        dr = d.reshape(-1)
        cc = (vals.unsqueeze(1) * vals.unsqueeze(0)).reshape(-1) / cov0
        nb = 30
        edges = torch.linspace(0, dr.max().item() + 1e-6, nb + 1)
        for k in range(nb):
            sel = (dr >= edges[k]) & (dr < edges[k + 1])
            if sel.any() and cc[sel].mean() < 0.5:
                return 0.5 * (edges[k] + edges[k + 1])
        return dr.max()


class BayesLinear(nn.Module):
    """Mean-field variational linear layer (Blundell et al. 2015, local reparam).

    forward(x, sample) returns (out, var): pre-activation mean (+ optional weight-noise
    sample) and the analytic pre-activation variance. ``kl()`` is the closed-form
    KL(q(W)||N(0,prior_std^2)) used as a GP_KL-like regulariser. Near-zero mean init so the
    Bayes head starts at ~no output, matching the deterministic head it replaces.
    """
    def __init__(self, in_dim, out_dim, prior_std=1.0, rho_init=-5.0):
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.prior_std = float(prior_std)
        self.w_mu = nn.Parameter(torch.empty(out_dim, in_dim))
        self.w_rho = nn.Parameter(torch.full((out_dim, in_dim), float(rho_init)))
        self.b_mu = nn.Parameter(torch.zeros(out_dim))
        self.b_rho = nn.Parameter(torch.full((out_dim,), float(rho_init)))
        nn.init.normal_(self.w_mu, std=1e-3)

    @property
    def w_sigma(self):
        return torch.nn.functional.softplus(self.w_rho)

    @property
    def b_sigma(self):
        return torch.nn.functional.softplus(self.b_rho)

    def forward(self, x, sample=True):
        mean = x @ self.w_mu.t() + self.b_mu
        var = x.pow(2) @ self.w_sigma.pow(2).t() + self.b_sigma.pow(2)
        if sample:
            mean = mean + var.clamp(min=1e-12).sqrt() * torch.randn_like(mean)
        return mean, var

    def kl(self):
        ps2 = self.prior_std ** 2
        def _kl(mu, sig):
            return (0.5 * (torch.log(torch.tensor(ps2, device=mu.device)) - 2.0 * torch.log(sig)
                           + (sig.pow(2) + mu.pow(2)) / ps2 - 1.0)).sum()
        return _kl(self.w_mu, self.w_sigma) + _kl(self.b_mu, self.b_sigma)


class BasalRFFField(nn.Module):
    """Deterministic coordinate-only spatial BASAL field: coords -> GP_dim latent mean.

    Fixed-bandwidth Random Fourier Features (Rahimi & Recht 2007) + shared MLP. NOT
    perturbation-conditioned (basal is one baseline). Drop-in replacement for the
    per-perturbation SVGP basal latent: cannot collapse (no learnable kernel scale).
    """
    def __init__(self, gp_dim, rff_features=128, rff_bandwidth=0.15,
                 hidden=128, coord_scale=20.0, tissue_dim=0, basal_bayes=False):
        super().__init__()
        self.coord_scale = float(coord_scale)
        self.tissue_dim = int(tissue_dim)
        self.bayes = bool(basal_bayes)
        W = torch.randn(2, rff_features) * rff_bandwidth
        b = torch.rand(rff_features) * 2.0 * math.pi
        self.register_buffer("rff_W", W)
        self.register_buffer("rff_b", b)
        in_dim = 2 * rff_features + self.tissue_dim
        if self.bayes:
            # body -> hidden; Bayes-linear head -> gp_dim (calibrated mean+var, GP_KL-like reg)
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
            self.bayes_head = BayesLinear(hidden, gp_dim)
        else:
            self.mlp = nn.Sequential(             # P0 deterministic head (byte-preserved)
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, gp_dim),
            )
            self.bayes_head = None
        self.last_var = None

    def forward(self, coords, tissue_emb=None):
        c = coords / self.coord_scale
        proj = c @ self.rff_W + self.rff_b               # (B, F)
        feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (B, 2F)
        if self.tissue_dim > 0 and tissue_emb is not None:
            feat = torch.cat([feat, tissue_emb], dim=-1)  # region identity -> spatial path
        h = self.mlp(feat)
        if self.bayes:
            mean, var = self.bayes_head(h, sample=self.training)
            self.last_var = var
            return mean                                  # (B, gp_dim)
        return h                                          # (B, gp_dim)

    def kl(self):
        if self.bayes:
            return self.bayes_head.kl()
        return torch.zeros((), device=self.rff_W.device)


class TimeEmbedding(nn.Module):
    """Continuous-time embedding: Fourier features of t∈[0,1] → small MLP.

    t is the normalized day (day/21 for the DSS course {0,3,9,21}). Fourier features
    make the map smooth yet nonlinear, so the learned time function can interpolate to
    unseen days (LOTO-D9) and extrapolate past the last observed day — unlike a
    categorical/LORD time code, which cannot represent times never seen in training.
    """

    def __init__(self, n_bands: int = 6, out_dim: int = 16):
        super().__init__()
        freqs = (2.0 ** torch.arange(n_bands).float()) * math.pi   # geometric bands
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_bands + 1, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1).float()                               # (B, 1)
        ff = torch.cat([t, torch.sin(t * self.freqs), torch.cos(t * self.freqs)], dim=-1)
        return self.mlp(ff)                                        # (B, out_dim)


class AmplitudeTime(nn.Module):
    """Smooth, low-dimensional non-negative inflammation amplitude a(t).

    Maps a time embedding to a single positive scalar per cell. Deliberately
    low-capacity (one hidden layer) so it cannot memorize per-cell snapshots —
    it must fit a smooth rise-to-peak-then-fall over the DSS course. Softplus
    output guarantees a(t) >= 0 so it only scales (never sign-flips) f_pert.

    Baseline anchoring (`anchor_offset` > 0, needs `ref_emb`): a(t) is measured
    RELATIVE to the healthy baseline time,

        a(t) = softplus(g(t) - g(t_ref) - offset)   =>   a(t_ref) = softplus(-offset) ~ 0

    so the pre-injury day-0 snapshot is reconstructed by the basal path alone. This
    is what identifies the SIGN of f_pert: with basal pinned to healthy, every later
    day's residual must ADD inflammation rather than the model storing inflammation
    in basal and having f_pert subtract it (the inverted mechanism job 31807288 found).
    a(t_ref) is a CONSTANT here — no gradient, nothing for the NB term to pull on — which
    a soft lambda*a(0)^2 penalty could not achieve: at lambda=10 the likelihood simply
    out-pulled it and a(0) ROSE 0.69 -> 0.82 (job 31812173).
    """
    def __init__(self, time_emb_dim: int, hidden: int = 16, anchor_offset: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        # start near a(t) ~ softplus(0) ~ 0.69 (mild, non-zero) everywhere
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.anchor_offset = float(anchor_offset)

    def forward(self, time_emb, ref_emb=None):
        g = self.mlp(time_emb)                                   # (B, 1)
        if self.anchor_offset > 0:
            if ref_emb is None:
                raise ValueError("anchored AmplitudeTime needs the baseline-time ref_emb")
            g = g - self.mlp(ref_emb) - self.anchor_offset       # (B, 1), broadcast over (1, 1)
        return torch.nn.functional.softplus(g).squeeze(-1)       # (B,)


class MembershipField(nn.Module):
    """Spatiotemporal inflammation-membership field π ∈ [0,1].

    π_i = sigmoid(MLP([gate_state_i, z_niche_i, coords_i, time_emb_i])) where
    gate_state = [gp_latent ; basal_latent] (spatial + cell identity), z_niche = the
    NicheDescriptor microenvironment code, coords = raw xy, time_emb = Fourier(t).
    Applied to EVERY cell (control and perturbed) so the inflamed FRACTION (composition)
    can grow/shrink with time: a control cell whose π rises at the injury-peak time in a
    susceptible niche enters the inflammation program. BCE-supervised by the observed
    inflammation label at each cell's observed time.
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # start π near the base rate for a stable BCE start (sigmoid(-2) ≈ 0.12)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(feats)).squeeze(-1)   # (B,)


class InflammationField(nn.Module):
    """Continuous, non-negative spatiotemporal inflammation field (v3).

    s_i = softplus(MLP([z_niche_i, coords_i, time_emb_i]))  — a continuous scalar ≥ 0.

    Two deliberate differences from MembershipField (which it supersedes):
      1. **Continuous, not binary.** softplus output (no sigmoid); supervised by the observed
         continuous inflammation gene-score via MSE (no BCE) → no saturation trap.
      2. **Generative, not reconstructive.** The input DROPS the per-cell own-latent
         (gate_state = [gp_latent ; basal_latent]); it reads ONLY (z_niche, coords, time_emb).
         The field can never copy the cell's own inflammation — it predicts inflammation fresh
         from spatial context + time, so re-decoding a cell at a new time t′ MUST move s.
    Applied to EVERY cell so the inflamed intensity/composition grows/shrinks with time.
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # start s near the base rate: softplus(0) ≈ 0.69 — a stable non-saturating MSE start.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.mlp(feats)).squeeze(-1)   # (B,) ≥ 0


class NicheDescriptor(nn.Module):
    """Unsupervised, transferable microenvironment code (batch-local, no graph file).

    z_niche_i = pool_MLP( distance-weighted kNN mean over batch neighbors j of proj(gs_j) ,
                          proj(gs_i) ).
    Uses ONLY the content latent gs (cell-state) + spatial coords — no labels, no precomputed
    graph — so it summarizes "what cell states surround me" and transfers across patches/slides.
    Neighbors are taken batch-locally over coords (same pattern as _knn_smooth); for the small
    pmap slides a training batch ≈ the whole slide, so this is the full neighborhood. Compact
    out_dim keeps it inspectable and overfit-resistant.
    """
    def __init__(self, gs_dim, out_dim=8, hidden=32, k=25, dispersion=False):
        super().__init__()
        self.k = int(k)
        self.dispersion = bool(dispersion)
        self.proj = nn.Linear(gs_dim, hidden)
        # dispersion adds a neighborhood 2nd-moment (std) block -> pool input grows 2H -> 3H.
        pool_in = (3 if self.dispersion else 2) * hidden
        self.pool = nn.Sequential(
            nn.Linear(pool_in, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )
        self.out_dim = out_dim

    def forward(self, gs, coords):
        # gs: (B, gs_dim) (detach upstream); coords: (B, 2)
        B = gs.shape[0]
        centre = self.proj(gs)                                   # (B,H)
        kk = int(min(self.k, B - 1))
        if kk < 1:
            surround = centre
            std = torch.zeros_like(centre)
        else:
            d2 = torch.cdist(coords, coords) ** 2                # (B,B)
            d2 = d2 + torch.eye(B, device=gs.device, dtype=gs.dtype) * 1e12
            dist, idx = torch.topk(d2, kk, dim=1, largest=False)  # (B,kk)
            w = 1.0 / (1.0 + dist)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            neigh = centre[idx]                                   # (B,kk,H)
            surround = (w.unsqueeze(-1) * neigh).sum(dim=1)       # (B,H) neighbor mean of proj(gs)
            if self.dispersion:
                var = (w.unsqueeze(-1) * (neigh - surround.unsqueeze(1)) ** 2).sum(dim=1)
                std = torch.sqrt(var + 1e-6)                      # (B,H) neighbor dispersion
        parts = [centre, surround, std] if self.dispersion else [centre, surround]
        return self.pool(torch.cat(parts, dim=-1))               # (B,out_dim)


def atomic_torch_save(obj, path: str) -> None:
    """torch.save via a temp file + rename, so a partial write can't destroy the
    previous checkpoint. Writing straight onto the target means one failed save
    (full disk or quota) leaves a truncated, unloadable checkpoint and no fallback."""
    tmp_path = path + ".tmp"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        # Leave any previous checkpoint untouched and drop the partial file.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class EarlyStopping:
    """Early stopping with best-weight saving based on validation loss."""

    def __init__(self, patience: int = 10, verbose: bool = False, modelfile: str = "model.pt",
                 patience_start_epoch: int = 0, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.model_file = modelfile
        self.patience_start_epoch = patience_start_epoch
        self.min_delta = float(min_delta)

    def __call__(self, loss: float, model: nn.Module, epoch: int = 0) -> None:
        if np.isnan(loss):
            self.early_stop = True
            return
        # Always keep the true-best checkpoint on disk (any new min val loss),
        # independent of the patience baseline below.
        if loss < self.loss_min:
            self._save(loss, model)

        if epoch < self.patience_start_epoch:
            # Still in warmup — checkpoint tracked above, but don't count patience
            score = -loss
            if self.best_score is None or score > self.best_score:
                self.best_score = score
            return

        score = -loss
        # Reset patience only on a *meaningful* improvement (> min_delta over the
        # baseline best); marginal/noise-level dips accumulate the counter so the
        # run actually stops at the plateau instead of grinding to maxiter.
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)

    def _save(self, loss: float, model: nn.Module) -> None:
        if self.verbose:
            logging.info("Validation loss improved: %.6f → %.6f (saving)", self.loss_min, loss)
        atomic_torch_save(model.state_dict(), self.model_file)
        self.loss_min = loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CONCERT(nn.Module):
    """CONCERT — GP-VAE for spatial counterfactual perturbation modeling.

    Latent factorization:
      • GP block (GP_dim): spatially structured latent processes via SVGP.
      • Gaussian block (Normal_dim): non-spatial latent effects.
    """

    def __init__(
        self,
        *,
        encoder_dim: int,
        GP_dim: int,
        Normal_dim: int,
        cell_atts: np.ndarray,
        num_genes: int,
        n_batch: int,
        n_slides: int = 1,
        encoder_layers: Iterable[int],
        decoder_layers: Iterable[int],
        noise: float,
        encoder_dropout: float,
        decoder_dropout: float,
        shared_dispersion: bool,
        fixed_inducing_points: bool,
        initial_inducing_points: np.ndarray,
        fixed_gp_params: bool,
        kernel_scale: np.ndarray,
        multi_kernel_mode: bool,
        mask_cutoff: np.ndarray,
        N_train: int,
        KL_loss: float,
        dynamicVAE: bool,
        init_beta: float,
        min_beta: float,
        max_beta: float,
        dtype: torch.dtype,
        device: str,
        # ── Perturbation Action Denoiser (PAD) ──────────────────────────
        use_pad: bool = True,
        pad_action_dim: int = 64,
        pad_lambda_mmd: float = 0.1,
        pad_lambda_adv: float = 0.01,
        pad_adv_hidden: int = 64,
        # ── explicit action correction ────────────────────────
        use_action_correction: bool = True,
        detach_mu_action_for_ko: bool = False,
        mu_action_anchor=None,
        mu_action_anchor_lambda: float = 0.0,
        # ── latent disentanglement (drop pert from encoder) ───
        # When True, the VAE encoder sees only [basal, tissue] from LORD —
        # the perturbation embedding is held back and applied as an explicit
        # gene-space additive log-fold term via f_pert. This makes z
        # structurally pert-blind, leaving f_pert as the sole counterfactual
        # mechanism (clean and interpretable).
        use_basal_encoder_only: bool = False,
        use_f_pert: bool = False,
        f_pert_init_std: float = 0.01,
        f_pert_clamp: float = 5.0,
        fpert_field: str = "constant",     # 'constant'|'rff'|'svgp'|'hybrid'
        fpert_rff_features: int = 128,
        fpert_rff_bandwidth: float = 0.15,
        fpert_hidden: int = 256,
        fpert_svgp_inducing: int = 32,
        fpert_svgp_ls_init: float = 5.0,
        fpert_coord_scale: float = 1.0,
        fpert_mb_bandwidths=None,
        fpert_tile_spacing: float = 0.0,   # C1 multi-slide: tile spacing S (0 = single-slide)
        # Niche-conditioned FiLM effect head (Tgfbr2 sign-flip fix). Transferable, annotation-free.
        use_niche_film: bool = False,
        niche_film_dim: int = 8,           # compact microenvironment code dim
        niche_film_k: int = 25,            # batch-local kNN neighborhood size (niche, not GP k=6)
        niche_film_gcap: float = 3.0,      # cap on per-pert modulation depth g_scale
        niche_film_l2: float = 1e-3,       # earned-departure regularizer on g_scale + beta
        niche_film_rank: int = 8,          # low-rank of the z_niche-driven per-gene beta residual
        niche_film_dispersion: bool = False,  # add neighborhood 2nd-moment (dispersion) to z_niche
        use_consistency_loss: bool = False,
        consistency_lambda: float = 1.0,
        # ── inductive expression encoder + collapse fixes ─────
        use_expr_encoder: bool = False,
        expr_hidden_dims: Optional[List[int]] = None,
        expr_dropout: float = 0.1,
        use_basal_consistency: bool = False,
        basal_consistency_lambda: float = 1.0,
        beta_warmup_epochs: int = 0,
        beta_target: float = 1.0,
        free_bits: float = 0.0,
        gp_free_bits: float = 0.0,
        use_pert_correction: bool = False,
        # NNGP local field
        use_nngp: bool = False,
        nngp_lambda: float = 1.0,
        # paracrine spillover field
        use_paracrine: bool = False,
        paracrine_l1: float = 0.0,
        # niche x pert interaction
        use_niche_interaction: bool = False,
        niche_rank: int = 8,
        niche_dim: Optional[int] = None,
        # multiplicative per-cell scalar gate on the global pert effect
        use_pert_gate: bool = False,
        gate_hidden: int = 64,
        # Time-varying inflammation composition (membership field)
        use_membership_field: bool = False,
        membership_hidden: int = 64,
        membership_bce_lambda: float = 1.0,
        membership_moran_lambda: float = 0.0,
        use_time_intensity: bool = True,   # keep the legacy time×pert residual unless off
        # continuous inflammation field (supersedes the binary membership field)
        use_inflammation_field: bool = False,
        field_mse_lambda: float = 1.0,
        field_moran_lambda: float = 0.0,
        # NORMALIZED day (t/21, matching TimeEmbedding's t∈[0,1]) at which to freeze the
        # inflammation field's time input; None = current behaviour (field sees the true t).
        # Makes s a function of architecture only; see the note at the field call site.
        field_time_fixed_t: Optional[float] = None,
        perturbed_code: int = 1,           # pert_map["perturbed"]; the shared inflamed program row
        # spatial-basal disentanglement + mini-bulk LFC supervision
        spatial_basal_only: bool = False,
        lfc_lambda: float = 0.0,
        disentangle_lambda: float = 0.0,
        moran_lambda: float = 0.0,
        deamortize_gp: bool = False,
        additive_decoder: bool = False,
        gp_field_aux_lambda: float = 0.0,
        basal_moran_output_lambda: float = 0.0,
        # Stage 1 GP guardrail: soft band prior on the SVGP kernel lengthscale
        ls_prior_lambda: float = 0.0,
        ls_prior_min: float = 3.0,
        ls_prior_max: float = 8.0,
        # ── P0: deterministic RFF basal field + scale-aware disentanglement ──
        basal_field: str = "svgp",
        basal_rff_features: int = 128,
        basal_rff_bandwidth: float = 0.15,
        basal_rff_hidden: int = 128,
        basal_coord_scale: float = 20.0,
        disentangle_radius: float = 0.0,
        basal_bayes: bool = False,
        # ── Continuous-time conditioning (gut DSS temporal case study) ──
        # Fourier-embed a per-cell time scalar t∈[0,1] (day_norm) and add a low-rank
        # time×pert term to the perturbation log-shift so the inflammation fold-change
        # evolves over the DSS course (peaks D9). Default OFF → epoch-0 == niche model.
        use_time_conditioning: bool = False,
        time_fourier_bands: int = 6,
        time_emb_dim: int = 16,
        time_rank: int = 8,
        time_l2: float = 1e-4,
        use_shared_basal: bool = False,
        n_tier1: Optional[int] = None,
        arch_dim: int = 20,
        use_timekernel: bool = False,
        amp_anchor_offset: float = 0.0,
        # FALLBACK (soft): penalise lambda*a(t_ref)^2 in the ELBO. Inert when the
        # by-construction anchor is on (a(t_ref) is then constant). Kept as a knob;
        # on its own it is NOT sufficient — see AmplitudeTime docstring.
        amp_anchor_lambda: float = 0.0,
        amp_anchor_t: float = 0.0,   # normalised time of the healthy baseline (day_norm = 0/21)
        marker_sign_positive: bool = False,
        marker_gene_idx: Optional[Sequence[int]] = None,
        time_in_field_mlp: bool = False,
    ) -> None:
        super().__init__()
        torch.set_default_dtype(dtype)

        # SVGP prior (only built for the legacy/default basal_field='svgp' path; the
        # deterministic RFF basal field replaces it when basal_field='rff').
        if str(basal_field) == "svgp":
            if multi_kernel_mode:
                self.svgp = SVGP(
                    fixed_inducing_points=fixed_inducing_points,
                    initial_inducing_points=initial_inducing_points,
                    fixed_gp_params=fixed_gp_params,
                    kernel_scale=kernel_scale,
                    multi_kernel_mode=multi_kernel_mode,
                    kernel_phi=1.0,
                    jitter=1e-6,
                    N_train=N_train,
                    dtype=dtype,
                    device=device,
                )
            else:
                self.svgp = SVGP(
                    fixed_inducing_points=fixed_inducing_points,
                    initial_inducing_points=initial_inducing_points,
                    fixed_gp_params=fixed_gp_params,
                    kernel_scale=kernel_scale[0],
                    multi_kernel_mode=multi_kernel_mode,
                    jitter=1e-6,
                    N_train=N_train,
                    dtype=dtype,
                    device=device,
                )
        else:
            self.svgp = None

        # Hyper-params
        self.encoder_dim = encoder_dim
        self.GP_dim = GP_dim
        self.Normal_dim = Normal_dim
        self.noise = noise
        self.device = device
        self.num_genes = num_genes
        self.N_train = int(N_train)
        self.n_batch = n_batch
        self.n_slides = int(n_slides)
        self.cell_atts = cell_atts
        self.shared_dispersion = shared_dispersion
        # 9 flags
        self.use_expr_encoder = use_expr_encoder
        self.use_basal_consistency = use_basal_consistency
        self.basal_consistency_lambda = float(basal_consistency_lambda)
        self.beta_warmup_epochs = int(beta_warmup_epochs)
        self.beta_target = float(beta_target)
        self.free_bits = float(free_bits)
        self.gp_free_bits = float(gp_free_bits)
        self.use_pert_correction = use_pert_correction

        # Dynamic beta scheduling
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss_target = KL_loss
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta

        # ── Control-anchored explicit action correction ──────────
        # 3-term decoder: log μ_g = f_basal(z_b) + 𝟙[has_guide]·μ_action_g + 𝟙[is_real_KO]·f_pert
        # μ_action is a learnable per-gene additive shift in log-count space.
        # Identifiability:
        #   unassigned cells → establish f_basal (both action terms off)
        #   control cells    → identify μ_action   (f_pert off, has_guide=1)
        #   gene_KO cells    → identify f_pert given μ_action from above
        self.use_action_correction = use_action_correction
        self.detach_mu_action_for_ko = detach_mu_action_for_ko
        self.mu_action_anchor_lambda = float(mu_action_anchor_lambda)
        # empirical anchor (logmean(NTC) − logmean(unassigned)) pins μ_action
        # to the pure technical direction so the NTC-only gradient (after detach)
        # can't drift. Registered once as a (possibly None) non-trainable buffer.
        self.register_buffer(
            "mu_action_anchor",
            torch.as_tensor(mu_action_anchor, dtype=dtype)
            if (use_action_correction and mu_action_anchor is not None) else None,
        )
        if use_action_correction:
            self.mu_action = nn.Parameter(
                torch.zeros(num_genes, dtype=dtype), requires_grad=True
            )

        # ── P0: deterministic RFF basal field + scale-aware disentanglement ──
        self.basal_field = str(basal_field)
        self.disentangle_radius = float(disentangle_radius)
        # Multi-slide guard: the Moran-scale penalty builds a dense Gaussian graph over the
        # minibatch; a radius >= the tile gap would connect cells across slides. loc_range=20.
        if self.n_slides > 1 and self.disentangle_radius > 0:
            _gap = float(fpert_tile_spacing) - 20.0
            if self.disentangle_radius >= _gap:
                raise ValueError(
                    f"disentangle_radius={self.disentangle_radius} >= tile gap {_gap}; "
                    f"would leak cross-slide autocorrelation. Use a smaller radius.")
        self.basal_bayes = bool(basal_bayes)
        if self.basal_field == "rff":
            # n_tissue derived as in n_labels_override (cell_atts[:,0].max()+1); feed the
            # LORD tissue embedding (size encoder_dim) into the spatial field iff >1 class.
            _n_tissue = int(np.asarray(cell_atts)[:, 0].max()) + 1 if np.asarray(cell_atts).size else 1
            _tissue_dim = encoder_dim if _n_tissue > 1 else 0
            self.basal_rff = BasalRFFField(
                gp_dim=self.GP_dim, rff_features=basal_rff_features,
                rff_bandwidth=basal_rff_bandwidth, hidden=basal_rff_hidden,
                coord_scale=basal_coord_scale, tissue_dim=_tissue_dim,
                basal_bayes=self.basal_bayes)
            # learnable per-dim log-variance for the (otherwise deterministic) spatial latent
            self.basal_rff_logvar = nn.Parameter(torch.full((self.GP_dim,), -6.0))
        else:
            self.basal_rff = None

        # ── spatial-basal disentanglement + LFC supervision ────────
        self.spatial_basal_only = bool(spatial_basal_only)
        self.lfc_lambda = float(lfc_lambda)
        self.disentangle_lambda = float(disentangle_lambda)
        self._last_disent = 0.0   # running value for the per-epoch log line
        self.moran_lambda = float(moran_lambda)
        self._last_moran = 0.0   # running value for the per-epoch log line
        self.additive_decoder = bool(additive_decoder)
        # Route 2: auxiliary supervision forcing the coord-driven GP field to reconstruct the
        # spatially-smoothed expression -> injects coordinate structure into the encoder gp_mu +
        # SVGP inducing points (attacks encoder-starvation) while staying fully amortized/inductive.
        self.gp_field_aux_lambda = float(gp_field_aux_lambda)
        self._last_gpaux = 0.0
        # RC2 spatial residualization: penalize spatial autocorrelation of the ADDITIVE basal
        # OUTPUT contribution (log_basal in gene space), not just the 2-dim latent. Forces the
        # smooth spatial component of expression out of the basal path → it can only be
        # reconstructed by the coord-driven GP path. Stronger/more direct than latent Moran.
        self.basal_moran_output_lambda = float(basal_moran_output_lambda)
        self._last_basal_out_moran = 0.0
        self._cache_log_basal = None
        if self.gp_field_aux_lambda > 0:
            self.gp_aux_head = nn.Linear(GP_dim, self.num_genes)
        # Lever 2 (de-amortize GP): free per-cell GP latent target, indexed by global cell
        # index, so z_gp is reconstruction-driven instead of competing inside the expr encoder.
        self.deamortize_gp = bool(deamortize_gp)
        if self.deamortize_gp:
            _n_cells = int(np.asarray(cell_atts).shape[0])
            self.gp_mu_free = nn.Parameter(torch.zeros(_n_cells, GP_dim, dtype=dtype))
        self.ls_prior_lambda = float(ls_prior_lambda)
        self.ls_prior_min = float(ls_prior_min)
        self.ls_prior_max = float(ls_prior_max)
        # observed per-(niche, pert_code) KO-vs-NTC pseudobulk LFC targets, (G,) tensors.
        # Set by run_concert_spac before .fit(); consumed in forward when lfc_lambda>0.
        self.lfc_targets: Optional[Dict[Tuple[int, int], torch.Tensor]] = None

        # ── latent disentanglement flags ─────────────────────────
        self.use_basal_encoder_only = use_basal_encoder_only
        self.use_f_pert = use_f_pert
        self.f_pert_clamp = float(f_pert_clamp)
        self.use_consistency_loss = use_consistency_loss
        self.consistency_lambda = float(consistency_lambda)

        # Number of LORD blocks fed to the VAE encoder:
        #   - basal_encoder_only: 2 blocks (basal + tissue)  → pert held out
        #   - legacy:             3 blocks (basal + tissue + perturbation)
        n_lord_blocks_to_encoder = 2 if use_basal_encoder_only else 3

        # ── Perturbation Action Denoiser (PAD) ──────────────────────────────
        # Now a soft regularizer; main action removal via μ_action above.
        # Downgrade λ values (mmd 0.1, adv 0.01) to avoid over-suppression.
        # With use_basal_encoder_only=True, PAD is redundant (pert is removed
        # structurally) but kept here for backward compat and optional backstop.
        self.use_pad = use_pad
        if use_pad:
            self.pad = PerturbActionDenoiser(
                action_dim=pad_action_dim,
                basal_dim=encoder_dim,
                lambda_mmd=pad_lambda_mmd,
                lambda_adv=pad_lambda_adv,
                adv_hidden=pad_adv_hidden,
            ).to(device)
            _encoder_input_dim = encoder_dim * n_lord_blocks_to_encoder + pad_action_dim
        else:
            self.pad = None  # type: ignore[assignment]
            _encoder_input_dim = encoder_dim * n_lord_blocks_to_encoder

        # f_pert: per-perturbation gene-space additive log-fold term.
        # Takes LORD's perturbation embedding (encoder_dim) → (num_genes).
        # Init near zero so training starts from "no pert effect".
        if use_f_pert:
            self.f_pert = nn.Linear(encoder_dim, num_genes, bias=False)
            nn.init.normal_(self.f_pert.weight, mean=0.0, std=f_pert_init_std)
        else:
            self.f_pert = None  # type: ignore[assignment]

        # Coordinate-conditioned footprint residual (added on top of the constant
        # self.f_pert core). 'constant' => no field => exact legacy behavior (anti-flat gate).
        self.fpert_field_kind = str(fpert_field)
        if use_f_pert and fpert_field != "constant":
            self.fpert_field = FpertField(
                encoder_dim=encoder_dim, num_genes=num_genes, field=fpert_field,
                rff_features=fpert_rff_features, rff_bandwidth=fpert_rff_bandwidth,
                hidden=fpert_hidden, svgp_inducing=fpert_svgp_inducing,
                svgp_ls_init=fpert_svgp_ls_init, coord_scale=fpert_coord_scale,
                mb_bandwidths=fpert_mb_bandwidths, tile_spacing=fpert_tile_spacing,
                ls_min=ls_prior_min, ls_max=ls_prior_max,
                time_emb_dim=time_emb_dim, time_conditioned_ls=use_timekernel,
                time_in_mlp=bool(use_timekernel and time_in_field_mlp),
            )
        else:
            self.fpert_field = None  # type: ignore[assignment]

        # ── NNGP local field over the kNN graph (pert-agnostic) ──
        self.use_nngp = use_nngp
        self.nngp_lambda = float(nngp_lambda)
        if use_nngp:
            from nngp import NNGPLocal
            self.nngp = NNGPLocal(gp_dim=GP_dim)
            # epoch-stale cache of per-spot GP posterior means (global-index keyed).
            # non-persistent: it is recoverable training state, not a model parameter.
            n_cells = int(np.asarray(cell_atts).shape[0])
            # Persistent so the NNGP local term is non-zero at inference/decompose time
            # (the non-persistent buffer was absent from the checkpoint -> NNGP dead).
            self.register_buffer("gp_cache", torch.zeros(n_cells, GP_dim, dtype=dtype),
                                 persistent=True)
        else:
            self.nngp = None  # type: ignore[assignment]
        # Spatial graph attached post-construction (depends on data order).
        self.graph = None  # type: ignore[assignment]

        # ── paracrine spillover (own gene-signature head + per-pert radius) ──
        self.use_paracrine = use_paracrine
        self.paracrine_l1 = float(paracrine_l1)
        if use_paracrine:
            # spillover gene signature: separate weights from f_pert, shared pert embedding
            self.s_signature = nn.Linear(encoder_dim, num_genes, bias=False)
            nn.init.normal_(self.s_signature.weight, mean=0.0, std=0.01)
            # learnable per-pert Cauchy signaling radius (softplus, r_min=0.1).
            # n_batch == number of perturbations; ko_p codes index into this.
            self._raw_radius = nn.Parameter(torch.zeros(n_batch))
        else:
            self.s_signature = None  # type: ignore[assignment]
            self._raw_radius = None  # type: ignore[assignment]

        # ── low-rank niche x pert interaction (niche = GP latent summary) ──
        self.use_niche_interaction = use_niche_interaction
        if use_niche_interaction:
            nd = int(niche_dim) if niche_dim is not None else GP_dim   # niche summary dim
            r = int(niche_rank)
            self.niche_proj = nn.Linear(GP_dim, nd, bias=False)        # niche(z) summary
            self.inter_U = nn.Linear(encoder_dim, r, bias=False)       # pert -> rank r
            self.inter_V = nn.Linear(nd, r, bias=False)                # niche -> rank r
            self.inter_out = nn.Linear(r, num_genes, bias=False)       # rank r -> genes
            # all three init small so interaction starts ~0 and doesn't destabilise early training
            nn.init.normal_(self.inter_U.weight, std=0.01)
            nn.init.normal_(self.inter_V.weight, std=0.01)
            nn.init.normal_(self.inter_out.weight, std=0.01)
        else:
            self.niche_proj = None      # type: ignore[assignment]
            self.inter_U = None         # type: ignore[assignment]
            self.inter_V = None         # type: ignore[assignment]
            self.inter_out = None       # type: ignore[assignment]

        # ── per-cell scalar gate on the global perturbation effect ──
        # gate_i = sigmoid(MLP([gp_latent ; basal_latent])) ∈ [0,1]; multiplies g_global(g).
        # Coordinate-free (uses GP latent, not raw x/y) → transfers across slides.
        self.use_pert_gate = use_pert_gate
        if use_pert_gate:
            gate_in = GP_dim + Normal_dim          # niche (location) + basal cell-state
            self.pert_gate = nn.Sequential(
                nn.Linear(gate_in, gate_hidden), nn.ELU(),
                nn.Linear(gate_hidden, 1),
            )
            # Init gate ≈ 1 (final weight=0, bias=+3 → sigmoid(3)≈0.95): near drop-in for
            # "pure g_global" at start; training pulls the gate down only where warranted.
            nn.init.zeros_(self.pert_gate[-1].weight)
            nn.init.constant_(self.pert_gate[-1].bias, 3.0)
        else:
            self.pert_gate = None       # type: ignore[assignment]

        # ── Time-varying inflammation-membership field (composition model) ──
        self.use_membership_field = bool(use_membership_field)
        self.membership_bce_lambda = float(membership_bce_lambda)
        self.membership_moran_lambda = float(membership_moran_lambda)
        self.use_time_intensity = bool(use_time_intensity)
        self.perturbed_code = int(perturbed_code)
        self._last_pi = None    # cache for the BCE term (set each forward)
        if self.use_membership_field:
            assert use_niche_film, "membership field requires --use_niche_film True (z_niche input)"
            assert use_time_conditioning, "membership field requires --use_time_conditioning True (time_emb input)"
            _mf_in = GP_dim + Normal_dim + int(niche_film_dim) + 2 + int(time_emb_dim)
            self.membership_field = MembershipField(in_dim=_mf_in, hidden=int(membership_hidden))
        else:
            self.membership_field = None

        # ── continuous inflammation field s = softplus(MLP([z_niche, coords, time_emb])) ──
        # Input DROPS gate_state (the per-cell own-latent) — the leak-prevention change.
        self.use_inflammation_field = bool(use_inflammation_field)
        self.field_mse_lambda = float(field_mse_lambda)
        self.field_time_fixed_t = (None if field_time_fixed_t is None
                                   else float(field_time_fixed_t))
        self.field_moran_lambda = float(field_moran_lambda)
        self._last_s = None      # cache for the field MSE term (set each forward)
        self._last_field_mse = None
        if self.use_inflammation_field:
            assert not self.use_membership_field, \
                "use_inflammation_field and use_membership_field are mutually exclusive"
            assert use_niche_film, "inflammation field requires --use_niche_film True (z_niche input)"
            assert use_time_conditioning, \
                "inflammation field requires --use_time_conditioning True (time_emb input)"
            _if_in = int(niche_film_dim) + 2 + int(time_emb_dim)   # z_niche + coords(2) + time_emb
            self.inflammation_field = InflammationField(in_dim=_if_in, hidden=int(membership_hidden))
        else:
            self.inflammation_field = None

        # Encoder / decoder
        # LORD produces (basal, tissue, perturbation) embeddings each of size encoder_dim
        # [+ PAD action_dim if use_pad=True]
        self.lord_encoder = LordEncoder(
            embedding_dim=[encoder_dim] * 3,
            num_genes=self.num_genes,
            labels=cell_atts,
            attributes=["tissue", "perturbation"],
            attributes_type=["categorical", "categorical"],
            noise=self.noise,
            device=device,
            use_expr_encoder=use_expr_encoder,
            expr_hidden_dims=expr_hidden_dims,
            expr_dropout=expr_dropout,
            # tissue cardinality from the data (spac → all-zeros → 1 class; perturbmap
            # → tumor/normal → 2 classes). Hardcoding 1 silently drops tissue disentanglement.
            n_labels_override=[
                int(np.asarray(cell_atts)[:, 0].max()) + 1 if np.asarray(cell_atts).size else 1,
                n_batch,
            ],
        )
        # ── Niche-FiLM head: transferable niche descriptor + per-pert sign-flip gamma ──
        self.use_niche_film = use_niche_film
        if use_niche_film:
            self.niche_descriptor = NicheDescriptor(Normal_dim, out_dim=niche_film_dim,
                                                    k=niche_film_k,
                                                    dispersion=niche_film_dispersion)
            self.niche_h = nn.Sequential(nn.Linear(encoder_dim + niche_film_dim, 64), nn.ReLU())
            self.gamma_head = nn.Linear(64, 1)
            nn.init.zeros_(self.gamma_head.weight)
            nn.init.zeros_(self.gamma_head.bias)              # tanh(0)=0 -> gamma≈1 at init
            # per-perturbation modulation DEPTH as a function of the pert embedding (no index
            # threading; generalizes to unseen perts). bias -4 -> softplus≈0.018 -> near-identity.
            self.g_scale_head = nn.Linear(encoder_dim, 1)
            nn.init.zeros_(self.g_scale_head.weight)
            nn.init.constant_(self.g_scale_head.bias, -4.0)
            self.g_cap = float(niche_film_gcap)
            self.niche_film_l2 = float(niche_film_l2)
            # low-rank z_niche-driven per-gene residual (beta_out zero-init -> beta=0 -> identity)
            _r = int(niche_film_rank)
            self.beta_U = nn.Linear(encoder_dim, _r, bias=False)
            self.beta_V = nn.Linear(niche_film_dim, _r, bias=False)
            self.beta_out = nn.Linear(_r, self.num_genes, bias=False)
            nn.init.normal_(self.beta_U.weight, std=0.01)
            nn.init.normal_(self.beta_V.weight, std=0.01)
            nn.init.zeros_(self.beta_out.weight)
        else:
            self.niche_descriptor = None

        # ── Continuous-time conditioning: low-rank time×pert log-shift ──────────
        # shift += time_out( time_U(pert_emb) ⊙ time_V(time_embed(t)) ).  time_out is
        # zero-init → the term is 0 at epoch 0, so a use_time_conditioning=True model
        # is byte-identical to the niche model until it learns temporal structure.
        # Gated at the return of _pert_log_shift_from_lord by is_real_KO, so only
        # perturbed cells carry a time-evolving effect (control = time-invariant baseline).
        self.use_time_conditioning = bool(use_time_conditioning)
        self.time_l2 = float(time_l2)
        if self.use_time_conditioning:
            self.time_embed = TimeEmbedding(n_bands=int(time_fourier_bands),
                                            out_dim=int(time_emb_dim))
            _tr = int(time_rank)
            self.time_U = nn.Linear(encoder_dim, _tr, bias=False)      # from pert_emb
            self.time_V = nn.Linear(int(time_emb_dim), _tr, bias=False)  # from time_emb
            self.time_out = nn.Linear(_tr, self.num_genes, bias=False)
            nn.init.normal_(self.time_U.weight, std=0.01)
            nn.init.normal_(self.time_V.weight, std=0.01)
            nn.init.zeros_(self.time_out.weight)                       # near-identity init

        # ── Timekernel: shared Tier1+architecture basal identity ────────────
        # Replace the free per-cell Gaussian basal latent with a SHARED lookup
        # embedding per Tier1 lineage plus a LINEAR map of the day-invariant
        # architectural descriptor. Two cells with the same (Tier1, arch_desc) get
        # IDENTICAL basal → the basal path physically cannot store one cell's
        # inflammation, forcing it into the perturbation term (Task 4; design
        # §4.1/§5 guarantee 1). No coords, no expression, no perturbation label,
        # no free latent enter this path — and, unlike the Leiden_neigh code it
        # replaces, the descriptor cannot explain BETWEEN-day variation either,
        # which is what identifies a(t) (arch-descriptor design §4).
        self.use_shared_basal = bool(use_shared_basal)
        self.n_tier1 = n_tier1
        self.arch_dim = int(arch_dim)
        if self.use_shared_basal:
            assert n_tier1 and self.arch_dim > 0, "shared basal needs n_tier1 and arch_dim"
            self.E_tier1 = nn.Embedding(int(n_tier1), encoder_dim)
            self.W_arch_basal = nn.Linear(self.arch_dim, encoder_dim)
            self.dec_basal_shared = nn.Linear(encoder_dim, self.num_genes)
            if not hasattr(self, "_mean_act"):
                self._mean_act = MeanAct()   # exp head (clamped >0); shared with additive decoder
        # Shared-basal mode has no gs_mu, so the batch-local NicheDescriptor cannot supply
        # z_niche for the FiLM head. Instead learn a linear map of the architectural
        # descriptor (arch_dim → niche_film_dim) that matches the FiLM head's z_niche
        # input contract (Task 6a).
        if self.use_shared_basal and self.use_niche_film:
            self.W_arch_field = nn.Linear(self.arch_dim, niche_film_dim)

        # ── Timekernel: smooth non-negative inflammation amplitude a(t) ──────
        # Scales the shared coord/niche-modulated program by a smooth scalar a(t)
        # for ALL cells (no is_real_KO gate); replaces the sigmoid inflammation field.
        self.use_timekernel = bool(use_timekernel)
        self.amp_anchor_lambda = float(amp_anchor_lambda)
        self.amp_anchor_t = float(amp_anchor_t)
        self.amp_anchor_offset = float(amp_anchor_offset)
        self._last_a0 = None     # cache for logging (set each forward when anchored)
        if self.use_timekernel:
            self.amp_time = AmplitudeTime(time_emb_dim=time_emb_dim,
                                          anchor_offset=self.amp_anchor_offset)

        # Marker-panel sign constraint. Buffer (not a plain list) so the indices move with
        # .to(device) and survive a state_dict round-trip; register_buffer accepts None, so
        # the attribute always exists and the branch below is a simple None check.
        self.marker_sign_positive = bool(marker_sign_positive)
        _mk = None
        if self.marker_sign_positive:
            if marker_gene_idx is None or len(marker_gene_idx) == 0:
                raise ValueError("marker_sign_positive=True requires a non-empty marker_gene_idx")
            _mk = torch.as_tensor(list(marker_gene_idx), dtype=torch.long)
        self.register_buffer("marker_gene_idx", _mk)

        self.encoder = DenseEncoder(
            input_dim=_encoder_input_dim,
            hidden_dims=list(encoder_layers),
            output_dim=GP_dim + Normal_dim,
            activation="elu",
            dropout=encoder_dropout,
        )
        self.decoder = buildNetwork([GP_dim + Normal_dim] + list(decoder_layers), activation="elu", dropout=decoder_dropout)
        last_dim = decoder_layers[-1] if len(list(decoder_layers)) > 0 else GP_dim + Normal_dim
        self.dec_mean = nn.Sequential(nn.Linear(last_dim, self.num_genes), MeanAct())

        # Option 2 — gated additive decoder. Reconstruct log μ additively from TWO separate
        # paths: a spatial path fed ONLY by the coord-driven GP latent z_gp, and a basal path
        # fed ONLY by the expression basal latent z_basal. Because the paths no longer share a
        # decoder MLP (where the high-capacity basal input dominates the first layer), z_gp gets
        # an un-shared gradient path to reconstruction. Paired with basal whitening (moran_lambda
        # + small Normal_dim), the smooth spatial component is forced through z_gp → GP becomes
        # load-bearing without de-amortizing (so z_gp stays coord-driven, preserving the
        # spatial-footprint goal). log μ_basal = dec_logmu_gp(z_gp) + dec_logmu_basal(z_basal).
        if self.additive_decoder:
            self.decoder_gp = buildNetwork([GP_dim] + list(decoder_layers), activation="elu", dropout=decoder_dropout)
            self.decoder_basal = buildNetwork([Normal_dim] + list(decoder_layers), activation="elu", dropout=decoder_dropout)
            _last_gp = decoder_layers[-1] if len(list(decoder_layers)) > 0 else GP_dim
            _last_basal = decoder_layers[-1] if len(list(decoder_layers)) > 0 else Normal_dim
            self.dec_logmu_gp = nn.Linear(_last_gp, self.num_genes)
            self.dec_logmu_basal = nn.Linear(_last_basal, self.num_genes)
            self._mean_act = MeanAct()
            self._last_logbasal_frac = 0.0  # running diag: basal share of total |log μ|

        # NB dispersion (shared or batch-specific)
        if shared_dispersion:
            self.dec_disp = nn.Parameter(torch.randn(self.num_genes), requires_grad=True)
        else:
            self.dec_disp = nn.Parameter(torch.randn(self.num_genes, n_batch), requires_grad=True)

        # C2 (multi-slide batch correction): per-slide additive gene-level log-mean bias.
        # Only created for >1 slide so single-slide models are byte-identical (no new param).
        if self.n_slides > 1:
            self.slide_bias = nn.Parameter(torch.zeros(self.n_slides, self.num_genes))
        else:
            self.slide_bias = None

        # Learnable mask cutoff vector
        self.mask_cutoff = nn.Parameter(torch.tensor(mask_cutoff, dtype=dtype), requires_grad=True)

        # Losses
        self.NB_loss = NBLoss().to(self.device)
        self.mse = nn.MSELoss(reduction="mean")
        self.to(device)

    def _decode_to_mean(self, z: torch.Tensor) -> torch.Tensor:
        """Map a latent sample z = [z_gp ; z_basal] to the count-space basal mean μ_basal.

        Standard mode: shared decoder MLP then exp head. Additive mode (Option 2): two
        separate paths summed in log space — log μ = dec_logmu_gp(z_gp) + dec_logmu_basal(z_basal)
        — so the GP path has an independent gradient route to reconstruction.
        """
        if not self.additive_decoder:
            return self.dec_mean(self.decoder(z))
        z_gp = z[:, : self.GP_dim]
        z_basal = z[:, self.GP_dim :]
        log_gp = self.dec_logmu_gp(self.decoder_gp(z_gp))
        log_basal = self.dec_logmu_basal(self.decoder_basal(z_basal))
        if getattr(self, "basal_moran_output_lambda", 0.0) > 0:
            self._cache_log_basal = log_basal   # RC2: penalized for spatial autocorr in forward
        if self.training:
            with torch.no_grad():
                tot = log_gp.abs().mean() + log_basal.abs().mean() + 1e-8
                self._last_logbasal_frac = float((log_basal.abs().mean() / tot).item())
        return self._mean_act(log_gp + log_basal)

    def _shared_basal_mean(self, tier1_id, arch_desc):
        """Basal (healthy) mean from the SHARED lineage embedding + a linear map of the
        day-invariant architectural descriptor only. No per-cell free latent, no coords,
        no expression, no pert label, no time -> cannot encode one cell's inflammation
        (design §4.1, §5 guarantee 1) and cannot encode its day (arch-descriptor §4).

        tier1_id (B,) long; arch_desc (B, arch_dim) float (already standardised)."""
        ident = self.E_tier1(tier1_id.long()) + self.W_arch_basal(arch_desc)
        return self._mean_act(self.dec_basal_shared(ident))          # (B, G), > 0

    # -------------------------
    # Persistence
    # -------------------------
    def save_model(self, path: str) -> None:
        atomic_torch_save(self.state_dict(), path)

    def load_model(self, path: str, allow_cell_size_mismatch: bool = False,
                   skip_per_cell: bool = False) -> None:
        state = torch.load(path, map_location=lambda storage, loc: storage)
        _per_cell = {"mask_cutoff", "lord_encoder.z_encoder.embedding.weight"}
        merged = dict(self.state_dict())
        for k, ckpt_v in state.items():
            if k not in merged:
                continue
            model_v = merged[k]
            if k in _per_cell and skip_per_cell and ckpt_v.shape != model_v.shape:
                # Held-out-mouse eval: the eval cells are DISJOINT from training, so neither
                # the checkpoint values nor a front-truncation (which grabs the FIRST N training
                # cells — different cells) is meaningful. Keep the fresh init sized to the eval
                # set: mask_cutoff -> 0.5 (the population default; only consumed by the SVGP
                # basal branch, not the rff-basal decode used here) and the per-cell basal lookup
                # -> unused because use_expr_encoder amortises basal from each cell's expression.
                logging.warning("[load_model] skip_per_cell: %s kept at fresh init %s (ckpt %s)",
                                k, tuple(model_v.shape), tuple(ckpt_v.shape))
                continue
            if ckpt_v.shape == model_v.shape:
                merged[k] = ckpt_v
            elif k == "slide_bias" and ckpt_v.shape[0] > model_v.shape[0]:
                # multi-slide checkpoint → fewer-slide model. Keep the leading rows;
                # target slide is row 0 by multi_sample_files convention.
                merged[k] = ckpt_v[:model_v.shape[0]].clone()
                logging.warning("[load_model] truncated slide_bias from %s to %s (kept row 0 = target slide)",
                                tuple(ckpt_v.shape), tuple(model_v.shape))
            elif allow_cell_size_mismatch and k in _per_cell and ckpt_v.shape[0] >= model_v.shape[0]:
                # Multi-slide checkpoint → single-slide eval: truncate to model size.
                # Correct because the target slide is always first in multi_sample_files.
                merged[k] = ckpt_v[:model_v.shape[0]]
                logging.warning("[load_model] truncated %s from %s to %s", k, tuple(ckpt_v.shape), tuple(model_v.shape))
            else:
                raise RuntimeError(
                    f"Size mismatch for {k}: checkpoint {tuple(ckpt_v.shape)} vs model {tuple(model_v.shape)}"
                )
        # Guard the silent-partial-load trap: any TRAINABLE parameter present in the model but
        # ABSENT from the checkpoint keeps its random init (no error). This silently broke the
        # 2-2 Bayes-basal CF eval (deterministic head built vs bayes checkpoint → mlp.2 random).
        # Warn loudly so an arch/flag mismatch (e.g. missing --basal_bayes) surfaces immediately.
        _missing = [k for k in dict(self.named_parameters()) if k not in state]
        if _missing:
            logging.warning(
                "[load_model] %d model parameter(s) NOT in checkpoint — left at RANDOM INIT "
                "(likely a build/flag mismatch vs training, e.g. --basal_bayes/--basal_field/--fpert_field): %s",
                len(_missing), _missing[:12],
            )
        self.load_state_dict(merged)

    def attach_graph(self, graph) -> None:
        """Attach the precomputed SpatialGraph (device tensors). Call after construction."""
        self.graph = graph

    # -------------------------
    # Helpers for disentangled routing
    # -------------------------
    def _encoder_input_from_lord(self, lord: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Assemble encoder input from LORD output, dropping pert if disentangled mode is on.

        Layout (in order):
            basal_latent + tissue            if use_basal_encoder_only
            basal_latent + tissue + pert     otherwise (legacy total_latent)
        """
        if self.use_basal_encoder_only:
            return torch.cat([lord["basal_latent"], lord["tissue"]], dim=-1)
        return lord["total_latent"]

    @property
    def paracrine_radius(self) -> torch.Tensor:
        """Per-pert Cauchy signaling radius (positive), softplus-parameterized, r_min=0.1."""
        return torch.nn.functional.softplus(self._raw_radius) + 0.1

    def _pert_corrected_expr(self, y, is_real_KO, cell_atts, sample_index):
        """Subtract the current estimate of the perturbation log-shift from expression
        before encoding (default-OFF fallback for residual leakage). EM-style: uses the
        model's own f_pert / mu_action, detached so it does not create a training cycle."""
        if not self.use_f_pert or self.f_pert is None or is_real_KO is None:
            return y
        with torch.no_grad():
            lord_tmp = self.lord_encoder.predict(
                sample_indices=sample_index, labels=cell_atts, batch_size=y.shape[0],
                expression=None,  # attribute embeddings only; basal unused here
            )
            shift = self._pert_log_shift_from_lord(lord_tmp, is_real_KO, niche=None)
        if shift is None:
            return y
        return y - shift.detach()

    def _apply_niche_film(self, shift: torch.Tensor, pert_emb: torch.Tensor,
                          z_niche: torch.Tensor) -> torch.Tensor:
        # Niche-conditioned FiLM: per-cell SCALAR gamma (can cross 0 -> sign flip) modulates
        # the whole core+field signature; low-rank z_niche-driven beta adds a per-gene residual.
        g_scale = torch.clamp(torch.nn.functional.softplus(self.g_scale_head(pert_emb)),
                              max=self.g_cap)                 # (B,1) per-pert modulation depth
        h = self.niche_h(torch.cat([pert_emb, z_niche], dim=-1))
        gamma = 1.0 + g_scale * torch.tanh(self.gamma_head(h))   # (B,1)
        shift = gamma * shift
        beta = self.beta_out(self.beta_U(pert_emb) * self.beta_V(z_niche))   # (B, G)
        shift = shift + beta
        return shift

    def _pert_log_shift_from_lord(
        self, lord: Dict[str, torch.Tensor], is_real_KO: Optional[torch.Tensor],
        niche: Optional[torch.Tensor] = None,
        gate_state: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        z_niche: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Gene-space per-perturbation log shift, gated by is_real_KO mask.

        f_pert(pert) is the niche-invariant core. When use_niche_interaction is on
        and a niche latent (GP summary) is given, a low-rank niche x pert term is
        added so the effect's fold-change can vary by niche. When use_pert_gate is on
        and gate_state ([gp_latent ; basal_latent]) is given, a per-cell scalar gate
        ∈ [0,1] multiplies the core (gate factorization). Returns (B, G) or None.
        """
        if not self.use_f_pert or self.f_pert is None:
            return None
        if self.use_membership_field or self.use_inflammation_field or self.use_timekernel:
            # Shared inflamed program: use the perturbed-code embedding for EVERY cell so a
            # control cell can be driven inflamed when the field rises. The field / a(t) (not
            # is_real_KO) gates it. Also sets B for the timekernel branch below.
            B = lord["perturbation"].shape[0]
            pert_table = self.lord_encoder.s_encoder["perturbation"].weight   # (n_pert, enc)
            pert_emb = pert_table[self.perturbed_code].unsqueeze(0).expand(B, -1)
        else:
            if is_real_KO is None:
                return None
            pert_emb = lord["perturbation"]                       # (B, encoder_dim)
        shift = self.f_pert(pert_emb)                             # program core (B, G)
        if getattr(self, "fpert_field", None) is not None and coords is not None:
            shift = shift + self.fpert_field(pert_emb, coords, time_emb=time_emb)  # coord-conditioned residual (B, G); ℓ(t)
        if self.use_niche_film and z_niche is not None:
            shift = self._apply_niche_film(shift, pert_emb, z_niche)
        elif self.use_niche_interaction and niche is not None and self.inter_out is not None:
            nb = self.niche_proj(niche)                           # (B, nd)
            inter = self.inter_out(self.inter_U(pert_emb) * self.inter_V(nb))   # (B, G)
            shift = shift + inter
        if self.use_time_conditioning and self.use_time_intensity and time_emb is not None:
            # time × pert low-rank residual: the perturbation (inflammation) fold-change
            # evolves over the DSS course. Coupled here (not a separate additive time
            # term) so it modulates the pert effect per cell state — this is what can
            # beat linear interpolation at the D9 peak.
            time_shift = self.time_out(self.time_U(pert_emb) * self.time_V(time_emb))  # (B, G)
            shift = shift + time_shift
        if self.use_timekernel:
            # Timekernel: shared program for ALL cells (control included) scaled by a smooth
            # non-negative amplitude a(t). The coord/niche field Φ modulates the shared program;
            # a(t) supplies the continuous field magnitude (replaces the sigmoid inflammation s).
            pert_table = self.lord_encoder.s_encoder["perturbation"].weight   # (n_pert, enc)
            pert_emb = pert_table[self.perturbed_code].unsqueeze(0).expand(B, -1)
            shift = self.f_pert(pert_emb)                          # (B, G) shared program
            if getattr(self, "fpert_field", None) is not None and coords is not None:
                shift = shift + self.fpert_field(pert_emb, coords, time_emb=time_emb)  # Φ; ℓ(t) inside
            if self.use_niche_film and z_niche is not None:
                shift = self._apply_niche_film(shift, pert_emb, z_niche)
            if self.marker_sign_positive and self.marker_gene_idx is not None:
                # Force the DSS panel's component of the program to point UP. index_copy is
                # OUT-of-place, so autograd sees a clean graph (an in-place shift[:, mk] = ...
                # on a tensor needed for grad would error). Applied BEFORE the clamp so the
                # cached diagnostic reads exactly what the decoder consumes, and before the
                # multiplier s_i so the constraint holds regardless of magnitude.
                mk = self.marker_gene_idx
                shift = shift.index_copy(
                    1, mk, torch.nn.functional.softplus(shift.index_select(1, mk)))
            shift = torch.clamp(shift, -self.f_pert_clamp, self.f_pert_clamp)
            self._last_shift = shift                               # cache for sign diagnostics
            if self.use_inflammation_field and z_niche is not None and time_emb is not None:
                _fixed = getattr(self, "field_time_fixed_t", None)
                s_time_emb = (self.time_embed(
                    time_emb.new_full((time_emb.shape[0],), float(_fixed))).detach()
                    if _fixed is not None else time_emb)
                s = self.inflammation_field(torch.cat([z_niche, coords, s_time_emb], dim=-1))
                self._last_s = s                                   # cache for the field MSE term
                self._last_a = s                                   # amplitude readout = s
                return s.unsqueeze(1) * shift                      # s_i·Φ·f_pert
            a_t = (self.amp_time(time_emb, ref_emb=self._baseline_time_emb(shift))
                   if time_emb is not None else shift.new_ones(B))
            self._last_a = a_t                                     # cache for logging
            return a_t.unsqueeze(1) * shift                       # a(t)·Φ·f_pert
        if self.use_inflammation_field:
            # continuous s ≥ 0 from [z_niche, coords, time_emb] ONLY (NO gate_state —
            # leak prevention). No is_real_KO gate, no sigmoid. s scales the shared program.
            if z_niche is None or coords is None or time_emb is None:
                raise ValueError("inflammation field needs z_niche, coords, time_emb")
            feats = torch.cat([z_niche, coords, time_emb], dim=-1)
            s = self.inflammation_field(feats)                    # (B,) ≥ 0
            self._last_s = s                                      # cache for the field MSE term
            shift = torch.clamp(shift, -self.f_pert_clamp, self.f_pert_clamp)
            return s.unsqueeze(1) * shift                         # (B, G)
        if self.use_membership_field:
            # π ∈ [0,1] from [gate_state, z_niche, coords, time_emb]; gates the shared program.
            if gate_state is None or z_niche is None or coords is None or time_emb is None:
                raise ValueError("membership field needs gate_state, z_niche, coords, time_emb")
            feats = torch.cat([gate_state, z_niche, coords, time_emb], dim=-1)
            pi = self.membership_field(feats)                     # (B,)
            self._last_pi = pi                                    # cache for BCE
            shift = torch.clamp(shift, -self.f_pert_clamp, self.f_pert_clamp)
            return pi.unsqueeze(1) * shift                        # (B, G)
        if self.use_pert_gate and self.pert_gate is not None and gate_state is not None:
            gate = torch.sigmoid(self.pert_gate(gate_state))      # (B, 1) ∈ [0,1]
            shift = shift * gate
        shift = torch.clamp(shift, -self.f_pert_clamp, self.f_pert_clamp)
        return is_real_KO.unsqueeze(1) * shift                     # (B, G)

    @staticmethod
    def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)) -> torch.Tensor:
        """Multi-bandwidth RBF MMD^2 between two batches of vectors (same dim).

        Uses the dot-product identity ‖a−b‖² = ‖a‖²+‖b‖²−2⟨a,b⟩ so
        memory cost is O(n²) not O(n²·G) — safe with G=10k genes.
        """
        def _sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a2 = (a * a).sum(-1, keepdim=True)   # (n, 1)
            b2 = (b * b).sum(-1, keepdim=True)   # (m, 1)
            ab = a @ b.t()                        # (n, m)
            return (a2 + b2.t() - 2.0 * ab).clamp(min=0.0)

        d_xx = _sq_dists(x, x)
        d_yy = _sq_dists(y, y)
        d_xy = _sq_dists(x, y)
        out = x.new_zeros(())
        for s in sigmas:
            g = 1.0 / (2.0 * s * s)
            out = out + (torch.exp(-g * d_xx).mean() + torch.exp(-g * d_yy).mean()
                         - 2.0 * torch.exp(-g * d_xy).mean())
        return out

    def _baseline_time_emb(self, like: torch.Tensor):
        """Fourier embedding of the healthy-baseline time t_ref, as a (1, time_emb_dim) row.

        Returns None unless the by-construction anchor is on, which keeps the un-anchored
        a(t) = softplus(g(t)) path byte-identical to the pre-fix model.
        """
        if self.amp_anchor_offset <= 0 or getattr(self, "time_embed", None) is None:
            return None
        t_ref = torch.full((1, 1), self.amp_anchor_t,
                           device=like.device, dtype=like.dtype)
        return self.time_embed(t_ref)

    # -------------------------
    # Core forward
    # -------------------------
    def forward(
        self,
        *,
        x: torch.Tensor,           # positions (+ batch one-hot appended)
        y: torch.Tensor,           # normalized counts
        batch: torch.Tensor,       # one-hot batch
        raw_y: torch.Tensor,       # raw counts
        sample_index: torch.Tensor,
        cell_atts: torch.Tensor,
        size_factors: torch.Tensor,
        cutoff: torch.Tensor,
        num_samples: int = 1,
        is_perturbed: Optional[torch.Tensor] = None,  # (B,) int 0/1 for PAD
        has_guide: Optional[torch.Tensor] = None,     # (B,) float 0/1 — action indicator
        is_real_KO: Optional[torch.Tensor] = None,   # (B,) float 0/1 — KO indicator
        niche_id: Optional[torch.Tensor] = None,     # (B,) long — niche bin id for LFC supervision
        slide_index: Optional[torch.Tensor] = None,   # (B,) long — per-cell slide id (C2)
        time_t: Optional[torch.Tensor] = None,        # (B,) float — normalized day t∈[0,1]
        infl_score: Optional[torch.Tensor] = None,    # (B,) float — observed continuous inflammation gene-score (v3 target)
        tier1_id: Optional[torch.Tensor] = None,      # (B,) long — Tier1 lineage code (timekernel)
        arch_desc: Optional[torch.Tensor] = None,     # (B, arch_dim) float — day-invariant architectural descriptor (timekernel; standardised)
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass computing ELBO terms and reconstructions."""
        self.train()
        time_emb = (self.time_embed(time_t)
                    if (self.use_time_conditioning and time_t is not None) else None)
        bsz = y.shape[0]

        # LORD encodes attribute-aware representation
        basal_expr = y
        if self.use_pert_correction:
            basal_expr = self._pert_corrected_expr(
                y, is_real_KO=is_real_KO, cell_atts=cell_atts, sample_index=sample_index)
        lord = self.lord_encoder.predict(
            sample_indices=sample_index, labels=cell_atts, batch_size=bsz,
            expression=basal_expr if self.use_expr_encoder else None,
        )
        # encoder input drops pert when use_basal_encoder_only=True
        y_embed = self._encoder_input_from_lord(lord)

        # PAD: append binary action embedding + compute denoising losses
        pad_loss = torch.tensor(0.0, device=self.device)
        if self.use_pad and is_perturbed is not None:
            s_action = self.pad(is_perturbed)                           # (B, action_dim)
            y_embed = torch.cat([y_embed, s_action], dim=-1)            # (B, …+action)
            pad_loss = self.pad.combined_loss(lord["basal_latent"], is_perturbed)

        if self.use_shared_basal:
            # ── Timekernel shared-basal identity path ───────────────────────
            # Basal (healthy) mean comes ONLY from the shared Tier1 lookup
            # embedding + a linear map of the day-invariant architectural
            # descriptor. The encoder, GP field, and free Gaussian latent (the
            # inflammation-leak vectors) are SKIPPED entirely; every
            # latent-derived ELBO contribution is set to zero. Two cells with
            # the same (Tier1, arch_desc) therefore get IDENTICAL basal, so the
            # basal path cannot store one cell's inflammation (Task 4).
            assert tier1_id is not None and arch_desc is not None, \
                "use_shared_basal requires tier1_id and arch_desc in forward()"
            mu_basal_shared = self._shared_basal_mean(tier1_id, arch_desc)   # (B,G) > 0
            gs_mu = None                     # no free per-cell basal latent in this mode
            _z0 = torch.zeros((), device=self.device)
            gp_KL = _z0
            gauss_KL = _z0
            gp_ce = _z0
            inside_elbo = _z0
            inside_elbo_recon = _z0
            inside_elbo_kl = _z0
            disent_loss = _z0
            moran_loss = _z0
            gp_aux_loss = _z0
            nngp_kl = torch.tensor(0.0, device=self.device)
            gp_p_m = torch.zeros((bsz, self.GP_dim), device=self.device)
            gp_p_v = torch.ones((bsz, self.GP_dim), device=self.device)
            q_mu = torch.zeros((bsz, self.GP_dim + self.Normal_dim), device=self.device)
            q_var = torch.ones((bsz, self.GP_dim + self.Normal_dim), device=self.device)
            latent_dist = None               # basal not sampled from a latent in shared mode
        else:
            # Inference network -> posterior params for GP & Gaussian latents
            q_mu, q_var = self.encoder(y_embed)
            q_mu  = torch.nan_to_num(q_mu,  nan=0.0)
            q_var = torch.nan_to_num(q_var, nan=1.0, posinf=1.0, neginf=1.0)
            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]
            if self.deamortize_gp:
                gp_mu = self.gp_mu_free[sample_index.long()]   # recon-driven free GP target

            if self.spatial_basal_only and (self.use_membership_field or self.use_inflammation_field) and is_real_KO is not None:
                is_una = is_real_KO < 0.5
                if not bool(is_una.any()):                               # all-inflamed batch guard
                    is_una = torch.ones(bsz, dtype=torch.bool, device=self.device)
            elif self.spatial_basal_only and has_guide is not None:
                is_una = has_guide < 0.5                                  # (B,) unassigned
                if not bool(is_una.any()):                               # all-perturbed batch guard
                    is_una = torch.ones(bsz, dtype=torch.bool, device=self.device)
            else:
                is_una = torch.ones(bsz, dtype=torch.bool, device=self.device)
            if self.spatial_basal_only:
                # zero the per-cell Normal latent for NTC/KO -> their basal is the spatial GP only.
                keep = is_una.unsqueeze(1).to(gs_mu.dtype)
                gs_mu = gs_mu * keep
                gs_var = gs_var * keep + (1.0 - keep)                     # NTC/KO -> unit-variance prior
                # NTC/KO are NON-INFORMATIVE observations (huge noise) so they cannot shape the
                # spatial field, but still receive a field prediction (gp_p_m) at their coordinate.
                # Keeping test==train==x preserves the per-cell cutoff sizing the SVGP kernel needs.
                gp_noise = torch.where(is_una.unsqueeze(1), gp_var, torch.full_like(gp_var, 1e6))
            else:
                gp_noise = gp_var

            # Spatial basal latent. RFF: deterministic coord(+tissue) field, no GP KL. SVGP:
            # per-pert GP posterior (legacy) — observations = all cells but NTC/KO carry ~0
            # precision (gp_noise), so only unassigned define the field.
            if self.basal_field == "rff":
                coords = x[:, :2]
                # tissue embedding from LORD (region identity) -> spatial field; None when 1 class
                tissue_emb = lord["tissue"] if self.basal_rff.tissue_dim > 0 else None
                gp_p_m = self.basal_rff(coords, tissue_emb=tissue_emb)  # (B, GP_dim) deterministic mean
                gp_p_v = torch.exp(self.basal_rff_logvar).expand_as(gp_p_m)
                inside_elbo_recon = torch.zeros((), device=self.device)
                inside_elbo_kl = torch.zeros((), device=self.device)
                inside_elbo = torch.zeros((), device=self.device)      # no GP KL for RFF basal
            else:
                elbo_recon_terms, elbo_kl_terms, gp_p_m, gp_p_v = [], [], [], []
                for l in range(self.GP_dim):
                    p_m_l, p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(
                        x, x, gp_mu[:, l], gp_noise[:, l], cutoff=cutoff
                    )
                    rec_l, kl_l = self.svgp.variational_loss(
                        x=x, y=gp_mu[:, l], noise=gp_noise[:, l], mu_hat=mu_hat_l, A_hat=A_hat_l, cutoff=cutoff
                    )
                    elbo_recon_terms.append(rec_l)
                    elbo_kl_terms.append(kl_l)
                    gp_p_m.append(p_m_l)
                    gp_p_v.append(p_v_l)

                inside_elbo_recon = torch.sum(torch.stack(elbo_recon_terms, dim=-1))
                inside_elbo_kl = torch.sum(torch.stack(elbo_kl_terms, dim=-1))
                inside_elbo = inside_elbo_recon - (bsz / self.svgp.N_train) * inside_elbo_kl
                # Guard: SVGP Cholesky can fail on ill-conditioned kernel (bad random init);
                # zero the term so GP has no gradient signal for that batch rather than NaN-ing all.
                inside_elbo = torch.nan_to_num(inside_elbo, nan=0.0, posinf=0.0, neginf=0.0)

                gp_p_m = torch.stack(gp_p_m, dim=1)
                gp_p_v = torch.stack(gp_p_v, dim=1)
                # Guard SVGP numerical instability: NaN posterior → zero mean, unit variance fallback
                gp_p_m = torch.nan_to_num(gp_p_m, nan=0.0)
                gp_p_v = torch.nan_to_num(gp_p_v, nan=1.0, posinf=1.0, neginf=1.0)

            # ── NNGP local field (epoch-stale cache for cross-batch neighbors) ──
            nngp_kl = torch.tensor(0.0, device=self.device)
            if self.use_nngp and self.nngp is not None and self.graph is not None:
                gidx = sample_index.long()
                knn_idx_b  = self.graph.knn_idx[gidx]      # (B, k) global neighbor indices
                knn_dist_b = self.graph.knn_dist[gidx]     # (B, k)
                neighbor_vals = self.gp_cache[knn_idx_b]   # (B, k, GP_dim) — detached cache
                local_mean, nngp_kl = self.nngp(knn_dist_b, neighbor_vals)
                # cache the PRE-smoothing GP means for future batches (avoid recursive feedback)
                # nan_to_num guards against SVGP instability poisoning the cache
                self.gp_cache[gidx] = torch.nan_to_num(gp_p_m.detach(), nan=0.0)
                gp_p_m = gp_p_m + local_mean

            # ── Route 2: auxiliary GP-field supervision ─────────────────────────
            # Force the coord-driven SVGP field gp_p_m to reconstruct the spatially-smoothed
            # expression (per-gene standardized so all genes weigh equally and λ is scale-free).
            # The gradient flows through gp_p_m into the encoder gp_mu AND the SVGP inducing
            # points, so they MUST carry coordinate structure -> un-starves the GP. Stays fully
            # amortized (no per-cell free params) -> works on unseen (GEX, coord) cells at test time.
            gp_aux_loss = torch.zeros((), device=self.device)
            if self.gp_field_aux_lambda > 0:
                coords = x[:, : (x.shape[1] - self.n_batch)]            # spatial coords only
                with torch.no_grad():
                    y_sm = _knn_smooth(y, coords, k=6)                  # (B,G) smoothed expr target
                    y_sm = (y_sm - y_sm.mean(dim=0, keepdim=True)) / (y_sm.std(dim=0, keepdim=True) + 1e-4)
                pred = self.gp_aux_head(gp_p_m)                         # (B,G) from coord-driven field
                gp_aux_loss = self.mse(pred, y_sm)
                self._last_gpaux = float(gp_aux_loss.detach().item())

            # GP prior KL via cross entropy (q||p). NTC/KO are pinned toward the field by this
            # term (helpful: trains the encoder to be pert-blind), while their z still comes from
            # gp_p_m (the field) — so no perturbation leaks into z.
            if self.basal_field == "rff":
                # RFF basal is deterministic + collapse-proof: no GP variational KL.
                gp_ce = torch.zeros((), device=self.device)
                gp_KL = torch.zeros((), device=self.device)
            else:
                gp_ce = self._gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var).sum()
                if self.gp_free_bits > 0:
                    # 9 v5a: per-dim GP free-bits floor (mirror of the Gaussian block).
                    # the earlier build left the GP block unfloored while the Normal block had a per-dim floor,
                    # so the encoder routed ~no signal into gp_mu and the SVGP smoothed a near-flat
                    # field (gp_latent_std ~1e-3). Decompose gp_KL per GP dim and clamp each to
                    # >= gp_free_bits * bsz: the first gp_free_bits nats/dim become KL-free so
                    # reconstruction can push the spatially-smooth cell-state component into the GP.
                    ce_per_dim = (0.5 * (torch.log(gp_var) - torch.log(gp_p_v)
                                         + (gp_p_v + (gp_p_m - gp_mu) ** 2) / gp_var - 1.0)
                                  ).sum(dim=0)                                    # (GP_dim,)
                    recon_per_dim = torch.stack(elbo_recon_terms)                 # (GP_dim,) scalars
                    kl_per_dim = torch.stack(elbo_kl_terms)                       # (GP_dim,) scalars
                    inside_per_dim = recon_per_dim - (bsz / self.svgp.N_train) * kl_per_dim
                    gp_KL_per_dim = ce_per_dim - inside_per_dim                   # (GP_dim,)
                    gp_KL = torch.clamp(gp_KL_per_dim, min=self.gp_free_bits * bsz).sum()
                    gp_KL = torch.nan_to_num(gp_KL, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    gp_KL = torch.nan_to_num(gp_ce - inside_elbo, nan=0.0, posinf=0.0, neginf=0.0)

            # Gaussian block KL (with optional free-bits floor so no Normal dim can fully
            # collapse to the prior — guards the posterior collapse fixed earlier).
            prior = Normal(torch.zeros_like(gs_mu), torch.ones_like(gs_var))
            post = Normal(gs_mu, torch.sqrt(gs_var))
            if self.free_bits > 0:
                gauss_KL = torch.clamp(
                    kl_divergence(post, prior).sum(dim=0), min=self.free_bits * bsz
                ).sum()
            else:
                gauss_KL = kl_divergence(post, prior).sum()

            # make the non-spatial latent gs independent of coordinates.
            # x is the per-cell coordinate tensor used by the SVGP above.
            if self.disentangle_lambda > 0:
                disent_loss = _hsic_biased(gs_mu, x)
                self._last_disent = float(disent_loss.detach().item())
            else:
                disent_loss = torch.zeros((), device=self.device)

            # penalize spatial autocorrelation of the basal latent so spatially-
            # predictable expression must be reconstructed through the coord-driven GP latent.
            if self.moran_lambda > 0:
                if self.disentangle_radius > 0:
                    moran_loss = _moran_i_scale(gs_mu, x, self.disentangle_radius)
                else:
                    moran_loss = _moran_i(gs_mu, x)
                self._last_moran = float(moran_loss.detach().item())
            else:
                moran_loss = torch.zeros((), device=self.device)

            # Reparameterized samples & NB recon loss
            gp_p_m = torch.nan_to_num(gp_p_m, nan=0.0)
            gp_p_v = torch.nan_to_num(gp_p_v, nan=1.0, posinf=1.0, neginf=1.0)
            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.clamp(torch.cat([gp_p_v, gs_var], dim=1), min=1e-6)
            latent_dist = Normal(p_m, torch.sqrt(p_v))

        # ── explicit gene-space log-shift ─────────────────────
        # log μ = log μ_basal + 𝟙[has_guide]·μ_action + 𝟙[is_real_KO]·f_pert(p)
        # In count space: μ = μ_basal * exp(action_shift + f_pert_shift)
        # Identifiable via the 3-class structure:
        #   unassigned → both shifts off  → establishes f_basal baseline
        #   control    → μ_action only    → identifies action effect
        #   gene_KO    → both terms       → identifies f_pert per perturbation
        action_shift = None
        if self.use_action_correction and has_guide is not None:
            mu_action_clamped = torch.clamp(self.mu_action, -5.0, 5.0)
            if self.detach_mu_action_for_ko and is_real_KO is not None:
                # Constrain μ_action to control (NTC) cells: KO cells still receive
                # μ_action in their decoder but with the gradient detached, so the
                # KO-vs-NTC signal is absorbed by f_pert instead of inflating μ_action.
                # μ_action then stays the pure guide-delivery effect (identified by
                # NTC), and f_pert(g) carries the FULL gene-KO-vs-NTC effect.
                rk = is_real_KO.float().unsqueeze(1)               # (B,1) KO cells
                hg = has_guide.float().unsqueeze(1)                # (B,1) guide cells
                action_shift = ((hg - rk) * mu_action_clamped.unsqueeze(0)
                                + rk * mu_action_clamped.detach().unsqueeze(0))
            else:
                action_shift = has_guide.unsqueeze(1) * mu_action_clamped.unsqueeze(0)  # (B, G)

        # detach: niche is a conditioning signal, not a gradient path through the SVGP posterior
        niche_latent = gp_p_m.detach() if self.use_niche_interaction else None
        # gate conditioning: [gp_latent ; basal_latent], detached (conditioning, not posterior grad path)
        # Shared-basal mode has no gs_mu; gate_state is unused by timekernel and the
        # real niche embedding (z_niche) is supplied by Task 6 — force both to None here.
        gate_state = (torch.cat([gp_p_m, gs_mu], dim=-1).detach()
                      if (not self.use_shared_basal
                          and (self.use_pert_gate or self.use_membership_field)) else None)
        # niche descriptor for the FiLM head. Shared-basal mode has no gs_mu, so use the
        # linear map of the architectural descriptor (Task 6a); else the batch-local
        # coord-kNN descriptor over gs_mu (detached — conditioning signal, not a
        # posterior gradient path).
        if self.use_niche_film:
            if self.use_shared_basal:
                z_niche_f = self.W_arch_field(arch_desc)              # (B, niche_film_dim)
            else:
                z_niche_f = self.niche_descriptor(gs_mu.detach(), x[:, :2])
        else:
            z_niche_f = None
        f_pert_shift = self._pert_log_shift_from_lord(
            lord, is_real_KO, niche=niche_latent, gate_state=gate_state,
            coords=x[:, :2], z_niche=z_niche_f, time_emb=time_emb)  # x first 2 cols are coords  # (B, G) or None

        mean_accum, disp_accum, recon_loss, recon_mse = [], [], 0.0, 0.0
        mu_basal_last: Optional[torch.Tensor] = None
        for _ in range(num_samples):
            if self.use_shared_basal:
                mu_basal = mu_basal_shared   # (B, G) — shared Tier1+niche identity, no latent sampling
            else:
                z = latent_dist.rsample()
                mu_basal = self._decode_to_mean(z)   # (B, G) — exp(pre-activation) in count space
            mu_basal_last = mu_basal       # keep last sample for consistency loss

            log_shift = torch.zeros_like(mu_basal)
            if action_shift is not None:
                log_shift = log_shift + action_shift
            if f_pert_shift is not None:
                log_shift = log_shift + f_pert_shift
            if self.use_paracrine and self.s_signature is not None and self.graph is not None:
                pert_emb_table = self.lord_encoder.s_encoder["perturbation"].weight   # (n_pert, enc)
                para = paracrine_log_shift(self.graph, sample_index.long(),
                                           self.paracrine_radius, self.s_signature,
                                           pert_emb_table, clamp=self.f_pert_clamp)
                log_shift = log_shift + para
            if self.slide_bias is not None and slide_index is not None:
                log_shift = log_shift + self.slide_bias[slide_index.long()]   # C2 per-slide bias
            mu = mu_basal * torch.exp(log_shift)

            if self.shared_dispersion:
                disp = torch.exp(torch.clamp(self.dec_disp, -15.0, 15.0)).T
            else:
                disp = torch.exp(torch.clamp(torch.matmul(self.dec_disp, batch.T), -15.0, 15.0)).T
            mean_accum.append(mu)
            disp_accum.append(disp)
            recon_loss = recon_loss + self.NB_loss(x=raw_y, mean=mu, disp=disp, scale_factor=size_factors)
            recon_mse = recon_mse + self.mse(mu, raw_y)

        recon_loss = recon_loss / max(1, num_samples)
        recon_mse = recon_mse / max(1, num_samples)

        # RC2 spatial residualization: spatial autocorrelation of the additive basal OUTPUT
        # contribution. Penalizing it forces the smooth spatial component of expression out of
        # the basal path (which otherwise wins reconstruction) and into the coord-driven GP path.
        basal_out_moran = torch.zeros((), device=self.device)
        if self.basal_moran_output_lambda > 0 and self._cache_log_basal is not None:
            basal_out_moran = _moran_i(self._cache_log_basal, x)
            self._last_basal_out_moran = float(basal_out_moran.detach().item())
            self._cache_log_basal = None

        # LORD penalty on basal latent
        unknown_pen = _unknown_attribute_penalty(lord["basal_latent"])  # encourages disentanglement
        lord_loss = unknown_pen + recon_mse

        # ── consistency loss — push mu_basal of KO and control
        # cells to share a distribution. Forces f_pert to absorb all per-pert
        # variance; mu_basal stays pert-blind. Batch-level MMD on log mu_basal.
        consistency_loss = torch.tensor(0.0, device=self.device)
        if (
            self.use_consistency_loss
            and mu_basal_last is not None
            and has_guide is not None
            and is_real_KO is not None
        ):
            is_ko_mask = is_real_KO.bool()
            is_ctrl_mask = has_guide.bool() & (~is_ko_mask)   # has guide but not real KO
            if int(is_ko_mask.sum().item()) > 1 and int(is_ctrl_mask.sum().item()) > 1:
                log_basal = torch.log(mu_basal_last + 1e-6)
                # L2-normalize each cell vector so squared distances are in [0, 4]
                # regardless of gene count → bandwidth selection stays sensible.
                lb_norm = torch.nn.functional.normalize(log_basal, dim=1)
                consistency_loss = self._mmd_rbf(lb_norm[is_ko_mask], lb_norm[is_ctrl_mask])

        # consistency directly on the basal latent (leakage happens at the
        # latent, so regularize it there, not only on downstream mu_basal).
        basal_consistency_loss = torch.tensor(0.0, device=self.device)
        if (
            self.use_basal_consistency
            and has_guide is not None
            and is_real_KO is not None
        ):
            is_ko_mask_b = is_real_KO.bool()
            is_ctrl_mask_b = has_guide.bool() & (~is_ko_mask_b)
            if int(is_ko_mask_b.sum().item()) > 1 and int(is_ctrl_mask_b.sum().item()) > 1:
                basal_lat = lord["basal_latent"]
                basal_consistency_loss = self._mmd_rbf(
                    basal_lat[is_ko_mask_b], basal_lat[is_ctrl_mask_b])

        # ── mini-bulk KO-vs-NTC LFC supervision ────────────────────
        # Model's KO-vs-NTC pseudobulk LFC over a niche IS f_pert_shift (mu_action is
        # shared with NTC and cancels). Pull it toward the observed per-(niche, pert) LFC.
        # Mini-bulk only — per-cell LFC is undefined on sparse spots.
        lfc_loss = torch.tensor(0.0, device=self.device)
        if (self.lfc_lambda > 0 and self.lfc_targets is not None
                and f_pert_shift is not None and is_real_KO is not None and niche_id is not None):
            ko = is_real_KO > 0.5
            pert_codes = cell_atts[:, 1].long()
            nid = niche_id.long()
            ko_idx = torch.where(ko)[0]
            if ko_idx.numel() > 0:
                pairs = torch.stack([nid[ko_idx], pert_codes[ko_idx]], dim=1).unique(dim=0)
                n_terms = 0
                for kp in pairs:
                    tgt = self.lfc_targets.get((int(kp[0].item()), int(kp[1].item())))
                    if tgt is None:
                        continue
                    m = ko & (nid == kp[0]) & (pert_codes == kp[1])
                    pred = f_pert_shift[m].mean(0)            # (G,) model KO-vs-NTC LFC in niche
                    lfc_loss = lfc_loss + ((pred - tgt) ** 2).mean()
                    n_terms += 1
                if n_terms > 0:
                    lfc_loss = lfc_loss / n_terms
                if os.environ.get("CONCERT_DBG_FPERT"):
                    _fpv = f_pert_shift.detach().abs().mean().item() if f_pert_shift is not None else float('nan')
                    logging.info("[dbg-fwd] lfc_n_terms=%d lfc_loss=%.4e n_ko=%d mean|fpert_shift|=%.4e",
                                 n_terms, float(lfc_loss.item()), int(ko.sum().item()), _fpv)

        elbo = (
            recon_loss
            + self.beta * (gp_KL + gauss_KL)
            + lord_loss
            + pad_loss
            + self.consistency_lambda * consistency_loss
            + self.basal_consistency_lambda * basal_consistency_loss
            + self.nngp_lambda * nngp_kl
            + self.lfc_lambda * lfc_loss
            + self.disentangle_lambda * disent_loss
            + self.moran_lambda * moran_loss
            + self.gp_field_aux_lambda * gp_aux_loss
            + self.basal_moran_output_lambda * basal_out_moran
        )
        # M2: Bayes-linear basal head KL (analytic, GP_KL-like). Amortized once per epoch
        # via (bsz / N_train) so it is commensurate with the batch-summed gauss/gp KLs.
        if (self.basal_field == "rff" and getattr(self, "basal_bayes", False)
                and self.basal_rff is not None):
            elbo = elbo + self.beta * (bsz / self.N_train) * self.basal_rff.kl()
        # Stage 1 GP guardrail: squared-hinge band prior keeping the SVGP kernel
        # lengthscale inside [ls_prior_min, ls_prior_max]. Data moves freely within
        # the band; only out-of-band excursions are penalized — this stops the
        # multi-slide runaway (L~27-60 vs spacing ~3.3) that made K_mm singular and
        # collapsed the GP. No-op when ls_prior_lambda == 0 (default / eval).
        if self.ls_prior_lambda > 0 and hasattr(self.svgp, "kernel"):
            ksc = getattr(self.svgp.kernel, "scale", None)
            if isinstance(ksc, torch.Tensor) and ksc.requires_grad:
                over = torch.clamp(ksc - self.ls_prior_max, min=0.0) ** 2
                under = torch.clamp(self.ls_prior_min - ksc, min=0.0) ** 2
                ls_prior = (over + under).mean()
                elbo = elbo + self.ls_prior_lambda * ls_prior
        # M1: same band guard on the per-pert GP perturbation lengthscale (gp_pp). Keeps the
        # interpretable footprint radius in [ls_prior_min, ls_prior_max]; stops drift/collapse.
        if (self.ls_prior_lambda > 0 and getattr(self, "fpert_field", None) is not None
                and getattr(self.fpert_field, "field", "") == "gp_pp"
                and getattr(self.fpert_field, "_last_ls", None) is not None):
            ls = self.fpert_field._last_ls
            over = torch.clamp(ls - self.ls_prior_max, min=0.0) ** 2
            under = torch.clamp(self.ls_prior_min - ls, min=0.0) ** 2
            elbo = elbo + self.ls_prior_lambda * (over + under).mean()
        if self.use_paracrine and self.paracrine_l1 > 0 and self.s_signature is not None:
            elbo = elbo + self.paracrine_l1 * self.s_signature.weight.abs().mean()
        if self.use_niche_film:
            # keep modulation depth + beta at 0 unless earned -> homogeneous perts stay at identity,
            # and g_scale stays a clean per-pert heterogeneity readout.
            pe = self.lord_encoder.s_encoder["perturbation"].weight
            gsc = torch.clamp(torch.nn.functional.softplus(self.g_scale_head(pe)), max=self.g_cap)
            elbo = elbo + self.niche_film_l2 * (gsc.mean() + self.beta_out.weight.pow(2).mean())
        if self.use_time_conditioning and self.time_l2 > 0:
            # earned-departure reg on the time×pert residual (same spirit as niche beta L2)
            elbo = elbo + self.time_l2 * self.time_out.weight.pow(2).mean()
        if self.use_timekernel and getattr(self, "time_embed", None) is not None:
            # Baseline anchor readout + optional soft penalty. a(t_ref) -> 0 means day 0
            # (pre-injury) is reconstructed by the basal path alone, which identifies the SIGN
            # of f_pert: with the healthy snapshot pinned to basal, every later day's residual
            # must ADD inflammation instead of the model storing inflammation in basal and
            # subtracting it back. Evaluated at the fixed t_ref (not the batch's times) so it is
            # batch-composition-invariant — a batch of day-9 cells reports the same a(t_ref).
            t_ref = torch.full((1, 1), self.amp_anchor_t, device=self.device, dtype=x.dtype)
            a0 = self.amp_time(self.time_embed(t_ref),
                               ref_emb=self._baseline_time_emb(x))       # (1,)
            if self.amp_anchor_lambda > 0:
                elbo = elbo + self.amp_anchor_lambda * a0.pow(2).mean()
            self._last_a0 = float(a0.detach().mean().item())
        if self.use_membership_field and self._last_pi is not None and is_real_KO is not None:
            # BCE: π must predict the observed inflammation label at the cell's observed time.
            pi = torch.clamp(self._last_pi, 1e-6, 1.0 - 1e-6)
            y = is_real_KO.float().to(pi.device)
            bce = torch.nn.functional.binary_cross_entropy(pi, y)
            elbo = elbo + self.membership_bce_lambda * bce
            self._last_bce = float(bce.detach().item())
            # Optional: reward spatial contiguity of π (Moran's I) -> contiguous inflamed domains.
            if self.membership_moran_lambda > 0:
                moran_pi = _moran_i(pi.unsqueeze(1), x)
                elbo = elbo - self.membership_moran_lambda * moran_pi   # subtract: high Moran is good
        # continuous field MSE — s must track the observed continuous inflammation gene-score.
        if self.use_inflammation_field and self._last_s is not None and infl_score is not None:
            s = self._last_s
            tgt = infl_score.to(s.device).float()
            field_mse = torch.nn.functional.mse_loss(s, tgt)
            elbo = elbo + self.field_mse_lambda * field_mse
            self._last_field_mse = float(field_mse.detach().item())
            # Optional: reward spatial contiguity of s (Moran's I) -> contiguous inflamed domains.
            if self.field_moran_lambda > 0:
                moran_s = _moran_i(s.unsqueeze(1), x)
                elbo = elbo - self.field_moran_lambda * moran_s   # subtract: high Moran is good
        # anchor μ_action to the empirical NTC−unassigned technical shift.
        if (self.use_action_correction and self.mu_action_anchor is not None
                and self.mu_action_anchor_lambda > 0):
            mu_anchor_loss = (
                (torch.clamp(self.mu_action, -5.0, 5.0) - self.mu_action_anchor) ** 2
            ).mean()
            elbo = elbo + self.mu_action_anchor_lambda * mu_anchor_loss
        return (
            elbo,
            recon_loss,
            gp_KL,
            gauss_KL,
            inside_elbo,
            gp_ce,
            gp_p_m,
            gp_p_v,
            q_mu,
            q_var,
            mean_accum,
            disp_accum,
            inside_elbo_recon,
            inside_elbo_kl,
            None,          # latent_samples (omitted)
            pad_loss,      # replaces noise_reg placeholder
            unknown_pen,
            recon_mse,
        )

    @staticmethod
    def _gauss_cross_entropy(m1: torch.Tensor, v1: torch.Tensor, m2: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Cross entropy between diagonal Gaussians N(m1,v1) and N(m2,v2)."""
        return 0.5 * (torch.log(v2) - torch.log(v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0).sum(dim=-1)

    # -------------------------
    # Public batching helpers
    # -------------------------
    @torch.no_grad()
    def batching_latent_samples(
        self, X: np.ndarray, sample_index: torch.Tensor, cell_atts: np.ndarray, batch_size: int = 512,
        Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Y = normalized log-expression (same matrix fed to forward); required when
        # use_expr_encoder is on so the basal latent is amortized, not read from the
        # untrained lookup. Falls back to lookup when Y is None.
        self.eval()
        cuts = self.mask_cutoff
        X_t = torch.tensor(X, dtype=torch.get_default_dtype())
        Y_t = torch.tensor(Y, dtype=torch.get_default_dtype()) if Y is not None else None
        atts_t = torch.tensor(cell_atts, dtype=torch.int)
        idx_t = torch.tensor(sample_index, dtype=torch.int)

        outs: List[torch.Tensor] = []
        N = X_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            xs = X_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            cs = atts_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            is_ = idx_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            cutoff = cuts[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            ys = Y_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device) if Y_t is not None else None

            lord = self.lord_encoder.predict(
                sample_indices=is_, batch_size=xs.shape[0], labels=cs,
                expression=ys if self.use_expr_encoder else None,
            )
            y_ = self._encoder_input_from_lord(lord)
            q_mu, q_var = self.encoder(y_)
            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            if self.deamortize_gp:
                gp_mu = self.gp_mu_free[is_.long()]

            if self.basal_field == "rff":
                # RFF basal: deterministic coord(+tissue) field — mirror forward (:952).
                tissue_emb = lord["tissue"] if self.basal_rff.tissue_dim > 0 else None
                gp_p_m = self.basal_rff(xs[:, :2], tissue_emb=tissue_emb)
            else:
                gp_p_m = []
                for l in range(self.GP_dim):
                    m_l, v_l, _, _ = self.svgp.approximate_posterior_params(xs, xs, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                    gp_p_m.append(m_l)
                gp_p_m = torch.stack(gp_p_m, dim=1)
            if self.use_nngp and self.nngp is not None and self.graph is not None:
                gidx = is_.long()
                knn_idx_b  = self.graph.knn_idx[gidx]
                knn_dist_b = self.graph.knn_dist[gidx]
                neighbor_vals = self.gp_cache[knn_idx_b]
                local_mean, _ = self.nngp(knn_dist_b, neighbor_vals)
                gp_p_m = gp_p_m + local_mean
            p_m = torch.cat([gp_p_m, q_mu[:, self.GP_dim :]], dim=1)
            outs.append(p_m.cpu())
        return torch.cat(outs, dim=0).numpy()

    @torch.no_grad()
    def batching_denoise_counts(
        self,
        X: np.ndarray,
        sample_index: torch.Tensor,
        cell_atts: np.ndarray,
        n_samples: int = 1,
        batch_size: int = 512,
        has_guide: Optional[np.ndarray] = None,
        is_real_KO: Optional[np.ndarray] = None,
        force_basal: bool = False,
        apply_paracrine: bool = True,
        Y: Optional[np.ndarray] = None,
        time_t: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Denoised mean expression.

        time_t (N,) — normalized day t∈[0,1]; when use_time_conditioning is on, the
        f_pert shift is decoded at this per-cell time. Pass a DIFFERENT t than a cell's
        observed day to impute its expression at an unobserved timepoint (LOTO / virtual
        longitudinal reconstruction).

        When force_basal=True, returns mu_basal only (no μ_action / f_pert
        shifts) — i.e., the cell's "what would it look like unperturbed" mean.
        Otherwise applies μ_action (if has_guide given) and f_pert (if
        is_real_KO given and module enabled) on top of mu_basal.
        """
        self.eval()
        cuts = self.mask_cutoff
        X_t = torch.tensor(X, dtype=torch.get_default_dtype())
        atts_t = torch.tensor(cell_atts, dtype=torch.int)
        idx_t = torch.tensor(sample_index, dtype=torch.int)
        hg_t = torch.tensor(has_guide,  dtype=torch.float32) if has_guide  is not None else None
        rk_t = torch.tensor(is_real_KO, dtype=torch.float32) if is_real_KO is not None else None
        Y_t = torch.tensor(Y, dtype=torch.get_default_dtype()) if Y is not None else None
        tt_t = (torch.tensor(time_t, dtype=torch.float32)
                if (self.use_time_conditioning and time_t is not None) else None)

        means: List[torch.Tensor] = []
        vars_: List[torch.Tensor] = []

        N = X_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            lo, hi = b * batch_size, min((b + 1) * batch_size, N)
            xs = X_t[lo:hi].to(self.device)
            cs = atts_t[lo:hi].to(self.device)
            is_ = idx_t[lo:hi].to(self.device)
            cutoff = cuts[lo:hi].to(self.device)
            hg_b = hg_t[lo:hi].to(self.device) if hg_t is not None else None
            rk_b = rk_t[lo:hi].to(self.device) if rk_t is not None else None
            ys = Y_t[lo:hi].to(self.device) if Y_t is not None else None
            time_emb_b = (self.time_embed(tt_t[lo:hi].to(self.device))
                          if tt_t is not None else None)

            lord = self.lord_encoder.predict(
                sample_indices=is_, batch_size=xs.shape[0], labels=cs,
                expression=ys if self.use_expr_encoder else None,
            )
            y_ = self._encoder_input_from_lord(lord)
            q_mu, q_var = self.encoder(y_)

            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]
            if self.deamortize_gp:
                gp_mu = self.gp_mu_free[is_.long()]

            if self.basal_field == "rff":
                # RFF basal: deterministic coord(+tissue) field — mirror forward (:952).
                tissue_emb = lord["tissue"] if self.basal_rff.tissue_dim > 0 else None
                gp_p_m = self.basal_rff(xs[:, :2], tissue_emb=tissue_emb)
                gp_p_v = torch.exp(self.basal_rff_logvar).expand_as(gp_p_m)
            else:
                gp_p_m, gp_p_v = [], []
                for l in range(self.GP_dim):
                    m_l, v_l, _, _ = self.svgp.approximate_posterior_params(xs, xs, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                    gp_p_m.append(m_l)
                    gp_p_v.append(v_l)
                gp_p_m = torch.stack(gp_p_m, dim=1)
                gp_p_v = torch.stack(gp_p_v, dim=1)
            if self.use_nngp and self.nngp is not None and self.graph is not None:
                gidx = is_.long()
                knn_idx_b  = self.graph.knn_idx[gidx]
                knn_dist_b = self.graph.knn_dist[gidx]
                neighbor_vals = self.gp_cache[knn_idx_b]
                local_mean, _ = self.nngp(knn_dist_b, neighbor_vals)
                gp_p_m = gp_p_m + local_mean

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.clamp(torch.cat([gp_p_v, gs_var], dim=1), min=1e-6)
            dist = Normal(p_m, torch.sqrt(p_v))

            # Pre-compute per-batch shift (independent of z samples)
            shift = None
            if not force_basal:
                if self.use_action_correction and hg_b is not None:
                    mu_action_clamped = torch.clamp(self.mu_action, -5.0, 5.0)
                    shift = hg_b.unsqueeze(1) * mu_action_clamped.unsqueeze(0)
                niche_lat = gp_p_m if self.use_niche_interaction else None
                gate_state = (torch.cat([gp_p_m, gs_mu], dim=-1).detach()
                          if (self.use_pert_gate or self.use_membership_field) else None)
                fp = self._pert_log_shift_from_lord(lord, rk_b, niche=niche_lat, gate_state=gate_state,
                                                    coords=xs[:, :2], time_emb=time_emb_b)
                if fp is not None:
                    shift = fp if shift is None else (shift + fp)
                if apply_paracrine and self.use_paracrine and self.s_signature is not None and self.graph is not None:
                    pert_emb_table = self.lord_encoder.s_encoder["perturbation"].weight
                    para = paracrine_log_shift(self.graph, is_.long(), self.paracrine_radius,
                                               self.s_signature, pert_emb_table, clamp=self.f_pert_clamp)
                    shift = para if shift is None else (shift + para)

            samples = [dist.sample() for _ in range(max(1, n_samples))]
            mu_stack = []
            for z in samples:
                mu_basal = self._decode_to_mean(z)
                mu = mu_basal if shift is None else mu_basal * torch.exp(shift)
                mu_stack.append(mu)
            mu_mean = torch.stack(mu_stack, dim=0).mean(dim=0)
            means.append(mu_mean.cpu())
            vars_.append(torch.sqrt(p_v).cpu())

        return torch.cat(means, dim=0).numpy(), torch.cat(vars_, dim=0).numpy()

    @torch.no_grad()
    def batching_recon_samples(self, Z: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.eval()
        Z_t = torch.tensor(Z, dtype=torch.get_default_dtype())
        outs: List[torch.Tensor] = []
        N = Z_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            z = Z_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            outs.append(self._decode_to_mean(z).cpu())
        return torch.cat(outs, dim=0).numpy()

    # -------------------------
    # Imputation / counterfactuals
    # -------------------------
    @torch.no_grad()
    def imputation(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
        Y_sample_index: torch.Tensor,
        Y_cell_atts: np.ndarray,
        n_samples: int = 1,
        batch_size: int = 512,
        Y_train: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Impute latents & counts on unseen locations by nearest-neighbor Gaussian part + GP posterior.

        Y_train = normalized log-expression of the TRAIN cells (same matrix fed to forward);
        required when use_expr_encoder is on so train latents are amortized via the basal
        encoder rather than read from the untrained per-cell lookup. Falls back when None.
        """
        self.eval()

        def nearest_idx(array: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return torch.argmin(torch.sum((array - value) ** 2, dim=1))

        cuts = self.mask_cutoff.to(self.device)
        X_te = torch.tensor(X_test, dtype=torch.get_default_dtype()).to(self.device)
        X_tr = torch.tensor(X_train, dtype=torch.get_default_dtype()).to(self.device)
        pos_te, sample_te = X_te[:, :2], X_te[:, 2:]
        pos_tr, sample_tr = X_tr[:, :2], X_tr[:, 2:]

        atts_tr = torch.tensor(Y_cell_atts, dtype=torch.int).to(self.device)
        idx_tr = torch.tensor(Y_sample_index, dtype=torch.int).to(self.device)
        Ytr_t = torch.tensor(Y_train, dtype=torch.get_default_dtype()) if Y_train is not None else None

        # Precompute q_mu/q_var on train
        q_mu_all, q_var_all = [], []
        N_tr = X_tr.shape[0]
        n_tr_batches = int(math.ceil(N_tr / batch_size))
        for b in range(n_tr_batches):
            cs = atts_tr[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device)
            is_ = idx_tr[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device)
            ys = Ytr_t[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device) if Ytr_t is not None else None
            lord = self.lord_encoder.predict(
                sample_indices=is_, batch_size=cs.shape[0], labels=cs,
                expression=ys if self.use_expr_encoder else None,
            )
            y_ = self._encoder_input_from_lord(lord)
            m, v = self.encoder(y_)
            q_mu_all.append(m)
            q_var_all.append(v)
        q_mu = torch.cat(q_mu_all, dim=0).to(self.device)
        q_var = torch.cat(q_var_all, dim=0).to(self.device)

        # Match each test to nearest train for Gaussian block; GP block via SVGP impute
        latents_out, means_out = [], []
        N_te = X_te.shape[0]
        n_te_batches = int(math.ceil(N_te / batch_size))
        x_train_sel = []
        for e in range(N_te):
            x_train_sel.append(nearest_idx(pos_tr, pos_te[e]))
        x_train_sel = torch.stack(x_train_sel)
        te_cutoff = cuts[x_train_sel.long()]
        te_sample = sample_tr[x_train_sel.long()]

        X_te = torch.cat([X_te, te_sample], dim=1)

        gp_mu_tr, gp_var_tr = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]

        for b in range(n_te_batches):
            x_te_b = X_te[b * batch_size : min((b + 1) * batch_size, N_te)].to(self.device)
            cut_te_b = te_cutoff[b * batch_size : min((b + 1) * batch_size, N_te)].to(self.device)
            cut_te_b = cut_te_b.view(-1)
            # Gaussian block (copy from nearest train point)
            sel_b = x_train_sel[b * batch_size : min((b + 1) * batch_size, N_te)].long()
            gs_mu = q_mu[sel_b, self.GP_dim :]
            gs_var = q_var[sel_b, self.GP_dim :]

            # GP block (impute)
            if self.basal_field == "rff":
                # RFF basal at the TEST coordinate (tissue_emb=None: spac has 1 tissue class;
                # impute path is not used in the P0 gate). Mirrors forward (:952).
                gp_p_m = self.basal_rff(x_te_b[:, :2], tissue_emb=None)
                gp_p_v = torch.exp(self.basal_rff_logvar).expand_as(gp_p_m)
            else:
                gp_p_m, gp_p_v = [], []
                for l in range(self.GP_dim):
                    assert x_te_b.shape[1] == X_tr.shape[1]
                    m_l, v_l, _, _ = self.svgp.approximate_posterior_params_impute(
                        index_points_test=x_te_b,
                        index_points_train=X_tr,
                        y=gp_mu_tr[:, l],
                        noise=gp_var_tr[:, l],
                        cutoff_test=cut_te_b,
                        cutoff_train=cuts.to(self.device),
                    )
                    gp_p_m.append(m_l)
                    gp_p_v.append(v_l)
                gp_p_m = torch.stack(gp_p_m, dim=1)
                gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.clamp(torch.cat([gp_p_v, gs_var], dim=1), min=1e-6)
            latents_out.append(p_m.cpu())

            dist = Normal(p_m, torch.sqrt(p_v))
            zs = [dist.sample() for _ in range(max(1, n_samples))]
            mu_stack = []
            for z in zs:
                mu_stack.append(self._decode_to_mean(z))
            means_out.append(torch.stack(mu_stack, dim=0).mean(dim=0).cpu())

        return torch.cat(latents_out, dim=0).numpy(), torch.cat(means_out, dim=0).numpy()

    @torch.no_grad()
    def counterfactualPrediction(
        self,
        *,
        X: np.ndarray,
        sample_index: torch.Tensor,
        cell_atts: np.ndarray,
        perturb_cell_id: torch.Tensor | np.ndarray | List[int] | None = None,
        target_cell_tissue: int | None = None,
        target_cell_perturbation: int | None = None,
        target_has_guide: bool | None = None,
        target_is_real_KO: bool | None = None,
        n_samples: int = 1,
        batch_size: int = 512,
        Y: Optional[np.ndarray] = None,
        time_t: Optional[np.ndarray] = None,
        tier1_id: Optional[np.ndarray] = None,
        arch_desc: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Counterfactual prediction by overwriting selected cells' attributes and decoding.

        time_t (N,) — per-cell normalized day t∈[0,1] at which to decode the f_pert shift
        (temporal case study). Set to a target day t' to evolve each cell across the DSS
        course (virtual longitudinal trajectory); when None, time conditioning is inert.

        Y = normalized log-expression of the cells in X (same matrix fed to forward);
        required when use_expr_encoder is on so the basal latent is amortized from the
        cell's own expression (enables held-out-cell and seen-cell pert-swap scenarios)
        rather than read from the untrained per-cell lookup. Falls back when None.

        tier1_id (N,) / arch_desc (N, arch_dim) — per-cell Tier1 lineage code and the
        STANDARDISED day-invariant architectural descriptor, aligned to X. Required when
        use_shared_basal is on: the basal (healthy) mean comes from the SHARED
        (Tier1, arch_desc) path (_shared_basal_mean) and the FiLM niche descriptor from
        W_arch_field(arch_desc) — never from the cell's own gs_mu/expression, guaranteeing
        a leakage-free re-decode at the target time. arch_desc MUST be standardised with the
        TRAINING scaler (arch_scaler.npz in the checkpoint dir); this method does not and
        must not re-derive scaler statistics from the eval subset.

        target_has_guide / target_is_real_KO: explicit overrides for whether the target
        perturbation carries the action (μ_action) and the KO footprint (f_pert). When
        None (default) the legacy heuristic ``target_pert > 0`` is used, which reserves
        code 0 as a basal/no-effect slot. Pass these when code 0 is itself a real KO
        (e.g. perturbmap Jak2=0) so the footprint can actually be requested.
        """
        self.eval()

        cuts = self.mask_cutoff
        X_t = torch.tensor(X, dtype=torch.get_default_dtype())
        idx_t = torch.tensor(sample_index, dtype=torch.int)
        Y_t = torch.tensor(Y, dtype=torch.get_default_dtype()) if Y is not None else None

        # Copy & apply targets
        pert_atts = np.array(cell_atts, copy=True)
        if perturb_cell_id is not None:
            if isinstance(perturb_cell_id, torch.Tensor):
                ids = perturb_cell_id.long().cpu().numpy().tolist()
            elif isinstance(perturb_cell_id, np.ndarray):
                ids = perturb_cell_id.astype(int).tolist()
            else:
                ids = list(map(int, perturb_cell_id))
            for i in ids:
                if target_cell_tissue is not None:
                    pert_atts[i, 0] = int(target_cell_tissue)
                if target_cell_perturbation is not None:
                    pert_atts[i, 1] = int(target_cell_perturbation)
        atts_t = torch.tensor(pert_atts, dtype=torch.int)

        # Per-cell guide / real-KO flags.
        #
        # The target semantics apply only to the cells named in perturb_cell_id. Cells that
        # were not edited keep the flags implied by their own perturbation code, exactly as
        # in forward(). Setting them batch-wide is correct only when every row passed in is
        # a row being edited; it makes the niche scenario -- edit one group of cells, read
        # an unedited group -- inexpressible, because the unedited cells would have their
        # own perturbation effect switched on and appear to respond to the edit.
        #
        # When perturb_cell_id names every row this is identical to the batch-wide form; when
        # it is None the batch-wide form is kept, so existing callers are unaffected.
        _target_pert_g = int(target_cell_perturbation) if target_cell_perturbation is not None else 0
        _per_cell_flags = perturb_cell_id is not None
        if _per_cell_flags:
            _edited = np.zeros(pert_atts.shape[0], dtype=bool)
            _edited[np.asarray(ids, dtype=int)] = True
            _own = (pert_atts[:, 1] > 0).astype(np.float32)
            _tgt_hg = float(bool(target_has_guide)) if target_has_guide is not None \
                else float(_target_pert_g > 0)
            _tgt_rk = float(bool(target_is_real_KO)) if target_is_real_KO is not None \
                else float(_target_pert_g > 0)
            _hg_all = torch.tensor(np.where(_edited, _tgt_hg, _own), dtype=torch.get_default_dtype())
            _rk_all = torch.tensor(np.where(_edited, _tgt_rk, _own), dtype=torch.get_default_dtype())

        tt_t = (torch.tensor(time_t, dtype=torch.float32)
                if (self.use_time_conditioning and time_t is not None) else None)

        means_out: List[torch.Tensor] = []
        N = X_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            xs = X_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            cs = atts_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            is_ = idx_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            cutoff = cuts[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            ys = Y_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device) if Y_t is not None else None
            time_emb_b = (self.time_embed(tt_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device))
                          if tt_t is not None else None)

            lord = self.lord_encoder.predict(
                sample_indices=is_, batch_size=xs.shape[0], labels=cs,
                expression=ys if self.use_expr_encoder else None,
            )
            y_ = self._encoder_input_from_lord(lord)

            # PAD: inject clean (is_perturbed=0) action embedding for all predict cells
            # Projects predictions into the unperturbed-action space (removes batch noise).
            if self.use_pad and self.pad is not None:
                s_action = self.pad.clean_action_embedding(xs.shape[0], self.device)
                y_ = torch.cat([y_, s_action], dim=-1)

            q_mu, q_var = self.encoder(y_)

            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]
            if self.deamortize_gp:
                gp_mu = self.gp_mu_free[is_.long()]

            if self.basal_field == "rff":
                # RFF basal: deterministic coord(+tissue) field — mirror forward (:952).
                tissue_emb = lord["tissue"] if self.basal_rff.tissue_dim > 0 else None
                gp_p_m = self.basal_rff(xs[:, :2], tissue_emb=tissue_emb)
                gp_p_v = torch.exp(self.basal_rff_logvar).expand_as(gp_p_m)
            else:
                gp_p_m, gp_p_v = [], []
                for l in range(self.GP_dim):
                    m_l, v_l, _, _ = self.svgp.approximate_posterior_params(xs, xs, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                    gp_p_m.append(m_l)
                    gp_p_v.append(v_l)
                gp_p_m = torch.stack(gp_p_m, dim=1)
                gp_p_v = torch.stack(gp_p_v, dim=1)
            if self.use_nngp and self.nngp is not None and self.graph is not None:
                gidx = is_.long()
                knn_idx_b  = self.graph.knn_idx[gidx]
                knn_dist_b = self.graph.knn_dist[gidx]
                neighbor_vals = self.gp_cache[knn_idx_b]
                local_mean, _ = self.nngp(knn_dist_b, neighbor_vals)
                gp_p_m = gp_p_m + local_mean

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.clamp(torch.cat([gp_p_v, gs_var], dim=1), min=1e-6)
            dist = Normal(p_m, torch.sqrt(p_v))

            # Apply gene-space additive shifts using the TARGET perturbation
            # (μ_action is global; f_pert uses LORD's pert_emb for the swapped
            # pert index — this is the clean counterfactual mechanism).
            # All target cells get has_guide=1 if target_cell_perturbation>0,
            # is_real_KO=1 if the target represents a real KO (here we treat
            # any non-zero target as a real KO; caller can override).
            B = xs.shape[0]
            target_pert = int(target_cell_perturbation) if target_cell_perturbation is not None else 0
            if _per_cell_flags:
                _lo = b * batch_size
                hg_b = _hg_all[_lo : _lo + xs.shape[0]].to(self.device)
                rk_b = _rk_all[_lo : _lo + xs.shape[0]].to(self.device)
            else:
                hg_b = torch.ones(B, device=self.device) if target_pert > 0 else torch.zeros(B, device=self.device)
                rk_b = hg_b  # treat target pert as a real KO unless target_pert==0
                # explicit overrides (needed when code 0 is itself a real KO, e.g. perturbmap)
                if target_has_guide is not None:
                    hg_b = torch.full((B,), float(bool(target_has_guide)), device=self.device)
                if target_is_real_KO is not None:
                    rk_b = torch.full((B,), float(bool(target_is_real_KO)), device=self.device)
            shift = None
            if self.use_action_correction:
                mu_action_clamped = torch.clamp(self.mu_action, -5.0, 5.0)
                shift = hg_b.unsqueeze(1) * mu_action_clamped.unsqueeze(0)
            niche_lat = gp_p_m if self.use_niche_interaction else None
            gate_state = (torch.cat([gp_p_m, gs_mu], dim=-1).detach()
                          if (self.use_pert_gate or self.use_membership_field) else None)
            # Diagnostic: _disable_fpert_field passes coords=None so _pert_log_shift_from_lord
            # skips ONLY the coord-conditioned FpertField residual (core f_pert / niche / gate
            # unchanged) — isolates whether the spatial field corrupts unseen/multimodal perts.
            _cf_coords = None if getattr(self, "_disable_fpert_field", False) else xs[:, :2]
            if self.use_shared_basal:
                adb = torch.as_tensor(
                    arch_desc[b * batch_size : min((b + 1) * batch_size, N)],
                    device=self.device, dtype=torch.get_default_dtype())
                z_niche_cf = self.W_arch_field(adb) if self.use_niche_film else None
            else:
                # z_niche keyed to each predicted cell's OWN neighborhood -> per-cell variation
                z_niche_cf = (self.niche_descriptor(gs_mu.detach(), xs[:, :2])
                              if self.use_niche_film else None)
            fp = self._pert_log_shift_from_lord(lord, rk_b, niche=niche_lat, gate_state=gate_state,
                                                coords=_cf_coords, z_niche=z_niche_cf, time_emb=time_emb_b)
            if fp is not None:
                shift = fp if shift is None else (shift + fp)
            if self.use_paracrine and self.s_signature is not None and self.graph is not None:
                pert_emb_table = self.lord_encoder.s_encoder["perturbation"].weight
                para = paracrine_log_shift(self.graph, is_.long(), self.paracrine_radius,
                                           self.s_signature, pert_emb_table, clamp=self.f_pert_clamp)
                shift = para if shift is None else (shift + para)

            zs = [dist.sample() for _ in range(max(1, n_samples))]
            # Shared-basal mode: basal (healthy) mean depends ONLY on (Tier1, arch_desc)
            # — no per-cell latent z — so it is identical across samples and cannot carry
            # the cell's own inflammation (leakage-free re-decode, Task 7). It is also
            # time-invariant, so re-decoding at a new t moves ONLY a(t)·Φ(ℓ(t))·f_pert.
            if self.use_shared_basal:
                t1b = torch.as_tensor(
                    tier1_id[b * batch_size : min((b + 1) * batch_size, N)],
                    device=self.device).long()
                adb = torch.as_tensor(
                    arch_desc[b * batch_size : min((b + 1) * batch_size, N)],
                    device=self.device, dtype=torch.get_default_dtype())
                mu_basal_shared = self._shared_basal_mean(t1b, adb)
            mu_stack = []
            for z in zs:
                if self.use_shared_basal:
                    mu_basal = mu_basal_shared
                else:
                    mu_basal = self._decode_to_mean(z)
                mu = mu_basal if shift is None else mu_basal * torch.exp(shift)
                mu_stack.append(mu)
            means_out.append(torch.stack(mu_stack, dim=0).mean(dim=0).cpu())

        return torch.cat(means_out, dim=0).numpy(), pert_atts

    @torch.no_grad()
    def predict_membership(
        self, *, X: np.ndarray, sample_index: torch.Tensor, cell_atts: np.ndarray,
        Y: np.ndarray, time_t: np.ndarray, batch_size: int = 512,
    ) -> np.ndarray:
        """Return π ∈ [0,1] per cell at the given normalized day(s) time_t.

        X = pos_batched (coords + slide one-hot); Y = normalized expression (expr encoder);
        time_t = per-cell normalized day at which to evaluate inflammation membership.
        Mirrors the counterfactual decode path but returns only the membership field.
        """
        assert self.use_membership_field and self.membership_field is not None
        self.eval()
        X_t = torch.tensor(X, dtype=torch.get_default_dtype())
        Y_t = torch.tensor(Y, dtype=torch.get_default_dtype())
        idx_t = torch.tensor(sample_index, dtype=torch.int)
        atts_t = torch.tensor(cell_atts, dtype=torch.int)
        tt_t = torch.tensor(time_t, dtype=torch.float32)
        out: List[torch.Tensor] = []
        N = X_t.shape[0]
        for b in range(int(math.ceil(N / batch_size))):
            sl = slice(b * batch_size, min((b + 1) * batch_size, N))
            xs = X_t[sl].to(self.device); ys = Y_t[sl].to(self.device)
            is_ = idx_t[sl].to(self.device); cs = atts_t[sl].to(self.device)
            time_emb_b = self.time_embed(tt_t[sl].to(self.device))
            lord = self.lord_encoder.predict(
                sample_indices=is_, batch_size=xs.shape[0], labels=cs,
                expression=ys if self.use_expr_encoder else None)
            y_ = self._encoder_input_from_lord(lord)
            if self.use_pad and self.pad is not None:
                y_ = torch.cat([y_, self.pad.clean_action_embedding(xs.shape[0], self.device)], dim=-1)
            q_mu, q_var = self.encoder(y_)
            gp_mu = q_mu[:, : self.GP_dim]; gs_mu = q_mu[:, self.GP_dim :]
            if self.basal_field == "rff":
                tissue_emb = lord["tissue"] if self.basal_rff.tissue_dim > 0 else None
                gp_p_m = self.basal_rff(xs[:, :2], tissue_emb=tissue_emb)
            else:
                gp_p_m = gp_mu   # fallback; membership uses it only as a conditioning summary
            gate_state = torch.cat([gp_p_m, gs_mu], dim=-1)
            z_niche = self.niche_descriptor(gs_mu, xs[:, :2])
            feats = torch.cat([gate_state, z_niche, xs[:, :2], time_emb_b], dim=-1)
            out.append(self.membership_field(feats).cpu())
        return torch.cat(out, dim=0).numpy()

    @torch.no_grad()
    def impute_and_counterfactual_fun1(
  	    self,
        target: int,
        tissue: int,
        X_test: np.ndarray,
        X_train: np.ndarray,
        Y_sample_index: np.ndarray,
        Y_cell_atts: np.ndarray,
        n_samples: int = 1,
        batch_size: int = 512,
        *,
        knn_k: int = 10,
        knn_sample: int = 5,
        Y_train: Optional[np.ndarray] = None,
	) -> Tuple[np.ndarray, np.ndarray]:
        """
        Impute latents & denoised counts at unseen coords. For the Gaussian block,
        pull neighbors *only from the specified perturbation* in the training set.

        Y_train = normalized log-expression of the TRAIN cells; required when
        use_expr_encoder is on so train latents are amortized via the basal encoder
        rather than the untrained per-cell lookup. Falls back when None.

        NOTE: NNGP local field and paracrine spillover are skipped for these
        unseen coordinates (no precomputed graph / KO-neighbor entries exist for
        them) — a deferred extension. This method is pure imputation (no
        cell-autonomous perturbation shift), so f_pert/niche do not apply here.
        """
        self.eval()
        X_te = torch.tensor(X_test, dtype=torch.get_default_dtype()).to(self.device)
        X_tr = torch.tensor(X_train, dtype=torch.get_default_dtype()).to(self.device)
        atts_tr = torch.tensor(Y_cell_atts, dtype=torch.int).to(self.device)
        idx_tr = torch.tensor(Y_sample_index, dtype=torch.int).to(self.device)
        cut = self.mask_cutoff

        N_tr = X_tr.shape[0]
        N_te = X_te.shape[0]
        n_tr_chunks = int(math.ceil(N_tr / batch_size))
        n_te_chunks = int(math.ceil(N_te / batch_size))

        target_mask = (Y_cell_atts[:, 1].astype(int) == int(target))
        X_tr_target_np = X_train[target_mask]
        X_tr_target = torch.tensor(X_tr_target_np, dtype=torch.get_default_dtype()).to(self.device)

        pos_te, sample_te = X_te[:, :2], X_te[:, 2:]
        pos_tr, sample_tr = X_tr[:, :2], X_tr[:, 2:]
        pos_tr_target, sample_tr_target = X_tr_target[:, :2], X_tr_target[:, 2:]

        def _nearest_idx(array: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return torch.argmin(torch.sum((array - value) ** 2, dim=1))

        nn_train = [_nearest_idx(pos_tr_target, pos_te[i]) for i in range(N_te)]
        nn_train = torch.stack(nn_train)
        test_cutoff_nn = cut[nn_train.long()]

        q_mu_list, q_var_list = [], []
        for b in range(n_tr_chunks):
            lo = b * batch_size
            hi = min((b + 1) * batch_size, N_tr)
            ca = atts_tr[lo:hi].to(self.device, dtype=torch.int)
            si = idx_tr[lo:hi].to(self.device, dtype=torch.int)
            ys = (torch.tensor(Y_train[lo:hi], dtype=torch.get_default_dtype()).to(self.device)
                  if Y_train is not None else None)
            lord = self.lord_encoder.predict(
                sample_indices=si, batch_size=hi - lo, labels=ca,
                expression=ys if self.use_expr_encoder else None,
            )
            y_ = self._encoder_input_from_lord(lord)
            m, v = self.encoder(y_)
            q_mu_list.append(m)
            q_var_list.append(v)
        q_mu = torch.cat(q_mu_list, dim=0)
        q_var = torch.cat(q_var_list, dim=0)

        gp_mu_tr = q_mu[:, : self.GP_dim]
        gp_var_tr = q_var[:, : self.GP_dim]

        def _knn_subset(array: torch.Tensor, value: torch.Tensor, k: int, sample_k: int) -> torch.Tensor:
            d = torch.sum((array - value) ** 2, dim=1)
            k = int(min(k, array.shape[0]))
            sample_k = int(min(sample_k, k))
            topk = torch.topk(d, k, largest=False).indices
            if sample_k < k:
                perm = torch.randperm(k, device=topk.device)[:sample_k]
                return topk[perm]
            return topk

        latents_out, means_out = [], []
        for b in range(n_te_chunks):
            lo = b * batch_size
            hi = min((b + 1) * batch_size, N_te)
            x_te_b = X_te[lo:hi].to(self.device)
            cutoff_test_b = test_cutoff_nn[lo:hi].to(self.device, dtype=torch.get_default_dtype())

            g_mu_rows, g_var_rows = [], []
            for i in range(x_te_b.shape[0]):
                if X_tr_target.shape[0] == 0:
                    idx = _nearest_idx(X_tr, x_te_b[i]).item()
                    g_mu_rows.append(q_mu[idx, self.GP_dim :])
                    g_var_rows.append(q_var[idx, self.GP_dim :])
                else:
                    inds = _knn_subset(X_tr_target, x_te_b[i], k=knn_k, sample_k=knn_sample)
                    full_inds = torch.nonzero(torch.tensor(target_mask, device=self.device), as_tuple=False).squeeze(1)[inds]
                    g_mu_rows.append(q_mu[full_inds, self.GP_dim :].mean(dim=0))
                    g_var_rows.append(q_var[full_inds, self.GP_dim :].mean(dim=0))
            g_mu = torch.stack(g_mu_rows, dim=0)
            g_var = torch.stack(g_var_rows, dim=0)

            if self.basal_field == "rff":
                # RFF basal at the TEST coordinate (tissue_emb=None; impute path not in P0 gate).
                gp_p_m = self.basal_rff(x_te_b[:, :2], tissue_emb=None)
                gp_p_v = torch.exp(self.basal_rff_logvar).expand_as(gp_p_m)
            else:
                gp_p_m_list, gp_p_v_list = [], []
                for l in range(self.GP_dim):
                    m_l, v_l, _, _ = self.svgp.approximate_posterior_params_impute(
                        index_points_test=x_te_b,
                        index_points_train=X_tr,
                        y=gp_mu_tr[:, l],
                        noise=gp_var_tr[:, l],
                        cutoff_test=cutoff_test_b,
                        cutoff_train=cut.to(self.device),
                    )
                    gp_p_m_list.append(m_l)
                    gp_p_v_list.append(v_l)
                gp_p_m = torch.stack(gp_p_m_list, dim=1)
                gp_p_v = torch.stack(gp_p_v_list, dim=1)

            p_m = torch.cat([gp_p_m, g_mu], dim=1)
            latents_out.append(p_m.detach().cpu())
            p_v = torch.clamp(torch.cat([gp_p_v, g_var], dim=1), min=1e-6)

            dist = Normal(p_m, torch.sqrt(p_v))
            mu_stack = []
            for _ in range(max(1, n_samples)):
                z = dist.sample()
                mu_stack.append(self._decode_to_mean(z))
            means_out.append(torch.stack(mu_stack, dim=0).mean(dim=0).detach().cpu())

        return torch.cat(latents_out, dim=0).numpy(), torch.cat(means_out, dim=0).numpy()

    @torch.no_grad()
    def impute_and_counterfactual_fun2(
        self,
        target: int,
        tissue: int,
        X_test: np.ndarray,
        X_train: np.ndarray,
        Y_sample_index: np.ndarray,
        Y_cell_atts: np.ndarray,
        n_samples: int = 1,
        batch_size: int = 512,
        Y_train: Optional[np.ndarray] = None,
	) -> Tuple[np.ndarray, np.ndarray]:
        """
        Counterfactual impute:
        • GP block: SVGP posterior from training GP latents.
        • Gaussian block: replace tissue/perturbation with specified targets.

        Y_train = normalized log-expression of the TRAIN cells; required when
        use_expr_encoder is on so train latents are amortized via the basal encoder
        rather than the untrained per-cell lookup. Falls back when None.
        """
        self.eval()
        X_te = torch.tensor(X_test, dtype=torch.get_default_dtype()).to(self.device)
        X_tr = torch.tensor(X_train, dtype=torch.get_default_dtype()).to(self.device)
        atts_tr = torch.tensor(Y_cell_atts, dtype=torch.int).to(self.device)
        idx_tr = torch.tensor(Y_sample_index, dtype=torch.int).to(self.device)
        cut = self.mask_cutoff

        N_tr = X_tr.shape[0]
        N_te = X_te.shape[0]
        n_tr_chunks = int(math.ceil(N_tr / batch_size))
        n_te_chunks = int(math.ceil(N_te / batch_size))

        target_mask = (Y_cell_atts[:, 1].astype(int) == int(target))
        X_tr_target_np = X_train[target_mask]
        X_tr_target = torch.tensor(X_tr_target_np, dtype=torch.get_default_dtype()).to(self.device)

        pos_te, sample_te = X_te[:, :2], X_te[:, 2:]
        pos_tr, sample_tr = X_tr[:, :2], X_tr[:, 2:]
        pos_tr_target, sample_tr_target = X_tr_target[:, :2], X_tr_target[:, 2:]

        def _nearest_idx(array: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return torch.argmin(torch.sum((array - value) ** 2, dim=1))

        nn_train = [_nearest_idx(pos_tr_target, pos_te[i]) for i in range(N_te)]
        nn_train = torch.stack(nn_train)
        test_cutoff_nn = cut[nn_train.long()]

        basal_list, q_mu_list, q_var_list = [], [], []
        for b in range(n_tr_chunks):
            lo = b * batch_size
            hi = min((b + 1) * batch_size, N_tr)
            ca = atts_tr[lo:hi].to(self.device, dtype=torch.int)
            si = idx_tr[lo:hi].to(self.device, dtype=torch.int)
            ys = (torch.tensor(Y_train[lo:hi], dtype=torch.get_default_dtype()).to(self.device)
                  if Y_train is not None else None)
            lord = self.lord_encoder.predict(
                sample_indices=si, batch_size=hi - lo, labels=ca,
                expression=ys if self.use_expr_encoder else None,
            )
            basal_list.append(lord["basal_latent"])
            y_ = self._encoder_input_from_lord(lord)
            m, v = self.encoder(y_)
            q_mu_list.append(m)
            q_var_list.append(v)

        basal_tr = torch.cat(basal_list, dim=0)
        q_mu = torch.cat(q_mu_list, dim=0)
        q_var = torch.cat(q_var_list, dim=0)

        latents_out, means_out = [], []
        gp_mu_tr = q_mu[:, : self.GP_dim]
        gp_var_tr = q_var[:, : self.GP_dim]

        for b in range(n_te_chunks):
            lo = b * batch_size
            hi = min((b + 1) * batch_size, N_te)
            x_te_b = X_te[lo:hi].to(self.device)
            idx_te_b = nn_train[lo:hi].long()
            test_cut_b = test_cutoff_nn[lo:hi].to(self.device, dtype=torch.get_default_dtype())

            basal_cf = basal_tr[idx_te_b]
            B = basal_cf.shape[0]
            l_tissue = self.lord_encoder.get_latent(att=torch.tensor(tissue, device=self.device), type_="tissue", batch_size=B)
            l_pert = self.lord_encoder.get_latent(att=torch.tensor(target, device=self.device), type_="perturbation", batch_size=B)
            # In disentangled mode, the encoder is fed [basal, tissue] only;
            # the target pert embedding is applied via f_pert in mean space.
            if self.use_basal_encoder_only:
                y_cf = torch.cat([basal_cf, l_tissue], dim=1)
            else:
                y_cf = torch.cat([basal_cf, l_tissue, l_pert], dim=1)

            g_mu_all, g_var_all = self.encoder(y_cf)
            g_mu = g_mu_all[:, self.GP_dim :]
            g_var = g_var_all[:, self.GP_dim :]

            if self.basal_field == "rff":
                # RFF basal at the TEST coordinate (tissue_emb=None; impute path not in P0 gate).
                gp_p_m = self.basal_rff(x_te_b[:, :2], tissue_emb=None)
                gp_p_v = torch.exp(self.basal_rff_logvar).expand_as(gp_p_m)
            else:
                gp_p_m_list, gp_p_v_list = [], []
                for l in range(self.GP_dim):
                    m_l, v_l, _, _ = self.svgp.approximate_posterior_params_impute(
                        index_points_test=x_te_b,
                        index_points_train=X_tr,
                        y=gp_mu_tr[:, l],
                        noise=gp_var_tr[:, l],
                        cutoff_test=test_cut_b,
                        cutoff_train=cut.to(self.device),
                    )
                    gp_p_m_list.append(m_l)
                    gp_p_v_list.append(v_l)
                gp_p_m = torch.stack(gp_p_m_list, dim=1)
                gp_p_v = torch.stack(gp_p_v_list, dim=1)

            p_m = torch.cat([gp_p_m, g_mu], dim=1)
            latents_out.append(p_m.detach().cpu())
            p_v = torch.clamp(torch.cat([gp_p_v, g_var], dim=1), min=1e-6)

            # Gene-space additive shift for the target perturbation
            target_is_KO = int(target) > 0
            shift = None
            if self.use_action_correction and target_is_KO:
                mu_action_clamped = torch.clamp(self.mu_action, -5.0, 5.0)
                shift = mu_action_clamped.unsqueeze(0).expand(B, -1)
            if self.use_f_pert and self.f_pert is not None and target_is_KO:
                fp = self.f_pert(l_pert)
                if getattr(self, "fpert_field", None) is not None:
                    fp = fp + self.fpert_field(l_pert, x_te_b[:, :2])  # coord-conditioned residual
                fp = torch.clamp(fp, -self.f_pert_clamp, self.f_pert_clamp)
                shift = fp if shift is None else (shift + fp)

            dist = Normal(p_m, torch.sqrt(p_v))
            mu_stack = []
            for _ in range(max(1, n_samples)):
                z = dist.sample()
                mu_basal = self._decode_to_mean(z)
                mu = mu_basal if shift is None else mu_basal * torch.exp(shift)
                mu_stack.append(mu)
            means_out.append(torch.stack(mu_stack, dim=0).mean(dim=0).detach().cpu())

        return torch.cat(latents_out, dim=0).numpy(), torch.cat(means_out, dim=0).numpy()

    # -------------------------
    # Training
    # -------------------------
    def train_model(
        self,
        *,
        pos: np.ndarray,
        batch: np.ndarray,
        ncounts: np.ndarray,
        raw_counts: np.ndarray,
        size_factors: np.ndarray,
        has_guide: Optional[np.ndarray] = None,   # (N,) float32 0/1 — action indicator
        is_real_KO: Optional[np.ndarray] = None,  # (N,) float32 0/1 — gene KO indicator
        niche_id: Optional[np.ndarray] = None,    # (N,) int — niche bin id for LFC supervision (-1 = none)
        sample_id: Optional[np.ndarray] = None,   # (N,) int — per-cell slide id (C2; None = single-slide)
        time_t: Optional[np.ndarray] = None,      # (N,) float — normalized day t∈[0,1] (temporal case study)
        infl_score: Optional[np.ndarray] = None,  # (N,) float — observed continuous inflammation gene-score (v3 target)
        tier1_id: Optional[np.ndarray] = None,    # (N,) int — Tier1 lineage code (timekernel)
        arch_desc: Optional[np.ndarray] = None,   # (N, arch_dim) float — standardised day-invariant architectural descriptor (timekernel)
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 512,
        num_samples: int = 1,
        train_size: float = 0.95,
        maxiter: int = 5000,
        patience: int = 200,
        earlystop_min_delta: float = 0.0,
        save_model: bool = True,
        model_weights: str = "model.pt",
        print_kernel_scale: bool = True,
        report_every: int = 50,
        perturb_name_map: Optional[Dict[int, str]] = None,
        # --- wandb hooks ---
        log_fn: Optional[Callable[[dict, int], None]] = None,
        wandb_run: Optional[object] = None,
        # --- DDP ---
        ddp_model=None,   # DDP-wrapped self; if set, used for forward to sync gradients
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Train the model with early stopping on validation ELBO and optional wandb logging."""
        self.train()

        N = ncounts.shape[0]
        sample_indices = torch.arange(N, dtype=torch.int)
        atts = torch.tensor(self.cell_atts, dtype=torch.int)

        # Sanity check: all cell-level arrays must have the same leading dim
        _shapes = {
            "pos": pos.shape[0], "ncounts": N, "raw_counts": raw_counts.shape[0],
            "size_factors": len(size_factors), "atts": atts.shape[0],
            "batch": batch.shape[0], "mask_cutoff": self.mask_cutoff.shape[0],
        }
        if has_guide is not None:
            _shapes["has_guide"] = len(has_guide)
        if len(set(_shapes.values())) != 1:
            raise ValueError(
                f"[train_model] Size mismatch across inputs — check that normalize() "
                f"is called with filter_min_counts=False. Shapes: {_shapes}"
            )

        # ── 3-class action indicators ────────────────────────────────────────
        # has_guide: True for control + gene_KO (any guide delivered)
        # is_real_KO: True for gene_KO only (not control, not unassigned)
        # is_perturbed (PAD flag): same as has_guide
        if has_guide is not None:
            has_guide_t  = torch.tensor(has_guide,  dtype=torch.float32)
            is_real_KO_t = torch.tensor(is_real_KO, dtype=torch.float32) if is_real_KO is not None \
                           else has_guide_t.clone()
            is_perturbed_all = has_guide_t.to(torch.int8)
        else:
            # Fallback: infer from perturb_code > 0
            perturb_codes    = torch.tensor(self.cell_atts[:, 1], dtype=torch.int8)
            is_perturbed_all = (perturb_codes > 0).to(torch.int8)
            has_guide_t      = is_perturbed_all.float()
            is_real_KO_t     = has_guide_t.clone()

        # niche id per cell for mini-bulk LFC supervision (-1 sentinel when not provided)
        if niche_id is not None:
            niche_id_t = torch.tensor(np.asarray(niche_id), dtype=torch.long)
        else:
            niche_id_t = torch.full((ncounts.shape[0],), -1, dtype=torch.long)

        # per-cell slide id (zeros for single-slide). Long tensor, new final slot.
        _N = ncounts.shape[0]
        slide_id_t = torch.tensor(
            np.asarray(sample_id) if sample_id is not None else np.zeros(_N),
            dtype=torch.long)

        # Continuous-time scalar per cell (normalized day). Zeros when not provided →
        # with use_time_conditioning off this slot is ignored downstream.
        time_t_t = torch.tensor(
            np.asarray(time_t) if time_t is not None else np.zeros(_N),
            dtype=torch.float32)

        # per-cell continuous inflammation gene-score target (zeros when not provided →
        # ignored downstream unless use_inflammation_field is on).
        infl_score_t = torch.tensor(
            np.asarray(infl_score) if infl_score is not None else np.zeros(_N),
            dtype=torch.float32)

        # Ensure all array-like inputs are plain numpy arrays before tensoring
        _to_np = lambda a: a.values if hasattr(a, "values") else np.asarray(a)

        # Timekernel: per-cell Tier1 lineage code + standardised architectural descriptor
        # (zeros when not provided → ignored downstream unless use_timekernel is on).
        tier1_id_t = (torch.tensor(_to_np(tier1_id), dtype=torch.long)
                      if tier1_id is not None else torch.zeros(N, dtype=torch.long))
        if arch_desc is not None:
            arch_desc_np = np.asarray(_to_np(arch_desc), dtype=np.float32)
            if arch_desc_np.ndim != 2 or arch_desc_np.shape[0] != N:
                raise ValueError(
                    f"[train_model] arch_desc must be (N, arch_dim) row-aligned to the cell "
                    f"set; got {arch_desc_np.shape} for N={N}.")
            if arch_desc_np.shape[1] != int(getattr(self, "arch_dim", arch_desc_np.shape[1])):
                raise ValueError(
                    f"[train_model] arch_desc width {arch_desc_np.shape[1]} != model arch_dim "
                    f"{self.arch_dim}.")
            arch_desc_t = torch.tensor(arch_desc_np, dtype=torch.float32)
        else:
            arch_desc_t = torch.zeros((N, int(getattr(self, "arch_dim", 20))), dtype=torch.float32)

        dataset = TensorDataset(
            torch.tensor(_to_np(pos)),
            torch.tensor(_to_np(ncounts)),
            torch.tensor(_to_np(raw_counts)),
            torch.tensor(_to_np(size_factors)),
            sample_indices,
            atts,
            torch.tensor(_to_np(batch)),
            self.mask_cutoff,
            is_perturbed_all,   # slot 8: PAD flag (has_guide)
            has_guide_t,        # slot 9: has_guide float
            is_real_KO_t,       # slot 10: is_real_KO float
            niche_id_t,         # slot 11: niche bin id (long, -1 = none)
            slide_id_t,         # slot 12: per-cell slide id (long, C2)
            time_t_t,           # slot 13: normalized day t∈[0,1] (temporal)
            infl_score_t,       # slot 14: continuous inflammation gene-score target (v3)
            tier1_id_t,         # slot 15: Tier1 lineage code (long)
            arch_desc_t,        # slot 16: standardised architectural descriptor (float, (N, arch_dim))
        )

        is_ddp = world_size > 1
        is_main = rank == 0

        # DataLoader worker config. NOTE: the TensorDataset tensors are CUDA-resident,
        # so num_workers>0 is INVALID here — forked workers cannot re-init CUDA
        # ("CUDA error: initialization error") and pin_memory can't pin GPU tensors.
        # Default 0 (known-good). The real per-sample-indexing bottleneck is addressed
        # separately via batched sampling, not workers. Left env-overridable only for
        # a hypothetical CPU-resident dataset variant.
        _n_workers = int(os.environ.get("CONCERT_NUM_WORKERS", "0"))
        _loader_kw = {}
        if _n_workers > 0:
            _loader_kw = dict(num_workers=_n_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

        if train_size < 1.0:
            train_set, val_set = random_split(dataset=dataset, lengths=[train_size, 1.0 - train_size])
            if is_ddp:
                val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
                val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, drop_last=False, **_loader_kw)
            else:
                val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False, **_loader_kw)
        else:
            train_set = dataset
            val_set = None
            val_loader = None

        drop_last = bool(ncounts.shape[0] * train_size > batch_size)
        if is_ddp:
            train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last)
            train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, drop_last=drop_last, **_loader_kw)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last, **_loader_kw)

        early = EarlyStopping(patience=patience, modelfile=model_weights,
                              patience_start_epoch=self.beta_warmup_epochs,
                              min_delta=earlystop_min_delta)

        # DDP: use the wrapped model for forward passes so gradient hooks fire.
        # Falls back to self when running single-GPU.
        _fwd = ddp_model if ddp_model is not None else self

        # Differential LR: speed up kernel scale tuning
        named = dict(self.named_parameters())
        scale_params = [p for n, p in named.items() if "kernel.scale" in n]
        other_params = [p for n, p in named.items() if "kernel.scale" not in n]
        optim = torch.optim.Adam(
            [
                {"params": other_params, "lr": lr * world_size, "weight_decay": weight_decay},
                {"params": scale_params, "lr": lr * world_size * 10.0, "weight_decay": 0.0},
            ]
        )

        q = deque([], maxlen=10)
        if is_main:
            logging.info("Training... (world_size=%d)", world_size)
        for epoch in range(maxiter):
            stats = dict(elbo=0.0, nb=0.0, gp_kl=0.0, gauss_kl=0.0, mse=0.0, unknown=0.0, pad=0.0)
            n_seen = 0

            # DistributedSampler needs epoch set so each rank gets a different shuffle
            if is_ddp:
                train_loader.sampler.set_epoch(epoch)

            # Ramp GRL alpha from 0→1 over training (Ganin et al. 2015)
            if self.use_pad and self.pad is not None:
                self.pad.set_grl_alpha(grl_alpha_schedule(epoch, maxiter))

            # bf16 autocast disabled: cuSOLVER Cholesky/inv don't support bf16,
            # and bf16 precision causes NaN in encoder latent means.
            # fp32 on H100 is fast enough — no autocast needed.

            for xb, yb, yr, sf, si, ca, bt, cf, ip, hg, rk, ni, sid_b, tt, isc, t1b, adb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yr = yr.to(self.device)
                sf = sf.to(self.device)
                si = si.to(self.device, dtype=torch.int)
                ca = ca.to(self.device, dtype=torch.int)
                bt = bt.to(self.device)
                cf = cf.to(self.device)
                ip = ip.to(self.device, dtype=torch.int)
                hg = hg.to(self.device)   # has_guide float
                rk = rk.to(self.device)   # is_real_KO float
                ni = ni.to(self.device, dtype=torch.long)   # niche bin id
                sid_b = sid_b.to(self.device, dtype=torch.long)   # per-cell slide id (C2)
                tt = tt.to(self.device)   # normalized day t∈[0,1]
                isc = isc.to(self.device)   # continuous inflammation gene-score target (v3)
                t1b = t1b.to(self.device, dtype=torch.long)   # Tier1 lineage code (timekernel)
                adb = adb.to(self.device)                     # (B, arch_dim) architectural descriptor

                with torch.autocast("cuda", enabled=False):
                    (
                        elbo, nb, gpkl, gskl, inside_elbo, gp_ce, p_m, p_v, q_mu, q_var,
                        mean_s, disp_s, rec_in, kl_in, _, pad_l, unknown_pen, mse
                    ) = _fwd(
                        x=xb,
                        y=yb,
                        batch=bt,
                        raw_y=yr,
                        sample_index=si,
                        cell_atts=ca,
                        size_factors=sf,
                        cutoff=cf,
                        num_samples=num_samples,
                        is_perturbed=ip if self.use_pad else None,
                        has_guide=hg if self.use_action_correction else None,
                        # f_pert is gated by use_f_pert (inside _pert_log_shift_from_lord),
                        # NOT use_action_correction: with μ_action OFF (perturbmap option 2)
                        # f_pert must still apply so it receives a reconstruction gradient.
                        is_real_KO=rk,
                        niche_id=ni if self.lfc_lambda > 0 else None,
                        slide_index=sid_b,
                        time_t=tt if self.use_time_conditioning else None,
                        infl_score=isc if self.use_inflammation_field else None,
                        tier1_id=t1b if getattr(self, "use_timekernel", False) else None,
                        arch_desc=adb if getattr(self, "use_timekernel", False) else None,
                    )

                self.zero_grad(set_to_none=True)
                elbo.backward(retain_graph=True)
                # --- NaN/Inf-safe gradient handling -------------------------------
                # clip_grad_norm_ is NOT nan-safe: a single non-finite grad makes
                # total_norm nan -> clip_coef nan -> ALL grads nan -> every param
                # corrupted after optim.step() (incl. the GP kernel `scale`, which
                # then stays nan forever since clamp(nan)=nan). Sanitize per-param
                # grads to 0 on the non-finite entries, then clip the finite result.
                bad_grad = False
                for p in self.parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        bad_grad = True
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                if bad_grad:
                    print("[grad-guard] non-finite gradients sanitized to 0 this step")
                if os.environ.get("CONCERT_DBG_FPERT") and (epoch % 100 == 0) and is_main:
                    import math as _m
                    def _gn(pfx):
                        s = 0.0
                        for _n, _p in self.named_parameters():
                            if _p.grad is not None and _n.startswith(pfx):
                                s += float(_p.grad.norm()) ** 2
                        return _m.sqrt(s)
                    logging.info("[dbg ep%d] f_pert.grad=%.3e fpert_field.grad=%.3e",
                                 epoch, _gn("f_pert."), _gn("fpert_field."))
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optim.step()
                # Belt-and-suspenders: if any parameter still went non-finite,
                # restore it so it cannot poison subsequent forward passes.
                with torch.no_grad():
                    for p in self.parameters():
                        if not torch.isfinite(p).all():
                            torch.nan_to_num_(p, nan=0.0, posinf=0.0, neginf=0.0)

                stats["elbo"] += float(elbo.item())
                stats["nb"] += float(nb.item())
                stats["gp_kl"] += float(gpkl.item())
                stats["gauss_kl"] += float(gskl.item())
                stats["mse"] += float(mse.item())
                stats["unknown"] += float(unknown_pen.item())
                stats["pad"] += float(pad_l.item())
                n_seen += xb.shape[0]

                # Beta schedule: linear warmup 0 -> beta_target takes precedence
                # (collapse insurance), then dynamic PID / static.
                if self.beta_warmup_epochs > 0 and epoch <= self.beta_warmup_epochs:
                    self.beta = self.beta_target * float(epoch) / float(self.beta_warmup_epochs)
                elif self.dynamicVAE:
                    # PID beta update (rank 0 owns the PID; broadcast to other ranks)
                    KL_val = (gpkl.item() + gskl.item()) / max(1, xb.shape[0])
                    q.append(KL_val)
                    if is_main:
                        avg_KL = float(np.mean(q))
                        self.beta, _ = self.PID.pid(self.KL_loss_target * (self.GP_dim + self.Normal_dim), avg_KL)
                    if is_ddp:
                        beta_t = torch.tensor(self.beta, device=self.device)
                        dist.broadcast(beta_t, src=0)
                        self.beta = float(beta_t.item())

            # Normalize stats (aggregate across ranks first in DDP)
            if is_ddp:
                stat_t = torch.tensor(
                    [stats[k] for k in stats] + [float(n_seen)], device=self.device
                )
                dist.all_reduce(stat_t, op=dist.ReduceOp.SUM)
                stat_vals = stat_t.tolist()
                total_seen = max(1, stat_vals[-1])
                for i, k in enumerate(stats):
                    stats[k] = stat_vals[i] / total_seen
            else:
                for k in list(stats.keys()):
                    stats[k] /= max(1, n_seen)

            if is_main:
                logging.info(
                    "Epoch %4d | ELBO %.6f | NB %.6f | GP_KL %.6f | G_KL %.6f | MSE %.6f | Basal %.6f | PAD %.6f | DISENT %.6f | MORAN %.6f | GPAUX %.6f | BMORAN %.6f | beta %.4f",
                    epoch + 1,
                    stats["elbo"],
                    stats["nb"],
                    stats["gp_kl"],
                    stats["gauss_kl"],
                    stats["mse"],
                    stats["unknown"],
                    stats["pad"],
                    self._last_disent,
                    self._last_moran,
                    self._last_gpaux,
                    self._last_basal_out_moran,
                    self.beta,
                )
                # Baseline anchor readout: a(0) must fall toward 0 (day 0 = basal alone).
                # If it plateaus well above 0 the anchor is too weak to identify the sign.
                if self._last_a0 is not None:
                    logging.info("           | a(t=0) %.4f (anchor offset %.3g, lambda %.3g)",
                                 self._last_a0, getattr(self, "amp_anchor_offset", 0.0),
                                 getattr(self, "amp_anchor_lambda", 0.0))
                # Periodic kernel/cutoff report
                if report_every > 0 and ((epoch + 1) % report_every == 0 or epoch == 0):
                    self._report_kernel_and_cutoff(epoch=epoch + 1, perturb_name_map=perturb_name_map)

                if print_kernel_scale and self.svgp is not None:
                    try:
                        logging.debug("kernel scale: %s", self.svgp.kernel.scale.data)
                        logging.debug(
                            "mask cutoff: [min=%.4f, max=%.4f]",
                            float(torch.min(self.mask_cutoff)),
                            float(torch.max(self.mask_cutoff)),
                        )
                    except Exception:  # pragma: no cover
                        pass

                # --- wandb logging ---
                if log_fn is not None:
                    log_fn(
                        {
                            "train/elbo": stats["elbo"],
                            "train/nb": stats["nb"],
                            "train/gp_kl": stats["gp_kl"],
                            "train/gauss_kl": stats["gauss_kl"],
                            "train/mse": stats["mse"],
                            "train/basal": stats["unknown"],
                            "train/beta": float(self.beta),
                        },
                        step=epoch + 1,
                    )
                elif (wandb is not None) and (wandb_run is not None):
                    try:
                        wandb_run.log(
                            {
                                "train/elbo": stats["elbo"],
                                "train/nb": stats["nb"],
                                "train/gp_kl": stats["gp_kl"],
                                "train/gauss_kl": stats["gauss_kl"],
                                "train/mse": stats["mse"],
                                "train/basal": stats["unknown"],
                                "train/beta": float(self.beta),
                            },
                            step=epoch + 1,
                        )
                    except Exception:
                        pass

            # Validation
            stop_flag = torch.tensor(0, device=self.device)
            if val_loader is not None:
                val_elbo, val_n = 0.0, 0
                for xb, yb, yr, sf, si, ca, bt, cf, ip, hg, rk, ni, sid_b, tt, isc, t1b, adb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    yr = yr.to(self.device)
                    sf = sf.to(self.device)
                    si = si.to(self.device, dtype=torch.int)
                    ca = ca.to(self.device, dtype=torch.int)
                    bt = bt.to(self.device)
                    cf = cf.to(self.device)
                    ip = ip.to(self.device, dtype=torch.int)
                    hg = hg.to(self.device)
                    rk = rk.to(self.device)
                    ni = ni.to(self.device, dtype=torch.long)
                    sid_b = sid_b.to(self.device, dtype=torch.long)   # per-cell slide id (C2)
                    tt = tt.to(self.device)   # normalized day t∈[0,1]
                    isc = isc.to(self.device)   # continuous inflammation gene-score target (v3)
                    t1b = t1b.to(self.device, dtype=torch.long)   # Tier1 lineage code (timekernel)
                    adb = adb.to(self.device)                    # (B, arch_dim) architectural descriptor

                    with torch.no_grad():
                        velbo, *_ = self.forward(
                            x=xb,
                            y=yb,
                            batch=bt,
                            raw_y=yr,
                            sample_index=si,
                            cell_atts=ca,
                            size_factors=sf,
                            cutoff=cf,
                            num_samples=num_samples,
                            is_perturbed=ip if self.use_pad else None,
                            has_guide=hg if self.use_action_correction else None,
                            is_real_KO=rk,   # see train-loop note: f_pert gated by use_f_pert, not action
                            niche_id=ni if self.lfc_lambda > 0 else None,
                            slide_index=sid_b,
                            time_t=tt if self.use_time_conditioning else None,
                            infl_score=isc if self.use_inflammation_field else None,
                            tier1_id=t1b if getattr(self, "use_timekernel", False) else None,
                            arch_desc=adb if getattr(self, "use_timekernel", False) else None,
                        )
                    val_elbo += float(velbo.item())
                    val_n += xb.shape[0]

                # Aggregate val ELBO across ranks
                if is_ddp:
                    val_t = torch.tensor([val_elbo, float(val_n)], device=self.device)
                    dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
                    val_elbo = float(val_t[0].item()) / max(1.0, float(val_t[1].item()))
                else:
                    val_elbo /= max(1, val_n)

                if is_main:
                    logging.info("          | Val ELBO %.6f", val_elbo)

                    # wandb validation log
                    if log_fn is not None:
                        log_fn({"val/elbo": val_elbo}, step=epoch + 1)
                    elif (wandb is not None) and (wandb_run is not None):
                        try:
                            wandb_run.log({"val/elbo": val_elbo}, step=epoch + 1)
                        except Exception:
                            pass

                    # early stopping — rank 0 decides
                    early(val_elbo, self, epoch=epoch)
                    if early.early_stop:
                        logging.info("Early stopping at epoch %d", epoch + 1)
                        stop_flag.fill_(1)

                # Broadcast early-stop decision to all ranks
                if is_ddp:
                    dist.broadcast(stop_flag, src=0)
                if bool(stop_flag.item()):
                    break

        if save_model and is_main:
            atomic_torch_save(self.state_dict(), model_weights)
            logging.info("Saved weights to %s", model_weights)
    
    def _report_kernel_and_cutoff(
        self,
        epoch: int,
        perturb_name_map: Optional[Dict[int, str]] = None,
        ) -> None:
        if self.svgp is None:
            logging.info("[epoch %d] basal_field=rff (no SVGP kernel); skipping kernel report.", epoch)
            return
        try:
            scale = getattr(self.svgp.kernel, "scale", None)
            if scale is None:
                logging.info("[epoch %d] Kernel has no `scale` attribute; skipping report.", epoch)
                return
            scale_np = scale.detach().float().cpu().numpy()
        except Exception:
            logging.info("[epoch %d] Could not read kernel scales; skipping report.", epoch)
            return

        perts = self.cell_atts[:, 1].astype(int)  # (N,) integer codes
        unique_codes = sorted(np.unique(perts).tolist())

        # If single-kernel mode, just print the shared scale once.
        if scale_np.ndim == 1:
            logging.info("[epoch %d] Kernel scale (shared): %s", epoch, np.array2string(scale_np, precision=4))
        else:
            # multi-kernel: scale_np shape (n_batch, spatial_dims)
            logging.info("[epoch %d] Kernel scales by perturbation:", epoch)

        cutoff_vec = self.mask_cutoff.detach().float().cpu().numpy()

        for code in unique_codes:
            name = (perturb_name_map or {}).get(code, f"pert={code}")
            idx = (perts == code)
            mean_cut = float(np.mean(cutoff_vec[idx])) if np.any(idx) else float("nan")

            # print kernel scale row if in multi-kernel mode, else show shared again with tag
            if scale_np.ndim == 2:
                if code < scale_np.shape[0]:
                    scale_row = scale_np[code]
                    logging.info("  • %-16s | scale=%s | mean_cutoff=%.4f",
                                 name, np.array2string(scale_row, precision=4), mean_cut)
                else:
                    logging.info("  • %-16s | scale=<out of range for batch idx %d> | mean_cutoff=%.4f",
                                 name, code, mean_cut)
            else:
                logging.info("  • %-16s | scale(shared)=%s | mean_cutoff=%.4f",
                             name, np.array2string(scale_np, precision=4), mean_cut)

