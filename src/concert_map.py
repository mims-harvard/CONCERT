# ==========================
# CONCERT for perturb-map
# ==========================
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, Iterable, List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset, random_split

# Optional wandb import (safe if not installed)
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from I_PID import PIDControl
from SVGP_Batch import SVGP
from VAE_utils import MeanAct, NBLoss, buildNetwork, DenseEncoder
from lord_batch import LordEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unknown_attribute_penalty(latent_unknown: torch.Tensor) -> torch.Tensor:
    """Quadratic penalty on the unknown/basal attributes (encourages small magnitude)."""
    return torch.sum(latent_unknown ** 2, dim=1).mean()


class EarlyStopping:
    """Early stopping with best-weight saving based on validation loss."""

    def __init__(self, patience: int = 10, verbose: bool = False, modelfile: str = "model.pt") -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.model_file = modelfile

    def __call__(self, loss: float, model: nn.Module) -> None:
        if np.isnan(loss):
            self.early_stop = True
            return
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self._save(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                logging.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self._save(loss, model)
            self.counter = 0

    def _save(self, loss: float, model: nn.Module) -> None:
        if self.verbose:
            logging.info("Validation loss improved: %.6f → %.6f (saving)", self.loss_min, loss)
        torch.save(model.state_dict(), self.model_file)
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
    ) -> None:
        super().__init__()
        torch.set_default_dtype(dtype)

        # SVGP prior
        if multi_kernel_mode:
            self.svgp = SVGP(
                fixed_inducing_points=fixed_inducing_points,
                initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params,
                kernel_scale=kernel_scale,
                multi_kernel_mode=multi_kernel_mode,
                kernel_phi=1.0,
                jitter=1e-8,
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
                jitter=1e-8,
                N_train=N_train,
                dtype=dtype,
                device=device,
            )

        # Hyper-params
        self.encoder_dim = encoder_dim
        self.GP_dim = GP_dim
        self.Normal_dim = Normal_dim
        self.noise = noise
        self.device = device
        self.num_genes = num_genes
        self.cell_atts = cell_atts
        self.shared_dispersion = shared_dispersion

        # Dynamic beta scheduling
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss_target = KL_loss
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta

        # Encoder / decoder
        # LORD produces (basal, tissue, perturbation) embeddings each of size encoder_dim
        self.lord_encoder = LordEncoder(
            embedding_dim=[encoder_dim] * 3,
            num_genes=self.num_genes,
            labels=cell_atts,
            attributes=["tissue", "perturbation"],
            attributes_type=["categorical", "categorical"],
            noise=self.noise,
            device=device,
        )
        self.encoder = DenseEncoder(
            input_dim=encoder_dim * 3,
            hidden_dims=list(encoder_layers),
            output_dim=GP_dim + Normal_dim,
            activation="elu",
            dropout=encoder_dropout,
        )
        self.decoder = buildNetwork([GP_dim + Normal_dim] + list(decoder_layers), activation="elu", dropout=decoder_dropout)
        last_dim = decoder_layers[-1] if len(list(decoder_layers)) > 0 else GP_dim + Normal_dim
        self.dec_mean = nn.Sequential(nn.Linear(last_dim, self.num_genes), MeanAct())

        # NB dispersion (shared or batch-specific)
        if shared_dispersion:
            self.dec_disp = nn.Parameter(torch.randn(self.num_genes), requires_grad=True)
        else:
            self.dec_disp = nn.Parameter(torch.randn(self.num_genes, n_batch), requires_grad=True)

        # Learnable mask cutoff vector
        self.mask_cutoff = nn.Parameter(torch.tensor(mask_cutoff, dtype=dtype), requires_grad=True)

        # Losses
        self.NB_loss = NBLoss().to(self.device)
        self.mse = nn.MSELoss(reduction="mean")
        self.to(device)

    # -------------------------
    # Persistence
    # -------------------------
    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, map_location=lambda storage, loc: storage)
        filtered = {k: v for k, v in state.items() if k in self.state_dict()}
        self.load_state_dict({**self.state_dict(), **filtered})

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
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass computing ELBO terms and reconstructions."""
        self.train()
        bsz = y.shape[0]

        # LORD encodes attribute-aware representation
        lord = self.lord_encoder.predict(sample_indices=sample_index, labels=cell_atts, batch_size=bsz)
        y_embed = lord["total_latent"]  # (B, 3*encoder_dim)

        # Inference network -> posterior params for GP & Gaussian latents
        q_mu, q_var = self.encoder(y_embed)
        gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
        gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]

        # SVGP posterior & variational loss (sum across GP dims)
        elbo_recon_terms, elbo_kl_terms, gp_p_m, gp_p_v = [], [], [], []
        for l in range(self.GP_dim):
            p_m_l, p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(
                x, x, gp_mu[:, l], gp_var[:, l], cutoff=cutoff
            )
            rec_l, kl_l = self.svgp.variational_loss(
                x=x, y=gp_mu[:, l], noise=gp_var[:, l], mu_hat=mu_hat_l, A_hat=A_hat_l, cutoff=cutoff
            )
            elbo_recon_terms.append(rec_l)
            elbo_kl_terms.append(kl_l)
            gp_p_m.append(p_m_l)
            gp_p_v.append(p_v_l)

        inside_elbo_recon = torch.sum(torch.stack(elbo_recon_terms, dim=-1))
        inside_elbo_kl = torch.sum(torch.stack(elbo_kl_terms, dim=-1))
        inside_elbo = inside_elbo_recon - (bsz / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # GP prior KL via cross entropy (q||p)
        gp_ce = self._gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var).sum()
        gp_KL = gp_ce - inside_elbo

        # Gaussian block KL
        prior = Normal(torch.zeros_like(gs_mu), torch.ones_like(gs_var))
        post = Normal(gs_mu, torch.sqrt(gs_var))
        gauss_KL = kl_divergence(post, prior).sum()

        # Reparameterized samples & NB recon loss
        p_m = torch.cat([gp_p_m, gs_mu], dim=1)
        p_v = torch.cat([gp_p_v, gs_var], dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))

        mean_accum, disp_accum, recon_loss, recon_mse = [], [], 0.0, 0.0
        for _ in range(num_samples):
            z = latent_dist.rsample()
            h = self.decoder(z)
            mu = self.dec_mean(h)
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

        # LORD penalty on basal latent
        unknown_pen = _unknown_attribute_penalty(lord["basal_latent"])  # encourages disentanglement
        lord_loss = unknown_pen + recon_mse

        elbo = recon_loss + self.beta * (gp_KL + gauss_KL) + lord_loss
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
            None,  # latent_samples (omitted here)
            torch.tensor(0.0, device=self.device),  # noise_reg placeholder
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
        self, X: np.ndarray, sample_index: torch.Tensor, cell_atts: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        self.eval()
        cuts = self.mask_cutoff
        X_t = torch.tensor(X, dtype=torch.get_default_dtype())
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

            lord = self.lord_encoder.predict(sample_indices=is_, batch_size=xs.shape[0], labels=cs)
            y_ = lord["total_latent"]
            q_mu, q_var = self.encoder(y_)
            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]

            gp_p_m = []
            for l in range(self.GP_dim):
                m_l, v_l, _, _ = self.svgp.approximate_posterior_params(xs, xs, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                gp_p_m.append(m_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        cuts = self.mask_cutoff
        X_t = torch.tensor(X, dtype=torch.get_default_dtype())
        atts_t = torch.tensor(cell_atts, dtype=torch.int)
        idx_t = torch.tensor(sample_index, dtype=torch.int)

        means: List[torch.Tensor] = []
        vars_: List[torch.Tensor] = []

        N = X_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            xs = X_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            cs = atts_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            is_ = idx_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            cutoff = cuts[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)

            lord = self.lord_encoder.predict(sample_indices=is_, batch_size=xs.shape[0], labels=cs)
            y_ = lord["total_latent"]
            q_mu, q_var = self.encoder(y_)

            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                m_l, v_l, _, _ = self.svgp.approximate_posterior_params(xs, xs, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                gp_p_m.append(m_l)
                gp_p_v.append(v_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            dist = Normal(p_m, torch.sqrt(p_v))

            samples = [dist.sample() for _ in range(max(1, n_samples))]
            mu_stack = []
            for z in samples:
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
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
            h = self.decoder(z)
            outs.append(self.dec_mean(h).cpu())
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Impute latents & counts on unseen locations by nearest-neighbor Gaussian part + GP posterior."""
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

        # Precompute q_mu/q_var on train
        q_mu_all, q_var_all = [], []
        N_tr = X_tr.shape[0]
        n_tr_batches = int(math.ceil(N_tr / batch_size))
        for b in range(n_tr_batches):
            cs = atts_tr[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device)
            is_ = idx_tr[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device)
            lord = self.lord_encoder.predict(sample_indices=is_, batch_size=cs.shape[0], labels=cs)
            y_ = lord["total_latent"]
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
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            latents_out.append(p_m.cpu())

            dist = Normal(p_m, torch.sqrt(p_v))
            zs = [dist.sample() for _ in range(max(1, n_samples))]
            mu_stack = []
            for z in zs:
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
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
        n_samples: int = 1,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Counterfactual prediction by overwriting selected cells' attributes and decoding."""
        self.eval()

        cuts = self.mask_cutoff
        X_t = torch.tensor(X, dtype=torch.get_default_dtype())
        idx_t = torch.tensor(sample_index, dtype=torch.int)

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

        means_out: List[torch.Tensor] = []
        N = X_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            xs = X_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            cs = atts_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            is_ = idx_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            cutoff = cuts[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)

            lord = self.lord_encoder.predict(sample_indices=is_, batch_size=xs.shape[0], labels=cs)
            y_ = lord["total_latent"]
            q_mu, q_var = self.encoder(y_)

            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                m_l, v_l, _, _ = self.svgp.approximate_posterior_params(xs, xs, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                gp_p_m.append(m_l)
                gp_p_v.append(v_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            dist = Normal(p_m, torch.sqrt(p_v))

            zs = [dist.sample() for _ in range(max(1, n_samples))]
            mu_stack = []
            for z in zs:
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
            means_out.append(torch.stack(mu_stack, dim=0).mean(dim=0).cpu())

        return torch.cat(means_out, dim=0).numpy(), pert_atts
    
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
	) -> Tuple[np.ndarray, np.ndarray]:
        """
        Impute latents & denoised counts at unseen coords. For the Gaussian block,
        pull neighbors *only from the specified perturbation* in the training set.
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
            lord = self.lord_encoder.predict(sample_indices=si, batch_size=hi - lo, labels=ca)
            y_ = lord["total_latent"]
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
            p_v = torch.cat([gp_p_v, g_var], dim=1)

            dist = Normal(p_m, torch.sqrt(p_v))
            mu_stack = []
            for _ in range(max(1, n_samples)):
                z = dist.sample()
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
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
	) -> Tuple[np.ndarray, np.ndarray]:
        """
        Counterfactual impute:
        • GP block: SVGP posterior from training GP latents.
        • Gaussian block: replace tissue/perturbation with specified targets.
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
            lord = self.lord_encoder.predict(sample_indices=si, batch_size=hi - lo, labels=ca)
            basal_list.append(lord["basal_latent"])
            y_ = lord["total_latent"]
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
            y_cf = torch.cat([basal_cf, l_tissue, l_pert], dim=1)

            g_mu_all, g_var_all = self.encoder(y_cf)
            g_mu = g_mu_all[:, self.GP_dim :]
            g_var = g_var_all[:, self.GP_dim :]

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
            p_v = torch.cat([gp_p_v, g_var], dim=1)

            dist = Normal(p_m, torch.sqrt(p_v))
            mu_stack = []
            for _ in range(max(1, n_samples)):
                z = dist.sample()
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
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
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 512,
        num_samples: int = 1,
        train_size: float = 0.95,
        maxiter: int = 5000,
        patience: int = 200,
        save_model: bool = True,
        model_weights: str = "model.pt",
        print_kernel_scale: bool = True,
        report_every: int = 50,
        perturb_name_map: Optional[Dict[int, str]] = None,
        # --- wandb hooks ---
        log_fn: Optional[Callable[[dict, int], None]] = None,   # recommended: driver passes wandb logger
        wandb_run: Optional[object] = None,                     # fallback: pass a wandb run object
    ) -> None:
        """Train the model with early stopping on validation ELBO and optional wandb logging."""
        self.train()

        sample_indices = torch.arange(ncounts.shape[0], dtype=torch.int)
        atts = torch.tensor(self.cell_atts, dtype=torch.int)

        dataset = TensorDataset(
            torch.tensor(pos),
            torch.tensor(ncounts),
            torch.tensor(raw_counts),
            torch.tensor(size_factors),
            sample_indices,
            atts,
            torch.tensor(batch),
            self.mask_cutoff,
        )

        if train_size < 1.0:
            train_set, val_set = random_split(dataset=dataset, lengths=[train_size, 1.0 - train_size])
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_set = dataset
            val_loader = None

        drop_last = bool(ncounts.shape[0] * train_size > batch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        early = EarlyStopping(patience=patience, modelfile=model_weights)

        # Differential LR: speed up kernel scale tuning
        named = dict(self.named_parameters())
        scale_params = [p for n, p in named.items() if "kernel.scale" in n]
        other_params = [p for n, p in named.items() if "kernel.scale" not in n]
        optim = torch.optim.Adam(
            [
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
                {"params": scale_params, "lr": lr * 10.0, "weight_decay": 0.0},
            ]
        )

        q = deque([], maxlen=10)
        logging.info("Training...")
        for epoch in range(maxiter):
            stats = dict(elbo=0.0, nb=0.0, gp_kl=0.0, gauss_kl=0.0, mse=0.0, unknown=0.0)
            n_seen = 0

            for xb, yb, yr, sf, si, ca, bt, cf in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yr = yr.to(self.device)
                sf = sf.to(self.device)
                si = si.to(self.device, dtype=torch.int)
                ca = ca.to(self.device, dtype=torch.int)
                bt = bt.to(self.device)
                cf = cf.to(self.device)

                (
                    elbo, nb, gpkl, gskl, inside_elbo, gp_ce, p_m, p_v, q_mu, q_var,
                    mean_s, disp_s, rec_in, kl_in, _, noise_reg, unknown_pen, mse
                ) = self.forward(
                    x=xb,
                    y=yb,
                    batch=bt,
                    raw_y=yr,
                    sample_index=si,
                    cell_atts=ca,
                    size_factors=sf,
                    cutoff=cf,
                    num_samples=num_samples,
                )

                self.zero_grad(set_to_none=True)
                elbo.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optim.step()

                stats["elbo"] += float(elbo.item())
                stats["nb"] += float(nb.item())
                stats["gp_kl"] += float(gpkl.item())
                stats["gauss_kl"] += float(gskl.item())
                stats["mse"] += float(mse.item())
                stats["unknown"] += float(unknown_pen.item())
                n_seen += xb.shape[0]

                # PID beta update
                if self.dynamicVAE:
                    KL_val = (gpkl.item() + gskl.item()) / max(1, xb.shape[0])
                    q.append(KL_val)
                    avg_KL = float(np.mean(q))
                    self.beta, _ = self.PID.pid(self.KL_loss_target * (self.GP_dim + self.Normal_dim), avg_KL)

            # Normalize stats
            for k in list(stats.keys()):
                stats[k] /= max(1, n_seen)

            logging.info(
                "Epoch %4d | ELBO %.6f | NB %.6f | GP_KL %.6f | G_KL %.6f | MSE %.6f | Basal %.6f | beta %.4f",
                epoch + 1,
                stats["elbo"],
                stats["nb"],
                stats["gp_kl"],
                stats["gauss_kl"],
                stats["mse"],
                stats["unknown"],
                self.beta,
            )
            # Periodic kernel/cutoff report
            if report_every > 0 and ((epoch + 1) % report_every == 0 or epoch == 0):
                self._report_kernel_and_cutoff(epoch=epoch + 1, perturb_name_map=perturb_name_map)

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
                # Direct logging if a wandb run is provided (fallback)
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

            if print_kernel_scale:
                try:
                    logging.debug("kernel scale: %s", self.svgp.kernel.scale.data)
                    logging.debug(
                        "mask cutoff: [min=%.4f, max=%.4f]",
                        float(torch.min(self.mask_cutoff)),
                        float(torch.max(self.mask_cutoff)),
                    )
                except Exception:  # pragma: no cover
                    pass

            # Validation
            if val_loader is not None:
                val_elbo, val_n = 0.0, 0
                for xb, yb, yr, sf, si, ca, bt, cf in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    yr = yr.to(self.device)
                    sf = sf.to(self.device)
                    si = si.to(self.device, dtype=torch.int)
                    ca = ca.to(self.device, dtype=torch.int)
                    bt = bt.to(self.device)
                    cf = cf.to(self.device)

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
                    )
                    val_elbo += float(velbo.item())
                    val_n += xb.shape[0]
                val_elbo /= max(1, val_n)
                logging.info("          | Val ELBO %.6f", val_elbo)

                # wandb validation log
                if log_fn is not None:
                    log_fn({"val/elbo": val_elbo}, step=epoch + 1)
                elif (wandb is not None) and (wandb_run is not None):
                    try:
                        wandb_run.log({"val/elbo": val_elbo}, step=epoch + 1)
                    except Exception:
                        pass

                # early stopping
                early(val_elbo, self)
                if early.early_stop:
                    logging.info("Early stopping at epoch %d", epoch + 1)
                    break

        if save_model:
            torch.save(self.state_dict(), model_weights)
            logging.info("Saved weights to %s", model_weights)
    
    def _report_kernel_and_cutoff(
        self,
        epoch: int,
        perturb_name_map: Optional[Dict[int, str]] = None,
        ) -> None:
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

