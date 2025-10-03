from __future__ import annotations
import logging
import math
from collections import deque
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset, random_split

# Optional wandb import (safe if absent)
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from SVGP_Batch_3d import SVGP
from I_PID import PIDControl
from VAE_utils import MeanAct, NBLoss, buildNetwork, DenseEncoder, gauss_cross_entropy
from lord_batch import Lord_encoder


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _unknown_attribute_penalty(latent_unknown: torch.Tensor) -> torch.Tensor:
    """Quadratic penalty to keep basal (unknown) attributes small."""
    return torch.sum(latent_unknown**2, dim=1).mean()


class EarlyStopping:
    """Early stopping with best-weight restore based on validation ELBO."""
    def __init__(self, patience: int = 10, verbose: bool = False, modelfile: str = "model.pt") -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
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
            logging.info("Val loss improved: %.6f → %.6f (saving)", self.loss_min, loss)
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


# ---------------------------------------------------------------------
# Model (3D stroke / binary-batched kernel)
# ---------------------------------------------------------------------
class CONCERT(nn.Module):
    """
    CONCERT for the 3D stroke dataset with **binary-batched** SVGP kernel.

    Notes
    -----
    - Works with 3D coordinates (x,y,z) that you typically scale per-batch in the driver.
    """

    def __init__(
        self,
        *,
        input_dim: int,
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
        initial_inducing_points: np.ndarray,   # (M*n_batch, 3 + n_batch) for batched kernel
        fixed_gp_params: bool,
        kernel_scale: float | np.ndarray,      # float or (n_batch, spatial_dims) if batched
        allow_batch_kernel_scale: bool,
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

        self.svgp = SVGP(
            fixed_inducing_points=fixed_inducing_points,
            initial_inducing_points=initial_inducing_points,
            fixed_gp_params=fixed_gp_params,
            kernel_scale=([kernel_scale] * n_batch) if allow_batch_kernel_scale and np.isscalar(kernel_scale) else kernel_scale,
            allow_batch_kernel_scale=bool(allow_batch_kernel_scale),
            jitter=1e-6,
            N_train=N_train,
            dtype=dtype,
            device=device,
        )

        # Hyperparams
        self.input_dim = int(input_dim)
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss_target = float(KL_loss)
        self.dynamicVAE = bool(dynamicVAE)
        self.beta = float(init_beta)
        self.dtype = dtype
        self.GP_dim = int(GP_dim)
        self.Normal_dim = int(Normal_dim)
        self.noise = float(noise)
        self.device = device
        self.num_genes = int(num_genes)
        self.cell_atts = cell_atts
        self.n_batch = int(n_batch)
        self.shared_dispersion = bool(shared_dispersion)

        # LORD encoder: attributes = ["perturbation"(categorical)]
        self.lord_encoder = Lord_encoder(
            embedding_dim=[128, 128],
            num_genes=self.num_genes,
            labels=cell_atts,
            attributes=["perturbation"],
            attributes_type=["categorical"],
            noise=self.noise,
            device=device,
        )

        # Inference & decoder
        self.encoder = DenseEncoder(
            input_dim=self.input_dim,
            hidden_dims=list(encoder_layers),
            output_dim=self.GP_dim + self.Normal_dim,
            activation="elu",
            dropout=encoder_dropout,
        )
        self.decoder = buildNetwork([self.GP_dim + self.Normal_dim] + list(decoder_layers), activation="elu", dropout=decoder_dropout)
        last_dim = decoder_layers[-1] if len(list(decoder_layers)) > 0 else (self.GP_dim + self.Normal_dim)
        self.dec_mean = nn.Sequential(nn.Linear(last_dim, self.num_genes), MeanAct())

        # NB dispersion
        if self.shared_dispersion:
            self.dec_disp = nn.Parameter(torch.randn(self.num_genes), requires_grad=True)
        else:
            self.dec_disp = nn.Parameter(torch.randn(self.num_genes, self.n_batch), requires_grad=True)

        # Losses
        self.NB_loss = NBLoss().to(device)
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
    # Forward
    # -------------------------
    def forward(
        self,
        *,
        x: torch.Tensor,               # (N, 3 + n_batch) positions with batch one-hot appended
        y: torch.Tensor,               # normalized counts
        batch: torch.Tensor,           # (N, n_batch) one-hot
        raw_y: torch.Tensor,           # raw counts
        sample_index: torch.Tensor,    # indices for LORD
        cell_atts: torch.Tensor,       # [perturb_code]
        size_factors: torch.Tensor,    # NB scale factors
        num_samples: int = 1,
    ):
        self.train()
        bsz = y.shape[0]

        # LORD latent → encoder
        lord = self.lord_encoder.predict(sample_indices=sample_index, labels=cell_atts, batch_size=bsz)
        y_ = lord["total_latent"]
        q_mu, q_var = self.encoder(y_)

        gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
        gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            p_m_l, p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x, gp_mu[:, l], gp_var[:, l])
            rec_l, kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l], noise=gp_var[:, l], mu_hat=mu_hat_l, A_hat=A_hat_l)
            inside_elbo_recon.append(rec_l)
            inside_elbo_kl.append(kl_l)
            gp_p_m.append(p_m_l)
            gp_p_v.append(p_v_l)

        inside_elbo_recon = torch.sum(torch.stack(inside_elbo_recon, dim=-1))
        inside_elbo_kl = torch.sum(torch.stack(inside_elbo_kl, dim=-1))
        inside_elbo = inside_elbo_recon - (bsz / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # KL terms
        gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var).sum()
        gp_KL_term = gp_ce_term - inside_elbo

        prior = Normal(torch.zeros_like(gs_mu), torch.ones_like(gs_var))
        post = Normal(gs_mu, torch.sqrt(gs_var))
        gaussian_KL_term = kl_divergence(post, prior).sum()

        # Decoder sampling
        p_m = torch.cat([gp_p_m, gs_mu], dim=1)
        p_v = torch.cat([gp_p_v, gs_var], dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))

        recon_nb, recon_mse = 0.0, 0.0
        mean_samples: List[torch.Tensor] = []
        disp_samples: List[torch.Tensor] = []

        for _ in range(max(1, num_samples)):
            z = latent_dist.rsample()
            h = self.decoder(z)
            mu = self.dec_mean(h)
            if self.shared_dispersion:
                disp = torch.exp(torch.clamp(self.dec_disp, -15.0, 15.0)).unsqueeze(0).expand_as(mu)
            else:
                # per-batch dispersion: (G, B) · (N, B)^T → (N, G)
                disp = torch.exp(torch.clamp(torch.matmul(self.dec_disp, batch.T), -15.0, 15.0)).T

            mean_samples.append(mu)
            disp_samples.append(disp)
            recon_nb = recon_nb + self.NB_loss(x=raw_y, mean=mu, disp=disp, scale_factor=size_factors)
            recon_mse = recon_mse + self.mse(mu, raw_y)

        recon_nb = recon_nb / max(1, num_samples)
        recon_mse = recon_mse / max(1, num_samples)

        # LORD penalty
        unknown_pen = _unknown_attribute_penalty(lord["basal_latent"])
        lord_loss = unknown_pen + recon_mse

        elbo = recon_nb + self.beta * (gp_KL_term + gaussian_KL_term) + lord_loss

        return (
            elbo,
            recon_nb,
            gp_KL_term,
            gaussian_KL_term,
            inside_elbo,
            gp_ce_term,
            gp_p_m,
            gp_p_v,
            q_mu,
            q_var,
            mean_samples,
            disp_samples,
            inside_elbo_recon,
            inside_elbo_kl,
            None,  # latent_samples (omit to save memory)
            torch.tensor(0.0, device=self.device),  # noise_reg placeholder
            unknown_pen,
            recon_mse,
        )

    # -------------------------
    # Batch helpers
    # -------------------------
    @torch.no_grad()
    def batching_latent_samples(
        self, X: np.ndarray, sample_index: np.ndarray, cell_atts: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        self.eval()
        X_t = torch.tensor(X, dtype=self.dtype)
        atts_t = torch.tensor(cell_atts, dtype=torch.int)
        idx_t = torch.tensor(sample_index, dtype=torch.int)

        outs: List[torch.Tensor] = []
        N = X_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            xb = X_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            si = idx_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device, dtype=torch.int)
            ca = atts_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device, dtype=torch.int)

            lord = self.lord_encoder.predict(sample_indices=si, batch_size=xb.shape[0], labels=ca)
            y_ = lord["total_latent"]
            q_mu, q_var = self.encoder(y_)

            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            gs_mu = q_mu[:, self.GP_dim :]

            gp_p_m = []
            for l in range(self.GP_dim):
                m_l, _, _, _ = self.svgp.approximate_posterior_params(xb, xb, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(m_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            outs.append(p_m.cpu())

        return torch.cat(outs, dim=0).numpy()

    @torch.no_grad()
    def batching_denoise_counts(
        self, X: np.ndarray, sample_index: np.ndarray, cell_atts: np.ndarray, n_samples: int = 1, batch_size: int = 512
    ) -> np.ndarray:
        self.eval()
        X_t = torch.tensor(X, dtype=self.dtype)
        atts_t = torch.tensor(cell_atts, dtype=torch.int)
        idx_t = torch.tensor(sample_index, dtype=torch.int)

        means: List[torch.Tensor] = []
        N = X_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            xb = X_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            si = idx_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device, dtype=torch.int)
            ca = atts_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device, dtype=torch.int)

            lord = self.lord_encoder.predict(sample_indices=si, batch_size=xb.shape[0], labels=ca)
            y_ = lord["total_latent"]
            q_mu, q_var = self.encoder(y_)

            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                m_l, v_l, _, _ = self.svgp.approximate_posterior_params(xb, xb, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(m_l)
                gp_p_v.append(v_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            dist = Normal(p_m, torch.sqrt(p_v))

            mu_stack = []
            for _ in range(max(1, n_samples)):
                z = dist.sample()
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
            means.append(torch.stack(mu_stack, dim=0).mean(dim=0).cpu())

        return torch.cat(means, dim=0).numpy()

    @torch.no_grad()
    def batching_recon_samples(self, Z: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.eval()
        Z_t = torch.tensor(Z, dtype=self.dtype)
        outs: List[torch.Tensor] = []
        N = Z_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            z = Z_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            outs.append(self.dec_mean(self.decoder(z)).cpu())
        return torch.cat(outs, dim=0).numpy()

    @torch.no_grad()
    def batching_predict_samples(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
        Y_sample_index: np.ndarray,
        Y_cell_atts: np.ndarray,
        n_samples: int = 1,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Impute latents & counts on unseen coordinates using GP posterior + NN Gaussian block."""
        self.eval()

        def nearest_idx(array: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return torch.argmin(torch.sum((array - value) ** 2, dim=1))

        X_te = torch.tensor(X_test, dtype=self.dtype)
        X_tr = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        atts_tr = torch.tensor(Y_cell_atts, dtype=torch.int)
        idx_tr = torch.tensor(Y_sample_index, dtype=torch.int)

        # Precompute q on train
        q_mu_all, q_var_all = [], []
        N_tr = X_tr.shape[0]
        n_tr_batches = int(math.ceil(N_tr / batch_size))
        for b in range(n_tr_batches):
            ca = atts_tr[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device, dtype=torch.int)
            si = idx_tr[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device, dtype=torch.int)
            lord = self.lord_encoder.predict(sample_indices=si, batch_size=ca.shape[0], labels=ca)
            y_ = lord["total_latent"]
            m, v = self.encoder(y_)
            q_mu_all.append(m)
            q_var_all.append(v)
        q_mu = torch.cat(q_mu_all, dim=0)
        q_var = torch.cat(q_var_all, dim=0)

        latents_out, means_out = [], []
        N_te = X_te.shape[0]
        n_te_batches = int(math.ceil(N_te / batch_size))
        x_train_sel = [nearest_idx(X_tr, X_te[e]) for e in range(N_te)]
        x_train_sel = torch.stack(x_train_sel)

        gp_mu_tr, gp_var_tr = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]

        for b in range(n_te_batches):
            x_te_b = X_te[b * batch_size : min((b + 1) * batch_size, N_te)].to(self.device)
            sel_b = x_train_sel[b * batch_size : min((b + 1) * batch_size, N_te)].long()
            gs_mu = q_mu[sel_b, self.GP_dim :]
            gs_var = q_var[sel_b, self.GP_dim :]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                m_l, v_l, _, _ = self.svgp.approximate_posterior_params(
                    index_points_test=x_te_b, index_points_train=X_tr, y=gp_mu_tr[:, l], noise=gp_var_tr[:, l]
                )
                gp_p_m.append(m_l)
                gp_p_v.append(v_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            latents_out.append(p_m.cpu())

            dist = Normal(p_m, torch.sqrt(p_v))
            mu_stack = []
            for _ in range(max(1, n_samples)):
                z = dist.sample()
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
            means_out.append(torch.stack(mu_stack, dim=0).mean(dim=0).cpu())

        return torch.cat(latents_out, dim=0).numpy(), torch.cat(means_out, dim=0).numpy()

    # -------------------------
    # Training (concert_map style: logs, early stop, optional wandb)
    # -------------------------
    def train_model(
        self,
        *,
        pos: np.ndarray,                    # (N, 3 + n_batch)
        batch: np.ndarray,                  # (N, n_batch) one-hot
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
        # logging hooks
        log_fn: Optional[Callable[[dict, int], None]] = None,
        wandb_run: Optional[object] = None,
        report_every: int = 50,
    ) -> None:
        """Train with early stopping on validation ELBO; periodic kernel-scale reporting."""
        self.train()

        dataset = TensorDataset(
            torch.tensor(pos, dtype=self.dtype),
            torch.tensor(ncounts, dtype=self.dtype),
            torch.tensor(raw_counts, dtype=self.dtype),
            torch.tensor(size_factors, dtype=self.dtype),
            torch.tensor(np.arange(ncounts.shape[0]), dtype=torch.int),
            torch.tensor(self.cell_atts, dtype=torch.int),
            torch.tensor(batch, dtype=self.dtype),
        )

        # split
        if train_size < 1.0:
            train_set, val_set = random_split(dataset=dataset, lengths=[train_size, 1.0 - train_size])
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_set = dataset
            val_loader = None

        drop_last = bool(ncounts.shape[0] * train_size > batch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        early = EarlyStopping(patience=patience, modelfile=model_weights)
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        q = deque([], maxlen=10)
        logging.info("Training...")
        for epoch in range(maxiter):
            stats = dict(elbo=0.0, nb=0.0, gp_kl=0.0, g_kl=0.0, mse=0.0, basal=0.0)
            n_seen = 0

            for xb, yb, yr, sf, si, ca, b_oh in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yr = yr.to(self.device)
                sf = sf.to(self.device)
                si = si.to(self.device, dtype=torch.int)
                ca = ca.to(self.device, dtype=torch.int)
                b_oh = b_oh.to(self.device)

                (
                    elbo, nb, gpkl, gskl, inside_elbo, gp_ce, p_m, p_v, q_mu, q_var,
                    mean_s, disp_s, rec_in, kl_in, _, noise_reg, unknown_pen, mse
                ) = self.forward(
                    x=xb,
                    y=yb,
                    batch=b_oh,
                    raw_y=yr,
                    sample_index=si,
                    cell_atts=ca,
                    size_factors=sf,
                    num_samples=num_samples,
                )

                self.zero_grad(set_to_none=True)
                elbo.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optim.step()

                stats["elbo"] += float(elbo.item())
                stats["nb"] += float(nb.item())
                stats["gp_kl"] += float(gpkl.item())
                stats["g_kl"] += float(gskl.item())
                stats["mse"] += float(mse.item())
                stats["basal"] += float(unknown_pen.item())
                n_seen += xb.shape[0]

                # PID beta update
                if self.dynamicVAE:
                    KL_val = (gpkl.item() + gskl.item()) / max(1, xb.shape[0])
                    q.append(KL_val)
                    avg_KL = float(np.mean(q))
                    self.beta, _ = self.PID.pid(self.KL_loss_target * (self.GP_dim + self.Normal_dim), avg_KL)

            # normalize stats
            for k in list(stats.keys()):
                stats[k] /= max(1, n_seen)

            logging.info(
                "Epoch %4d | ELBO %.6f | NB %.6f | GP_KL %.6f | G_KL %.6f | MSE %.6f | Basal %.6f | beta %.4f",
                epoch + 1,
                stats["elbo"],
                stats["nb"],
                stats["gp_kl"],
                stats["g_kl"],
                stats["mse"],
                stats["basal"],
                self.beta,
            )

            # Periodic kernel-scale report
            if print_kernel_scale and report_every > 0 and ((epoch + 1) % report_every == 0 or epoch == 0):
                try:
                    scale = getattr(self.svgp.kernel, "scale", None)
                    if scale is not None:
                        s_np = torch.clamp(F.softplus(scale), min=1e-10, max=1e4).detach().cpu().numpy()
                        if s_np.ndim == 1:
                            logging.info("[epoch %d] Kernel scale (shared): %s", epoch + 1, np.array2string(s_np, precision=4))
                        else:
                            logging.info("[epoch %d] Kernel scales by batch:", epoch + 1)
                            for i, row in enumerate(s_np):
                                logging.info("  • batch=%d | scale=%s", i, np.array2string(row, precision=4))
                except Exception:  # pragma: no cover
                    pass

            # Optional logging
            if log_fn is not None:
                log_fn(
                    {
                        "train/elbo": stats["elbo"],
                        "train/nb": stats["nb"],
                        "train/gp_kl": stats["gp_kl"],
                        "train/gauss_kl": stats["g_kl"],
                        "train/mse": stats["mse"],
                        "train/basal": stats["basal"],
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
                            "train/gauss_kl": stats["g_kl"],
                            "train/mse": stats["mse"],
                            "train/basal": stats["basal"],
                            "train/beta": float(self.beta),
                        },
                        step=epoch + 1,
                    )
                except Exception:
                    pass

            # Validation
            if val_loader is not None:
                val_elbo, val_n = 0.0, 0
                for xb, yb, yr, sf, si, ca, b_oh in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    yr = yr.to(self.device)
                    sf = sf.to(self.device)
                    si = si.to(self.device, dtype=torch.int)
                    ca = ca.to(self.device, dtype=torch.int)
                    b_oh = b_oh.to(self.device)

                    velbo, *_ = self.forward(
                        x=xb,
                        y=yb,
                        batch=b_oh,
                        raw_y=yr,
                        sample_index=si,
                        cell_atts=ca,
                        size_factors=sf,
                        num_samples=num_samples,
                    )
                    val_elbo += float(velbo.item())
                    val_n += xb.shape[0]
                val_elbo /= max(1, val_n)
                logging.info("          | Val ELBO %.6f", val_elbo)

                if log_fn is not None:
                    log_fn({"val/elbo": val_elbo}, step=epoch + 1)
                elif (wandb is not None) and (wandb_run is not None):
                    try:
                        wandb_run.log({"val/elbo": val_elbo}, step=epoch + 1)
                    except Exception:
                        pass

                early(val_elbo, self)
                if early.early_stop:
                    logging.info("Early stopping at epoch %d", epoch + 1)
                    break

        if save_model:
            torch.save(self.state_dict(), model_weights)
            logging.info("Saved weights to %s", model_weights)

    # -------------------------
    # Counterfactual prediction
    # -------------------------
    @torch.no_grad()
    def counterfactualPrediction(
        self,
        X: np.ndarray,
        sample_index: np.ndarray,
        cell_atts: np.ndarray,
        *,
        perturb_cell_id: Optional[np.ndarray] = None,
        target_cell_perturbation: Optional[int] = None,
        n_samples: int = 1,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Counterfactual counts after modifying selected cells' perturbation."""
        self.eval()

        X_t = torch.tensor(X, dtype=self.dtype)
        idx_t = torch.tensor(sample_index, dtype=torch.int)

        # clone & modify attributes
        pert_atts = np.array(cell_atts, copy=True)
        if perturb_cell_id is not None and target_cell_perturbation is not None:
            for i in np.asarray(perturb_cell_id).tolist():
                pert_atts[:, 0][i] = int(target_cell_perturbation)
        atts_t = torch.tensor(pert_atts, dtype=torch.int)

        means: List[torch.Tensor] = []
        N = X_t.shape[0]
        n_batches = int(math.ceil(N / batch_size))
        for b in range(n_batches):
            xb = X_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device)
            si = idx_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device, dtype=torch.int)
            ca = atts_t[b * batch_size : min((b + 1) * batch_size, N)].to(self.device, dtype=torch.int)

            lord = self.lord_encoder.predict(sample_indices=si, batch_size=xb.shape[0], labels=ca)
            y_ = lord["total_latent"]
            q_mu, q_var = self.encoder(y_)

            gp_mu, gp_var = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]
            gs_mu, gs_var = q_mu[:, self.GP_dim :], q_var[:, self.GP_dim :]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                m_l, v_l, _, _ = self.svgp.approximate_posterior_params(xb, xb, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(m_l)
                gp_p_v.append(v_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            dist = Normal(p_m, torch.sqrt(p_v))

            mu_stack = []
            for _ in range(max(1, n_samples)):
                z = dist.sample()
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
            means.append(torch.stack(mu_stack, dim=0).mean(dim=0).cpu())

        return torch.cat(means, dim=0).numpy(), pert_atts

    @torch.no_grad()
    def batching_predict_samples_cp(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
        Y_sample_index: np.ndarray,
        Y_cell_atts: np.ndarray,
        *,
        perturb_cell_id: Optional[np.ndarray] = None,
        target_cell_perturbation: Optional[int] = None,
        n_samples: int = 1,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Counterfactual imputation at NEW coordinates (like batching_predict_samples + CF edits)."""
        self.eval()

        # Apply counterfactual edits on the training attributes used for Gaussian block NN
        pert_atts = np.array(Y_cell_atts, copy=True)
        if perturb_cell_id is not None and target_cell_perturbation is not None:
            for i in np.asarray(perturb_cell_id).tolist():
                pert_atts[:, 0][i] = int(target_cell_perturbation)

        X_te = torch.tensor(X_test, dtype=self.dtype)
        X_tr = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        atts_tr = torch.tensor(pert_atts, dtype=torch.int)
        idx_tr = torch.tensor(Y_sample_index, dtype=torch.int)

        # Precompute q on train (with edited attributes)
        q_mu_all, q_var_all = [], []
        N_tr = X_tr.shape[0]
        n_tr_batches = int(math.ceil(N_tr / batch_size))
        for b in range(n_tr_batches):
            ca = atts_tr[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device, dtype=torch.int)
            si = idx_tr[b * batch_size : min((b + 1) * batch_size, N_tr)].to(self.device, dtype=torch.int)
            lord = self.lord_encoder.predict(sample_indices=si, batch_size=ca.shape[0], labels=ca)
            y_ = lord["total_latent"]
            m, v = self.encoder(y_)
            q_mu_all.append(m)
            q_var_all.append(v)
        q_mu = torch.cat(q_mu_all, dim=0)
        q_var = torch.cat(q_var_all, dim=0)

        def nearest_idx(array: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return torch.argmin(torch.sum((array - value) ** 2, dim=1))

        latents_out, means_out = [], []
        N_te = X_te.shape[0]
        n_te_batches = int(math.ceil(N_te / batch_size))
        x_train_sel = [nearest_idx(X_tr, X_te[e]) for e in range(N_te)]
        x_train_sel = torch.stack(x_train_sel)

        gp_mu_tr, gp_var_tr = q_mu[:, : self.GP_dim], q_var[:, : self.GP_dim]

        for b in range(n_te_batches):
            x_te_b = X_te[b * batch_size : min((b + 1) * batch_size, N_te)].to(self.device)
            sel_b = x_train_sel[b * batch_size : min((b + 1) * batch_size, N_te)].long()
            gs_mu = q_mu[sel_b, self.GP_dim :]
            gs_var = q_var[sel_b, self.GP_dim :]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                m_l, v_l, _, _ = self.svgp.approximate_posterior_params(
                    index_points_test=x_te_b, index_points_train=X_tr, y=gp_mu_tr[:, l], noise=gp_var_tr[:, l]
                )
                gp_p_m.append(m_l)
                gp_p_v.append(v_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            latents_out.append(p_m.cpu())

            dist = Normal(p_m, torch.sqrt(p_v))
            mu_stack = []
            for _ in range(max(1, n_samples)):
                z = dist.sample()
                h = self.decoder(z)
                mu_stack.append(self.dec_mean(h))
            means_out.append(torch.stack(mu_stack, dim=0).mean(dim=0).cpu())

        return torch.cat(latents_out, dim=0).numpy(), torch.cat(means_out, dim=0).numpy()
