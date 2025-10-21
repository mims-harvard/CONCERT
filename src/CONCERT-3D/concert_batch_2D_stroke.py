from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader, TensorDataset, random_split

# Optional W&B
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

# Project modules
from SVGP_Batch import SVGP
from I_PID import PIDControl
from VAE_utils import NBLoss, DenseEncoder, buildNetwork, MeanAct

# LORD encoder (support both names)
try:
    from lord_batch import LordEncoder  # newer name
except Exception:
    try:
        from lord_batch import Lord_encoder as LordEncoder  # legacy export
    except Exception:  # pragma: no cover
        LordEncoder = None


# ----------------------------- Numerics ------------------------------

_EPS = 1e-8
_MAX_VAR = 1e6
_MAX_EXP_CLAMP = 15.0


def _nan_guard(name: str, t: torch.Tensor):
    if t is None:
        return
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
        raise ValueError(f"[NaN/Inf] in tensor '{name}' at {bad[:5].tolist()}")


def safe_softplus(x: torch.Tensor) -> torch.Tensor:
    y = F.softplus(x)
    y = torch.clamp(y, min=_EPS, max=_MAX_VAR)
    return torch.nan_to_num(y, nan=_EPS, posinf=_MAX_VAR, neginf=_EPS)


def safe_sqrt_var(v: torch.Tensor) -> torch.Tensor:
    v = torch.clamp(v, min=_EPS, max=_MAX_VAR)
    v = torch.nan_to_num(v, nan=_EPS, posinf=_MAX_VAR, neginf=_EPS)
    return torch.sqrt(v)


def safe_exp_clamped(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, -_MAX_EXP_CLAMP, _MAX_EXP_CLAMP))


# --------------------------- EarlyStopping ---------------------------

class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


# ------------------------------- Model --------------------------------

class CONCERT3D(nn.Module):
    """
    CONCERT model (SVGP + VAE + LORD) for 3D spatial inputs.

    Notes
    -----
    * Dimension-agnostic: as long as `SVGP` is instantiated with 3D index points
      (or 3D + one-hot batch appended), this model just forwards tensors through.
    * forward() returns exactly: (elbo, recon_loss, gp_KL_term, gaussian_KL_term)
    * Loss normalization matches the “old” behavior (per-cell scale, NB / genes).
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
        encoder_layers: List[int],
        decoder_layers: List[int],
        noise: float,
        encoder_dropout: float,
        decoder_dropout: float,
        shared_dispersion: bool,
        fixed_inducing_points: bool,
        initial_inducing_points: np.ndarray,
        fixed_gp_params: bool,
        kernel_scale: float | List[float],
        allow_batch_kernel_scale: bool,
        N_train: int,
        KL_loss: float,
        dynamicVAE: bool,
        init_beta: float,
        min_beta: float,
        max_beta: float,
        dtype: torch.dtype,
        device: str,
        lord_embedding_dim: Tuple[int, int] = (128, 128),
        lord_attributes: List[str] = ["perturbation"],
        lord_attr_types: List[str] = ["categorical"],
        init_dispersion_log: float = -2.0,  # closer to old code’s stable start
    ):
        super().__init__()
        if LordEncoder is None:
            raise ImportError("Could not import LordEncoder / Lord_encoder from lord_batch.")

        torch.set_default_dtype(dtype)
        self.device = device
        self.dtype = dtype

        # --- SVGP block (agnostic to index point dimensionality)
        if allow_batch_kernel_scale:
            self.svgp = SVGP(
                fixed_inducing_points=fixed_inducing_points,
                initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params,
                kernel_scale=([kernel_scale] * n_batch) if not isinstance(kernel_scale, list) else kernel_scale,
                allow_batch_kernel_scale=allow_batch_kernel_scale,
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
                kernel_scale=kernel_scale if not isinstance(kernel_scale, list) else float(kernel_scale[0]),
                allow_batch_kernel_scale=allow_batch_kernel_scale,
                jitter=1e-8,
                N_train=N_train,
                dtype=dtype,
                device=device,
            )

        # --- VAE
        self.input_dim = input_dim
        self.GP_dim = GP_dim
        self.Normal_dim = Normal_dim
        self.num_genes = num_genes
        self.noise = noise
        self.shared_dispersion = shared_dispersion

        self.encoder = DenseEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_layers,
            output_dim=GP_dim + Normal_dim,  # utils returns (mu, var_raw)
            activation="elu",
            dropout=encoder_dropout,
        )
        self.decoder = buildNetwork([GP_dim + Normal_dim] + decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], self.num_genes), MeanAct())
        else:
            self.dec_mean = nn.Sequential(nn.Linear(GP_dim + Normal_dim, self.num_genes), MeanAct())

        if self.shared_dispersion:
            self.dec_disp = nn.Parameter(torch.full((self.num_genes,), init_dispersion_log), requires_grad=True)
        else:
            self.dec_disp = nn.Parameter(torch.full((self.num_genes, n_batch), init_dispersion_log), requires_grad=True)

        # --- LORD encoder (attribute-aware)
        self.cell_atts = cell_atts
        self.lord_encoder = LordEncoder(
            embedding_dim=list(lord_embedding_dim),
            num_genes=self.num_genes,
            attributes=lord_attributes,
            attributes_type=lord_attr_types,
            labels=cell_atts,
            device=device,
            noise=self.noise,
        )
        logging.info("LORD encoder initialized: noise=%.3f", float(self.noise))

        # --- Loss & beta control
        self.NB_loss = NBLoss().to(self.device)
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta

        self.to(device)

    # ----------------------- Persistence -----------------------

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    # ------------------------- Forward -------------------------

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: torch.Tensor,
        raw_y: torch.Tensor,
        sample_index: torch.Tensor,
        cell_atts: torch.Tensor,
        size_factors: torch.Tensor,
        num_samples: int = 1,
    ):
        """
        Forward pass — returns (elbo, recon_loss, gp_KL_term, gaussian_KL_term)
        """
        self.train()
        b = y.shape[0]

        # LORD latents
        lord_latents = self.lord_encoder.predict(
            sample_indices=sample_index, labels=cell_atts, batch_size=b
        )
        y_in = lord_latents["total_latent"]
        _nan_guard("lord_total_latent", y_in)

        # Encoder -> mean, var (DenseEncoder returns (mu, var_raw))
        q_mu, q_var_raw = self.encoder(y_in)
        _nan_guard("enc_mu_raw", q_mu)
        _nan_guard("enc_var_raw", q_var_raw)
        q_var = safe_softplus(q_var_raw)

        gp_mu = q_mu[:, :self.GP_dim]
        gp_var = q_var[:, :self.GP_dim]
        gs_mu = q_mu[:, self.GP_dim:]
        gs_var = q_var[:, self.GP_dim:]

        # SVGP terms per GP dim
        inside_elbo_recon_list, inside_elbo_kl_list = [], []
        gp_p_m_list, gp_p_v_list = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(
                x, x, gp_mu[:, l], gp_var[:, l]
            )
            elbo_rec_l, elbo_kl_l = self.svgp.variational_loss(
                x=x, y=gp_mu[:, l], noise=gp_var[:, l], mu_hat=mu_hat_l, A_hat=A_hat_l
            )
            inside_elbo_recon_list.append(elbo_rec_l)
            inside_elbo_kl_list.append(elbo_kl_l)
            gp_p_m_list.append(gp_p_m_l)
            gp_p_v_list.append(gp_p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon_list).sum()
        inside_elbo_kl = torch.stack(inside_elbo_kl_list).sum()
        inside_elbo = inside_elbo_recon - (b / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m_list, dim=1)
        gp_p_v = torch.stack(gp_p_v_list, dim=1)
        _nan_guard("gp_post_m", gp_p_m)
        _nan_guard("gp_post_v", gp_p_v)

        gp_ce_term = self._gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var).sum()
        gp_KL_term = gp_ce_term - inside_elbo

        # Non-GP latent KL
        prior = Normal(torch.zeros_like(gs_mu), torch.ones_like(gs_var))
        post = Normal(gs_mu, safe_sqrt_var(gs_var))
        gaussian_KL_term = kl_divergence(post, prior).sum()

        # Reconstruction
        # latent Normal: mean=[gp_p_m, gs_mu], var=[gp_p_v, gs_var]
        p_m = torch.cat([gp_p_m, gs_mu], dim=1)
        p_v = torch.cat([gp_p_v, gs_var], dim=1)
        latent_dist = Normal(p_m, safe_sqrt_var(p_v))

        # Dispersion
        if self.shared_dispersion:
            disp = safe_exp_clamped(self.dec_disp).T  # (G,) -> broadcast to (N,G) in loss
        else:
            disp = safe_exp_clamped(self.dec_disp @ batch.T).T  # (N,G)

        recon_loss = 0.0
        for _ in range(num_samples):
            z = latent_dist.rsample()
            h = self.decoder(z)
            mean = self.dec_mean(h)
            _nan_guard("mean", mean)
            recon_loss = recon_loss + self.NB_loss(x=raw_y, mean=mean, disp=disp, scale_factor=size_factors)
        recon_loss = recon_loss / num_samples

        # --- normalization to “old” scale ---
        cells, genes = raw_y.shape
        recon_loss = recon_loss / genes          # per-cell (NB averaged over genes)
        gp_KL_term = gp_KL_term / cells          # KLs per-cell
        gaussian_KL_term = gaussian_KL_term / cells

        elbo = recon_loss + self.beta * gp_KL_term + self.beta * gaussian_KL_term

        # Final sanity
        _nan_guard("elbo", elbo)
        _nan_guard("recon_loss", recon_loss)
        _nan_guard("gp_KL_term", gp_KL_term)
        _nan_guard("gaussian_KL_term", gaussian_KL_term)

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term

    @staticmethod
    def _gauss_cross_entropy(m_p: torch.Tensor, v_p: torch.Tensor, m_q: torch.Tensor, v_q: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of diagonal Gaussians N_p || N_q (elementwise)."""
        v_q = torch.clamp(v_q, min=_EPS, max=_MAX_VAR)
        term = 0.5 * (torch.log(2 * math.pi * v_q) + (v_p + (m_p - m_q) ** 2) / v_q)
        return torch.sum(term, dim=-1)

    # --------------------------- Inference utils ---------------------------

    @torch.no_grad()
    def batching_latent_samples(self, X, sample_index, cell_atts, batch_size=512):
        self.eval()
        X = torch.tensor(X, dtype=self.dtype)
        cell_atts = torch.tensor(cell_atts, dtype=torch.int64)
        total_sample_indices = torch.tensor(sample_index, dtype=torch.int64)

        out = []
        n = X.shape[0]
        n_batches = int(math.ceil(n / batch_size))
        for i in range(n_batches):
            xb = X[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            sb = total_sample_indices[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            ab = cell_atts[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            bsz = xb.size(0)

            lord_lat = self.lord_encoder.predict(sample_indices=sb, batch_size=bsz, labels=ab)
            y_in = lord_lat["total_latent"]

            q_mu, q_var_raw = self.encoder(y_in)
            q_var = safe_softplus(q_var_raw)
            gp_mu = q_mu[:, :self.GP_dim]
            gs_mu = q_mu[:, self.GP_dim:]

            gp_p_m = []
            for l in range(self.GP_dim):
                pm, _, _, _ = self.svgp.approximate_posterior_params(xb, xb, gp_mu[:, l], q_var[:, l])
                gp_p_m.append(pm)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            out.append(p_m.cpu())

        return torch.cat(out, dim=0).numpy()

    @torch.no_grad()
    def batching_denoise_counts(self, X, sample_index, cell_atts, n_samples=1, batch_size=512):
        self.eval()
        X = torch.tensor(X, dtype=self.dtype)
        cell_atts = torch.tensor(cell_atts, dtype=torch.int64)
        total_sample_indices = torch.tensor(sample_index, dtype=torch.int64)

        means = []
        n = X.shape[0]
        n_batches = int(math.ceil(n / batch_size))
        for i in range(n_batches):
            xb = X[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            sb = total_sample_indices[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            ab = cell_atts[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            bsz = xb.size(0)

            lord_lat = self.lord_encoder.predict(sample_indices=sb, batch_size=bsz, labels=ab)
            y_in = lord_lat["total_latent"]

            q_mu, q_var_raw = self.encoder(y_in)
            q_var = safe_softplus(q_var_raw)
            gp_mu = q_mu[:, :self.GP_dim]
            gs_mu = q_mu[:, self.GP_dim:]
            gp_var = q_var[:, :self.GP_dim]
            gs_var = q_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                pm, pv, _, _ = self.svgp.approximate_posterior_params(xb, xb, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(pm)
                gp_p_v.append(pv)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            dist = Normal(p_m, safe_sqrt_var(p_v))

            m_list = []
            for _ in range(int(max(1, n_samples))):
                z = dist.sample()
                h = self.decoder(z)
                m = self.dec_mean(h)
                m_list.append(m)
            m_avg = torch.stack(m_list, dim=0).mean(0)
            means.append(m_avg.cpu())

        return torch.cat(means, dim=0).numpy()

    @torch.no_grad()
    def batching_recon_samples(self, Z, batch_size=512):
        self.eval()
        Z = torch.tensor(Z, dtype=self.dtype)
        out = []
        n = Z.shape[0]
        n_batches = int(math.ceil(n / batch_size))
        for i in range(n_batches):
            zb = Z[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            h = self.decoder(zb)
            m = self.dec_mean(h)
            out.append(m.cpu())
        return torch.cat(out, dim=0).numpy()

    @torch.no_grad()
    def imputation(self, X_test, X_train, Y_sample_index, Y_cell_atts, n_samples=1, batch_size=512):
        self.eval()
        X_test = torch.tensor(X_test, dtype=self.dtype)
        X_train = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        train_atts = torch.tensor(Y_cell_atts, dtype=torch.int64)
        train_idx = torch.tensor(Y_sample_index, dtype=torch.int64)

        # pre-encode train attributes
        q_mu_list, q_var_list = [], []
        n = X_train.shape[0]
        nb = int(math.ceil(n / batch_size))
        for i in range(nb):
            ab = train_atts[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            sb = train_idx[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            lord_lat = self.lord_encoder.predict(sample_indices=sb, batch_size=ab.shape[0], labels=ab)
            y_in = lord_lat["total_latent"]
            mu, vr = self.encoder(y_in)
            q_mu_list.append(mu)
            q_var_list.append(vr)
        q_mu = torch.cat(q_mu_list, dim=0)
        q_var = safe_softplus(torch.cat(q_var_list, dim=0))

        def nearest(a: torch.Tensor, v: torch.Tensor):
            return torch.argmin(torch.sum((a - v) ** 2, dim=1))

        latents, means = [], []
        m = X_test.shape[0]
        mb = int(math.ceil(m / batch_size))
        for i in range(mb):
            xt = X_test[i * batch_size: min((i + 1) * batch_size, m)].to(self.device)
            gp_mu = q_mu[:, :self.GP_dim]
            gp_var = q_var[:, :self.GP_dim]

            select = []
            for e in range(xt.shape[0]):
                select.append(nearest(X_train, xt[e]))
            select = torch.stack(select)
            gs_mu = q_mu[select.long(), self.GP_dim:]
            gs_var = q_var[select.long(), self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                pm, pv, _, _ = self.svgp.approximate_posterior_params(
                    index_points_test=xt, index_points_train=X_train, y=gp_mu[:, l], noise=gp_var[:, l]
                )
                gp_p_m.append(pm)
                gp_p_v.append(pv)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            latents.append(p_m.cpu())

            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            dist = Normal(p_m, safe_sqrt_var(p_v))

            ms = []
            for _ in range(int(max(1, n_samples))):
                z = dist.sample()
                h = self.decoder(z)
                m_ = self.dec_mean(h)
                ms.append(m_)
            means.append(torch.stack(ms, dim=0).mean(0).cpu())

        return torch.cat(latents, dim=0).numpy(), torch.cat(means, dim=0).numpy()

    @torch.no_grad()
    def counterfactualPrediction(
        self, X, sample_index, cell_atts, n_samples=1, batch_size=512,
        perturb_cell_id=None, target_cell_perturbation=None
    ):
        self.eval()
        X = torch.tensor(X, dtype=self.dtype)
        total_sample_indices = torch.tensor(sample_index, dtype=torch.int64)

        # apply counterfactual attribute
        pert_cell_atts = np.array(cell_atts, copy=True)
        if perturb_cell_id is not None and target_cell_perturbation is not None:
            idxs = np.array(perturb_cell_id, dtype=int).tolist()
            for i in idxs:
                pert_cell_atts[i, 0] = int(target_cell_perturbation)

        pert_cell_atts = torch.tensor(pert_cell_atts, dtype=torch.int64)

        means = []
        n = X.shape[0]
        nb = int(math.ceil(n / batch_size))
        for i in range(nb):
            xb = X[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            ab = pert_cell_atts[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            sb = total_sample_indices[i * batch_size: min((i + 1) * batch_size, n)].to(self.device)
            bsz = xb.shape[0]

            lord_lat = self.lord_encoder.predict(sample_indices=sb, batch_size=bsz, labels=ab)
            y_in = lord_lat["total_latent"]
            q_mu, q_var_raw = self.encoder(y_in)
            q_var = safe_softplus(q_var_raw)

            gp_mu = q_mu[:, :self.GP_dim]
            gp_var = q_var[:, :self.GP_dim]
            gs_mu = q_mu[:, self.GP_dim:]
            gs_var = q_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                pm, pv, _, _ = self.svgp.approximate_posterior_params(xb, xb, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(pm)
                gp_p_v.append(pv)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat([gp_p_m, gs_mu], dim=1)
            p_v = torch.cat([gp_p_v, gs_var], dim=1)
            dist = Normal(p_m, safe_sqrt_var(p_v))

            ms = []
            for _ in range(int(max(1, n_samples))):
                z = dist.sample()
                h = self.decoder(z)
                m_ = self.dec_mean(h)
                ms.append(m_)
            means.append(torch.stack(ms, dim=0).mean(0).cpu())

        return torch.cat(means, dim=0).numpy(), pert_cell_atts.cpu().numpy()

    @torch.no_grad()
    def batching_predict_samples_cp(
        self, X_test, X_train, Y_sample_index, Y_cell_atts, n_samples=1, batch_size=512,
        perturb_cell_id=None, target_cell_perturbation=None
    ):
        # identical idea to batching_predict_samples but with perturbed attributes
        pert_cell_atts = np.array(Y_cell_atts, copy=True)
        if perturb_cell_id is not None and target_cell_perturbation is not None:
            idxs = np.array(perturb_cell_id, dtype=int).tolist()
            for i in idxs:
                pert_cell_atts[i, 0] = int(target_cell_perturbation)
        return self.batching_predict_samples(X_test, X_train, Y_sample_index, pert_cell_atts, n_samples, batch_size)

    # ------------------------------ Training ------------------------------

    def train_model(
        self,
        pos,
        batch,
        ncounts,
        raw_counts,
        size_factors,
        lr=1e-4,
        weight_decay=1e-6,
        batch_size=512,
        num_samples=1,
        train_size=0.95,
        maxiter=5000,
        patience=200,
        save_model=True,
        model_weights="model.pt",
        print_kernel_scale=True,
        # W&B
        wandb_run=None,
        wandb_cfg: dict | None = None,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ):
        self.train()

        # Clamp size factors
        if isinstance(size_factors, np.ndarray):
            size_factors = np.clip(size_factors, a_min=_EPS, a_max=np.inf)
        else:
            size_factors = np.maximum(size_factors, _EPS)

        sample_indices = torch.arange(ncounts.shape[0], dtype=torch.int64)
        cell_atts = self.cell_atts

        dataset = TensorDataset(
            torch.tensor(pos, dtype=self.dtype),
            torch.tensor(ncounts, dtype=self.dtype),
            torch.tensor(raw_counts, dtype=self.dtype),
            torch.tensor(size_factors, dtype=self.dtype),
            sample_indices,
            torch.tensor(cell_atts, dtype=torch.int64),
            torch.tensor(batch, dtype=self.dtype),
        )

        validate_dataloader = None
        if train_size < 1.0:
            n_train = int(round(len(dataset) * float(train_size)))
            n_val = len(dataset) - n_train
            train_dataset, validate_dataset = random_split(dataset, [n_train, n_val])
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        else:
            train_dataset = dataset

        drop_last = (len(train_dataset) > batch_size)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        early_stopping = EarlyStopping(patience=patience, modelfile=model_weights)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        # W&B init if requested
        run = wandb_run
        if run is None and wandb_cfg is not None and _WANDB_AVAILABLE:
            run = wandb.init(
                project=wandb_project or wandb_cfg.get("project", "concert"),
                name=wandb_run_name or wandb_cfg.get("name"),
                config=wandb_cfg,
            )

        kl_queue = []
        print("Training")

        for epoch in range(maxiter):
            self.train()
            elbo_sum = rec_sum = gpkl_sum = gskl_sum = 0.0
            n_samples_seen = 0

            for (x_b, y_b, y_raw_b, sf_b, samp_b, atts_b, b_b) in dataloader:
                x_b = x_b.to(self.device, dtype=self.dtype)
                y_b = y_b.to(self.device, dtype=self.dtype)
                y_raw_b = y_raw_b.to(self.device, dtype=self.dtype)
                sf_b = torch.clamp(sf_b.to(self.device, dtype=self.dtype), min=_EPS)
                b_b = b_b.to(self.device, dtype=self.dtype)
                samp_b = samp_b.to(self.device, dtype=torch.int64)
                atts_b = atts_b.to(self.device, dtype=torch.int64)

                elbo, rec, gpkl, gskl = self.forward(
                    x=x_b, y=y_b, batch=b_b, raw_y=y_raw_b,
                    sample_index=samp_b, cell_atts=atts_b,
                    size_factors=sf_b, num_samples=num_samples
                )

                _nan_guard("train/elbo", elbo)

                optimizer.zero_grad(set_to_none=True)
                elbo.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                bs = x_b.size(0)
                elbo_sum += float(elbo.item()) * bs
                rec_sum += float(rec.item()) * bs
                gpkl_sum += float(gpkl.item()) * bs
                gskl_sum += float(gskl.item()) * bs
                n_samples_seen += bs

                if self.dynamicVAE:
                    kl_val = (gpkl.item() + gskl.item()) / max(1, bs)
                    kl_queue.append(kl_val)
                    if len(kl_queue) > 10:
                        kl_queue.pop(0)
                    avg_kl = float(np.mean(kl_queue))
                    self.beta, _ = self.PID.pid(self.KL_loss * (self.GP_dim + self.Normal_dim), avg_kl)

            n = max(1, n_samples_seen)
            metrics = {
                "train/elbo": elbo_sum / n,
                "train/nb": rec_sum / n,
                "train/gpkl": gpkl_sum / n,
                "train/gskl": gskl_sum / n,
                "train/beta": float(self.beta),
            }
            print("Epoch {}: {}".format(
                epoch + 1,
                ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
            ))

            if print_kernel_scale:
                try:
                    ks = torch.clamp(F.softplus(self.svgp.kernel.scale), min=_EPS, max=1e4).detach().cpu().numpy()
                    metrics["kernel_scale/mean"] = float(ks.mean())
                except Exception:
                    pass

            if run is not None and _WANDB_AVAILABLE:
                wandb.log(metrics, step=epoch)

            # Validation + early stopping
            if validate_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    v_elbo_sum, v_n = 0.0, 0
                    for (vx, vy, vyr, vsf, vsamp, vatts, vb) in validate_dataloader:
                        vx = vx.to(self.device, dtype=self.dtype)
                        vy = vy.to(self.device, dtype=self.dtype)
                        vyr = vyr.to(self.device, dtype=self.dtype)
                        vsf = torch.clamp(vsf.to(self.device, dtype=self.dtype), min=_EPS)
                        vb = vb.to(self.device, dtype=self.dtype)
                        vsamp = vsamp.to(self.device, dtype=torch.int64)
                        vatts = vatts.to(self.device, dtype=torch.int64)

                        velbo, _, _, _ = self.forward(
                            x=vx, y=vy, batch=vb, raw_y=vyr,
                            sample_index=vsamp, cell_atts=vatts,
                            size_factors=vsf, num_samples=num_samples
                        )
                        _nan_guard("val/elbo", velbo)
                        bs = vx.size(0)
                        v_elbo_sum += float(velbo.item()) * bs
                        v_n += bs

                    v_elbo = v_elbo_sum / max(1, v_n)
                    if run is not None and _WANDB_AVAILABLE:
                        wandb.log({"val/elbo": v_elbo}, step=epoch)
                    print(f"  val/elbo={v_elbo:.6f}")

                    early_stopping(v_elbo, self)
                    if early_stopping.early_stop:
                        print(f"EarlyStopping at epoch {epoch+1}")
                        break

        if save_model:
            torch.save(self.state_dict(), model_weights)

        if run is not None and _WANDB_AVAILABLE:
            run.finish()