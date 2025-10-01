# ================================================================================
# Sparse Variational Gaussian Process (SVGP) model for batched spatial kernels.
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from kernel import (
    CauchyKernel,
    BatchedCauchyKernel_CONCERT_flex,
    SampleKernel,
)

def _add_diagonal_jitter(matrix: torch.Tensor, jitter: float = 1e-8) -> torch.Tensor:
    """Safely add jitter to the diagonal of a (b, m, m) or (m, m) matrix."""
    eye = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype)
    # Broadcast eye to batch if needed
    if matrix.dim() == 3 and eye.dim() == 2:
        eye = eye.expand(matrix.size(0), -1, -1)
    return matrix + jitter * eye


class SVGP(nn.Module):
    """
    Sparse Variational Gaussian Process (SVGP) for batched spatial kernels.

    Parameters
    ----------
    fixed_inducing_points : bool
        If True, inducing locations are fixed; otherwise they are learnable parameters.
    initial_inducing_points : np.ndarray | torch.Tensor
        Initial inducing locations with appended one-hot batch columns:
        shape (M, 2 + n_batches). First 2 columns must be spatial coords.
    fixed_gp_params : bool
        If True, kernel hyperparameters (e.g., scale) are fixed.
    kernel_scale : np.ndarray | torch.Tensor | float
        Kernel length-scale(s). For multi-kernel mode, pass an array of shape (n_batch, spatial_dims).
    multi_kernel_mode : bool
        If True, use batched kernel across sample groups/batches.
    jitter : float
        Numerical jitter for PD adjustments.
    N_train : int
        Total number of training samples for scaling the ELBO terms.
    dtype : torch.dtype
    device : str
    kernel_phi : float, optional
        Extra kernel parameter used by batched Cauchy kernels.
    """

    def __init__(
        self,
        fixed_inducing_points: bool,
        initial_inducing_points,
        fixed_gp_params: bool,
        kernel_scale,
        multi_kernel_mode: bool,
        jitter: float,
        N_train: int,
        dtype: torch.dtype,
        device: str,
        kernel_phi: float = 1.0,
    ) -> None:
        super().__init__()
        self.N_train = torch.tensor(N_train, dtype=dtype, device=device)
        self.multi_kernel_mode = multi_kernel_mode
        self.jitter = float(jitter)
        self.dtype = dtype
        self.device = device
        self.kernel_phi = float(kernel_phi)

        # Inducing points
        init_ip = torch.as_tensor(initial_inducing_points, dtype=dtype, device=device)
        if fixed_inducing_points:
            self.inducing_index_points = init_ip  # fixed tensor
        else:
            self.inducing_index_points = nn.Parameter(init_ip, requires_grad=True)

        # Kernel
        if multi_kernel_mode:
            # Batched kernel across sample groups with flexible scale
            self.kernel = BatchedCauchyKernel_CONCERT_flex(
                scale=kernel_scale, fixed_scale=fixed_gp_params, phi=self.kernel_phi,
                dtype=dtype, device=device
            ).to(device)
        else:
            # Single (shared) kernel
            self.kernel = CauchyKernel(
                scale=kernel_scale, fixed_scale=fixed_gp_params, dtype=dtype, device=device
            ).to(device)

        self.sample_kernel = SampleKernel().to(device)

    # ---------------------------------------------------------------------
    # Kernel matrix helpers
    # ---------------------------------------------------------------------
    def kernel_matrix(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_inducing: bool = True,
        y_inducing: bool = True,
        diag_only: bool = False,
        cutoff: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """
        Compute K(x, y) for spatial coords (first 2 dims) with appended one-hot sample/batch dims.

        Inputs
        ------
        x, y : (N, 2 + B) and (M, 2 + B)
            First two columns are spatial positions; remaining columns are one-hot sample indicators.
        diag_only : bool
            If True, return only the diagonal (vector) of K(x, y) assuming x=y.
        cutoff : float or tensor
            Spatial mask cutoff per point (broadcastable).

        Returns
        -------
        K : (N, M) or (N,) if diag_only=True.
        """
        pos_x, sample_x = x[:, :2], x[:, 2:]
        pos_y, sample_y = y[:, :2], y[:, 2:]

        if self.multi_kernel_mode:
            if x_inducing and y_inducing:
                if diag_only:
                    return self.kernel.forward_diag_samples(pos_x, pos_y, sample_x, sample_y, cutoff=torch.tensor(0.0, device=self.device))
                return self.kernel.forward_samples(pos_x, pos_y, sample_x, sample_y, cutoff=torch.tensor(0.0, device=self.device))
            if not x_inducing and y_inducing:
                return self.kernel.forward_samples_points(pos_x, pos_y, sample_x, sample_y, cutoff=cutoff)
            # not x_inducing and not y_inducing
            if diag_only:
                return self.kernel.forward_diag_samples(pos_x, pos_y, sample_x, sample_y, cutoff=cutoff)
            return self.kernel.forward_samples(pos_x, pos_y, sample_x, sample_y, cutoff=cutoff)

        # Single-kernel mode
        if diag_only:
            return self.kernel.forward_diag(pos_x, pos_y)
        return self.kernel(pos_x, pos_y)

    def kernel_matrix_impute(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_inducing: bool = True,
        y_inducing: bool = True,
        diag_only: bool = False,
        x_cutoff: torch.Tensor | float = 0.0,
        y_cutoff: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """
        Compute K(x, y) for imputation, allowing different cutoffs for x and y domains.
        """
        pos_x, sample_x = x[:, :2], x[:, 2:]
        pos_y, sample_y = y[:, :2], y[:, 2:]

        if self.multi_kernel_mode:
            if x_inducing and y_inducing:
                if diag_only:
                    return self.kernel.forward_diag_samples(pos_x, pos_y, sample_x, sample_y, cutoff=torch.tensor(0.0, device=self.device))
                return self.kernel.forward_samples(pos_x, pos_y, sample_x, sample_y, cutoff=torch.tensor(0.0, device=self.device))
            if not x_inducing and y_inducing:
                return self.kernel.forward_samples_points(pos_x, pos_y, sample_x, sample_y, cutoff=x_cutoff)
            # not x_inducing and not y_inducing
            if diag_only:
                return self.kernel.forward_diag_samples_impute(
                    pos_x, pos_y, sample_x, sample_y, x_cutoff=x_cutoff, y_cutoff=y_cutoff
                )
            return self.kernel.forward_samples_impute(
                pos_x, pos_y, sample_x, sample_y, x_cutoff=x_cutoff, y_cutoff=y_cutoff
            )

        # Single-kernel mode
        if diag_only:
            return self.kernel.forward_diag(pos_x, pos_y)
        return self.kernel(pos_x, pos_y)

    # ---------------------------------------------------------------------
    # Variational terms
    # ---------------------------------------------------------------------
    def variational_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        noise: torch.Tensor,
        mu_hat: torch.Tensor,
        A_hat: torch.Tensor,
        cutoff: torch.Tensor | float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute variational loss terms (sum and KL) for a batch.

        Returns
        -------
        L_3_sum_term : torch.Tensor  (scalar)
        KL_term      : torch.Tensor  (scalar)
        """
        b = x.shape[0]
        m = self.inducing_index_points.shape[0]

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points, cutoff=cutoff)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))

        K_nn = self.kernel_matrix(x, x, x_inducing=False, y_inducing=False, diag_only=True, cutoff=cutoff)

        K_nm = self.kernel_matrix(x, self.inducing_index_points, x_inducing=False, cutoff=cutoff)
        K_mn = K_nm.transpose(0, 1)

        # KL(q(u) || p(u))
        K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
        S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))
        K_mm_log_det = 2.0 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))
        S_log_det = 2.0 * torch.sum(torch.log(torch.diagonal(S_chol)))

        KL_term = 0.5 * (
            K_mm_log_det - S_log_det - m
            + torch.trace(torch.matmul(K_mm_inv, A_hat))
            + torch.sum(mu_hat * torch.matmul(K_mm_inv, mu_hat))
        )

        # Precision & diag(K_tilde)
        precision = 1.0 / noise
        K_tilde_diag = K_nn - torch.diagonal(torch.matmul(K_nm, torch.matmul(K_mm_inv, K_mn)))

        # Lambda: (b, m, m)
        lambda_mat = torch.matmul(K_nm.unsqueeze(2), K_nm.unsqueeze(1))
        lambda_mat = torch.matmul(K_mm_inv, torch.matmul(lambda_mat, K_mm_inv))

        # Trace terms: (b,)
        trace_terms = precision * torch.einsum("bii->b", torch.matmul(A_hat, lambda_mat))

        # Mean vector for reconstruction piece
        mean_vector = torch.matmul(K_nm, torch.matmul(K_mm_inv, mu_hat))

        # L_3 sum term
        L_3_sum_term = -0.5 * (
            torch.sum(precision * K_tilde_diag)
            + torch.sum(trace_terms)
            + torch.sum(torch.log(noise))
            + b * np.log(2.0 * np.pi)
            + torch.sum(precision * (y - mean_vector) ** 2)
        )

        return L_3_sum_term, KL_term

    # ---------------------------------------------------------------------
    # Posterior parameter approximations
    # ---------------------------------------------------------------------
    def approximate_posterior_params(
        self,
        index_points_test: torch.Tensor,
        index_points_train: torch.Tensor,
        y: torch.Tensor,
        noise: torch.Tensor,
        cutoff: torch.Tensor | float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and diagonal variance for f_* at test points, and variational q(u) params.

        Returns
        -------
        mean_vector : (X, )
        B           : (X, ) diagonal of posterior covariance at test points
        mu_hat      : (m, ) variational mean of inducing outputs
        A_hat       : (m, m) variational covariance of inducing outputs
        """
        b = index_points_train.shape[0]

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points, cutoff=cutoff)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))

        K_xx = self.kernel_matrix(index_points_test, index_points_test, x_inducing=False, y_inducing=False, diag_only=True, cutoff=cutoff)
        K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points, x_inducing=False, cutoff=cutoff)
        K_mx = K_xm.transpose(0, 1)

        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points, x_inducing=False, cutoff=cutoff)
        K_mn = K_nm.transpose(0, 1)

        # Sigma_l and its inverse
        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, K_nm / noise[:, None])
        sigma_l_inv = torch.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))

        # Predictive mean at X_*
        mean_vector = (self.N_train / b) * torch.matmul(
            K_xm, torch.matmul(sigma_l_inv, torch.matmul(K_mn, y / noise))
        )

        # Predictive variance diag at X_*
        K_xm_Sigma_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        B = K_xx + torch.diagonal(-torch.matmul(K_xm, torch.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_K_mx)

        # q(u) params
        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mn)), y / noise)
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))

        # Diagnostics
        for name, tensor in {
            "K_xx": K_xx, "K_xm": K_xm, "K_mx": K_mx, "K_mm": K_mm, "K_mm_inv": K_mm_inv,
            "K_xm_Sigma_K_mx": K_xm_Sigma_K_mx, "B": B
        }.items():
            if torch.isnan(tensor).any():
                logging.warning("NaN detected in %s", name)
            if torch.isinf(tensor).any():
                logging.warning("Inf detected in %s", name)
        if (B <= 0).any():
            logging.warning("Posterior variance had non-positive entries; clamping may be required.")

        return mean_vector, B, mu_hat, A_hat

    def approximate_posterior_params_impute(
        self,
        index_points_test: torch.Tensor,
        index_points_train: torch.Tensor,
        y: torch.Tensor,
        noise: torch.Tensor,
        x_cutoff: torch.Tensor | float = 0.0,
        y_cutoff: torch.Tensor | float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Like `approximate_posterior_params` but supports different spatial cutoffs for train vs test domains.
        """
        b = index_points_train.shape[0]

        K_mm = self.kernel_matrix_impute(self.inducing_index_points, self.inducing_index_points, x_cutoff=x_cutoff, y_cutoff=y_cutoff)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))

        K_xx = self.kernel_matrix_impute(index_points_test, index_points_test, x_inducing=False, y_inducing=False, diag_only=True, x_cutoff=x_cutoff, y_cutoff=y_cutoff)
        K_xm = self.kernel_matrix_impute(index_points_test, self.inducing_index_points, x_inducing=False, x_cutoff=x_cutoff, y_cutoff=y_cutoff)
        K_mx = K_xm.transpose(0, 1)

        K_nm = self.kernel_matrix_impute(index_points_train, self.inducing_index_points, x_inducing=False, x_cutoff=x_cutoff, y_cutoff=y_cutoff)
        K_mn = K_nm.transpose(0, 1)

        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, K_nm / noise[:, None])
        sigma_l_inv = torch.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))

        mean_vector = (self.N_train / b) * torch.matmul(
            K_xm, torch.matmul(sigma_l_inv, torch.matmul(K_mn, y / noise))
        )

        K_xm_Sigma_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        B = K_xx + torch.diagonal(-torch.matmul(K_xm, torch.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_K_mx)

        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mn)), y / noise)
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))

        for name, tensor in {
            "K_xx": K_xx, "K_xm": K_xm, "K_mx": K_mx, "K_mm": K_mm, "K_mm_inv": K_mm_inv,
            "K_xm_Sigma_K_mx": K_xm_Sigma_K_mx, "B": B
        }.items():
            if torch.isnan(tensor).any():
                logging.warning("NaN detected in %s", name)
            if torch.isinf(tensor).any():
                logging.warning("Inf detected in %s", name)
        if (B <= 0).any():
            logging.warning("Posterior variance had non-positive entries (impute); clamping may be required.")

        return mean_vector, B, mu_hat, A_hat
