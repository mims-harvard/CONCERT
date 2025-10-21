# SVGP_Batch_3D_stable.py  (drop-in replacement for your SVGP 3D module)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kernel import (
    EQKernel, CauchyKernel, BatchedCauchyKernel,
    CauchyKernel3d, BatchedCauchyKernel3d, SampleKernel
)

# ----------------------------
# Helpers: robust numerics
# ----------------------------

def _add_diagonal_jitter(A: torch.Tensor, jitter: float) -> torch.Tensor:
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    return A + jitter * I

def _symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

def _make_spd(A: torch.Tensor, base_jitter: float = 1e-6, max_tries: int = 7):
    """
    Symmetrize A, then attempt Cholesky; on failure, escalate jitter.
    Fall back to eigen clamp only if all tries fail.
    Returns (A_repaired, chol(A_repaired)).
    """
    A = torch.nan_to_num(A)
    A = _symmetrize(A)
    jitter = float(base_jitter)

    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(_add_diagonal_jitter(A, jitter))
            return A, L
        except RuntimeError:
            jitter *= 10.0  # escalate

    # Rare fallback: eigen clamp
    evals, evecs = torch.linalg.eigh(A)  # eigh is okay once we escalated jitter a lot
    evals = torch.clamp(evals, min=1e-12)
    A = evecs @ torch.diag(evals) @ evecs.transpose(-1, -2)
    L = torch.linalg.cholesky(_add_diagonal_jitter(A, jitter))
    return A, L

def _safe_diag_of_quad(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    diag(A @ B) computed stably: einsum('ij,ji->i', A, B)
    A: (n, m), B: (m, n)
    returns: (n,)
    """
    return torch.einsum('ij,ji->i', A, B)

# ----------------------------
# SVGP (3D with batch gating)
# ----------------------------

class SVGP(nn.Module):
    """
    Sparse Variational GP with 3D (x,y,z) kernel + per-batch gating features.

    Inputs to kernel_matrix(x, y):
      x[:, :3] -> spatial coords
      x[:, 3:] -> one-hot batch gates (same for y)
    """

    def __init__(
        self,
        fixed_inducing_points,
        initial_inducing_points,
        fixed_gp_params,
        kernel_scale,
        allow_batch_kernel_scale,
        jitter,
        N_train,
        dtype,
        device,
    ):
        super().__init__()
        self.N_train = torch.tensor(float(N_train), dtype=dtype, device=device)
        self.allow_batch_kernel_scale = bool(allow_batch_kernel_scale)
        self.jitter = float(jitter)
        self.dtype = dtype
        self.device = device

        # Inducing points
        Z = torch.tensor(initial_inducing_points, dtype=dtype, device=device)
        if fixed_inducing_points:
            self.inducing_index_points = Z
        else:
            self.inducing_index_points = nn.Parameter(Z, requires_grad=True)

        # Kernels
        if self.allow_batch_kernel_scale:
            self.kernel = BatchedCauchyKernel3d(
                scale=kernel_scale,
                fixed_scale=fixed_gp_params,
                dtype=dtype,
                device=device,
            ).to(device)
        else:
            self.kernel = CauchyKernel3d(
                scale=kernel_scale,
                fixed_scale=fixed_gp_params,
                dtype=dtype,
                device=device,
            ).to(device)

        self.sample_kernel = SampleKernel().to(device)

    # ----------------------------
    # Kernels
    # ----------------------------
    def kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        K(X, Y) with gating: K = K_batch(sample_x, sample_y) * K_space(pos_x, pos_y[, batch])
        """
        pos_x = x[:, :3]
        pos_y = y[:, :3]
        sample_x = x[:, 3:]
        sample_y = y[:, 3:]

        if self.allow_batch_kernel_scale:
            if diag_only:
                k_batch = self.sample_kernel.forward_diag(sample_x, sample_y)            # (n,)
                k_space = self.kernel.forward_diag(pos_x, pos_y, sample_x, sample_y)     # (n,)
                K = k_batch * k_space
            else:
                K = self.sample_kernel(sample_x, sample_y) * self.kernel(pos_x, pos_y, sample_x, sample_y)
        else:
            if diag_only:
                k_batch = self.sample_kernel.forward_diag(sample_x, sample_y)            # (n,)
                k_space = self.kernel.forward_diag(pos_x, pos_y)                         # (n,)
                K = k_batch * k_space
            else:
                K = self.sample_kernel(sample_x, sample_y) * self.kernel(pos_x, pos_y)

        # Scrub NaNs/Infs defensively
        if diag_only:
            K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=0.0)
        else:
            K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=0.0)

        return K

    # ----------------------------
    # Variational lower bound terms
    # ----------------------------
    def variational_loss(self, x, y, noise, mu_hat, A_hat):
        """
        L3 (reconstruction-like) term and KL(q(u)||p(u)) for inducing variables.
        x: (b, 3 + B)
        y: (b,)
        noise: (b,)  -- MUST be positive
        mu_hat: (m,)
        A_hat: (m, m)  -- covariance for q(u)
        """
        b = x.shape[0]
        m = self.inducing_index_points.shape[0]

        # Clamp noise
        noise = torch.clamp(noise, min=1e-6)

        # K_mm
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)
        K_mm, K_mm_chol = _make_spd(K_mm, base_jitter=self.jitter)
        K_mm_inv = torch.cholesky_inverse(K_mm_chol)

        # Diagonal K_nn and cross covariances
        K_nn = self.kernel_matrix(x, x, x_inducing=False, y_inducing=False, diag_only=True)  # (b,)
        K_nm = self.kernel_matrix(x, self.inducing_index_points, x_inducing=False)           # (b, m)
        K_mn = K_nm.transpose(0, 1)                                                          # (m, b)

        # Repair A_hat (covariance of q(u))
        A_hat = _symmetrize(A_hat)
        A_hat, S_chol = _make_spd(A_hat, base_jitter=self.jitter)
        S = A_hat

        # log|K_mm| and log|S| from Cholesky
        K_mm_log_det = 2.0 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))
        S_log_det    = 2.0 * torch.sum(torch.log(torch.diagonal(S_chol)))

        # KL(q(u)||p(u))
        trace_val = torch.trace(K_mm_inv @ S)
        quad_mu   = mu_hat @ (K_mm_inv @ mu_hat)
        KL_term = 0.5 * (K_mm_log_det - S_log_det - m + trace_val + quad_mu)

        # Mean vector of f at data locations
        mean_vector = K_nm @ (K_mm_inv @ mu_hat)  # (b,)

        # Expected log likelihood (SVGP L3 decomposition)
        precision = 1.0 / noise  # (b,)

        # diag(K_nm @ K_mm_inv @ K_mn)
        diag_proj = _safe_diag_of_quad(K_nm, K_mm_inv @ K_mn)  # (b,)
        K_tilde_terms = precision * (K_nn - diag_proj)         # (b,)

        # trace term: tr(S K_mm_inv K_nm^T diag(precision) K_nm K_mm_inv)
        # Build middle Gram with weighting by precision
        G = K_nm.T @ (precision[:, None] * K_nm)               # (m, m)
        lambda_mat = K_mm_inv @ G @ K_mm_inv                   # (m, m)
        trace_terms = torch.einsum('ij,ji->', S, lambda_mat)   # scalar

        # data fit term
        resid = y - mean_vector
        data_term = precision * (resid ** 2)

        L_3_sum_term = -0.5 * (
            torch.sum(K_tilde_terms)
            + trace_terms
            + torch.sum(torch.log(noise))
            + b * np.log(2.0 * np.pi)
            + torch.sum(data_term)
        )

        # Gently clamp kernel scale to avoid runaway
        if hasattr(self.kernel, "scale"):
            with torch.no_grad():
                self.kernel.scale.data = torch.clamp(self.kernel.scale.data, min=1e-6, max=1e6)

        return L_3_sum_term, KL_term

    # ----------------------------
    # Posterior at test/train points (per GP dim)
    # ----------------------------
    def approximate_posterior_params(self, index_points_test, index_points_train=None, y=None, noise=None):
        """
        Returns:
          mean_vector: (n_test,)
          B:           (n_test,) predictive diagonal variance
          mu_hat:      (m,)
          A_hat:       (m, m)
        """
        if index_points_train is None:
            index_points_train = index_points_test
        assert y is not None and noise is not None, "y and noise must be provided"

        # Clamp noise
        noise = torch.clamp(noise, min=1e-6)

        b = index_points_train.shape[0]  # train batch size
        Z = self.inducing_index_points

        # K_mm (+ SPD)
        K_mm = self.kernel_matrix(Z, Z)
        K_mm, K_mm_chol = _make_spd(K_mm, base_jitter=self.jitter)
        K_mm_inv = torch.cholesky_inverse(K_mm_chol)

        # Train/Test covariances
        K_xx = self.kernel_matrix(index_points_test, index_points_test, x_inducing=False, y_inducing=False, diag_only=True)  # (n_test,)
        K_xm = self.kernel_matrix(index_points_test, Z, x_inducing=False)                                                    # (n_test, m)
        K_mx = K_xm.transpose(0, 1)                                                                                          # (m, n_test)

        K_nm = self.kernel_matrix(index_points_train, Z, x_inducing=False)                                                   # (b, m)
        K_mn = K_nm.transpose(0, 1)                                                                                          # (m, b)

        # Sigma_l = K_mm + (N_train / b) * K_mn @ (K_nm / noise[:,None])
        Sigma_l = K_mm + (self.N_train / b) * (K_mn @ (K_nm / noise[:, None]))
        Sigma_l, Sigma_l_chol = _make_spd(Sigma_l, base_jitter=self.jitter)
        Sigma_l_inv = torch.cholesky_inverse(Sigma_l_chol)

        # Predictive mean at X_test
        mean_vector = (self.N_train / b) * (K_xm @ (Sigma_l_inv @ (K_mn @ (y / noise))))

        # Predictive variance diagonal: K_xx - diag(K_xm K_mm^{-1} K_mx) + diag(K_xm Sigma_l^{-1} K_mx)
        diag1 = _safe_diag_of_quad(K_xm, K_mm_inv @ K_mx)
        diag2 = _safe_diag_of_quad(K_xm, Sigma_l_inv @ K_mx)
        B = torch.clamp(K_xx - diag1 + diag2, min=1e-6)

        # Variational parameters of q(u) = N(mu_hat, A_hat)
        mu_hat = (self.N_train / b) * (K_mm @ (Sigma_l_inv @ (K_mn @ (y / noise))))
        A_hat  = _symmetrize(K_mm @ (Sigma_l_inv @ K_mm))  # SPD by construction (up to numerics)

        return mean_vector, B, mu_hat, A_hat