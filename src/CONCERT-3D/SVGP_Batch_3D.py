import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from kernel import EQKernel, CauchyKernel, BatchedCauchyKernel, CauchyKernel3d, BatchedCauchyKernel3d, SampleKernel

def _add_diagonal_jitter(matrix, jitter=1e-8):
    Eye = torch.eye(matrix.size(-1), device=matrix.device).expand(matrix.shape)
    return matrix + jitter * Eye

def _enforce_symmetry(A_hat, vec):
    is_symmetric = torch.allclose(A_hat, A_hat.T, atol=1e-8)  
    if not is_symmetric:
        A_hat = (A_hat + A_hat.T) / 2  # Ensure symmetry
    return A_hat

def _enforse_positive_definite(A_hat, vec):
    eigenvalues, eigenvectors = torch.linalg.eigh(A_hat)
    if eigenvalues.min().item() < 1e-6:
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        A_hat = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return A_hat

def _check_conditioing(A_hat, vec, jitter=1e-6):
    det_A_hat = torch.linalg.det(A_hat)
    cond_A_hat = torch.linalg.cond(A_hat)
    if det_A_hat < 1e-10 or cond_A_hat > 1e10:
        A_hat = _add_diagonal_jitter(A_hat, jitter)
    return A_hat

class SVGP(nn.Module):
    def __init__(self, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, allow_batch_kernel_scale, jitter, N_train, dtype, device):
        super(SVGP, self).__init__()
        self.N_train = torch.tensor(N_train, dtype=dtype).to(device)
        self.allow_batch_kernel_scale = allow_batch_kernel_scale
        self.jitter = jitter
        self.dtype = dtype
        self.device = device

        # inducing points
        if fixed_inducing_points:
            self.inducing_index_points = torch.tensor(initial_inducing_points, dtype=dtype).to(device)
        else:
            self.inducing_index_points = nn.Parameter(torch.tensor(initial_inducing_points, dtype=dtype).to(device), requires_grad=True)

        # length scale of the kernel
        if allow_batch_kernel_scale:
            self.kernel = BatchedCauchyKernel3d(scale=kernel_scale, fixed_scale=fixed_gp_params, dtype=dtype, device=device).to(device)
        else:
            self.kernel = CauchyKernel3d(scale=kernel_scale, fixed_scale=fixed_gp_params, dtype=dtype, device=device).to(device)
        self.sample_kernel = SampleKernel().to(device)

    def kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y,z).
        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """
        pos_x = x[:, :3]
        pos_y = y[:, :3]

        sample_x = x[:, 3:]
        sample_y = y[:, 3:]

        if self.allow_batch_kernel_scale:
            if diag_only:
                matrix_diag_1 = self.sample_kernel.forward_diag(sample_x, sample_y)
                matrix_diag_2 = self.kernel.forward_diag(pos_x, pos_y, sample_x, sample_y)
                matrix = matrix_diag_1 * matrix_diag_2
            else:
                matrix = self.sample_kernel(sample_x, sample_y) * self.kernel(pos_x, pos_y, sample_x, sample_y)
        else:
            if diag_only:
                matrix_diag_1 = self.sample_kernel.forward_diag(sample_x, sample_y)
                matrix_diag_2 = self.kernel.forward_diag(pos_x, pos_y)
                matrix = matrix_diag_1 * matrix_diag_2
            else:
                matrix = self.sample_kernel(sample_x, sample_y) * self.kernel(pos_x, pos_y)
        return matrix

    def variational_loss(self, x, y, noise, mu_hat, A_hat):
        b = x.shape[0]
        m = self.inducing_index_points.shape[0]

        # Compute Kernel Matrices
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)
        K_mm = _enforce_symmetry(K_mm, "K_mm")
        K_mm = _enforse_positive_definite(K_mm, "K_mm")
        K_mm = _check_conditioing(K_mm, "K_mm", jitter=self.jitter)

        try:
            K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
            K_mm_inv = torch.cholesky_inverse(K_mm_chol)  # More numerically stable than direct inversion
        except RuntimeError:
            print("Warning: K_mm Cholesky failed, increasing jitter.")
            K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter * 10))
            K_mm_inv = torch.cholesky_inverse(K_mm_chol)

        K_nn = self.kernel_matrix(x, x, x_inducing=False, y_inducing=False, diag_only=True)
        K_nm = self.kernel_matrix(x, self.inducing_index_points, x_inducing=False)
        K_mn = torch.transpose(K_nm, 0, 1)

        # Enforce Symmetry and Positive Definiteness
        A_hat = _enforce_symmetry(A_hat, "A_hat")
        A_hat = _enforse_positive_definite(A_hat, "A_hat")
        A_hat = _check_conditioing(A_hat, "A_hat", jitter=self.jitter)

        S = A_hat

        # Compute Mean Vector
        mean_vector = torch.matmul(K_nm, torch.matmul(K_mm_inv, mu_hat))

        # Cholesky Decomposition Stability
        try:
            S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))
        except RuntimeError:
            print("Warning: Increasing jitter for A_hat Cholesky")
            S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter * 10))
        
        # Log Determinants with Clamping
        K_mm_log_det = 2 * torch.sum(torch.log(torch.clamp(torch.diagonal(K_mm_chol), min=1e-6)))
        S_log_det = 2 * torch.sum(torch.log(torch.clamp(torch.diagonal(S_chol), min=1e-6)))

        # Clamp trace value
        trace_val = torch.trace(torch.matmul(K_mm_inv, A_hat))
        trace_val = torch.clamp(trace_val, min=-1e6, max=1e6)
        
        KL_term = 0.5 * (K_mm_log_det - S_log_det - m + trace_val +
                     torch.sum(torch.clamp(mu_hat * torch.matmul(K_mm_inv, mu_hat), min=-1e6, max=1e6)))

        precision = 1 / torch.clamp(noise, min=1e-6)
        K_tilde_terms = precision * (K_nn - torch.diagonal(torch.matmul(K_nm, torch.matmul(K_mm_inv, K_mn))))
        lambda_mat = torch.matmul(K_nm.unsqueeze(2), torch.transpose(K_nm.unsqueeze(2), 1, 2))
        lambda_mat = torch.matmul(K_mm_inv, torch.matmul(lambda_mat, K_mm_inv))
        trace_terms = precision * torch.einsum('bii->b', torch.matmul(S, lambda_mat))
        L_3_sum_term = -0.5 * (torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                           torch.sum(torch.log(torch.clamp(noise, min=1e-6))) + b * np.log(2 * np.pi) +
                           torch.sum(precision * (y - mean_vector) ** 2))
        
        if hasattr(self.kernel, "scale"):
            self.kernel.scale.data = torch.clamp(self.kernel.scale.data, min=1e-6, max=1e6)

        return L_3_sum_term, KL_term

    def approximate_posterior_params(self, index_points_test, index_points_train=None, y=None, noise=None):
        """
        Computes parameters of q_S for the 3D kernel with numerical stability fixes.
    
        :param index_points_test: X_* (test locations)
        :param index_points_train: X_Train (training locations)
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP
        :return: posterior mean at index points,
             (diagonal of) posterior covariance matrix at index points
        """

        b = index_points_train.shape[0]  # Batch size
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m, m)
        K_mm = _enforce_symmetry(K_mm, "K_mm")
        K_mm = _enforse_positive_definite(K_mm, "K_mm")
        K_mm = _check_conditioing(K_mm, "K_mm", jitter=self.jitter)

        try:
            K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
            K_mm_inv = torch.cholesky_inverse(K_mm_chol)
        except RuntimeError:
            print("Warning: K_mm Cholesky failed, increasing jitter.")
            K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter * 10))
            K_mm_inv = torch.cholesky_inverse(K_mm_chol)

        K_xx = self.kernel_matrix(index_points_test, index_points_test, x_inducing=False, y_inducing=False, diag_only=True)  # (x)
        K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points, x_inducing=False)  # (x, m)
        K_mx = torch.transpose(K_xm, 0, 1)  # (m, x)

        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points, x_inducing=False)  # (N, m)
        K_mn = torch.transpose(K_nm, 0, 1)  # (m, N)

        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, K_nm / noise[:, None])

        sigma_l = _enforce_symmetry(sigma_l, "sigma_l")
        sigma_l = _enforse_positive_definite(sigma_l, "sigma_l")
        sigma_l = _check_conditioing(sigma_l, "sigma_l", jitter=self.jitter)

        sigma_l_chol = torch.linalg.cholesky(_add_diagonal_jitter(sigma_l, self.jitter * 100))
        sigma_l_inv = torch.cholesky_inverse(sigma_l_chol)
        
        mean_vector = (self.N_train / b) * torch.matmul(K_xm, torch.matmul(sigma_l_inv, torch.matmul(K_mn, y / noise)))
        K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        B = torch.clamp(
            K_xx + torch.diagonal(-torch.matmul(K_xm, torch.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_l_K_mx),
            min=1e-6
        )

        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mn)), y / noise)
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))
        
        return mean_vector, B, mu_hat, A_hat

