import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import os


def sq_dist(x1, x2, x1_eq_x2=False):
    """Equivalent to the square of `torch.cdist` with p=2."""
    # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

    # Zero out negative values
    return res.clamp_min_(0)


def dist(x1, x2, x1_eq_x2=False):
    """
    Equivalent to `torch.cdist` with p=2, but clamps the minimum element to 1e-15.
    """
    if not x1_eq_x2:
        res = torch.cdist(x1, x2)
        return res.clamp_min(1e-15)
    res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
    return res.clamp_min_(1e-30).sqrt_()


class MaternKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, nu=1.5, dtype=torch.float32, device="cpu"):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernel, self).__init__()
        self.nu = nu
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        mean = x.mean(dim=-2, keepdim=True)

        x_ = (x - mean).div(self.scale)
        y_ = (y - mean).div(self.scale)
        distance = dist(x_, y_)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return constant_component * exp_component

    def forward_diag(self, x, y):
        mean = x.mean(dim=-2, keepdim=True)

        x_ = (x - mean).div(self.scale)
        y_ = (y - mean).div(self.scale)
        distance = ((x_-y_)**2).sum(dim=1)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return constant_component * exp_component


class MultiMaternKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, nu=1.5, dim=1, dtype=torch.float32, device="cpu"):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MultiMaternKernel, self).__init__()
        self.nu = nu
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(np.repeat(scale, dim), dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y, l):
        mean = x.mean(dim=-2, keepdim=True)

        x_ = (x - mean).div(self.scale[l])
        y_ = (y - mean).div(self.scale[l])
        distance = dist(x_, y_)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return constant_component * exp_component

    def forward_diag(self, x, y, l):
        mean = x.mean(dim=-2, keepdim=True)

        x_ = (x - mean).div(self.scale[l])
        y_ = (y - mean).div(self.scale[l])
        distance = ((x_-y_)**2).sum(dim=1)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return constant_component * exp_component


class EQKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(EQKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = torch.exp(-d/self.scale)
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = torch.exp(-d/self.scale)
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)


class MultiEQKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dim=1, dtype=torch.float32, device="cpu"):
        super(MultiEQKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(np.repeat(scale, dim), dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y, l):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = torch.exp(-d/self.scale[l])
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale[l]), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y, l):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = torch.exp(-d/self.scale[l])
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale[l]), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)



class CauchyKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(CauchyKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = 1/(1+d/self.scale)
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = 1/(1+d/self.scale)
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)

class CauchyKernel3d(nn.Module):
    def __init__(self, scale=1.0, fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(CauchyKernel3d, self).__init__()
        self.fixed_scale = fixed_scale

        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        """
        x: Tensor of shape (batch_size, num_points_x, 3) -> 3D coordinates (x, y, z)
        y: Tensor of shape (batch_size, num_points_y, 3) -> 3D coordinates (x, y, z)
        """

        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)  # Shape: (batch_size, num_points_x, 1)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)  # Shape: (batch_size, 1, num_points_y)

        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))  # Shape: (batch_size, num_points_x, num_points_y)

        d = torch.clamp(sq_norms_x + sq_norms_y - 2.0 * dotprods, min=1e-10, max=1e6)  # Ensure non-negative

        if self.fixed_scale:
            scale = torch.clamp(self.scale, min=1e-6, max=1e6)  # Ensure stability
        else:
            scale = torch.clamp(F.softplus(self.scale), min=1e-6, max=1e6)  # Ensure learnable scale is positive

        res = 1 / (1 + d / scale)

        if not self.fixed_scale:
            self.scale.data = torch.clamp(self.scale.data, min=1e-6, max=1e6)

        return res

    def forward_diag(self, x, y):
        """
        Compute the diagonal kernel values (self-similarity between x and y).
        x: Tensor of shape (batch_size, num_points, 3)
        y: Tensor of shape (batch_size, num_points, 3)
        """
        # Squared Euclidean distance between corresponding points
        d = ((x - y) ** 2).sum(dim=-1)  # Shape: (batch_size, num_points)

        if self.fixed_scale:
            res = 1 / (1 + d / self.scale)
        else:
            scale = torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4)
            res = 1 / (1 + d / scale)

        return res

    def print_scale(self):
        print(self.scale)


class MultiCauchyKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dim=1, dtype=torch.float32, device="cpu"):
        super(MultiCauchyKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(np.repeat(scale, dim), dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(np.repeat(scale, dim), dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y, l):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = 1/(1+d/self.scale[l])
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale[l]), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y, l):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = 1/(1+d/self.scale[l])
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale[l]), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)


class LaplacianKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(LaplacianKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        d = (x.unsqueeze(1) - y.unsqueeze(0).repeat(x.shape[0], 1, 1)).abs().sum(dim=-1)
        if self.fixed_scale:
            res = torch.exp(-d/self.scale)
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y):
        d = ((x-y).abs()).sum(dim=1)
        if self.fixed_scale:
            res = torch.exp(-d/self.scale)
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)


class BatchedCauchyKernel(nn.Module):
    def __init__(self, scale=[], fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(BatchedCauchyKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y, sample_x, sample_y):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            scale_x = torch.matmul(sample_x, self.scale.unsqueeze(dim=1))
            scale_y = torch.matmul(sample_y, self.scale.unsqueeze(dim=1))
            scale_xy = torch.sqrt(torch.matmul(scale_x, scale_y.T))
            res = 1/(1+d/scale_xy)
        else:
            scale_x = torch.clamp(F.softplus(torch.matmul(sample_x, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4)
            scale_y = torch.clamp(F.softplus(torch.matmul(sample_y, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4)
            scale_xy = torch.sqrt(torch.matmul(scale_x, scale_y.T))
            res = 1/(1+d/scale_xy)
        return res

    def forward_diag(self, x, y, sample_x, sample_y):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            scale_x = torch.matmul(sample_x, self.scale.unsqueeze(dim=1)).squeeze()
            scale_y = torch.matmul(sample_y, self.scale.unsqueeze(dim=1)).squeeze()
            scale_xy = torch.sqrt(scale_x * scale_y)
            res = 1/(1+d/scale_xy)
        else:
            scale_x = torch.clamp(F.softplus(torch.matmul(sample_x, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4).squeeze()
            scale_y = torch.clamp(F.softplus(torch.matmul(sample_y, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4).squeeze()
            scale_xy = torch.sqrt(scale_x * scale_y)
            res = 1/(1+d/scale_xy)
        return res

    def print_scale(self):
        print(self.scale)

class BatchedCauchyKernel_CONCERT(nn.Module):
    def __init__(self, scale=[], fixed_scale=False, dtype=torch.float32, cutoff=[], phi=1., device="cpu"):
        super(BatchedCauchyKernel_CONCERT, self).__init__()
        self.fixed_scale = fixed_scale
        self.device = device
        self.dtype = dtype
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype).to(device)
            self.scale0 = self.scale[0]
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=dtype).to(device), requires_grad=True)
            self.scale0 = self.scale[0]
        self.cutoff = nn.Parameter(torch.tensor([cutoff], dtype=dtype), requires_grad=True)
        self.cutoff = torch.tensor([cutoff], dtype=dtype)
        self.phi = torch.tensor(phi, dtype=dtype)

    def forward_samples(self, x, y, sample_x, sample_y):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            scale_x = torch.matmul(sample_x, self.scale.unsqueeze(dim=1))
            scale_y = torch.matmul(sample_y, self.scale.unsqueeze(dim=1))
            scale_xy = torch.sqrt(torch.matmul(scale_x, scale_y.T))
            res = 1/(1+d/scale_xy)
        else:
            scale_x = torch.clamp(F.softplus(torch.matmul(sample_x, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4)
            scale_y = torch.clamp(F.softplus(torch.matmul(sample_y, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4)
            scale_xy = torch.sqrt(torch.matmul(scale_x, scale_y.T))
            res = 1/(1+d/scale_xy)
        if self.cutoff.mean() > 0:
            cut_mask = torch.sigmoid(self.phi * (res - self.cutoff.clamp(min=0., max=1e3)))
            res = res * cut_mask
        return res

    def forward_diag_samples(self, x, y, sample_x, sample_y):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            scale_x = torch.matmul(sample_x, self.scale.unsqueeze(dim=1)).squeeze()
            scale_y = torch.matmul(sample_y, self.scale.unsqueeze(dim=1)).squeeze()
            scale_xy = torch.sqrt(scale_x * scale_y)
            res = 1/(1+d/scale_xy)
        else:
            scale_x = torch.clamp(F.softplus(torch.matmul(sample_x, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4).squeeze()
            scale_y = torch.clamp(F.softplus(torch.matmul(sample_y, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4).squeeze()
            scale_xy = torch.sqrt(scale_x * scale_y)
            res = 1/(1+d/scale_xy)
        if self.cutoff.mean() > 0:
            cut_mask = torch.sigmoid(self.phi * (res - self.cutoff.clamp(min=0., max=1e3)))
            res = res * cut_mask
        return res
    
    def print_scale(self):
        print(self.scale)

class BatchedCauchyKernel_CONCERT_flex(nn.Module):
    def __init__(self, scale=[], fixed_scale=False, dtype=torch.float32, phi=1., device="cpu"):
        super(BatchedCauchyKernel_CONCERT_flex, self).__init__()
        self.fixed_scale = fixed_scale
        self.device = device
        self.dtype = dtype
        self.phi = torch.tensor(phi, dtype=dtype).to(device)
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype).to(device) #[B,D]
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=dtype).to(device),requires_grad=True)
        #print("scale: ", self.scale)
        # TI_KERNEL=1 -> translation-invariant: scale the DIFFERENCE (geom-mean of per-point
        # inverse scales) instead of scaling x,y separately. Fixes the multi-slide tiled-coord
        # blow-up when s_x != s_y. Default OFF (legacy behaviour) so existing runs are unchanged.
        self._ti = os.environ.get("TI_KERNEL", "0") == "1"

    def forward_samples(self, x, y, sample_x, sample_y, cutoff):
        scale = self.scale.clamp(min=1e-6, max=1e6)
        if torch.isnan(scale).any():
            print("forward_samples nan in scale")
        scale_x = sample_x @ scale  # (N, B) @ (B, D) = (N_x, D)
        scale_y = sample_y @ scale
        if self._ti:
            inv_pair = torch.sqrt((1.0/scale_x.clamp(min=1e-6))[:, None, :]
                                  * (1.0/scale_y.clamp(min=1e-6))[None, :, :])  # (N_x,N_y,D)
            diff = x[:, None, :] - y[None, :, :]
            d = torch.sum(diff**2 * inv_pair, dim=-1).clamp(min=1e-6)
        else:
            inv_sqrt_scale_x = (1.0 / torch.sqrt(scale_x.clamp(min=1e-6)))  # (N_x, D)
            inv_sqrt_scale_y = (1.0 / torch.sqrt(scale_y.clamp(min=1e-6)))  # (N_y, D)
            x_scaled = x * inv_sqrt_scale_x  # (N_x, D)
            y_scaled = y * inv_sqrt_scale_y  # (N_y, D)
            diff = x_scaled[:, None, :] - y_scaled[None, :, :]  # (N_x, N_y, D)
            d = torch.sum(diff**2, dim=-1).clamp(min=1e-6)
        res = 1 / (1 + d)
        if cutoff.mean() > 0:
            cutoff = cutoff.clamp(min=0.1, max=0.9)
            cutoff_matrix = (cutoff + cutoff.T) / 2.0  # (N_x, N_y)
            cut_mask = torch.sigmoid((self.phi * (res-cutoff_matrix).clamp(-1,1))).clamp(min=0.01)
            res = res * cut_mask
        return res

    def forward_diag_samples(self, x, y, sample_x, sample_y, cutoff):
        scale = self.scale.clamp(min=1e-6, max=1e6)
        if torch.isnan(scale).any():
            print("forward_diag_samples nan in scale")
        scale_x = sample_x @ scale  # (N_x, D)
        scale_y = sample_y @ scale
        if self._ti:
            inv_pair = torch.sqrt((1.0/scale_x.clamp(min=1e-6)) * (1.0/scale_y.clamp(min=1e-6)))  # (N,D)
            diff = x - y
            d = torch.sqrt((diff**2 * inv_pair).sum(1).clamp(min=1e-12)).clamp(min=1e-6)
        else:
            inv_sqrt_scale_x = (1.0 / torch.sqrt(scale_x.clamp(min=1e-6)))  # (N_x, D)
            inv_sqrt_scale_y = (1.0 / torch.sqrt(scale_y.clamp(min=1e-6)))  # (N_y, D)
            x_scaled = x * inv_sqrt_scale_x  # (N_x, D)
            y_scaled = y * inv_sqrt_scale_y  # (N_y, D)
            d = torch.norm(x_scaled - y_scaled, dim=1).clamp(min=1e-6)
        res = 1 / (1 + d)
        if cutoff.mean() > 0:
            cutoff = cutoff.clamp(min=0.1, max=0.9)
            cutoff_matrix = (cutoff + cutoff.T) / 2.0  # (N_x, N_y)
            cut_mask = torch.sigmoid((self.phi * (res-cutoff_matrix).clamp(-1,1))).clamp(min=0.01)
            res = res * cut_mask
        return res
    
    def forward_samples_points(self,  x, y, sample_x, sample_y, cutoff):
        scale = self.scale.clamp(min=1e-6, max=1e6)
        if torch.isnan(scale).any():
            print("forward_samples nan in scale")
        scale_x = sample_x @ scale  # (N, B) @ (B, D) = (N_x, D)
        scale_y = sample_y @ scale
        if self._ti:
            inv_pair = torch.sqrt((1.0/scale_x.clamp(min=1e-6))[:, None, :]
                                  * (1.0/scale_y.clamp(min=1e-6))[None, :, :])  # (N_x,N_y,D)
            diff = x[:, None, :] - y[None, :, :]
            d = torch.sum(diff**2 * inv_pair, dim=-1).clamp(min=1e-6)
        else:
            inv_sqrt_scale_x = (1.0 / torch.sqrt(scale_x.clamp(min=1e-6)))  # (N_x, D)
            inv_sqrt_scale_y = (1.0 / torch.sqrt(scale_y.clamp(min=1e-6)))  # (N_y, D)
            x_scaled = x * inv_sqrt_scale_x  # (N_x, D)
            y_scaled = y * inv_sqrt_scale_y  # (N_y, D)
            diff = x_scaled[:, None, :] - y_scaled[None, :, :]  # (N_x, N_y, D)
            d = torch.sum(diff**2, dim=-1).clamp(min=1e-6)
        res = 1 / (1 + d)
        if cutoff.mean() > 0:
            cutoff = cutoff.clamp(min=0.1, max=0.9)
            cutoff0 = torch.zeros(y.shape[0], dtype=self.dtype).to(self.device)
            cutoff_matrix = (cutoff.view(x.shape[0],1) + cutoff0.view(1,y.shape[0])) / 2.0  # (N_x, N_y)
            cut_mask = torch.sigmoid((self.phi * (res-cutoff_matrix).clamp(-1,1))).clamp(min=0.01)
            res = res * cut_mask
        return res
    
    def forward_diag_samples_impute(self, x, y, sample_x, sample_y, x_cutoff, y_cutoff):
        # Stable positive scales (B, D)
        scale = self.scale.clamp(min=1e-6, max=1e6)

        # Project per-sample batch-weights into per-dimension scales
        # (N, B) @ (B, D) -> (N, D)
        scale_x = sample_x @ scale
        scale_y = sample_y @ scale

        inv_sqrt_scale_x = (1.0 / torch.sqrt(scale_x.clamp(min=1e-6)))
        inv_sqrt_scale_y = (1.0 / torch.sqrt(scale_y.clamp(min=1e-6)))

        # Per-dimension scaling
        x_scaled = x * inv_sqrt_scale_x
        y_scaled = y * inv_sqrt_scale_y

        # Diagonal distance and Cauchy kernel
        d = torch.norm(x_scaled - y_scaled, dim=1).clamp(min=1e-6)   # (N,)
        res = 1.0 / (1.0 + d)                                       # (N,)

        # Per-pair cutoff mask using separate x/y cutoffs
        if (x_cutoff is not None) and (y_cutoff is not None):
            x_cutoff = x_cutoff.view(-1, 1)
            y_cutoff = y_cutoff.view(1, -1)
            x_c = x_cutoff.to(res.dtype).to(res.device).clamp(min=0.1, max=0.9)
            y_c = y_cutoff.to(res.dtype).to(res.device).clamp(min=0.1, max=0.9)
            pair_cut = torch.diagonal((x_c + y_c) * 0.5)   # (N,)
            # Smooth mask; clip argument to avoid extreme logits
            cut_mask = torch.sigmoid((self.phi * (res - pair_cut).clamp(-1.0, 1.0))).clamp(min=0.01)
            res = res * cut_mask

        return res
    
    def forward_samples_impute(self, x, y, sample_x, sample_y, x_cutoff, y_cutoff):
        # Stable positive scales (B, D)
        scale = self.scale.clamp(min=1e-6, max=1e6)

        # Project per-sample batch-weights into per-dimension scales
        # (N_x, B) @ (B, D) -> (N_x, D), (N_y, B) @ (B, D) -> (N_y, D)
        scale_x = sample_x @ scale
        scale_y = sample_y @ scale

        inv_sqrt_scale_x = 1.0 / torch.sqrt(scale_x.clamp(min=1e-6))
        inv_sqrt_scale_y = 1.0 / torch.sqrt(scale_y.clamp(min=1e-6))

        # Per-dimension scaling
        x_scaled = x * inv_sqrt_scale_x                     # (N_x, D)
        y_scaled = y * inv_sqrt_scale_y                     # (N_y, D)

        # Pairwise squared distances in scaled space and Cauchy kernel
        diff = x_scaled[:, None, :] - y_scaled[None, :, :]  # (N_x, N_y, D)
        d2 = torch.sum(diff ** 2, dim=-1).clamp(min=1e-6)   # (N_x, N_y)
        res = 1.0 / (1.0 + d2)                              # (N_x, N_y)

        # Cutoff mask using separate vectors for X and Y: pair_cut = (x_i + y_j)/2
        if (x_cutoff is not None) and (y_cutoff is not None):
            x_cutoff = x_cutoff.view(-1, 1)
            y_cutoff = y_cutoff.view(1, -1)
            x_c = x_cutoff.to(res.dtype).to(res.device).clamp(min=0.1, max=0.9).view(-1, 1)  # (N_x, 1)
            y_c = y_cutoff.to(res.dtype).to(res.device).clamp(min=0.1, max=0.9).view(1, -1)  # (1, N_y)
            pair_cut = 0.5 * (x_c + y_c)                                                     # (N_x, N_y)
            # Smooth gating; clip logits for numerical stability
            cut_mask = torch.sigmoid((self.phi * (res - pair_cut).clamp(-1.0, 1.0))).clamp(min=0.01)
            res = res * cut_mask

        return res

    def print_params(self):
        print(f"Scale Parameters: {self.scale}")
        print(f"Scale Parameters: {self.cutoff}")

class BatchedCauchyKernel3d(nn.Module):
    def __init__(self, scale=[], fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(BatchedCauchyKernel3d, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype, device=device)
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=dtype, device=device), requires_grad=True)

    def forward(self, x, y, sample_x, sample_y):
        """
        x: Tensor of shape (batch_size, num_points_x, 3)
        y: Tensor of shape (batch_size, num_points_y, 3)
        sample_x: Tensor of shape (batch_size, num_points_x, feature_dim)
        sample_y: Tensor of shape (batch_size, num_points_y, feature_dim)
        Returns:
          res: Kernel matrix of shape (batch_size, num_points_x, num_points_y)
        """

        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)

        dotprods = torch.matmul(x, y.transpose(-2, -1))

        d = torch.clamp(sq_norms_x + sq_norms_y - 2.0 * dotprods, min=1e-10, max=1e6)

        if self.fixed_scale:
            scale = torch.clamp(self.scale, min=1e-6, max=1e6)
            scale_x = torch.matmul(sample_x, scale.unsqueeze(-1))  # (batch_size, num_points_x, 1)
            scale_y = torch.matmul(sample_y, scale.unsqueeze(-1))  # (batch_size, num_points_y, 1)
        else:
            raw_scale = torch.clamp(F.softplus(self.scale), min=1e-6, max=1e6)
            scale_x = torch.matmul(sample_x, raw_scale.unsqueeze(-1))  # (batch_size, num_points_x, 1)
            scale_y = torch.matmul(sample_y, raw_scale.unsqueeze(-1))  # (batch_size, num_points_y, 1)

        scale_x = torch.clamp(scale_x, min=1e-10, max=1e6)
        scale_y = torch.clamp(scale_y, min=1e-10, max=1e6)

        scale_xy = torch.clamp(torch.matmul(scale_x, scale_y.transpose(-2, -1)), min=1e-10, max=1e12)
        scale_xy = torch.sqrt(scale_xy)  # shape: (batch_size, num_points_x, num_points_y)

        res = 1.0 / (1.0 + d / scale_xy)

        return res

    def forward_diag(self, x, y, sample_x, sample_y):
        """
        Compute the diagonal kernel values (self-similarity between x and y).
        """
        d = ((x - y) ** 2).sum(dim=-1)  # Squared Euclidean distance, Shape: (batch_size, num_points)

        if self.fixed_scale:
            scale_x = torch.matmul(sample_x, self.scale.unsqueeze(dim=1)).squeeze(-1)
            scale_y = torch.matmul(sample_y, self.scale.unsqueeze(dim=1)).squeeze(-1)
            scale_xy = torch.sqrt(scale_x * scale_y)
            res = 1 / (1 + d / scale_xy)
        else:
            scale_x = torch.clamp(F.softplus(torch.matmul(sample_x, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4).squeeze(-1)
            scale_y = torch.clamp(F.softplus(torch.matmul(sample_y, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4).squeeze(-1)
            scale_xy = torch.sqrt(scale_x * scale_y)
            res = 1 / (1 + d / scale_xy)

        return res

    def print_scale(self):
        print(self.scale)


class SampleKernel(nn.Module):
    def __init__(self):
        super(SampleKernel, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y.T)

    def forward_diag(self, x, y):
        return (x*y).sum(dim=1)


# ============================================================================
# Phase 2.5: Mahalanobis Cauchy Kernel (full rotational anisotropy)
# Phase 2.4: Embedding-conditioned kernel scale
# ============================================================================

class MahalanobisCauchyKernel(nn.Module):
    """Per-perturbation anisotropic Cauchy kernel via Cholesky metric tensor.

    Phase 2.5: Replaces per-perturbation diagonal scale with a full 2×2
    positive-definite metric tensor L_i @ L_i^T per perturbation, where L_i
    is a lower-triangular Cholesky factor. This allows the kernel to rotate
    to align with tissue structure (e.g., gland boundaries, tumor-stroma
    interfaces).

    Distance: d(x, y; i) = (x - y)^T M_i (x - y),  M_i = L_i L_i^T (PD)
    Kernel:   k(x, y; i) = 1 / (1 + d(x, y; i))

    Parameters
    ----------
    n_perts   : int  -- number of perturbations (= n_batch)
    d_pos     : int  -- spatial dimensionality (2 for Visium HD)
    phi       : float -- sharpness of soft cutoff gate
    dtype, device : standard

    Usage (forward_samples interface matches BatchedCauchyKernel_CONCERT_flex)
    -----
    kernel = MahalanobisCauchyKernel(n_perts=537, d_pos=2)
    K = kernel.forward_samples(x, y, sample_x, sample_y, cutoff)
    """

    def __init__(
        self,
        n_perts: int,
        d_pos: int = 2,
        phi: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.n_perts = n_perts
        self.d_pos   = d_pos
        self.phi     = torch.tensor(phi, dtype=dtype, device=device)

        # Cholesky factors: lower-triangular elements of L_i
        # For 2×2: 3 elements per pert = [L00, L10, L11]
        # L_i = [[L00_i, 0], [L10_i, L11_i]]
        # M_i = L_i @ L_i^T is guaranteed PD (diagonal elements forced positive)
        n_chol = d_pos * (d_pos + 1) // 2  # lower-triangular elements
        # Initialize near identity: diagonal ~ 1, off-diagonal ~ 0
        init = torch.zeros(n_perts, n_chol, dtype=dtype, device=device)
        # diagonal log = 0 -> exp(0) = 1 -> L diagonal = 1 -> M ≈ I
        self.chol_params = nn.Parameter(init, requires_grad=True)

    def _build_metric(self, sample_weights: torch.Tensor) -> torch.Tensor:
        """Build per-cell metric matrix M from Cholesky params.

        Parameters
        ----------
        sample_weights : (N, n_perts) float -- perturbation one-hot (or soft weights)

        Returns
        -------
        M : (N, d_pos, d_pos) float -- PD metric tensor per cell
        """
        N = sample_weights.shape[0]
        d = self.d_pos

        # Per-cell Cholesky params via weighted average (or one-hot lookup)
        # chol_cell: (N, n_chol)
        chol_cell = sample_weights @ self.chol_params  # (N, n_chol)

        # Assemble lower-triangular matrix L for each cell
        L = torch.zeros(N, d, d, dtype=chol_cell.dtype, device=chol_cell.device)

        # Fill lower-triangular elements
        idx = 0
        for row in range(d):
            for col in range(row + 1):
                if row == col:
                    # Diagonal: force positive via exp
                    L[:, row, col] = torch.exp(chol_cell[:, idx].clamp(-5, 5))
                else:
                    # Off-diagonal: unconstrained
                    L[:, row, col] = chol_cell[:, idx]
                idx += 1

        # M = L @ L^T (PD by construction)
        M = torch.bmm(L, L.transpose(-1, -2))  # (N, d, d)
        return M

    def _mahal_dist(
        self,
        x: torch.Tensor,  # (N_x, d)
        y: torch.Tensor,  # (N_y, d)
        M_x: torch.Tensor,  # (N_x, d, d)
        M_y: torch.Tensor,  # (N_y, d, d)
    ) -> torch.Tensor:
        """Symmetric Mahalanobis: d(x,y) = (x-y)^T ((M_x + M_y)/2) (x-y)."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N_x, N_y, d)
        # Average metric: M_pair = (M_x_i + M_y_j) / 2
        M_pair = 0.5 * (M_x.unsqueeze(1) + M_y.unsqueeze(0))  # (N_x, N_y, d, d)
        # Quadratic form: diff^T M_pair diff
        # Einsum: (N_x, N_y, d) @ (N_x, N_y, d, d) @ (N_x, N_y, d)
        Md = torch.einsum("ijk,ijkl->ijl", diff, M_pair)  # (N_x, N_y, d)
        d2 = torch.einsum("ijk,ijk->ij", Md, diff).clamp(min=1e-6)  # (N_x, N_y)
        return d2

    def forward_samples(
        self,
        x: torch.Tensor,        # (N_x, d) spatial positions
        y: torch.Tensor,        # (N_y, d)
        sample_x: torch.Tensor, # (N_x, n_perts) one-hot
        sample_y: torch.Tensor, # (N_y, n_perts)
        cutoff: torch.Tensor,   # (N_x, N_y) or scalar
    ) -> torch.Tensor:
        M_x = self._build_metric(sample_x)  # (N_x, d, d)
        M_y = self._build_metric(sample_y)  # (N_y, d, d)
        d2  = self._mahal_dist(x, y, M_x, M_y)   # (N_x, N_y)
        res = 1.0 / (1.0 + d2)

        if cutoff is not None and cutoff.numel() > 1 and cutoff.mean() > 0:
            cutoff = cutoff.clamp(min=0.1, max=0.9)
            cut_mask = torch.sigmoid(
                (self.phi * (res - cutoff).clamp(-1, 1))
            ).clamp(min=0.01)
            res = res * cut_mask
        return res

    def forward_diag_samples(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_x: torch.Tensor,
        sample_y: torch.Tensor,
        cutoff: torch.Tensor,
    ) -> torch.Tensor:
        """Diagonal kernel (N_x,) -- used for variational term."""
        M_x = self._build_metric(sample_x)  # (N, d, d)
        M_y = self._build_metric(sample_y)
        diff = x - y   # (N, d)
        M_avg = 0.5 * (M_x + M_y)  # (N, d, d)
        Md  = torch.einsum("ni,nij->nj", diff, M_avg)  # (N, d)
        d2  = torch.einsum("ni,ni->n", Md, diff).clamp(min=1e-6)  # (N,)
        res = 1.0 / (1.0 + d2)
        if cutoff is not None and cutoff.numel() > 1 and cutoff.mean() > 0:
            cutoff_d = cutoff.clamp(min=0.1, max=0.9)
            if isinstance(cutoff_d, torch.Tensor) and cutoff_d.shape == res.shape:
                cut_mask = torch.sigmoid(
                    (self.phi * (res - cutoff_d).clamp(-1, 1))
                ).clamp(min=0.01)
                res = res * cut_mask
        return res


class EmbeddingConditionedKernel(nn.Module):
    """Per-cell kernel scale predicted from perturbation embedding (Phase 2.4).

    Replaces the one-hot lookup (sample_x @ scale_matrix) in
    BatchedCauchyKernel_CONCERT_flex with an MLP that takes a continuous
    perturbation embedding as input. This allows the kernel scale to vary
    smoothly across related perturbations (e.g., nearby genes in embedding
    space share similar spatial lengthscales).

    The MLP maps:  emb_dim -> hidden -> d_pos (positive via softplus)

    Parameters
    ----------
    emb_dim  : int  -- perturbation embedding dimension (= encoder_dim in LORD)
    d_pos    : int  -- number of spatial dimensions to scale (= 2)
    hidden   : int  -- MLP hidden size
    phi      : float -- cutoff gate sharpness
    dtype, device : standard
    """

    def __init__(
        self,
        emb_dim: int,
        d_pos: int = 2,
        hidden: int = 64,
        phi: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.d_pos = d_pos
        self.phi   = torch.tensor(phi, dtype=dtype, device=device)

        # MLP: emb -> scale (positive via softplus)
        self.scale_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, d_pos),
        ).to(device)
        # Initialize to produce scale ≈ 10.0 (same as default in config)
        with torch.no_grad():
            self.scale_mlp[-1].bias.fill_(np.log(np.exp(10.0) - 1))  # softplus⁻¹(10)

    def _scale_from_emb(self, emb: torch.Tensor) -> torch.Tensor:
        """Map (N, emb_dim) -> (N, d_pos) positive scale."""
        return F.softplus(self.scale_mlp(emb)).clamp(min=1e-4, max=1e4)

    def forward_samples(
        self,
        x: torch.Tensor,       # (N_x, d_pos) spatial positions
        y: torch.Tensor,       # (N_y, d_pos)
        emb_x: torch.Tensor,   # (N_x, emb_dim) perturbation embeddings
        emb_y: torch.Tensor,   # (N_y, emb_dim)
        cutoff: torch.Tensor,
    ) -> torch.Tensor:
        scale_x = self._scale_from_emb(emb_x)  # (N_x, d_pos)
        scale_y = self._scale_from_emb(emb_y)  # (N_y, d_pos)

        # Geometric mean of scales: per-dimension, broadcast over the pair
        scale_xy = torch.sqrt(scale_x.unsqueeze(1) * scale_y.unsqueeze(0))  # (N_x, N_y, d_pos)

        # Mahalanobis-like distance with diagonal scale matrix
        diff = x.unsqueeze(1) - y.unsqueeze(0)   # (N_x, N_y, d_pos)
        d2   = (diff ** 2 / scale_xy.clamp(min=1e-6)).sum(-1).clamp(min=1e-6)  # (N_x, N_y)
        res  = 1.0 / (1.0 + d2)

        if cutoff is not None and cutoff.numel() > 1 and cutoff.mean() > 0:
            cutoff = cutoff.clamp(min=0.1, max=0.9)
            cut_mask = torch.sigmoid(
                (self.phi * (res - cutoff).clamp(-1, 1))
            ).clamp(min=0.01)
            res = res * cut_mask
        return res
