import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


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
            self.scale = torch.tensor(scale, dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=dtype).to(device), requires_grad=True)

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

        # Compute squared norms
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)  # Shape: (batch_size, num_points_x, 1)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)  # Shape: (batch_size, 1, num_points_y)

        # Compute dot products
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))  # Shape: (batch_size, num_points_x, num_points_y)

        # Compute squared Euclidean distance with clamping
        d = torch.clamp(sq_norms_x + sq_norms_y - 2.0 * dotprods, min=1e-10, max=1e6)  # Ensure non-negative

        # Scale handling
        if self.fixed_scale:
            scale = torch.clamp(self.scale, min=1e-6, max=1e6)  # Ensure stability
        else:
            scale = torch.clamp(F.softplus(self.scale), min=1e-6, max=1e6)  # Ensure learnable scale is positive

        # Compute Cauchy Kernel
        res = 1 / (1 + d / scale)

        # Clip scale if it's a learnable parameter to prevent instability
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
    
    # def forward_points(self, x, y):
    #     sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
    #     sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
    #     dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
    #     d = sq_norms_x + sq_norms_y - 2. * dotprods
    #     if self.fixed_scale:
    #         scale0 = self.scale0
    #         res = 1/(1+d/scale0)
    #     else:
    #         scale0 = torch.mean(torch.clamp(F.softplus(self.scale0), min=1e-10, max=1e4))
    #         res = 1/(1+d/scale0)
    #     return res

    # def forward_diag_points(self, x, y):
    #     d = ((x-y)**2).sum(dim=1)
    #     if self.fixed_scale:
    #         scale0 = self.scale0
    #         res = 1/(1+d/scale0)
    #     else:
    #         scale0 = torch.mean(torch.clamp(F.softplus(self.scale0), min=1e-10, max=1e4))
    #         res = 1/(1+d/scale0)       
    #     return res
    
    # def forward_samples_points(self, x, y, sample_x):
    #     b = y.shape[0]
    #     sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
    #     sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
    #     dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
    #     d = sq_norms_x + sq_norms_y - 2. * dotprods
    #     if self.fixed_scale:
    #         scale_x = torch.matmul(sample_x, self.scale.unsqueeze(dim=1))
    #         scale_y = self.scale0.expand(b, 1)
    #         scale_xy = torch.sqrt(torch.matmul(scale_x, scale_y.T))
    #         res = 1/(1+d/scale_x)
    #     else:
    #         scale_x = torch.clamp(F.softplus(torch.matmul(sample_x, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4)
    #         scale_y = torch.clamp(F.softplus(self.scale0.expand(b, 1)), min=1e-10, max=1e4)
    #         scale_xy = torch.sqrt(torch.matmul(scale_x, scale_y.T))
    #         res = 1/(1+d/scale_xy)
    #     if self.cutoff.mean() > 0:
    #         cut_mask = torch.sigmoid(self.phi * (res - self.cutoff.clamp(min=0., max=1e3)))
    #         res = res * cut_mask
    #     return res

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
            #self.scale0 = torch.tensor(scale[0,:], dtype=dtype).to(device) #D
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=dtype).to(device),requires_grad=True)
            #self.scale0 = torch.tensor(scale[0,:], dtype=dtype).to(device) #D
        #print("scale: ", self.scale)

    def forward_samples(self, x, y, sample_x, sample_y, cutoff):
        scale = self.scale.clamp(min=1e-6, max=1e6)
        if torch.isnan(scale).any():
            print("forward_samples nan in scale")
        scale_x = sample_x @ scale  # (N, B) @ (B, D) = (N_x, D)
        scale_y = sample_y @ scale
        inv_sqrt_scale_x = (1.0 / torch.sqrt(scale_x.clamp(min=1e-6)))  # (N_x, D)
        inv_sqrt_scale_y = (1.0 / torch.sqrt(scale_y.clamp(min=1e-6)))  # (N_y, D)
        x_scaled = x * inv_sqrt_scale_x  # (N_x, D)
        y_scaled = y * inv_sqrt_scale_y  # (N_y, D)
        diff = x_scaled[:, None, :] - y_scaled[None, :, :]  # (N_x, N_y, D)
        d = torch.sum(diff**2, dim=-1).clamp(min=1e-6) 
        res = 1 / (1 + d)
        if cutoff.mean() > 0:
            cutoff = cutoff.clamp(min=0.0001, max=0.9999)
            cutoff_matrix = (cutoff + cutoff.T) / 2.0  # (N_x, N_y)
            cut_mask = torch.sigmoid((self.phi * (res-cutoff_matrix).clamp(-1,1))).clamp(min=1e-6)
            res = res * cut_mask
        return res

    def forward_diag_samples(self, x, y, sample_x, sample_y, cutoff):
        scale = self.scale.clamp(min=1e-6, max=1e6)
        if torch.isnan(scale).any():
            print("forward_diag_samples nan in scale")
        scale_x = sample_x @ scale  # (N_x, D)
        scale_y = sample_y @ scale
        inv_sqrt_scale_x = (1.0 / torch.sqrt(scale_x.clamp(min=1e-6)))  # (N_x, D)
        inv_sqrt_scale_y = (1.0 / torch.sqrt(scale_y.clamp(min=1e-6)))  # (N_y, D)
        x_scaled = x * inv_sqrt_scale_x  # (N_x, D)
        y_scaled = y * inv_sqrt_scale_y  # (N_y, D)
        d = torch.norm(x_scaled - y_scaled, dim=1).clamp(min=1e-6)
        res = 1 / (1 + d)
        if cutoff.mean() > 0:
            cutoff = cutoff.clamp(min=0.001, max=0.999)
            cutoff_matrix = (cutoff + cutoff.T) / 2.0  # (N_x, N_y)
            cut_mask = torch.sigmoid((self.phi * (res-cutoff_matrix).clamp(-1,1))).clamp(min=1e-6)
            res = res * cut_mask
        return res
    
    def forward_samples_points(self,  x, y, sample_x, sample_y, cutoff):
        scale = self.scale.clamp(min=1e-6, max=1e6)
        if torch.isnan(scale).any():
            print("forward_samples nan in scale")
        scale_x = sample_x @ scale  # (N, B) @ (B, D) = (N_x, D)
        scale_y = sample_y @ scale
        inv_sqrt_scale_x = (1.0 / torch.sqrt(scale_x.clamp(min=1e-6)))  # (N_x, D)
        inv_sqrt_scale_y = (1.0 / torch.sqrt(scale_y.clamp(min=1e-6)))  # (N_y, D)
        x_scaled = x * inv_sqrt_scale_x  # (N_x, D)
        y_scaled = y * inv_sqrt_scale_y  # (N_y, D)
        diff = x_scaled[:, None, :] - y_scaled[None, :, :]  # (N_x, N_y, D)
        d = torch.sum(diff**2, dim=-1).clamp(min=1e-6) 
        res = 1 / (1 + d)
        if cutoff.mean() > 0:
            cutoff = cutoff.clamp(min=0.0001, max=0.9999)
            cutoff0 = torch.zeros(y.shape[0], dtype=self.dtype).to(self.device)
            cutoff_matrix = (cutoff.view(x.shape[0],1) + cutoff0.view(1,y.shape[0])) / 2.0  # (N_x, N_y)
            cut_mask = torch.sigmoid((self.phi * (res-cutoff_matrix).clamp(-1,1))).clamp(min=0.001)
            res = res * cut_mask
        return res
    
    # def forward_points(self, x, y):
    #     scale0 = self.scale0.unsqueeze(0).expand(x.shape[0], -1) # reshape self.scale0 in (D) to (N, D)
    #     inv_sqrt_scale = (1.0 / torch.sqrt(scale0))  # (N_x, D)
    #     x_scaled = x * inv_sqrt_scale  # (N_x, D)
    #     y_scaled = y * inv_sqrt_scale  # (N_y, D)
    #     diff = x_scaled[:, None, :] - y_scaled[None, :, :]  # (N_x, N_y, D)
    #     d = torch.sum(diff**2, dim=-1).clamp(min=1e-6) 
    #     res = 1 / (1 + d)
    #     return res

    # def forward_diag_points(self, x, y):
    #     scale0 = self.scale0.unsqueeze(0).expand(x.shape[0], -1) # reshape self.scale0 in (D) to (N, D)
    #     inv_sqrt_scale = (1.0 / torch.sqrt(scale0))  # (N_x, D)
    #     x_scaled = x * inv_sqrt_scale  # (N_x, D)
    #     y_scaled = y * inv_sqrt_scale  # (N_y, D)
    #     d = torch.norm(x_scaled - y_scaled, dim=1).clamp(min=1e-6)
    #     res = 1 / (1 + d)
    #     return res
    
    # def forward_samples_points(self, x, y, sample_x):
    #     scale = self.scale.clamp(min=1e-6, max=1e6)
    #     if torch.isnan(scale).any():
    #         print("forward_samples_points nan in scale")
    #     scale_x = sample_x @ scale  # (N, B) @ (B, D) = (N_x, D)
    #     scale_y = self.scale0.unsqueeze(0).expand(y.shape[0], -1) # reshape self.scale0 in (D) to (N_y, D)
    #     inv_sqrt_scale_x = (1.0 / torch.sqrt(scale_x.clamp(min=1e-6)))  # (N_x, D)
    #     inv_sqrt_scale_y = (1.0 / torch.sqrt(scale_y.clamp(min=1e-6)))  # (N_y, D)
    #     x_scaled = x * inv_sqrt_scale_x  # (N_x, D)
    #     y_scaled = y * inv_sqrt_scale_y  # (N_y, D)
    #     diff = x_scaled[:, None, :] - y_scaled[None, :, :]  # (N_x, N_y, D)
    #     d = torch.sum(diff**2, dim=-1).clamp(min=1e-6) 
    #     res = 1 / (1 + d)
    #     if self.cutoff.mean() > 0:
    #         cutoff = self.cutoff.clamp(min=0., max=1e3)
    #         cutoff_x = sample_x @ cutoff.view(-1, 1)  # (N_x, 1)
    #         cutoff_matrix = cutoff_x.expand(-1, y.shape[0])  # reshape to (N_x, N_y)
    #         cut_mask = torch.sigmoid((self.phi * (res - cutoff_matrix)).clamp(-30, 30))
    #         res = res * cut_mask
    #    return res

    def print_params(self):
        print(f"Scale Parameters: {self.scale}")
        print(f"Scale Parameters: {self.cutoff}")

class BatchedCauchyKernel3d(nn.Module):
    def __init__(self, scale=[], fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(BatchedCauchyKernel3d, self).__init__()
        self.fixed_scale = fixed_scale

        # Scale initialization
        if fixed_scale:
            # For fixed scale, ensure it's a torch tensor
            self.scale = torch.tensor(scale, dtype=dtype, device=device)
        else:
            # Make scale learnable
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

        # 1) Compute squared norms of x and y
        # Shape: (batch_size, num_points_x, 1)
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        # Shape: (batch_size, 1, num_points_y)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)

        # 2) Dot products => shape: (batch_size, num_points_x, num_points_y)
        dotprods = torch.matmul(x, y.transpose(-2, -1))

        # 3) Squared Euclidean distance with clamping
        #    Clamping prevents negative distances (due to float precision) and overly large values.
        d = torch.clamp(sq_norms_x + sq_norms_y - 2.0 * dotprods, min=1e-10, max=1e6)

        # 4) Compute per-point scale factors from sample_x, sample_y
        #    => shape: (batch_size, num_points_x, 1) and (batch_size, num_points_y, 1)
        if self.fixed_scale:
            # If scale is fixed, clamp it for safety
            scale = torch.clamp(self.scale, min=1e-6, max=1e6)
            scale_x = torch.matmul(sample_x, scale.unsqueeze(-1))  # (batch_size, num_points_x, 1)
            scale_y = torch.matmul(sample_y, scale.unsqueeze(-1))  # (batch_size, num_points_y, 1)
        else:
            # If scale is learnable, use softplus for positivity, then clamp
            raw_scale = torch.clamp(F.softplus(self.scale), min=1e-6, max=1e6)
            scale_x = torch.matmul(sample_x, raw_scale.unsqueeze(-1))  # (batch_size, num_points_x, 1)
            scale_y = torch.matmul(sample_y, raw_scale.unsqueeze(-1))  # (batch_size, num_points_y, 1)

        # Optional: clamp scale_x, scale_y to avoid extreme values
        scale_x = torch.clamp(scale_x, min=1e-10, max=1e6)
        scale_y = torch.clamp(scale_y, min=1e-10, max=1e6)

        # 5) Compute combined scale for each pair (x_i, y_j)
        # => shape: (batch_size, num_points_x, num_points_y)
        # We clamp the product before sqrt to avoid negative or extremely large values.
        scale_xy = torch.clamp(torch.matmul(scale_x, scale_y.transpose(-2, -1)), min=1e-10, max=1e12)
        scale_xy = torch.sqrt(scale_xy)  # shape: (batch_size, num_points_x, num_points_y)

        # 6) Compute final Cauchy Kernel
        # => shape: (batch_size, num_points_x, num_points_y)
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
