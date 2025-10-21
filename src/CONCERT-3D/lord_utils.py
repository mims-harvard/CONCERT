import logging
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch import nn

logger = logging.getLogger(__name__)
Number = Union[int, float]

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

def buildNetwork(layers, activation="relu", dropout=0.2, BN=True, dtype=torch.float32):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if i == len(layers)-1:
            break
        if BN:
            net.append(nn.BatchNorm1d(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

class NBLoss_(nn.Module):
    def __init__(self):
        super(NBLoss_, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        result = t1 + t2
        result = torch.mean(torch.sum(result, dim=1))
        return result

class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, eps=1e-8):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        # means of the negative binomial (has to be positive support)
        mu = mean
        # inverse dispersion parameter (has to be positive support)
        theta = disp

        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(x + 1.0)
            - torch.lgamma(x + theta + eps)
        )
        t2 = (theta + x) * torch.log(1.0 + (mu / (theta + eps))) + (
            x * (torch.log(theta + eps) - torch.log(mu + eps))
        )
        final = t1 + t2
        final = _nan2inf(final)

        return torch.mean(final)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, gamma=3, reduction="mean") -> None:
        """Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .

        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target):
        """Compute the FocalLoss

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        from torchvision.ops import focal_loss

        loss = focal_loss.sigmoid_focal_loss(
            inputs,
            target,
            reduction=self.reduction,
            gamma=self.gamma,
            alpha=self.alpha,
        )
        return loss

class GaussianLoss(torch.nn.Module):
    """
    Gaussian log-likelihood loss. It assumes targets `y` with n rows and d
    columns, but estimates `yhat` with n rows and 2d columns. The columns 0:d
    of `yhat` contain estimated means, the columns d:2*d of `yhat` contain
    estimated variances. This module assumes that the estimated variances are
    positive---for numerical stability, it is recommended that the minimum
    estimated variance is greater than a small number (1e-3).
    """

    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, mean, var, y):
        term1 = var.log().div(2)
        term2 = (y - mean).pow(2).div(var.mul(2))
        return (term1 + term2).mean()


class GaussianNLLLoss(torch.nn.Module):
    def __init__(self):
        super(GaussianNLLLoss, self).__init__()

    def forward(self, mean, var, y):
        GNLL = torch.nn.GaussianNLLLoss()
        loss = GNLL(input=mean, var=var, target=y)
        loss = torch.nn.functional.relu(loss)
        return loss

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.

    Careful: if activation is set to ReLU, ReLU is only applied to the first half of NN outputs!
    """

    def __init__(
        self,
        sizes,
        batch_norm=True,
        last_layer_act="linear",
        append_layer_width=None,
        append_layer_position=None,
    ):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        # We add another layer either at the front / back of the sequential model. It gets a different name
        # `append_XXX`. The naming of the other layers stays consistent.
        # This allows us to load the state dict of the "non_appended" MLP without errors.
        if append_layer_width:
            assert append_layer_position in ("first", "last")
            if append_layer_position == "first":
                layers_dict = OrderedDict()
                layers_dict["append_linear"] = torch.nn.Linear(
                    append_layer_width, sizes[0]
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[0])
                layers_dict["append_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
            else:
                layers_dict = OrderedDict(
                    {str(i): module for i, module in enumerate(layers)}
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[-1])
                layers_dict["append_relu"] = torch.nn.ReLU()
                layers_dict["append_linear"] = torch.nn.Linear(
                    sizes[-1], append_layer_width
                )
        else:
            layers_dict = OrderedDict(
                {str(i): module for i, module in enumerate(layers)}
            )

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, dim, device, nonlin="sigm"):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm or None. If None, then the doser is disabled and just returns the dosage unchanged.
        """
        super(GeneralizedSigmoid, self).__init__()
        assert nonlin in ("sigm", "logsigm", None)
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim, device=device), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim, device=device), requires_grad=True
        )

    def forward(self, x, idx=None):
        if self.nonlin == "logsigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][idx]
                beta = self.beta[0][idx]
                c0 = bias.sigmoid()
                return (torch.log1p(x) * beta + bias).sigmoid() - c0
        elif self.nonlin == "sigm":
            if idx is None:
                c0 = self.bias.sigmoid()
                return (x * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][idx]
                beta = self.beta[0][idx]
                c0 = bias.sigmoid()
                return (x * beta + bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == "logsigm":
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x

# class ZINB_loss(torch.nn.Module):
#     def __init__(self):
#         super(ZINB_loss, self).__init__()
    
#     def forward(
#         self,
#         x: torch.Tensor,
#         px_rate: torch.Tensor,
#         px_r: torch.Tensor,
#         px_dropout: torch.Tensor,
#         ) -> torch.Tensor:
#         """
#         Compute likelihood loss for zero-inflated negative binomial distribution.

#         Args:
#         ----
#             x: Input data.
#             px_rate: Mean of distribution.
#             px_r: Inverse dispersion.
#             px_dropout: Logits scale of zero inflation probability.

#         Returns
#         -------
#             Negative log likelihood (reconstruction loss) for each data point. If number
#             of latent samples == 1, the tensor has shape `(batch_size, )`. If number
#             of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
#         """
#         recon_loss = (
#             -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
#             .log_prob(x)
#             .sum(dim=-1)
#         )
#         return recon_loss

class ZINB_Loss_(nn.Module):
    def __init__(self):
        super(ZINB_Loss_, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result

def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and prior Gaussian.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
            prior_mean: Mean of the prior Gaussian.
            prior_var: Variance of the prior Gaussian.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

def latent_kl_divergence_(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-6
        variational_var = torch.clamp(variational_var, min=-10, max=10)
        kld = torch.mean(-0.5 * torch.sum(1 + variational_var - variational_mean ** 2 - variational_var.exp(), dim = 1), dim = 0)
        kld = torch.clamp(kld, min=eps)
        return kld

# def library_kl_divergence(
#         use_observed_lib_size: bool,
#         #batch_index: torch.Tensor,
#         variational_library_mean: torch.Tensor,
#         variational_library_var: torch.Tensor,
#         library: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Compute KL divergence between library size variational posterior and prior.

#         Both the variational posterior and prior are Log-Normal.
#         Args:
#         ----
#             batch_index: Batch indices for batch-specific library size mean and
#                 variance.
#             variational_library_mean: Mean of variational Log-Normal.
#             variational_library_var: Variance of variational Log-Normal.
#             library: Sampled library size.

#         Returns
#         -------
#             KL divergence for each data point. If number of latent samples == 1,
#             the tensor has shape `(batch_size, )`. If number of latent
#             samples > 1, the tensor has shape `(n_samples, batch_size)`.
#         """
#         if not use_observed_lib_size:
#             (
#                 local_library_log_means,
#                 local_library_log_vars,
#             ) = _compute_local_library_params(batch_index)

#             kl_library = kl(
#                 Normal(variational_library_mean, variational_library_var.sqrt()),
#                 Normal(local_library_log_means, local_library_log_vars.sqrt()),
#             )
#         else:
#             kl_library = torch.zeros_like(library)
#         return kl_library.sum(dim=-1)

def library_kl_divergence_(
        library: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between library size variational posterior and prior.

        Both the variational posterior and prior are Log-Normal.
        Args:
        ----
            batch_index: Batch indices for batch-specific library size mean and
                variance.
            variational_library_mean: Mean of variational Log-Normal.
            variational_library_var: Variance of variational Log-Normal.
            library: Sampled library size.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def _gaussian_kernel_matrix(x, y, device):
    sigmas = torch.tensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6], device = device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result

def maximum_mean_discrepancy(xs, batch_ids, device, ref_batch = None): #Function to calculate MMD value
    # number of cells
    assert batch_ids.shape[0] == xs.shape[0]
    batches = torch.unique(batch_ids, sorted = True)
    nbatches = batches.shape[0]
    if ref_batch is None:
        # select the first batch, the batches are equal sizes
        # batch 0 is also the largest batch
        ref_batch = batches[0]
    cost = 0
    # within batch
    for batch in batches:
        xs_batch = xs[batch_ids == batch, :]
        if batch == ref_batch:
            cost += (nbatches - 1) * torch.mean(_gaussian_kernel_matrix(xs_batch, xs_batch, device))
        else:
            cost += torch.mean(_gaussian_kernel_matrix(xs_batch, xs_batch, device))
    
    # between batches
    xs_refbatch = xs[batch_ids == ref_batch]
    for batch in batches:
        if batch != ref_batch:
            xs_batch = xs[batch_ids == batch, :]
            cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(xs_refbatch, xs_batch, device))
    
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.tensor([0.0], device = device)

    return cost

class GaussianKLDLoss(nn.Module): #multivariate Guassion kl divergence
    def __init__(self, reduction='mean', eps=1e-6):
        super(GaussianKLDLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, mu, logvar, prior_mu, prior_logvar):
        # Clamp the log-variances to avoid extremely small or large values
        logvar = torch.clamp(logvar, min=-10, max=10)
        prior_logvar = torch.clamp(prior_logvar, min=-10, max=10)

        # Calculate the KL divergence between two multivariate Gaussian distributions
        var_ratio = (logvar - prior_logvar).exp()
        t1 = (mu - prior_mu).pow(2) / (prior_logvar.exp() + self.eps)
        t2 = var_ratio * (1 - t1)
        kld = -0.5 * torch.sum(1 + logvar - prior_logvar - t1 - t2, dim=-1)

        if self.reduction == 'mean':
            kld = torch.mean(kld)
        elif self.reduction == 'sum':
            kld = torch.sum(kld)

        kld = torch.clamp(kld, min=self.eps)

        return kld