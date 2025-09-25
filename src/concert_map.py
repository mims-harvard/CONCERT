import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import numpy as np
import pandas as pd
from SVGP_Batch import SVGP
from I_PID import PIDControl
from VAE_utils import *
from collections import deque
from lord_batch import Lord_encoder
from torch.distributions.log_normal import LogNormal

def unknown_attribute_penalty_loss(latent_unknown_attributes: torch.Tensor) -> float:
        """Computes the content penalty term in the loss."""
        return torch.sum(latent_unknown_attributes**2, dim=1).mean()

class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
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
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class CONCERT(nn.Module):
    def __init__(self, encoder_dim, GP_dim, Normal_dim, cell_atts, num_genes, n_batch, encoder_layers, decoder_layers, noise, encoder_dropout, decoder_dropout, 
                    shared_dispersion, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, multi_kernel_mode, mask_cutoff,
                    N_train, KL_loss, dynamicVAE, init_beta, min_beta, max_beta, dtype, device):
        super(CONCERT, self).__init__()
        torch.set_default_dtype(dtype)
        if multi_kernel_mode:
            self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, multi_kernel_mode=multi_kernel_mode, 
                kernel_phi=1., jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        else:
            self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale[0], multi_kernel_mode=multi_kernel_mode,
                jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.encoder_dim = encoder_dim
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss          # expected KL loss value
        self.dynamicVAE = dynamicVAE
        self.shared_dispersion = shared_dispersion
        self.beta = init_beta           # beta controls the weight of reconstruction loss
        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.noise = noise              # intensity of random noise
        self.device = device
        self.num_genes = num_genes
        self.cell_atts = cell_atts
        self.encoder = DenseEncoder(input_dim=encoder_dim*3, hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = buildNetwork([GP_dim+Normal_dim]+decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], self.num_genes), MeanAct())
        else:
            self.dec_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, self.num_genes), MeanAct())
        self.shared_dispersion = shared_dispersion
        if self.shared_dispersion:
            self.dec_disp = nn.Parameter(torch.randn(self.num_genes), requires_grad=True)
        else:
            self.dec_disp = nn.Parameter(torch.randn(self.num_genes, n_batch), requires_grad=True)

        self.lord_encoder = Lord_encoder(embedding_dim=[encoder_dim] * 3, num_genes = self.num_genes, labels = cell_atts, 
                                         attributes=["tissue", "perturbation"], attributes_type = ["categorical", "categorical"], 
                                         noise=self.noise, device=device)
        self.mask_cutoff = nn.Parameter(torch.tensor(mask_cutoff, dtype=self.dtype),requires_grad=True)

        self.NB_loss = NBLoss().to(self.device)
        self.mse = nn.MSELoss(reduction='mean')
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def forward(self, x, y, batch, raw_y, sample_index, cell_atts, size_factors, cutoff, num_samples=1):
        """
        Forward pass.

        Parameters:
        -----------
        x: mini-batch of positions.
        y: mini-batch of preprocessed counts.
        batch: mini-batch of one-hot encoded batch IDs.
        raw_y: mini-batch of raw counts.
        sample_index: index of cells/spots.
        cell_atts: matrix of cell/spot's attributes.
        size_factor: mini-batch of size factors.
        cutoff: learnable vector of cutoffs for cell-cell/spot-spot's dependencies.
        num_samples: number of samplings of the posterior distribution of latent embedding.

        raw_y and size_factor are used for reconstruction loss.
        """ 

        self.train()
        b = y.shape[0]
        lord_latents =self.lord_encoder.predict(sample_indices=sample_index, labels = cell_atts, batch_size = b)
        y_ = lord_latents["total_latent"]
        qnet_mu, qnet_var = self.encoder(y_)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l], cutoff=cutoff)

            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l],
                                                                    noise=gp_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l, cutoff=cutoff)

            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            gp_p_m.append(gp_p_m_l)
            gp_p_v.append(gp_p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)

        inside_elbo = inside_elbo_recon - (b / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # cross entropy term
        gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var)
        gp_ce_term = torch.sum(gp_ce_term)

        # KL term of GP prior
        gp_KL_term = gp_ce_term - inside_elbo

        # KL term of Gaussian prior
        gaussian_prior_dist = Normal(torch.zeros_like(gaussian_mu), torch.ones_like(gaussian_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        gaussian_KL_term = kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))
        latent_samples = []
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        recon_loss = 0
        recon_loss_mse = 0
        for f in latent_samples:
            hidden_samples = self.decoder(f)
            mean_samples_ = self.dec_mean(hidden_samples)
            if self.shared_dispersion:
                disp_samples_ = torch.exp(torch.clamp(self.dec_disp, -15., 15.)).T
            else:
                disp_samples_ = torch.exp(torch.clamp(torch.matmul(self.dec_disp, batch.T), -15., 15.)).T

            mean_samples.append(mean_samples_)
            disp_samples.append(disp_samples_)
            recon_loss += self.NB_loss(x=raw_y, mean=mean_samples_, disp=disp_samples_, scale_factor=size_factors)
            recon_loss_mse += self.mse(mean_samples_, raw_y)

        recon_loss = recon_loss / num_samples
        recon_loss_mse = recon_loss_mse / num_samples

        noise_reg = 0

        #unknown attribute penalty
        latent_unknown_attributes = lord_latents["basal_latent"]
        unknown_attribute_penalty = unknown_attribute_penalty_loss(latent_unknown_attributes)

        #lord loss
        lord_loss = unknown_attribute_penalty + recon_loss_mse
        
        elbo = recon_loss + self.beta * gp_KL_term + self.beta * gaussian_KL_term + lord_loss

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, gp_p_m, gp_p_v, qnet_mu, qnet_var, \
            mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg, unknown_attribute_penalty, recon_loss_mse


    def batching_latent_samples(self, X, sample_index, cell_atts, batch_size=512):
        """
        Output latent embedding.

        Parameters:
        -----------
        X: Location information (n_spots, 2).
        sample_index: index of cells/spots (n_spots, 1).
        cell_atts: matrix of cell/spot's attributes (n_spots, n_atts)
        """ 

        self.eval()
        cutoffs = self.mask_cutoff

        X = torch.tensor(X, dtype=self.dtype)
        cell_atts = torch.tensor(cell_atts, dtype=torch.int)
        total_sample_indices = torch.tensor(sample_index, dtype=torch.int)

        latent_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            sample_index = total_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=torch.int)
            cell_atts_batch = cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=torch.int)
            b = xbatch.shape[0]
            cutoff = cutoffs[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=self.dtype)
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            ybatch_ = lord_latents["total_latent"]

            qnet_mu, qnet_var = self.encoder(ybatch_)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)

        return latent_samples.numpy()


    def batching_denoise_counts(self, X, sample_index, cell_atts, n_samples=1, batch_size=512):
        """
        Output denoised counts.

        Parameters:
        -----------
        X: Location information (n_spots, 2).
        sample_index: index of cells/spots (n_spots, 1).
        cell_atts: matrix of cell/spot's attributes (n_spots, n_atts)
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()
        cutoffs = self.mask_cutoff

        X = torch.tensor(X, dtype=self.dtype)
        cell_atts = torch.tensor(cell_atts, dtype=torch.int)
        total_sample_indices = torch.tensor(sample_index, dtype=torch.int)

        mean_samples = []
        var_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            cell_atts_batch = cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=torch.int)
            sample_index = total_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=torch.int)
            b = xbatch.shape[0]
            cutoff = cutoffs[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=self.dtype)
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            ybatch_ = lord_latents["total_latent"]

            qnet_mu, qnet_var = self.encoder(ybatch_)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            mean_samples_ = []
            for f in latent_samples:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.data.cpu().detach())
            var_samples.append(torch.sqrt(p_v).data.cpu().detach())

        mean_samples = torch.cat(mean_samples, dim=0)
        #gp_latent = torch.cat(gp_latent, dim=0)
        var_samples = torch.cat(var_samples, dim=0)

        return mean_samples.numpy(), var_samples.numpy()


    def batching_recon_samples(self, Z, batch_size=512):
        self.eval()

        Z = torch.tensor(Z, dtype=self.dtype)

        recon_samples = []

        num = Z.shape[0]
        num_batch = int(math.ceil(1.0*Z.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            zbatch = Z[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            h = self.decoder(zbatch)
            mean_batch = self.dec_mean(h)
            recon_samples.append(mean_batch.data.cpu().detach())

        recon_samples = torch.cat(recon_samples, dim=0)

        return recon_samples.numpy()


    def batching_predict_samples(self, X_test, X_train, Y_sample_index, Y_cell_atts, n_samples=1, batch_size=512):
        """
        Impute latent representations and denoised counts on unseen testing locations.

        Parameters:
        -----------
        X_test: Location information of testing set (n_test_spots, 2).
        X_train: Location information of training set (n_train_spots, 2).
        Y_sample_index: index of cells/spots (n_spots, 1).
        Y_cell_atts: matrix of cell/spot's attributes (n_spots, n_atts)
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx

        cutoffs = self.mask_cutoff

        X_test = torch.tensor(X_test, dtype=self.dtype)
        X_train = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        train_cell_atts = torch.tensor(Y_cell_atts, dtype=torch.int)
        train_sample_indices = torch.tensor(Y_sample_index, dtype=torch.int)

        latent_samples = []
        mean_samples = []

        train_num = X_train.shape[0]
        train_num_batch = int(math.ceil(1.0*X_train.shape[0]/batch_size))
        test_num = X_test.shape[0]
        test_num_batch = int(math.ceil(1.0*X_test.shape[0]/batch_size))

        x_train_select = []
        for e in range(X_test.shape[0]):
            x_train_select.append(find_nearest(X_train, X_test[e]))
        x_train_select = torch.stack(x_train_select)
        y_cutoff = cutoffs[x_train_select.long()]

        qnet_mu, qnet_var = [], []
        for batch_idx in range(train_num_batch):
            cell_atts_batch = train_cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            sample_index = train_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            b = cell_atts_batch.shape[0]
            cutoff_train_batch = cutoffs[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=self.dtype)
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            Y_train_batch_ = lord_latents["total_latent"]

            qnet_mu_, qnet_var_ = self.encoder(Y_train_batch_)
            qnet_mu.append(qnet_mu_)
            qnet_var.append(qnet_var_)
        qnet_mu = torch.cat(qnet_mu, dim=0)
        qnet_var = torch.cat(qnet_var, dim=0)

        for batch_idx in range(test_num_batch):
            x_test_batch = X_test[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)
            cutoff_test_batch = y_cutoff[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device).to(dtype=self.dtype)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            # x_train_select_batch represents the nearest X_train spots to x_test_batch
            x_train_select_batch = []
            for e in range(x_test_batch.shape[0]):
                x_train_select_batch.append(find_nearest(X_train, x_test_batch[e]))
            x_train_select_batch = torch.stack(x_train_select_batch)
            gaussian_mu = qnet_mu[x_train_select_batch.long(), self.GP_dim:]
            gaussian_var = qnet_var[x_train_select_batch.long(), self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params_impute(index_points_test=x_test_batch, index_points_train=X_train, 
                                        y=gp_mu[:, l], noise=gp_var[:, l], x_cutoff=cutoff_train_batch, y_cutoff=cutoff_test_batch)
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.data.cpu().detach())
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            latent_samples_ = []
            for _ in range(n_samples):
                f = latent_dist.sample()
                latent_samples_.append(f)

            mean_samples_ = []
            for f in latent_samples_:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)
        mean_samples = torch.cat(mean_samples, dim=0)

        return latent_samples.numpy(), mean_samples.numpy()

    def train_model(self, pos, batch, ncounts, raw_counts, size_factors, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, 
            train_size=0.95, maxiter=5000, patience=200, save_model=True, model_weights="model.pt", print_kernel_scale=True):
        """
        Model training.

        Parameters:
        -----------
        pos: array_like, shape (n_spots, 2)
            Location information.
        ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        raw_counts: array_like, shape (n_spots, n_genes)
            Raw count matrix.
        size_factor: array_like, shape (n_spots)
            The size factor of each spot, which need for the NB loss.
        lr: float, defalut = 0.001
            Learning rate for the opitimizer.
        weight_decay: float, default = 0.001
            Weight decay for the opitimizer.
        train_size: float, default = 0.95
            proportion of training size, the other samples are validations.
        maxiter: int, default = 5000
            Maximum number of iterations.
        patience: int, default = 200
            Patience for early stopping.
        model_weights: str
            File name to save the model weights.
        print_kernel_scale: bool
            Whether to print current kernel scale during training steps.
        """

        self.train()
        
        sample_indices = torch.tensor(np.arange(ncounts.shape[0]), dtype=torch.int)
        cell_atts = self.cell_atts

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(ncounts, dtype=self.dtype), 
                        torch.tensor(raw_counts, dtype=self.dtype), torch.tensor(size_factors, dtype=self.dtype),
                        sample_indices, torch.tensor(cell_atts, dtype=torch.int), torch.tensor(batch, dtype=self.dtype),
                        self.mask_cutoff)

        if train_size < 1:
            train_dataset, validate_dataset = random_split(dataset=dataset, lengths=[train_size, 1.-train_size])
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = dataset

        if ncounts.shape[0]*train_size > batch_size:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        early_stopping = EarlyStopping(patience=patience, modelfile=model_weights)

        named_params = dict(self.named_parameters())
        scale_params = [p for n, p in named_params.items() if "kernel.scale" in n]
        other_params = [p for n, p in named_params.items() if "kernel.scale" not in n]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": scale_params, "lr": lr*10, "weight_decay": 0.0}
            ])  
        
        queue = deque()

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            gaussian_KL_term_val = 0
            noise_reg_val = 0
            unknown_loss_val = 0
            mse_loss_val = 0
            num = 0
            for batch_idx, (x_batch, y_batch, y_raw_batch, sf_batch, sample_index_batch, cell_atts_batch, batch, cutoff) in enumerate(dataloader):
                x_batch = x_batch.to(self.device).to(dtype=self.dtype)
                y_batch = y_batch.to(self.device).to(dtype=self.dtype)
                y_raw_batch = y_raw_batch.to(self.device).to(dtype=self.dtype)
                sf_batch = sf_batch.to(self.device).to(dtype=self.dtype)
                b_batch = batch.to(self.device).to(dtype=self.dtype)
                sample_index_batch = sample_index_batch.to(self.device).to(dtype=torch.int)
                cell_atts_batch = cell_atts_batch.to(self.device).to(dtype=torch.int)
                cutoff_batch = cutoff.to(self.device).to(dtype=self.dtype)
                
                elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg, unknown_loss, mse_loss = \
                    self.forward(x=x_batch, y=y_batch, raw_y=y_raw_batch, size_factors=sf_batch, batch=b_batch, num_samples=num_samples, 
                                 sample_index=sample_index_batch, cell_atts=cell_atts_batch, cutoff=cutoff_batch)

                self.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                elbo.backward(retain_graph=True)
                for name, param in self.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN detected in gradient of {name}")
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                unknown_loss_val += unknown_loss.item()
                mse_loss_val += mse_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                gaussian_KL_term_val += gaussian_KL_term.item()

                num += x_batch.shape[0]

                if self.dynamicVAE:
                    KL_val = (gp_KL_term.item() + gaussian_KL_term.item()) / x_batch.shape[0]
                    queue.append(KL_val)
                    avg_KL = np.mean(queue)
                    self.beta, _ = self.PID.pid(self.KL_loss*(self.GP_dim+self.Normal_dim), avg_KL)
                    if len(queue) >= 10:
                        queue.popleft()

            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            unknown_loss_val = unknown_loss_val/num
            mse_loss_val = mse_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            gaussian_KL_term_val = gaussian_KL_term_val/num
            noise_reg_val = noise_reg_val/num

            print('Training epoch {}, ELBO:{:.8f}, NB loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f}, noise regularization:{:8f}, basal loss: {:8f}, MSE loss: {:8f}'.format(epoch+1, 
                  elbo_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val, noise_reg_val, unknown_loss_val, mse_loss_val))
            
            print('Current beta', self.beta)
            if print_kernel_scale:
                print('Current kernel scale', self.svgp.kernel.scale.data)
                print('Current min cutoff', torch.min(self.mask_cutoff))
                print('Current max cutoff', torch.max(self.mask_cutoff))

            if train_size < 1:
                validate_elbo_val = 0
                validate_num = 0
                for _, (validate_x_batch, validate_y_batch, validate_y_raw_batch, validate_sf_batch, validate_sample_index_batch, validate_cell_atts_batch, validate_batch, validate_cutoff) in enumerate(validate_dataloader):
                    validate_x_batch = validate_x_batch.to(self.device)
                    validate_y_batch = validate_y_batch.to(self.device)
                    validate_y_raw_batch = validate_y_raw_batch.to(self.device)
                    validate_sf_batch = validate_sf_batch.to(self.device)
                    validate_batch = validate_batch.to(self.device)

                    validate_sample_index_batch = validate_sample_index_batch.to(self.device).to(dtype=torch.int)
                    validate_cell_atts_batch = validate_cell_atts_batch.to(self.device).to(dtype=torch.int)
                    validate_cutoff_batch = validate_cutoff.to(self.device).to(dtype=self.dtype)


                    validate_elbo, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                        self.forward(x=validate_x_batch, y=validate_y_batch, raw_y=validate_y_raw_batch, size_factors=validate_sf_batch, num_samples=num_samples, batch=validate_batch,
                                     sample_index=validate_sample_index_batch, cell_atts=validate_cell_atts_batch, cutoff=validate_cutoff_batch)

                    validate_elbo_val += validate_elbo.item()
                    validate_num += validate_x_batch.shape[0]

                validate_elbo_val = validate_elbo_val / validate_num

                print("Training epoch {}, validating ELBO:{:.8f}".format(epoch+1, validate_elbo_val))
                early_stopping(validate_elbo_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch+1))
                    break

        if save_model:
            torch.save(self.state_dict(), model_weights)
        return 

    def counterfactualPrediction(self, X, sample_index, cell_atts,
                                 perturb_cell_id = None,
                                 target_cell_tissue = None,
                                 target_cell_perturbation = None,
                                  n_samples=1, batch_size=512
                                 ):
        """
        Counterfactual Prediction

        General Parameters:
        -----------
        X: Location information (n_spots, 2).
        sample_index: index of cells/spots (n_spots, 1).
        cell_atts: matrix of cell/spot's attributes (n_spots, n_atts)
        perturb_cell_id: cells/spots to perturb (counterfactual prediction).
        
        Dataset specific Parameters:
        -----------
        target_cell_tissue: target tissue (cell) type for counterfactual prediction.
        target_cell_perturbation: target perturbation state for counterfactual prediction.
        """ 

        self.eval()

        cutoffs = self.mask_cutoff

        X = torch.tensor(X, dtype=self.dtype)
        total_sample_indices = torch.tensor(sample_index, dtype=torch.int)

        #add perturbation
        pert_cell_atts = cell_atts
        for i in perturb_cell_id.tolist():
            pert_cell_atts[:,0][i] = target_cell_tissue
            pert_cell_atts[:,1][i] = target_cell_perturbation

        pert_cell_atts = torch.tensor(pert_cell_atts, dtype=torch.int)

        mean_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            cell_atts_batch = pert_cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=torch.int)
            sample_index = total_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=torch.int)
            b = xbatch.shape[0]
            cutoff = cutoffs[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=self.dtype)
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            ybatch_ = lord_latents["total_latent"]

            qnet_mu, qnet_var = self.encoder(ybatch_)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l], cutoff=cutoff)
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            mean_samples_ = []
            for f in latent_samples:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.data.cpu().detach())

        mean_samples = torch.cat(mean_samples, dim=0)

        return mean_samples.numpy(), pert_cell_atts.numpy()
        
    def batching_predict_samples_target(self, target, X_test, X_train, 
                                        Y_sample_index, Y_cell_atts, 
                                        n_samples=1, batch_size=512):
        """
        Impute latent representations and denoised counts on unseen testing locations.

        Parameters:
        -----------
        target: target perturbation state of the imputted cells/spots.
        X_test: Location information of testing set (n_test_spots, 2).
        X_train: Location information of training set (n_train_spots, 2).
        sample_index: index of cells/spots (n_spots, 1).
        cell_atts: matrix of cell/spot's attributes (n_spots, n_atts)
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx

        cutoffs = self.mask_cutoff

        target_cell_ind = np.where(Y_cell_atts[:,1]==target)[0]
        X_train_target = X_train[target_cell_ind]
        print(f'Extract {len(target_cell_ind)} cells in {target} for in-painting')

        X_test = torch.tensor(X_test, dtype=self.dtype)
        X_train = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        X_train_target = torch.tensor(X_train_target, dtype=self.dtype).to(self.device)
        train_cell_atts = torch.tensor(Y_cell_atts, dtype=torch.int)
        train_sample_indices = torch.tensor(Y_sample_index, dtype=torch.int)

        latent_samples = []
        mean_samples = []

        train_num = X_train.shape[0]
        train_num_batch = int(math.ceil(1.0*X_train.shape[0]/batch_size))
        test_num = X_test.shape[0]
        test_num_batch = int(math.ceil(1.0*X_test.shape[0]/batch_size))

        x_train_select = []
        for e in range(X_test.shape[0]):
            x_train_select.append(find_nearest(X_train, X_test[e]))
        x_train_select = torch.stack(x_train_select)
        y_cutoff = cutoffs[x_train_select.long()]

        qnet_mu, qnet_var = [], []
        for batch_idx in range(train_num_batch):
            cell_atts_batch = train_cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            sample_index = train_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            b = cell_atts_batch.shape[0]
            cutoff_train_batch = cutoffs[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=self.dtype)
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            Y_train_batch_ = lord_latents["total_latent"]

            qnet_mu_, qnet_var_ = self.encoder(Y_train_batch_)
            qnet_mu.append(qnet_mu_)
            qnet_var.append(qnet_var_)
        qnet_mu = torch.cat(qnet_mu, dim=0)
        qnet_var = torch.cat(qnet_var, dim=0)

        def find_nearest_k(array, value, k=10, sample_size=5):
            distances = torch.sum((array - value) ** 2, dim=1)
            nearest_indices = torch.topk(distances, k, largest=False).indices  # Get k smallest distances
            selected_indices = nearest_indices[torch.randperm(k)[:sample_size]]
            return selected_indices

        for batch_idx in range(test_num_batch):
            x_test_batch = X_test[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)
            cutoff_test_batch = y_cutoff[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device).to(dtype=self.dtype)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            # x_train_select_batch represents the nearest X_train spots to x_test_batch
            x_train_gaussian_mu = []
            x_train_gaussian_var = []
            for e in range(x_test_batch.shape[0]):
                ind = find_nearest_k(X_train_target, x_test_batch[e])
                gaussian_mu_batch = qnet_mu[ind, self.GP_dim:]
                gaussian_var_batch = qnet_var[ind, self.GP_dim:]
                x_train_gaussian_mu.append(gaussian_mu_batch.mean(dim=0))
                x_train_gaussian_var.append(gaussian_var_batch.mean(dim=0))

            gaussian_mu = torch.stack(x_train_gaussian_mu)
            gaussian_var = torch.stack(x_train_gaussian_var)

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params_impute(index_points_test=x_test_batch, index_points_train=X_train, 
                                        y=gp_mu[:, l], noise=gp_var[:, l], x_cutoff=cutoff_train_batch, y_cutoff=cutoff_test_batch)
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.data.cpu().detach())
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            latent_samples_ = []
            for _ in range(n_samples):
                f = latent_dist.sample()
                latent_samples_.append(f)

            mean_samples_ = []
            for f in latent_samples_:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)
        mean_samples = torch.cat(mean_samples, dim=0)

        return latent_samples.numpy(), mean_samples.numpy()

    def batching_predict_samples_target2(self, target, tissue, 
                                         X_test, X_train, 
                                         Y_sample_index, Y_cell_atts, 
                                         n_samples=1, batch_size=512
                                        ):
        """
        Impute latent representations and denoised counts on unseen testing locations.

        Parameters:
        -----------
        target: target perturbation state of the imputted cells/spots.
        tissue: target tissue type of the imputted cells/spots.
        X_test: Location information of testing set (n_test_spots, 2).
        X_train: Location information of training set (n_train_spots, 2).
        sample_index: index of cells/spots (n_spots, 1).
        cell_atts: matrix of cell/spot's attributes (n_spots, n_atts)
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx

        cutoffs = self.mask_cutoff

        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx
        
        X_test = torch.tensor(X_test, dtype=self.dtype).to(self.device)
        X_train = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        train_cell_atts = torch.tensor(Y_cell_atts, dtype=torch.int)
        train_sample_indices = torch.tensor(Y_sample_index, dtype=torch.int)
        
        x_test_neb = []
        for e in range(X_test.shape[0]):
            x_test_neb.append(find_nearest(X_train, X_test[e]))
        x_test_neb = torch.stack(x_test_neb)
        y_cutoff = cutoffs[x_test_neb.long()]

        Y_test_index = x_test_neb.to(self.device)
        Y_test_target = torch.tensor(target, device=self.device)
        Y_test_tissue = torch.tensor(tissue, device=self.device)

        latent_samples = []
        mean_samples = []

        train_num = X_train.shape[0]
        train_num_batch = int(math.ceil(1.0*X_train.shape[0]/batch_size))
        test_num = X_test.shape[0]
        test_num_batch = int(math.ceil(1.0*X_test.shape[0]/batch_size))

        train_basal = []
        qnet_mu, qnet_var = [], []
        for batch_idx in range(train_num_batch):
            cell_atts_batch = train_cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            sample_index = train_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            b = cell_atts_batch.shape[0]
            cutoff_train_batch = cutoffs[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=self.dtype)
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            train_basal.append(lord_latents["basal_latent"])
            Y_train_batch_ = lord_latents["total_latent"]
            qnet_mu_, qnet_var_ = self.encoder(Y_train_batch_)
            qnet_mu.append(qnet_mu_)
            qnet_var.append(qnet_var_)
        
        train_basal = torch.cat(train_basal, dim=0)
        test_neb_basal = train_basal[Y_test_index]
        qnet_mu = torch.cat(qnet_mu, dim=0)
        qnet_var = torch.cat(qnet_var, dim=0)
        qnet_mu_test, qnet_var_test = [], []
        for batch_idx in range(test_num_batch):
            test_neb_basal_batch = test_neb_basal[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)
            cutoff_test_batch = y_cutoff[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device).to(dtype=self.dtype)
            b = test_neb_basal_batch.shape[0]
            l_tissue =self.lord_encoder.get_latent(att = Y_test_tissue, type='tissue', batch_size = b)
            l_tissue = l_tissue.repeat(b, 1)
            l_pert =self.lord_encoder.get_latent(att = Y_test_target, type='perturbation', batch_size = b)
            l_pert = l_pert.repeat(b, 1)
            Y_train_batch_ = torch.cat((test_neb_basal_batch, l_tissue, l_pert), dim=1)

            qnet_mu_, qnet_var_ = self.encoder(Y_train_batch_)
            qnet_mu_test.append(qnet_mu_)
            qnet_var_test.append(qnet_var_)
        
        qnet_mu_test = torch.cat(qnet_mu_test, dim=0)
        qnet_var_test = torch.cat(qnet_var_test, dim=0)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu_test[:, self.GP_dim:]
        gaussian_var = qnet_var_test[:, self.GP_dim:]

        for batch_idx in range(test_num_batch):
            x_test_batch = X_test[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params_impute(index_points_test=x_test_batch, index_points_train=X_train, 
                                        y=gp_mu[:, l], noise=gp_var[:, l], x_cutoff=cutoff_train_batch, y_cutoff=cutoff_test_batch)
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.data.cpu().detach())
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            latent_samples_ = []
            for _ in range(n_samples):
                f = latent_dist.sample()
                latent_samples_.append(f)

            mean_samples_ = []
            for f in latent_samples_:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.data.cpu().detach())
        latent_samples = torch.cat(latent_samples, dim=0)
        mean_samples = torch.cat(mean_samples, dim=0)

        return latent_samples.numpy(), mean_samples.numpy()
