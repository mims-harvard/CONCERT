import math
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
from SVGP import SVGP
from I_PID import PIDControl
from VAE_utils import *
from collections import deque
from lord_batch import Lord_encoder

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
    def __init__(self, num_genes, input_dim, GP_dim, Normal_dim, cell_atts, encoder_layers, decoder_layers, noise, encoder_dropout, decoder_dropout, 
                    fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, 
                    KL_loss, dynamicVAE, init_beta, min_beta, max_beta, dtype, device):
        super(CONCERT, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.input_dim = input_dim
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss          # expected KL loss value
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta           # beta controls the weight of reconstruction loss
        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.noise = noise              # intensity of random noise
        self.device = device
        self.num_genes = num_genes
        self.encoder = DenseEncoder(input_dim=input_dim, hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = buildNetwork([GP_dim+Normal_dim]+decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], self.num_genes), MeanAct())
        else:
            self.dec_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, self.num_genes), MeanAct())
        self.dec_disp = nn.Parameter(torch.randn(self.num_genes), requires_grad=True)       # trainable dispersion parameter for NB loss
        self.lord_encoder = Lord_encoder(embedding_dim=[256, 256, 256], num_genes = self.num_genes, labels = cell_atts, 
                                         attributes=["tissue", "perturbation"], attributes_type = ["categorical", "categorical"], 
                                         noise=self.noise, device=device)

        #encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=1)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

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


    def forward(self, x, y, raw_y, sample_index, cell_atts, size_factors, num_samples=1):
        """
        Forward pass.

        Parameters:
        -----------
        x: mini-batch of positions.
        y: mini-batch of preprocessed counts.
        raw_y: mini-batch of raw counts.
        size_factor: mini-batch of size factors.
        num_samples: number of samplings of the posterior distribution of latent embedding.

        raw_y and size_factor are used for NB likelihood.
        """ 

        self.train()
        b = y.shape[0]
        lord_latents =self.lord_encoder.predict(sample_indices=sample_index, labels = cell_atts, batch_size = b)
        y_ = lord_latents["total_latent"]
        #print(y_.shape)
        #y_ = self.transformer_encoder(y_)
        #y_ = y_.view(-1, self.input_dim) 
        #print(y_.shape)
        qnet_mu, qnet_var = self.encoder(y_)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l])
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l],
                                                                    noise=gp_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l)

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
            disp_samples_ = (torch.exp(torch.clamp(self.dec_disp, -15., 15.))).unsqueeze(0)

            mean_samples.append(mean_samples_)
            disp_samples.append(disp_samples_)
            recon_loss += self.NB_loss(x=raw_y, mean=mean_samples_, disp=disp_samples_, scale_factor=size_factors)
            recon_loss_mse += self.mse(mean_samples_, raw_y)
        
        recon_loss = recon_loss / num_samples
        recon_loss_mse = recon_loss_mse / num_samples

        noise_reg = 0
#         if self.noise > 0:
#             for _ in range(num_samples):
#                 qnet_mu_, qnet_var_ = self.encoder(y + torch.randn_like(y)*self.noise)
#                 gp_mu_ = qnet_mu_[:, 0:self.GP_dim]
#                 gp_var_ = qnet_var_[:, 0:self.GP_dim]

# #                gaussian_mu_ = qnet_mu_[:, self.GP_dim:]
# #                gaussian_var_ = qnet_var_[:, self.GP_dim:]

#                 gp_p_m_, gp_p_v_ = [], []
#                 for l in range(self.GP_dim):
#                     gp_p_m_l_, gp_p_v_l_, _, _ = self.svgp.approximate_posterior_params(x, x,
#                                                                     gp_mu_[:, l], gp_var_[:, l])
#                     gp_p_m_.append(gp_p_m_l_)
#                     gp_p_v_.append(gp_p_v_l_)

#                 gp_p_m_ = torch.stack(gp_p_m_, dim=1)
#                 gp_p_v_ = torch.stack(gp_p_v_, dim=1)
#                 noise_reg += torch.sum((gp_p_m - gp_p_m_)**2)
#             noise_reg = noise_reg / num_samples

        latent_unknown_attributes = lord_latents["basal_latent"]
        unknown_attribute_penalty = unknown_attribute_penalty_loss(latent_unknown_attributes)
        
        #lord loss
        lord_loss = unknown_attribute_penalty + recon_loss_mse

        # # ELBO
        # if self.noise > 0 :
        #     elbo = recon_loss + noise_reg * self.input_dim / self.GP_dim + self.beta * gp_KL_term + self.beta * gaussian_KL_term
        # else:
        #     elbo = recon_loss + self.beta * gp_KL_term + self.beta * gaussian_KL_term

        elbo = recon_loss + self.beta * gp_KL_term + self.beta * gaussian_KL_term + lord_loss

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, gp_p_m, gp_p_v, qnet_mu, qnet_var, \
            mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg, unknown_attribute_penalty, recon_loss_mse


    def batching_latent_samples(self, X, sample_index, cell_atts, batch_size=512):
        """
        Output latent embedding.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        """ 

        self.eval()

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
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            ybatch_ = lord_latents["total_latent"]

            qnet_mu, qnet_var = self.encoder(ybatch_)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
#            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l])
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
        X: array_like, shape (n_spots, 2)
            Location information.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        cell_atts = torch.tensor(cell_atts, dtype=torch.int)
        total_sample_indices = torch.tensor(sample_index, dtype=torch.int)

        mean_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            cell_atts_batch = cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=torch.int)
            sample_index = total_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device).to(dtype=torch.int)
            b = xbatch.shape[0]
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            ybatch_ = lord_latents["total_latent"]

            qnet_mu, qnet_var = self.encoder(ybatch_)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l])
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

        return mean_samples.numpy()


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
        X_test: array_like, shape (n_test_spots, 2)
            Location information of testing set.
        X_train: array_like, shape (n_train_spots, 2)
            Location information of training set.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

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

        qnet_mu, qnet_var = [], []
        for batch_idx in range(train_num_batch):
            cell_atts_batch = train_cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            sample_index = train_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            b = cell_atts_batch.shape[0]
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            Y_train_batch_ = lord_latents["total_latent"]

            qnet_mu_, qnet_var_ = self.encoder(Y_train_batch_)
            qnet_mu.append(qnet_mu_)
            qnet_var.append(qnet_var_)
        qnet_mu = torch.cat(qnet_mu, dim=0)
        qnet_var = torch.cat(qnet_var, dim=0)

        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx

        for batch_idx in range(test_num_batch):
            x_test_batch = X_test[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)

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
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(index_points_test=x_test_batch, index_points_train=X_train, 
                                        y=gp_mu[:, l], noise=gp_var[:, l])
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


    def differential_expression(self, group1_idx, group2_idx, sample_index, cell_atts, num_denoise_samples=10000, batch_size=512, pos=None, ncounts=None, 
            gene_name=None, raw_counts=None, estimate_pseudocount=True):
        """
        Differential expression analysis.

        Parameters:
        -----------
        group1_idx: array_like, shape (n_group1)
            Index of group1.
        group2_idx: array_like, shape (n_group2)
            Index of group2.
        num_denoise_samples: Number of samplings in each group.
        pos: array_like, shape (n_spots, 2)
            Location information.
        ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        gene_name: array_like, shape (n_genes)
            gene names.
        raw_counts: array_like, shape (n_spots, n_genes)
            Raw count matrix.
        estimate_pseudocount: Whether to estimate pseudocount from data, otherwise use default value 0.05.
        """ 

        group1_idx_sampling = group1_idx[np.random.randint(group1_idx.shape[0], size=num_denoise_samples)]
        group2_idx_sampling = group2_idx[np.random.randint(group2_idx.shape[0], size=num_denoise_samples)]

        group1_denoised_counts = self.batching_denoise_counts(X=pos[group1_idx_sampling], sample_index = sample_index[group1_idx_sampling], cell_atts=cell_atts[group1_idx_sampling], batch_size=batch_size, n_samples=1)
        group2_denoised_counts = self.batching_denoise_counts(X=pos[group2_idx_sampling], sample_index = sample_index[group2_idx_sampling], cell_atts=cell_atts[group2_idx_sampling], batch_size=batch_size, n_samples=1)


        if estimate_pseudocount:
            group1_where_zero = np.max(raw_counts[group1_idx], axis=0) == 0
            group2_where_zero = np.max(raw_counts[group2_idx], axis=0) == 0
            group1_max_denoised_counts = np.max(group1_denoised_counts, axis=0)
            group2_max_denoised_counts = np.max(group2_denoised_counts, axis=0)

            if group1_where_zero.sum() >= 1:
                group1_atefact_count = group1_max_denoised_counts[group1_where_zero]
                group1_eps = np.quantile(group1_atefact_count, q=0.9)
            else:
                group1_eps = 1e-10
            if group2_where_zero.sum() >= 1:
                group2_atefact_count = group2_max_denoised_counts[group2_where_zero]
                group2_eps = np.quantile(group2_atefact_count, q=0.9)
            else:
                group2_eps = 1e-10

            eps = np.maximum(group1_eps, group2_eps)
            eps = np.clip(eps, a_min=0.05, a_max=0.5)
            print("Estimated pseudocounts", eps)
        else:
            eps = 0.05


        group1_denoised_mean = np.mean(group1_denoised_counts, axis=0)
        group2_denoised_mean = np.mean(group2_denoised_counts, axis=0)

        lfc = np.log2(group1_denoised_mean + eps) - np.log2(group2_denoised_mean + eps)
        p_lfc = np.log2(group1_denoised_counts + eps) - np.log2(group2_denoised_counts + eps)
        mean_lfc = np.mean(p_lfc, axis=0)
        median_lfc = np.median(p_lfc, axis=0)
        sd_lfc = np.std(p_lfc, axis=0)
        delta = gmm_fit(data=mean_lfc)
        print("LFC delta:", delta)
        is_de = (np.abs(p_lfc) >= delta).mean(0)
        not_de = (np.abs(p_lfc) < delta).mean(0)
        bayes_factor = np.log(is_de + 1e-10) - np.log(not_de + 1e-10)


        group1_raw_mean = np.mean(raw_counts[group1_idx], axis=0)
        group2_raw_mean = np.mean(raw_counts[group2_idx], axis=0)

        res_dat = pd.DataFrame(data={'LFC': lfc, 'mean_LFC': mean_lfc, 'median_LFC': median_lfc, 'sd_LFC': sd_lfc,
                                     'prob_DE': is_de, 'prob_not_DE': not_de, 'bayes_factor': bayes_factor,
                                     'denoised_mean1': group1_denoised_mean, 'denoised_mean2': group2_denoised_mean,
                                    'raw_mean1': group1_raw_mean, 'raw_mean2': group2_raw_mean}, index=gene_name)

        return res_dat


    def train_model(self, pos, cell_atts, sample_indices, ncounts, raw_counts, size_factors, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, 
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

        # print(torch.tensor(pos, dtype=self.dtype).shape)
        # print(torch.tensor(ncounts, dtype=self.dtype).shape)
        # print(torch.tensor(raw_counts, dtype=self.dtype).shape)
        # print(torch.tensor(size_factors, dtype=self.dtype).shape)
        # print(sample_indices.shape)
        # print(torch.tensor(cell_atts).shape)

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(ncounts, dtype=self.dtype), 
                        torch.tensor(raw_counts, dtype=self.dtype), torch.tensor(size_factors, dtype=self.dtype),
                        torch.tensor(sample_indices, dtype=torch.int), torch.tensor(cell_atts, dtype=torch.int))

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

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

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
            for batch_idx, (x_batch, y_batch, y_raw_batch, sf_batch, sample_index_batch, cell_atts_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device).to(dtype=self.dtype)
                y_batch = y_batch.to(self.device).to(dtype=self.dtype)
                y_raw_batch = y_raw_batch.to(self.device).to(dtype=self.dtype)
                sf_batch = sf_batch.to(self.device).to(dtype=self.dtype)
                sample_index_batch = sample_index_batch.to(self.device).to(dtype=torch.int)
                cell_atts_batch = cell_atts_batch.to(self.device).to(dtype=torch.int)

                elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg, unknown_loss, mse_loss = \
                    self.forward(x=x_batch, y=y_batch, raw_y=y_raw_batch, size_factors=sf_batch, num_samples=num_samples, sample_index=sample_index_batch, cell_atts=cell_atts_batch)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                unknown_loss_val += unknown_loss.item()
                mse_loss_val += mse_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                gaussian_KL_term_val += gaussian_KL_term.item()
                #if self.noise > 0:
                #    noise_reg_val += noise_reg.item()

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
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

            if train_size < 1:
                validate_elbo_val = 0
                validate_num = 0
                for _, (validate_x_batch, validate_y_batch, validate_y_raw_batch, validate_sf_batch, validate_sample_index_batch, validate_cell_atts_batch) in enumerate(validate_dataloader):
                    validate_x_batch = validate_x_batch.to(self.device)
                    validate_y_batch = validate_y_batch.to(self.device)
                    validate_y_raw_batch = validate_y_raw_batch.to(self.device)
                    validate_sf_batch = validate_sf_batch.to(self.device)
                    validate_sample_index_batch = validate_sample_index_batch.to(self.device).to(dtype=torch.int)
                    validate_cell_atts_batch = validate_cell_atts_batch.to(self.device).to(dtype=torch.int)

                    validate_elbo, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                        self.forward(x=validate_x_batch, y=validate_y_batch, raw_y=validate_y_raw_batch, size_factors=validate_sf_batch, num_samples=num_samples,
                                     sample_index=validate_sample_index_batch, cell_atts=validate_cell_atts_batch)

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

    def counterfactualPrediction(self, X, sample_index, cell_atts, 
                                 n_samples=1, batch_size=512,
                                 perturb_cell_id = None,
                                 target_cell_tissue = None,
                                 target_cell_perturbation = None
                                 ):
        """
        Counterfactual Prediction

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        """ 

        self.eval()

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
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            ybatch_ = lord_latents["total_latent"]

            qnet_mu, qnet_var = self.encoder(ybatch_)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l])
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

        return mean_samples.numpy()
    
    def batching_predict_samples_cp(self, X_test, X_train, Y_sample_index, Y_cell_atts, n_samples=1, batch_size=512,
                                    perturb_cell_id = None, target_cell_tissue = None, target_cell_perturbation = None):
        """
        Impute latent representations and denoised counts on unseen testing locations.

        Parameters:
        -----------
        X_test: array_like, shape (n_test_spots, 2)
            Location information of testing set.
        X_train: array_like, shape (n_train_spots, 2)
            Location information of training set.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()
        #add perturbation
        pert_cell_atts = Y_cell_atts
        for i in perturb_cell_id.tolist():
            pert_cell_atts[:,0][i] = target_cell_tissue
            pert_cell_atts[:,1][i] = target_cell_perturbation

        X_test = torch.tensor(X_test, dtype=self.dtype)
        X_train = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        train_cell_atts = torch.tensor(pert_cell_atts, dtype=torch.int)
        train_sample_indices = torch.tensor(Y_sample_index, dtype=torch.int)

        latent_samples = []
        mean_samples = []

        train_num = X_train.shape[0]
        train_num_batch = int(math.ceil(1.0*X_train.shape[0]/batch_size))
        test_num = X_test.shape[0]
        test_num_batch = int(math.ceil(1.0*X_test.shape[0]/batch_size))

        qnet_mu, qnet_var = [], []
        for batch_idx in range(train_num_batch):
            cell_atts_batch = train_cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            sample_index = train_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            b = cell_atts_batch.shape[0]
            lord_latents =self.lord_encoder.predict(sample_indices=sample_index, batch_size = b, labels = cell_atts_batch)
            Y_train_batch_ = lord_latents["total_latent"]

            qnet_mu_, qnet_var_ = self.encoder(Y_train_batch_)
            qnet_mu.append(qnet_mu_)
            qnet_var.append(qnet_var_)
        qnet_mu = torch.cat(qnet_mu, dim=0)
        qnet_var = torch.cat(qnet_var, dim=0)

        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx

        for batch_idx in range(test_num_batch):
            x_test_batch = X_test[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)

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
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(index_points_test=x_test_batch, index_points_train=X_train, y=gp_mu[:, l], noise=gp_var[:, l])
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

    def batching_predict_samples_target(self, target, tissue, X_test, X_train, Y_sample_index, Y_cell_atts, n_samples=1, batch_size=512):
        """
        Impute latent representations and denoised counts on unseen testing locations.

        Parameters:
        -----------
        X_test: array_like, shape (n_test_spots, 2)
            Location information of testing set.
        X_train: array_like, shape (n_train_spots, 2)
            Location information of training set.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

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

        qnet_mu, qnet_var = [], []
        for batch_idx in range(train_num_batch):
            cell_atts_batch = train_cell_atts[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            sample_index = train_sample_indices[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device).to(dtype=torch.int)
            b = cell_atts_batch.shape[0]
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
        
        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx

        for batch_idx in range(test_num_batch):
            x_test_batch = X_test[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)

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
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(index_points_test=x_test_batch, index_points_train=X_train, 
                                        y=gp_mu[:, l], noise=gp_var[:, l])
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

    def batching_predict_samples_target2(self, target, tissue, X_test, X_train, Y_sample_index, Y_cell_atts, n_samples=1, batch_size=512):
        """
        Impute latent representations and denoised counts on unseen testing locations.

        Parameters:
        -----------
        X_test: array_like, shape (n_test_spots, 2)
            Location information of testing set.
        X_train: array_like, shape (n_train_spots, 2)
            Location information of training set.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

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
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(index_points_test=x_test_batch, index_points_train=X_train, 
                                        y=gp_mu[:, l], noise=gp_var[:, l])
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