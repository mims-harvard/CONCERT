import numpy as np
import torch
import torch.nn.functional as F
from lord_utils import *
from torch import nn
import random
import os
from scvi.nn import FCLayers

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class RegularizedEmbedding(nn.Module):
    """Regularized embedding module."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        sigma: float
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_input,
            embedding_dim=n_output,
        )
        self.sigma = sigma

    def forward(self, x):
        """Forward pass."""
        x_ = self.embedding(x)
        if self.training and self.sigma != 0:
            #print("Adding noise")
            noise = torch.zeros_like(x_)
            noise.normal_(mean=0, std=self.sigma)
            x_ = x_ + noise
        return x_

def _move_inputs(*inputs, device="cuda"):
    def mv_input(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return [mv_input(y) for y in x]

    return [mv_input(x) for x in inputs]

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class Lord_encoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: List[int],
        num_genes: int,
        attributes: List[str],
        attributes_type: List[str],
        labels: List[List],
        device="cuda:0",
        append_layer_width=None,
        multi_task: bool = False,
        noise: float = 0.1,
    ):
        super(Lord_encoder, self).__init__()
        # set generic attributes
        self.embedding_dim = embedding_dim
        self.num_genes = num_genes
        self.device = device
        # early-stopping
        self.best_score = -1e3
        self.patience_trials = 0
        self.multi_task = multi_task
        self.px_r = torch.nn.Parameter(torch.randn(num_genes))
        self.noise = noise
        # set hyperparameters
        self.labels = labels
        self.num_covariates = labels.shape[1]
        print("Number of covariates: ", self.num_covariates)
        assert self.num_covariates < 10, "Too many covariates"
        self.unique_labels = [np.unique(labels[:,i]) for i in range(self.num_covariates)]
        self.attribute = attributes
        self.attribute_type = attributes_type
        self.attribute_type_dic = {}
        for i in range(self.num_covariates):
            self.attribute_type_dic[attributes[i]] = attributes_type[i]
        
        self.embedding_dim_map = {} 
        for i in range(self.num_covariates):
            self.embedding_dim_map[self.attribute[i]] = self.embedding_dim[i]

        self.categorical_attributes_list = attributes[attributes_type == "categorical"]
        self.categorical_attributes_map = {}
        for i in range(self.num_covariates):
            self.categorical_attributes_map[self.categorical_attributes_list[i]] = self.unique_labels[i]

        self.s_encoder = nn.ModuleDict()
        for i in range(self.num_covariates):
            if attributes_type[i] == "categorical":
                self.s_encoder[attributes[i]] = torch.nn.Embedding(
                    len(self.unique_labels[i]),
                    self.embedding_dim[i],
                )
            elif attributes_type[i] == "ordinal":
                self.s_encoder[attributes[i]] = FCLayers(
                    n_in=1,
                    n_out=self.embedding_dim[i],
                    n_layers=3,
                    n_hidden=128,
                    dropout_rate=0.3,
                    bias=False,
                    use_activation=True,
                )
        
        self.z_encoder = RegularizedEmbedding(
                    n_input = len(self.labels[:,0]),
                    n_output=self.embedding_dim[-1],
                    sigma=self.noise
                    )

        if append_layer_width:
            self.num_genes = append_layer_width
        
        self.apply(init_weights)

        self.to(self.device)

        self.history = {"epoch": [], "stats_epoch": []}

    def predict(
        self,
        sample_indices,
        batch_size,
        labels,
    ):
          
        label_dic = {}
        for i in range(self.num_covariates):
            label_dic[self.attribute[i]] = labels[:,i]
        
        z = self.z_encoder(sample_indices)
        if torch.isnan(z).any():
            print("nan in z")

        s = {}
        for attribute_, embedding_ in self.s_encoder.items():
            if self.attribute_type_dic[attribute_] == "ordinal":
                input = label_dic[attribute_].unsqueeze(1).float()
                s_ = embedding_(input)
            else:
                s_ = embedding_(label_dic[attribute_])
            s_ = s_.view(batch_size, self.embedding_dim_map[attribute_]).unsqueeze(0)
            if torch.isnan(s_).any():
                print(f"nan in {attribute_}")
            s[attribute_] = s_

        inference_output = {}
        latent_basal = z.squeeze()
        latent_vecs = [latent_basal]
        for key_, latent_ in s.items():
                latent_vecs.append(latent_.squeeze())
                inference_output[key_] = latent_.squeeze()
        latent = torch.cat(latent_vecs, dim=-1)
        inference_output["total_latent"] = latent
        inference_output["basal_latent"] = latent_basal
        return inference_output
    
    def get_latent(self, att, type, batch_size):
        if type ==  "perturbation":
            z = self.s_encoder['perturbation'](att)
        elif type == "tissue":
            z = self.s_encoder['tissue'](att)
        elif type == "batch":
            z = self.s_encoder['batch'](att)
        else:
            print("Attribute type not found")
        return z
