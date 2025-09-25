import math, os
from time import time

import torch
from concert_batch_gut import CONCERT
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import h5py
import scanpy as sc
from preprocess import normalize, geneSelection
import pandas as pd

# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Spatial dependency-aware variational autoencoder',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--data_index', default='x')
    parser.add_argument('--outdir', default='./outputs/')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--noise', default=0.25, type=float)
    parser.add_argument('--dropoutE', default=0., type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0., type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64, 32], type=int)
    parser.add_argument('--GP_dim', default=2, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=8, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[64, 128], type=int)
    parser.add_argument('--dynamicVAE', default=True, type=bool, 
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=5, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=6, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--denoised_counts_file', default='denoised_counts.txt')
    parser.add_argument('--num_denoise_samples', default=10000, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--allow_batch_kernel_scale', default=False, type=bool)
    parser.add_argument('--shared_dispersion', default=True, type=bool)
    parser.add_argument('--pert_cells', default="D73", type=str)
    parser.add_argument('--target_cell_day', default=13, type=float)
    parser.add_argument('--target_cell_perturbation', default="0.0", type=str)
    parser.add_argument('--stage', default="train", type=str)

    args = parser.parse_args()

    def str_list_to_unique_index(str_list):
        original_numbers = np.array([sum(ord(char) for char in s) for s in str_list])
        renumbered = {num: idx + 1 for idx, num in enumerate(sorted(set(original_numbers)))}
        new_numbers = [renumbered[num] for num in original_numbers]
        return np.array(new_numbers)

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).T.astype('float32') # count matrix
    loc = np.array(data_mat['pos']).T.astype('float32') # location information
    loc = loc[:, :2]
    region_ = np.array(data_mat['region']).astype('str') # tissue information
    region = str_list_to_unique_index(region_) - 1
    perturbation_ = np.array(data_mat['perturbation']).astype('str') # perturbation information
    perturbation = str_list_to_unique_index(perturbation_) - 1
    day_ = np.array(data_mat['day']).astype('str') # perturbation information
    day = []
    batch = []
    for i in day_:
        if i == "D0":
            day.append(1.)
            batch.append(0)
        elif i == "D12":
            day.append(12.)
            batch.append(1)
        elif i == "D30":
            day.append(30.)
            batch.append(2)
        elif i == "D73":
            day.append(73.)
            batch.append(3)

    day = np.array(day)
    num_classes = len(np.unique(batch))
    batch = np.eye(num_classes)[batch]
    print(batch.shape)

    #cell_atts = np.concatenate((region[:, None], perturbation[:, None], day[:, None]), axis=1)
    cell_atts = np.concatenate((perturbation[:, None], day[:, None]), axis=1)
    sample_indices = torch.tensor(np.arange(x.shape[0]), dtype=torch.int)

    data_mat.close()

    print(np.unique(day_, return_counts=True))
    print(np.unique(perturbation_, return_counts=True))

    pert_dic = {perturbation_[i]: perturbation[i] for i in range(len(perturbation_))}
    day_dic = {day_[i]: day[i] for i in range(len(day_))}
    print(day_dic)

    if args.batch_size == "auto":
        if x.shape[0] <= 1024:
            args.batch_size = 128
        elif x.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)

    print(args)

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]
        np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

    n_batch = batch.shape[1]
    # scale locations per batch
    loc_scaled = np.zeros(loc.shape, dtype=np.float64)
    for i in range(n_batch):
        scaler = MinMaxScaler()
        b_loc = loc[batch[:,i]==1., :]
        b_loc = scaler.fit_transform(b_loc) * args.loc_range
        loc_scaled[batch[:,i]==1., :] = b_loc
    loc = loc_scaled

    loc = np.concatenate((loc, batch), axis=1)

    print(x.shape)
    print(loc.shape)

# build inducing point matrix with batch index
    eps = 1e-5
    initial_inducing_points_0_ = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
    initial_inducing_points_0 = np.tile(initial_inducing_points_0_, (n_batch, 1))
    initial_inducing_points_1 = []
    for i in range(n_batch):
        initial_inducing_points_1_ = np.zeros((initial_inducing_points_0_.shape[0], n_batch))
        initial_inducing_points_1_[:, i] = 1
        initial_inducing_points_1.append(initial_inducing_points_1_)
    initial_inducing_points_1 = np.concatenate(initial_inducing_points_1, axis=0)
    initial_inducing_points = np.concatenate((initial_inducing_points_0, initial_inducing_points_1), axis=1)
    print(initial_inducing_points.shape)

    adata = sc.AnnData(x, dtype="float32")

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    model = CONCERT(cell_atts=cell_atts, num_genes=adata.n_vars, input_dim=192, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, n_batch=n_batch, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD, shared_dispersion=args.shared_dispersion,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, allow_batch_kernel_scale=args.allow_batch_kernel_scale,
        N_train=adata.n_obs, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE, init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta, 
        dtype=torch.float32, device=args.device)

    print(str(model))

    if args.stage == 'train':
        t0 = time()
        model.train_model(pos=loc, ncounts=adata.X, raw_counts=adata.raw.X, size_factors=adata.obs.size_factors, batch=batch,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
        print('Training time: %d seconds.' % int(time() - t0))
    else:
        model.load_model(args.model_file)
        pert_ind = np.where(day_ == args.pert_cells)[0]

        final_latent = model.batching_latent_samples(X=loc, sample_index=sample_indices, cell_atts=cell_atts, batch_size=args.batch_size)
        #np.savetxt(args.outdir + data_index+ '_final_latent.txt', final_latent, delimiter=",")

        # denoised_counts = model.batching_denoise_counts(X=loc, sample_index=sample_indices, cell_atts=cell_atts, batch_size=args.batch_size, n_samples=25)
        #save as h5ad
        #adata = sc.AnnData(denoised_counts, obs=adata.obs)
        #adata.write(args.outdir + "res_gut_" + args.pert_cells + "_N" + str(args.noise) + '_denoised_counts.h5ad')

        perturbed_counts, perturbed_atts = model.counterfactualPrediction(X=loc, sample_index=sample_indices, cell_atts=cell_atts, batch_size=args.batch_size, n_samples=25,
                                                      perturb_cell_id = pert_ind,
                                                      target_cell_day = args.target_cell_day, 
                                                      target_cell_perturbation = pert_dic[args.target_cell_perturbation]
                                                      )

        #save as h5ad
        perturbed_counts = perturbed_counts[pert_ind, :]
        perturbed_atts = perturbed_atts[pert_ind, :]
        perturbed_atts = pd.DataFrame(perturbed_atts, columns=["perturbation", "day"])
        adata = sc.AnnData(perturbed_counts, obs=perturbed_atts)
        adata.write(args.outdir + "res_gut_" + args.pert_cells + "_D" + str(args.target_cell_day) + "_P" + args.target_cell_perturbation + "_N" + str(args.noise) + '_perturbed_counts.h5ad')

