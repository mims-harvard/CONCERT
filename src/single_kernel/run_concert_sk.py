import math, os
from time import time

import torch
from concert_sk import CONCERT
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import h5py
import scanpy as sc
from preprocess import normalize, geneSelection


# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Spatial-aware perturbation prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--sample', default='sample')
    parser.add_argument('--data_index', default='x')
    parser.add_argument('--outdir', default='./outputs/')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--noise', default=0, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--GP_dim', default=2, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=8, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--dynamicVAE', default=True, type=bool, 
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=4, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=None, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--denoised_counts_file', default='denoised_counts.txt')
    parser.add_argument('--num_denoise_samples', default=10000, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pert_cells', default="patch_jak2.txt", type=str)
    parser.add_argument('--target_cell_tissue', default="tumor", type=str)
    parser.add_argument('--target_cell_perturbation', default="Jak2", type=str)

    args = parser.parse_args()

    def str_list_to_unique_index(str_list):
        original_numbers = np.array([sum(ord(char) for char in s) for s in str_list])
        renumbered = {num: idx + 1 for idx, num in enumerate(sorted(set(original_numbers)))}
        new_numbers = [renumbered[num] for num in original_numbers]
        return np.array(new_numbers)

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).astype('float32') # count matrix
    loc = np.array(data_mat['pos']).T.astype('float32') # location information
    tissue_ = np.array(data_mat['tissue']).astype('str') # tissue information
    tissue = str_list_to_unique_index(tissue_) - 1
    perturbation_ = np.array(data_mat['perturbation']).astype('str') # perturbation information
    perturbation = str_list_to_unique_index(perturbation_) - 1
    cell_atts = np.concatenate((tissue[:, None], perturbation[:, None]), axis=1)
    sample_indices = torch.tensor(np.arange(x.shape[0]), dtype=torch.int)

    data_mat.close()

    print(np.unique(tissue_, return_counts=True))
    print(np.unique(perturbation_, return_counts=True))

    tissue_dic = {tissue_[i]: tissue[i] for i in range(len(tissue_))}
    pert_dic = {perturbation_[i]: perturbation[i] for i in range(len(perturbation_))}
    pert_ind = np.loadtxt(args.pert_cells, dtype=int) - 1

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

    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * args.loc_range

    print(x.shape)
    print(loc.shape)

    if args.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
        print(initial_inducing_points.shape)
    else:
        loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(loc)
        initial_inducing_points = loc_kmeans.cluster_centers_

    adata = sc.AnnData(x, dtype="float32")

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    model = CONCERT(cell_atts=cell_atts, num_genes=adata.n_vars, input_dim=768, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata.n_obs, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE, 
        init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta, dtype=torch.float32, device=args.device)

    print(str(model))

    if not os.path.isfile(args.model_file):
        t0 = time()
        model.train_model(cell_atts=cell_atts, sample_indices=sample_indices,
                pos=loc, ncounts=adata.X, raw_counts=adata.raw.X, size_factors=adata.obs.size_factors,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
        print('Training time: %d seconds.' % int(time() - t0))
    else:
        model.load_model(args.model_file)

    data_index = args.sample + "_" + args.data_index

    #final_latent = model.batching_latent_samples(X=loc, sample_index=sample_indices, cell_atts=cell_atts, batch_size=args.batch_size)
    #np.savetxt(args.outdir + data_index+ '_final_latent.txt', final_latent, delimiter=",")

    #denoised_counts = model.batching_denoise_counts(X=loc, sample_index=sample_indices, cell_atts=cell_atts, batch_size=args.batch_size, n_samples=25)
    #adata = sc.AnnData(denoised_counts)
    #adata.write(args.outdir + data_index + '_denoised_counts.h5ad')

    perturbed_counts = model.counterfactualPrediction(X=loc, sample_index=sample_indices, cell_atts=cell_atts, batch_size=args.batch_size, n_samples=25, perturb_cell_id = pert_ind, 
                                                      target_cell_tissue = tissue_dic[args.target_cell_tissue], target_cell_perturbation = pert_dic[args.target_cell_perturbation])
    # save h5ad
    adata = sc.AnnData(perturbed_counts)
    adata.obs['perturbed'] = [1 if i in pert_ind else 0 for i in range(adata.n_obs)]
    adata.write(args.outdir + data_index + "_" + args.target_cell_tissue + "_" + args.target_cell_perturbation + '_perturbed_counts.h5ad')
    print("Done")
    






