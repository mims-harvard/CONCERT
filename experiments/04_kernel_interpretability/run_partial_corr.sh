#!/bin/bash -l
#SBATCH -J pfish_partial
#SBATCH -p kempner_requeue
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=16G
#SBATCH --time 0-00:20
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/perturbfish_interp/pfish_partial_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/perturbfish_interp/pfish_partial_%j.err

CODEBASE_ROOT=/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/CONCERT/codebase
source "${CODEBASE_ROOT}/config/paths.sh"
activate_env

W=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/perturbfish_interp
SPAC=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_spac1_niche_seen_lsfix/kernel_param_analysis

python -u "${W}/partial_corr_perturbfish.py" \
    --params_csv "${W}/pfish_tumor_niche/kernel_param_analysis/per_perturbation_kernel_params.csv" \
    --spac_eval_csv "${SPAC}/kernel_param_correlations_eval.csv" \
    --out_csv "${W}/pfish_tumor_niche/kernel_param_analysis/kernel_param_correlations_partial.csv"
