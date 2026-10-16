#!/bin/bash -l
#SBATCH -J pfish_partial
#SBATCH -p <partition>
#SBATCH --mem=16G
#SBATCH --time 0-00:20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -o pfish_partial_%j.out
#SBATCH -e pfish_partial_%j.err

# Locate config/paths.sh. Under sbatch the batch script is copied to a spool directory, so a
# path relative to this file does not resolve there; CODEBASE_ROOT is the fallback.
_PATHS="$(dirname "${BASH_SOURCE[0]}")/../../config/paths.sh"
[ -f "$_PATHS" ] || _PATHS="${CODEBASE_ROOT:-}/config/paths.sh"
[ -f "$_PATHS" ] || { echo "[FATAL] cannot find config/paths.sh; sbatch --export=ALL,CODEBASE_ROOT=/path/to/repo $0"; exit 1; }
source "$_PATHS"
activate_env

# Partial rank correlations of the Perturb-FISH kernel parameters (steps 2 and 3 of the README).
# Defaults read the shipped tables in data/perturbfish/; PFISH_ANALYSIS=<dir> reads a fresh
# run_perturbfish_interp.sh output (<its OUT>/kernel_param_analysis).
EXP="${CODEBASE_ROOT}/experiments/04_kernel_interpretability"
ANA="${PFISH_ANALYSIS:-${PERTURBFISH_DIR}/kernel_param_analysis}"
SPAC="${CONCERT_DATA_DIR}/spacseq/kernel_params"
OUT="${PFISH_FIG_OUT:-${CONCERT_WORKDIR}/perturbfish}"
mkdir -p "$OUT"

python -u "${EXP}/partial_corr_perturbfish.py" \
    --params_csv "${ANA}/per_perturbation_kernel_params.csv" \
    --spac_eval_csv "${SPAC}/kernel_param_correlations_eval.csv" \
    --out_csv "${OUT}/kernel_param_correlations_partial.csv"

python -u "${EXP}/make_fig_perturbfish_consistency.py" \
    --pfish_corr "${OUT}/kernel_param_correlations_partial.csv" \
    --pfish_params "${ANA}/per_perturbation_kernel_params.csv" \
    --spac_corr "${SPAC}/kernel_param_correlations_eval.csv" \
    --out_pdf "${OUT}/fig_perturbfish_consistency.pdf" \
    --out_stats "${OUT}/fig_perturbfish_consistency_stats.json"
