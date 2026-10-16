#!/bin/bash -l
#SBATCH -J spac_sens_ss
#SBATCH -p <partition>
#SBATCH --mem=200G
#SBATCH --time 0-12:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o spac_niche_%x_%j.out
#SBATCH -e spac_niche_%x_%j.err

# spac1 NICHE-FiLM training (code dir = CONCERT-spac-niche) with the niche-conditioned FiLM head:
#   --use_niche_interaction False --use_niche_film True ...
#   Data, holdout, hvg, inducing=16, ls band [3,18], encoder, basal RFF and gp_pp as in the base config.
#   NicheDescriptor uses batch-local coord-kNN of gs_mu (no graph dep; --use_nngp/paracrine False here anyway).
# Usage: sbatch sub_train_spac_niche.sh <regime:seen|unseen>
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DS="spac1"
REGIME="${1:?need regime seen|unseen}"
NS="${CONCERT_WORKDIR}"
DATA="${CONCERT_DATA_DIR}"
SP=${DATA}/spacseq/raw
HOCO=${DATA}/benchmark_assets/holdout_coords

FILES="${SP}/raw_output_1-1.h5ad"   # SINGLE SLIDE -- timing arm for the SI sensitivity analysis
IND=6; MINCELLS=25; LABELS=""; ACTCORR=True
HOLD="${HOCO}/holdout_spac_${REGIME}.json"
[ -f "$HOLD" ] || { echo "[FATAL] holdout file missing: $HOLD"; exit 1; }

OUT="${NS}/outputs/ms_${DS}_sens_ss_${REGIME}_${TAG:-base}"
mkdir -p "$OUT" "${NS}/logs"
cd ${CODEBASE_ROOT}/models/concert_spac_niche

GUARD="--ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 --fpert_svgp_inducing 16 --fpert_svgp_ls_init 0.25 --fpert_ls_min 0.15 --fpert_ls_max 0.90"
# Niche-FiLM replaces the additive niche interaction (matches the Perturb-map niche hyperparams).
NICHE="--use_niche_interaction False --use_niche_film True --niche_film_dim 8 --niche_film_k 25 --niche_film_gcap 3.0 --niche_film_l2 1e-3"

echo "[$(date)] spac NICHE-FiLM train ${DS}/${REGIME} HOLD=${HOLD} -> ${OUT}"
python -u run_concert_spac.py \
    --config config.yaml --sample "ms_${DS}" --stage train \
    --multi_sample_files "${FILES}" --outdir "$OUT" --model_file "model.pt" --project_index spac \
    --holdout_file "${HOLD}" \
    --hvg_max_rank 2000 --seed ${SEED:-42} \
    --inducing_point_steps ${IND} --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True \
    --maxiter 1500 --patience 50 --earlystop_min_delta 0.05 --lr 1e-4 --batch_size 2048 --train_size 0.95 \
    --use_action_correction ${ACTCORR} --use_pad False ${LABELS} --min_cells_per_pert ${MINCELLS} \
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
    --dynamicVAE False --beta_warmup_epochs 20 --beta_target 1.0 --free_bits 0.5 --gp_free_bits 1.0 \
    --use_nngp False --use_paracrine False ${NICHE} \
    --use_pert_gate True --gate_hidden 64 \
    --use_f_pert True --f_pert_init_std 0.01 --f_pert_clamp 5.0 \
    --spatial_basal_only True --lfc_lambda 1.0 \
    --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0 \
    --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
    ${GUARD} \
    --resume ${RESUME:-False} \
    --verbosity 1
echo "[$(date)] spac NICHE-FiLM train ${DS}/${REGIME} done -> ${OUT}"
