#!/bin/bash -l
#SBATCH -J concert_spac_seq
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 2-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_spac_seq_%j.out
#SBATCH -e concert_spac_seq_%j.err
#
# Train CONCERT on the SPAC-seq slides (five sections, 1-1 to 1-5).
#
# Usage: sbatch train_spac_seq.sh <seen|unseen>
#   seen   perturbations held out within a slide; the perturbation is observed on other slides
#   unseen perturbations held out from every slide
#
# The holdout file is produced by ../data_preparation/build_v3_holdout_coords.py.

# Locate config/paths.sh. Under sbatch the batch script is copied to a spool directory, so a
# path relative to this file does not resolve there; CODEBASE_ROOT is the fallback. Failing
# loudly matters here: if this source silently fails, activate_env never runs, `set -e` is never
# enabled, and every step below fails while the script still reports success.
_PATHS="$(dirname "${BASH_SOURCE[0]}")/../../../config/paths.sh"
[ -f "$_PATHS" ] || _PATHS="${CODEBASE_ROOT:-}/config/paths.sh"
[ -f "$_PATHS" ] || {
    echo "[FATAL] cannot find config/paths.sh."
    echo "        Submitting with sbatch? Pass the codebase root explicitly:"
    echo "          sbatch --export=ALL,CODEBASE_ROOT=/path/to/codebase $0 ..."
    exit 1
}
source "$_PATHS"
activate_env

REGIME="${1:?usage: sbatch train_spac_seq.sh <seen|unseen>}"
MODEL_DIR="${CODEBASE_ROOT}/models/concert_benchmark"
OUT="${OUTPUT_ROOT}/spac_seq_${REGIME}"
HOLDOUT="${SPAC_SEQ_DIR}/holdout_spac_${REGIME}.json"

FILES=$(for i in 1 2 3 4 5; do echo -n "${SPAC_SEQ_DIR}/raw_output_1-${i}.h5ad,"; done | sed 's/,$//')
[ -f "$HOLDOUT" ] || { echo "[FATAL] holdout file missing: $HOLDOUT"; exit 1; }
mkdir -p "$OUT"
cd "$MODEL_DIR"

# Spatial field over perturbation effects. The lengthscale band is given in normalized
# coordinates, matching the units the field itself uses.
GUARD="--ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 \
       --fpert_svgp_inducing 16 --fpert_svgp_ls_init 0.25 --fpert_ls_min 0.15 --fpert_ls_max 0.90"

# Niche-conditioned FiLM head: the predicted response is rescaled by a factor that depends on
# the local microenvironment, with a per-perturbation gain controlling how strong that
# dependence may become.
NICHE="--use_niche_interaction False --use_niche_film True --niche_film_dim 8 \
       --niche_film_k 25 --niche_film_gcap 3.0 --niche_film_l2 1e-3"

python -u run_concert_spac.py \
    --config config.yaml --sample "spac_seq" --stage train \
    --multi_sample_files "${FILES}" --outdir "$OUT" --model_file "model.pt" --project_index spac \
    --holdout_v3_file "${HOLDOUT}" \
    --hvg_max_rank 2000 --seed 42 \
    --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 \
    --multi_kernel_mode True \
    --maxiter 1500 --patience 50 --earlystop_min_delta 0.05 --lr 1e-4 --batch_size 2048 \
    --train_size 0.95 \
    --use_action_correction True --use_pad False --min_cells_per_pert 25 \
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
    --dynamicVAE False --beta_warmup_epochs 20 --beta_target 1.0 --free_bits 0.5 --gp_free_bits 1.0 \
    --use_nngp False --use_paracrine False ${NICHE} \
    --use_pert_gate True --gate_hidden 64 \
    --use_f_pert True --f_pert_init_std 0.01 --f_pert_clamp 5.0 \
    --spatial_basal_only True --lfc_lambda 1.0 \
    --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0 \
    --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
    ${GUARD} \
    --resume "${RESUME:-False}" \
    --verbosity 1

echo "[$(date)] SPAC-seq ${REGIME} training finished -> ${OUT}"
