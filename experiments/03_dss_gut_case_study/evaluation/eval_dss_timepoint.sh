#!/bin/bash -l
#SBATCH -J concert_dss_eval
#SBATCH -p gpu
#SBATCH --mem=128G
#SBATCH --time 0-03:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_dss_eval_%j.out
#SBATCH -e concert_dss_eval_%j.err
#
# Leave-one-timepoint-out evaluation. The model is trained with one timepoint removed
# (train_dss.sh on a manifest subset), then asked to predict that timepoint, and scored
# against the measured sections for it. The comparison baseline is linear interpolation
# between the two flanking timepoints.
#
# Usage: sbatch eval_dss_timepoint.sh <Healthy|DSS3|DSS9|DSS21> [TAG]
#
# The model is rebuilt here with the same construction flags used at training, because the
# architecture has to match the checkpoint before the weights can be loaded.

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

HOLDOUT_TP="${1:?usage: sbatch eval_dss_timepoint.sh <Healthy|DSS3|DSS9|DSS21> [TAG]}"
TAG="${2:-loto_${HOLDOUT_TP}}"

case "$HOLDOUT_TP" in
    Healthy) TARGET_DAY=0  ;;
    DSS3)    TARGET_DAY=3  ;;
    DSS9)    TARGET_DAY=9  ;;
    DSS21)   TARGET_DAY=21 ;;
    *) echo "[FATAL] timepoint must be Healthy, DSS3, DSS9 or DSS21"; exit 1 ;;
esac

MODEL_DIR="${CODEBASE_ROOT}/models/concert_temporal"
MANIFEST="${DSS_DIR}/manifest.csv"
OUT="${OUTPUT_ROOT}/dss_${TAG}"
[ -f "${OUT}/model.pt" ] || { echo "[FATAL] checkpoint missing: ${OUT}/model.pt"; exit 1; }

# Manifest column 4 is the sample type; the held-out timepoint's sections are the ground truth.
TRAINFILES=$(awk -F, -v hold="$HOLDOUT_TP" 'NR>1 && $4!=hold {print $1}' "$MANIFEST" \
             | sed "s#^#${DSS_DIR}/#" | paste -sd, -)
HOLDFILES=$(awk -F, -v hold="$HOLDOUT_TP" 'NR>1 && $4==hold {print $1}' "$MANIFEST" \
            | sed "s#^#${DSS_DIR}/#" | paste -sd, -)

cd "$MODEL_DIR"

GUARD="--ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 \
       --fpert_svgp_inducing 16 --fpert_svgp_ls_init 5.0"
NICHE="--use_niche_interaction False --use_niche_film True --niche_film_dim 8 \
       --niche_film_k 25 --niche_film_gcap 3.0 --niche_film_l2 1e-3"
TIME="--use_time_conditioning True --time_col day_norm --time_fourier_bands 6 --time_emb_dim 16"

python -u run_concert_spac.py --stage loto_eval \
    --config config.yaml --sample "dss_${TAG}" \
    --multi_sample_files "${TRAINFILES}" --outdir "$OUT" --model_file "model.pt" --project_index dss \
    --loto_holdout_files "${HOLDFILES}" --loto_target_day ${TARGET_DAY} --loto_day_denom 21 \
    --loto_n_samples 10 --loto_batch_size 512 \
    --hvg_max_rank 2000 --seed 42 \
    --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 \
    --multi_kernel_mode True \
    --batch_size 2048 --train_size 0.95 \
    --use_action_correction False --use_pad False --min_cells_per_pert 25 \
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
    --dynamicVAE False --beta_warmup_epochs 20 --beta_target 1.0 --free_bits 0.5 --gp_free_bits 1.0 \
    --use_nngp False --use_paracrine False ${NICHE} ${TIME} \
    --use_pert_gate True --gate_hidden 64 \
    --use_f_pert True --f_pert_init_std 0.01 --f_pert_clamp 5.0 \
    --spatial_basal_only True --lfc_lambda 0.0 \
    --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0 \
    --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
    ${GUARD} \
    --verbosity 1

echo "[$(date)] DSS leave-one-timepoint-out evaluation finished (${HOLDOUT_TP}) -> ${OUT}"
