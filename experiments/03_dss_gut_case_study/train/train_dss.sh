#!/bin/bash -l
#SBATCH -J concert_dss
#SBATCH -p gpu
#SBATCH --mem=256G
#SBATCH --time 1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_dss_%j.out
#SBATCH -e concert_dss_%j.err
#
# Train the temporal CONCERT model on the DSS colitis MERFISH time course
# (Healthy, day 3, day 9, day 21; 52 sections listed in the data manifest).
#
# Usage: sbatch train_dss.sh [TAG]
#   HOLDOUT_MICE="m1,m2"  optional; excludes every section of those animals from training,
#                         giving a leave-one-animal-out checkpoint
#
# The response is written as amplitude(t) x spatial field(lengthscale(t)) x perturbation
# effect, so a single model covers all four timepoints and can be asked for a timepoint it was
# not trained on. Three settings below fix the direction and magnitude of the inflammation
# response, which are otherwise not identified by the likelihood alone:
#   - a per-cell inflammation field supervised against the observed marker score, so the
#     magnitude is tied to measured inflammation rather than free
#   - the field's time input is frozen, because each section carries exactly one day and
#     "healthy architecture at day 9" therefore never appears in the data; time still enters
#     through the spatial field, which is where the rise and resolution are learned
#   - the sign of the effect is constrained to be positive on the inflammation marker panel
#     only; the remaining genes stay free, since resolution genuinely lowers some of them

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

TAG="${1:-dss}"
MODEL_DIR="${CODEBASE_ROOT}/models/concert_temporal"
MANIFEST="${DSS_DIR}/manifest.csv"
ARCH_DIR="${DSS_DIR}_arch"
OUT="${OUTPUT_ROOT}/dss_${TAG}"

# Manifest columns: filename, ..., mouse id, sample type.
FILELIST=$(awk -F, -v hm="${HOLDOUT_MICE:-}" '
    BEGIN { n = split(hm, a, ","); for (i = 1; i <= n; i++) if (a[i] != "") holdout[a[i]] = 1 }
    NR > 1 { if (!($3 in holdout)) print $1 }' "$MANIFEST")
FILES=$(echo "$FILELIST" | sed "s#^#${DSS_DIR}/#" | paste -sd, -)
echo "[data] $(echo "$FILELIST" | wc -l) sections (held-out animals: ${HOLDOUT_MICE:-none})"

mkdir -p "$OUT"
cd "$MODEL_DIR"

TIMEKERNEL="--use_timekernel True --use_shared_basal True --use_time_intensity False \
            --arch_desc_dir ${ARCH_DIR} \
            --use_inflammation_field True --field_mse_lambda 1.0 --field_time_fixed_t 0.0 \
            --marker_sign_positive True --time_in_field_mlp True \
            --amp_anchor_offset 0.0 --amp_anchor_lambda 0.0 --amp_anchor_t 0.0"
GUARD="--ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 \
       --fpert_svgp_inducing 16 --fpert_svgp_ls_init 5.0"
NICHE="--use_niche_interaction False --use_niche_film True --niche_film_dim 8 \
       --niche_film_k 25 --niche_film_gcap 3.0 --niche_film_l2 1e-3"
TIME="--use_time_conditioning True --time_col day_norm --time_fourier_bands 6 --time_emb_dim 16"

python -u run_concert_spac.py --stage train \
    --config config.yaml --sample "dss_${TAG}" \
    --multi_sample_files "${FILES}" --outdir "$OUT" --model_file model.pt --project_index dss \
    --maxiter 1000 --patience 50 --earlystop_min_delta 0.05 --lr 1e-4 --resume False \
    --hvg_max_rank 2000 --seed 42 \
    --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 \
    --multi_kernel_mode True \
    --batch_size 1024 --train_size 0.95 \
    --use_action_correction False --use_pad False --min_cells_per_pert 25 \
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
    --dynamicVAE False --beta_warmup_epochs 2 --beta_target 1.0 --free_bits 0.5 --gp_free_bits 1.0 \
    --use_nngp False --use_paracrine False ${TIMEKERNEL} ${NICHE} ${TIME} \
    --use_pert_gate False \
    --use_f_pert True --f_pert_init_std 0.01 --f_pert_clamp 5.0 \
    --spatial_basal_only True --lfc_lambda 0.0 \
    --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0 \
    --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
    ${GUARD} --verbosity 1

echo "[$(date)] DSS training finished -> ${OUT}"
