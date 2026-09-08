#!/bin/bash -l
#SBATCH -J concert_perturb_seq
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 0-06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_perturb_seq_%j.out
#SBATCH -e concert_perturb_seq_%j.err
#
# Train CONCERT on the spatial Perturb-seq slides (four sections, s0 to s3).
#
# Usage: sbatch train_perturb_seq.sh <seen|unseen>
#
# The model is the same one used for SPAC-seq (models/concert_benchmark). The settings differ
# only where the two assays differ: this panel has no non-targeting control class, spots are
# denser so the spatial field needs more inducing points and a wider lengthscale band, and no
# action correction is applied because guide assignment is already unambiguous here.
#
# The inducing-point count and the patience below are not free choices. Carrying over SPAC-seq's
# 16 inducing points and short patience costs about a third of the rank correlation on this
# assay; 64 points with patience 300 is what the reported numbers were produced with.

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

REGIME="${1:?usage: sbatch train_perturb_seq.sh <seen|unseen>}"
MODEL_DIR="${CODEBASE_ROOT}/models/concert_benchmark"
OUT="${OUTPUT_ROOT}/perturb_seq_${REGIME}"
HOLDOUT="${PERTURB_SEQ_DIR}/holdout_perturb_seq_${REGIME}.json"

FILES=$(for s in 0 1 2 3; do echo -n "${PERTURB_SEQ_DIR}/perturbmap_s${s}_spacfmt.h5ad,"; done | sed 's/,$//')
[ -f "$HOLDOUT" ] || { echo "[FATAL] holdout file missing: $HOLDOUT"; exit 1; }
mkdir -p "$OUT"
cd "$MODEL_DIR"

# Spatial field over perturbation effects. The band is given in the normalized coordinates the
# field itself uses, so it must be read together with --fpert_coord_scale below: at scale 70 the
# band [0.01, 0.60] is [0.7, 42] in section units, a section being 20 units wide. It is wide
# enough that the learned lengthscales settle inside it rather than against either end, which is
# what lets them be read as learned. The same values must be passed at evaluation time, because
# the band defines the squashing function inside the field, not just a training penalty.
GUARD="--ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 \
       --fpert_svgp_inducing 64 --fpert_svgp_ls_init 0.04 \
       --fpert_ls_min 0.01 --fpert_ls_max 0.60"
NICHE="--use_niche_interaction False --use_niche_film True --niche_film_dim 8 \
       --niche_film_k 25 --niche_film_gcap 3.0 --niche_film_l2 1e-3"
LABELS="--excluded_label= --unassigned_label unperturbed --control_label __NONE__"

python -u run_concert_spac.py \
    --config config.yaml --sample "perturb_seq" --stage train \
    --multi_sample_files "${FILES}" --outdir "$OUT" --model_file "model.pt" --project_index spac \
    --holdout_v3_file "${HOLDOUT}" \
    --hvg_max_rank 2000 --seed 42 \
    --inducing_point_steps 4 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 \
    --multi_kernel_mode True \
    --maxiter 1500 --patience 300 --lr 1e-4 --batch_size 2048 --train_size 0.95 \
    --use_action_correction False --use_pad False ${LABELS} --min_cells_per_pert 0 \
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
    --dynamicVAE False --beta_warmup_epochs 20 --beta_target 1.0 --free_bits 0.5 --gp_free_bits 1.0 \
    --use_nngp False --use_paracrine False ${NICHE} \
    --use_pert_gate True --gate_hidden 64 \
    --use_f_pert True --f_pert_init_std 0.01 --f_pert_clamp 5.0 \
    --spatial_basal_only True --lfc_lambda 1.0 \
    --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0 \
    --fpert_field gp_pp --fpert_coord_scale 70.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
    ${GUARD} \
    --resume "${RESUME:-False}" \
    --verbosity 1

echo "[$(date)] Perturb-seq ${REGIME} training finished -> ${OUT}"
