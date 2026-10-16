#!/bin/bash -l
#SBATCH -J pfish_interp
#SBATCH -p <partition>
#SBATCH --mem=64G
#SBATCH --time 0-04:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o pfish_interp_%j.out
#SBATCH -e pfish_interp_%j.err
#
# Small perturb-FISH test of experiment 04 (are the learned spatial parameters interpretable?).
#
# Three steps: put the preprocessed perturb-FISH file into the layout the loader reads,
# train the same within-slide configuration used for the SPAC-seq sections on the single
# perturb-FISH section, then run the experiment-04 readout on the resulting checkpoint.

# Locate config/paths.sh. Under sbatch the batch script is copied to a spool directory, so a
# path relative to this file does not resolve there; CODEBASE_ROOT is the fallback.
_PATHS="$(dirname "${BASH_SOURCE[0]}")/../../config/paths.sh"
[ -f "$_PATHS" ] || _PATHS="${CODEBASE_ROOT:-}/config/paths.sh"
[ -f "$_PATHS" ] || { echo "[FATAL] cannot find config/paths.sh; sbatch --export=ALL,CODEBASE_ROOT=/path/to/repo $0"; exit 1; }
source "$_PATHS"
activate_env

RAW="${PERTURBFISH_DIR}/perturbfish_tumor_count.h5ad"
OUT="${PFISH_OUT:-${CONCERT_WORKDIR}/perturbfish}"
PREP="${OUT}/perturbfish_tumor_concert.h5ad"
mkdir -p "$OUT"

echo "[$(date)] step 1: prepare"
python -u "${CODEBASE_ROOT}/experiments/04_kernel_interpretability/perturbfish_prepare_data.py" --in_h5ad "$RAW" --out_h5ad "$PREP"

echo "[$(date)] step 2: train"
cd "${CODEBASE_ROOT}/models/concert_benchmark"

# eval_utils is imported by the LFC target precomputation whenever lfc_lambda > 0, but
# it ships in benchmark_evaluation/ rather than alongside the model, so put that on the
# path instead of copying the file into the release tree.
export PYTHONPATH="${CODEBASE_ROOT}/benchmark_evaluation:${PYTHONPATH}"

GUARD="--ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 \
       --fpert_svgp_inducing 16 --fpert_svgp_ls_init 0.25 --fpert_ls_min 0.15 --fpert_ls_max 0.90"
NICHE="--use_niche_interaction False --use_niche_film True --niche_film_dim 8 \
       --niche_film_k 25 --niche_film_gcap 3.0 --niche_film_l2 1e-3"

python -u run_concert_spac.py \
    --config config.yaml --sample "pfish" --stage train \
    --multi_sample_files "${PREP}" --outdir "$OUT" --model_file "model.pt" --project_index pfish \
    --hvg_max_rank 2000 --seed 42 \
    --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 \
    --multi_kernel_mode True \
    --maxiter 800 --patience 50 --earlystop_min_delta 0.05 --lr 1e-4 --batch_size 2048 \
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

echo "[$(date)] step 3: kernel parameter readout"
cd "${CODEBASE_ROOT}/experiments/04_kernel_interpretability"
python -u 01_extract_kernel_params.py \
    --model_file "${OUT}/model.pt" \
    --pert_map   "${OUT}/pert_map.json" \
    --cutoffs_csv "${OUT}/pfish_pfish_cutoffs.csv" \
    --multi_sample_files "${PREP}" \
    --ls_min 0.15 --ls_max 0.90 \
    --outdir "${OUT}/kernel_param_analysis" \
    2>&1 | tee "${OUT}/kernel_param_analysis.log"

echo "[$(date)] done -> ${OUT}"
