#!/bin/bash -l
#SBATCH -J stroke_demo
#SBATCH --mem=64G
#SBATCH --time 0-04:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o stroke_%x_%j.out
#SBATCH -e stroke_%x_%j.err

# 2D capability demos: lesions no real experiment can deliver.
# Builds the perturbation sets, then runs ONE counterfactual per set through the same tested
# --stage eval path, plus a single matched control (__basal__). Perturbed spots KEEP their own
# anatomy (__KEEP__) so the surface-vs-deep comparison is not flattened to one tissue code.
# Usage: sbatch run_capability_demos.sh [arm]   (default 2dflat)
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"

ARM="${1:-2dflat}"
NS=${CONCERT_WORKDIR}
SL=${NS}/stroke3d/slides
FILES="${SL}/stroke_paste2_PT1.h5ad,${SL}/stroke_paste2_PT2.h5ad,${SL}/stroke_paste2_PT3.h5ad,${SL}/stroke_paste2_PT4.h5ad,${SL}/stroke_paste2_Sham1.h5ad,${SL}/stroke_paste2_Sham2.h5ad,${SL}/stroke_paste2_Sham3.h5ad,${SL}/stroke_paste2_Sham4.h5ad"
OUT="${NS}/outputs/stroke_${ARM}"
SETS="${OUT}/pert_sets"
cd ${CODEBASE_ROOT}/models/concert_spac_niche

# Sham3 carries the richest anatomy of the four sham sections -- CTX 812 / TH 400 / STR 303 /
# HY 215 -- and TH (thalamus) is a genuine deep structure, which Sham1 lacks entirely.
python -u ${CODEBASE_ROOT}/experiments/07_stroke_case_study/prep/make_demo_sets.py --files "${FILES}" \
    --target_slide "${TARGET_SLIDE:-Sham3}" --outdir "${SETS}" \
    --min_region_spots ${MIN_REGION:-150} --max_regions ${MAX_REGIONS:-4} \
    --region_patch_n ${PATCH_N:-150}

COMMON="--config config.yaml --stage eval --multi_sample_files ${FILES} --outdir ${OUT}
  --model_file model.pt --project_index stroke --hvg_max_rank 2000 --seed 42
  --unassigned_label unperturbed --control_label __NONE__ --min_cells_per_pert 25
  --use_action_correction False --use_pad False
  --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True
  --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True
  --dynamicVAE False --free_bits 0.5 --gp_free_bits 1.0
  --use_nngp False --use_paracrine ${PARACRINE:-False} ${GRAPH_ARGS:-}
  --use_niche_interaction False --use_niche_film True --niche_film_dim 8 --niche_film_k 25
  --niche_film_gcap 3.0 --niche_film_l2 1e-3
  --use_pert_gate True --gate_hidden 64
  --use_f_pert True --f_pert_clamp 5.0
  --spatial_basal_only True --lfc_ref_tissue CTX
  --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0
  --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15
  --fpert_ls_min 0.05 --fpert_ls_max 0.60
  --tile_by_stack False --fpert_n_z_levels 1
  --ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18
  --fpert_svgp_inducing 16 --fpert_svgp_ls_init 5.0 --verbosity 1"

# Matched control once: same decode path, perturbation switched to the basal slot (a no-op on
# these already-basal Sham spots), so every set's delta is measured against the same baseline.
BIG=$(ls -S ${SETS}/*.txt | head -1)
echo "[$(date)] control (__basal__) using ${BIG}"
python -u run_concert_spac.py ${COMMON} --sample "demo_CONTROL" \
    --pert_cells "${BIG}" --target_cell_tissue __KEEP__ --target_cell_perturbation "__basal__"

for F in ${SETS}/*.txt; do
  NAME=$(basename "$F" .txt)
  echo "[$(date)] set ${NAME}"
  python -u run_concert_spac.py ${COMMON} --sample "demo_${NAME}" \
      --pert_cells "$F" --target_cell_tissue __KEEP__ --target_cell_perturbation "stroke"
done
echo "[$(date)] demos done -> ${OUT}"
