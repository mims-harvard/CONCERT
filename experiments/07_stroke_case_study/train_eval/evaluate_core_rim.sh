#!/bin/bash -l
#SBATCH -J stroke_evcr
#SBATCH -p <partition>
#SBATCH --mem=64G
#SBATCH --time 0-06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o stroke_%x_%j.out
#SBATCH -e stroke_%x_%j.err
#
# Compartment-aware counterfactual: the imposed lesion gets a CORE and a RIM with DIFFERENT
# perturbation labels, which is the whole point of the _cr retrain.
#
# Three decode passes over the same cells, differing only in the perturbation attribute:
#   oncore : ICA       on the imposed core spots  (shape_core.txt)
#   onrim  : PIA       on the imposed rim spots   (shape_rim.txt)
#   off    : __basal__ on every sham spot         (matched control)
#
# Assembling delta[core] from `oncore` and delta[rim] from `onrim` is EXACT, not an
# approximation: NicheDescriptor (concert_spac.py:413) builds the microenvironment code from the
# expression latent and coords only -- it never reads neighbours' perturbation labels -- so a core
# spot's decode is unaffected by whether the rim spots were switched on in the same pass.
# Usage: sbatch -p gpu_test evaluate_core_rim.sh <2d|3d>
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"

ARM="${1:?need arm: 2d|3d}"
case "$ARM" in 3d) NZ=4 ;; 2d) NZ=1 ;; *) echo "[FATAL] arm must be 2d or 3d"; exit 1 ;; esac

NS=${CONCERT_WORKDIR}
SL=${NS}/stroke3d/slides_cr
FILES="${SL}/stroke_paste2_PT1.h5ad,${SL}/stroke_paste2_PT2.h5ad,${SL}/stroke_paste2_PT3.h5ad,${SL}/stroke_paste2_PT4.h5ad,${SL}/stroke_paste2_Sham1.h5ad,${SL}/stroke_paste2_Sham2.h5ad,${SL}/stroke_paste2_Sham3.h5ad,${SL}/stroke_paste2_Sham4.h5ad"
OUT="${NS}/outputs/stroke_${ARM}${CRTAG:-_cr}"
SHAPE="${NS}/outputs/stroke_3d/shape_sets"     # geometry only; identical across arms
cd ${NS}/CONCERT-spac-niche

python -u ${CODEBASE_ROOT}/experiments/07_stroke_case_study/prep/make_sham_index.py --files "${FILES}" --out "${OUT}/sham_cells.txt"
cp -r "${SHAPE}" "${OUT}/shape_sets" 2>/dev/null || true

for SPEC in oncore:ICA:${SHAPE}/shape_core.txt onrim:PIA:${SHAPE}/shape_rim.txt off:__basal__:${OUT}/sham_cells.txt; do
  T="${SPEC%%:*}"; REST="${SPEC#*:}"; TARGET="${REST%%:*}"; CELLS="${REST##*:}"
  echo "[$(date)] eval arm=${ARM} pass=${T} target=${TARGET} cells=$(basename ${CELLS})"
  python -u run_concert_spac.py \
      --config config.yaml --sample "stroke_${ARM}${CRTAG:-_cr}_${T}" --stage eval \
      --multi_sample_files "${FILES}" --outdir "$OUT" --model_file "model.pt" --project_index stroke \
      --hvg_max_rank 2000 --seed 42 \
      --unassigned_label "unperturbed" --control_label "__NONE__" --min_cells_per_pert 25 \
      --use_action_correction False --use_pad False \
      --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True \
      --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
      --dynamicVAE False --free_bits 0.5 --gp_free_bits 1.0 \
      --use_nngp False --use_paracrine False \
      --use_niche_interaction False --use_niche_film True --niche_film_dim 8 --niche_film_k 25 \
      --niche_film_gcap 3.0 --niche_film_l2 1e-3 \
      --use_pert_gate True --gate_hidden 64 \
      --use_f_pert True --f_pert_clamp 5.0 \
      --spatial_basal_only True --lfc_ref_tissue CTX \
      --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0 \
      --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
      --fpert_ls_min 0.05 --fpert_ls_max 0.60 \
      --tile_by_stack True --z_stack_obs mouse --z_index_obsm section_index \
      --fpert_n_z_levels ${NZ} --fpert_z_gain 0.25 \
      --ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 \
      --fpert_svgp_inducing ${INDUCING:-16} --fpert_svgp_ls_init 5.0 \
      --pert_cells "${CELLS}" \
      --target_cell_tissue CTX --target_cell_perturbation "${TARGET}" \
      --verbosity 1
done
echo "[$(date)] stroke eval_cr arm=${ARM} done -> ${OUT}"
