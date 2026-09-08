#!/bin/bash -l
#SBATCH -J stroke_ev
#SBATCH -p kempner_requeue
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=64G
#SBATCH --time 0-06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/stroke_%x_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/stroke_%x_%j.err

# Counterfactual: give the Sham mouse a stroke.
# Two eval passes over the SAME cells through the SAME decode path, differing only in the
# perturbation attribute -- so the delta is the model's stroke effect and nothing else:
#   ON  : target_cell_perturbation=stroke      (+ tissue forced to CTX, the lesion's anatomy)
#   OFF : target_cell_perturbation=__basal__   (no-op re-decode = matched control)
# The OFF label is `__basal__`, NOT `unperturbed`: `unperturbed` is the raw *unassigned* label in
# the h5ad, and build_attributes folds every unassigned spot into the reserved basal slot, so
# pert_map holds {__basal__: 0, stroke: 1} and the runner rejects any other name.
# Usage: sbatch sub_eval_stroke.sh <2d|3d>
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"

ARM="${1:?need arm: 2d|3d}"
case "$ARM" in 3d) NZ=4 ;; 2d) NZ=1 ;; *) echo "[FATAL] arm must be 2d or 3d"; exit 1 ;; esac

NS=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
SL=${NS}/stroke3d/slides
FILES="${SL}/stroke_paste2_PT1.h5ad,${SL}/stroke_paste2_PT2.h5ad,${SL}/stroke_paste2_PT3.h5ad,${SL}/stroke_paste2_PT4.h5ad,${SL}/stroke_paste2_Sham1.h5ad,${SL}/stroke_paste2_Sham2.h5ad,${SL}/stroke_paste2_Sham3.h5ad,${SL}/stroke_paste2_Sham4.h5ad"
OUT="${NS}/outputs/stroke_${ARM}${TAG:-}"
cd ${NS}/CONCERT-spac-niche

# 1-based row indices of the Sham cells, in the concatenation order the loader uses.
python -u ${NS}/stroke3d/make_sham_index.py --files "${FILES}" --out "${OUT}/sham_cells.txt"

for PAIR in on:stroke off:__basal__; do
  TAG="${PAIR%%:*}"; TARGET="${PAIR##*:}"
  echo "[$(date)] eval arm=${ARM} tag=${TAG} target=${TARGET}"
  python -u run_concert_spac.py \
      --config config.yaml --sample "stroke_${ARM}_${TAG}" --stage eval \
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
      --fpert_svgp_inducing 16 --fpert_svgp_ls_init 5.0 \
      --pert_cells "${OUT}/sham_cells.txt" \
      --target_cell_tissue CTX --target_cell_perturbation "${TARGET}" \
      --verbosity 1
done
echo "[$(date)] stroke eval arm=${ARM} done -> ${OUT}"
