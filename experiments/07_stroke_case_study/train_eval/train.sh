#!/bin/bash -l
#SBATCH -J stroke_tr
#SBATCH -p <partition>
#SBATCH --mem=64G
#SBATCH --time 1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o stroke_%x_%j.out
#SBATCH -e stroke_%x_%j.err

# CONCERT stroke run. Two arms, identical except whether f_pert can resolve the z axis:
#   3d : --fpert_n_z_levels 4  -> footprint field is (x, y, ordinal section)
#   2d : --fpert_n_z_levels 1  -> footprint field is (x, y); sections superimposed
# BOTH tile by z-STACK (mouse), so the 4 registered sections of a mouse share one XY frame --
# that is what lets a footprint learned on the PT lesion be evaluated at Sham coordinates.
# The 2d and 3d arms default to the core/rim configuration reported in the paper: slides with
# separate core (ICA) and rim (PIA) lesion labels (stroke3d/slides_cr, from prepare_slides.sh)
# and output tag _cr. The single-label variant is trained with
#   TAG= SLIDES=${CONCERT_WORKDIR}/stroke3d/slides sbatch train.sh <2d|3d>
# Usage: sbatch train.sh <2d|3d>
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARM="${1:?need arm: 2d|3d}"
# 2dflat is the standalone 2D task (per-slide tiles, no z-stack).
# 2d is the z-ablation arm -- same as 3d but with the z axis removed, sections superimposed.
case "$ARM" in
  3d)     NZ=4; TILE=True ;;
  2d)     NZ=1; TILE=True ;;
  2dflat) NZ=1; TILE=False ;;
  2dpara) NZ=1; TILE=False ;;   # 2dflat + paracrine spillover (systemic-effect demo)
  *)  echo "[FATAL] arm must be 2d, 2dflat, 2dpara or 3d"; exit 1 ;;
esac

NS=${CONCERT_WORKDIR}
case "$ARM" in
  2d|3d) SL=${SLIDES:-${NS}/stroke3d/slides_cr}; TAG=${TAG-_cr} ;;
  *)     SL=${SLIDES:-${NS}/stroke3d/slides} ;;
esac
FILES="${SL}/stroke_paste2_PT1.h5ad,${SL}/stroke_paste2_PT2.h5ad,${SL}/stroke_paste2_PT3.h5ad,${SL}/stroke_paste2_PT4.h5ad,${SL}/stroke_paste2_Sham1.h5ad,${SL}/stroke_paste2_Sham2.h5ad,${SL}/stroke_paste2_Sham3.h5ad,${SL}/stroke_paste2_Sham4.h5ad"
OUT="${NS}/outputs/stroke_${ARM}${TAG:-}"
mkdir -p "$OUT" "${NS}/logs"
cd ${CODEBASE_ROOT}/models/concert_spac_niche

# No non-targeting control in this dataset: unperturbed spots are the `unassigned` class, so
# action correction is off and the LFC reference is non-lesion cortex (CTX) rather than "tumor".
echo "[$(date)] stroke train arm=${ARM} (n_z_levels=${NZ}) -> ${OUT}"
python -u run_concert_spac.py \
    --config config.yaml --sample "stroke_${ARM}" --stage train \
    --multi_sample_files "${FILES}" --outdir "$OUT" --model_file "model.pt" --project_index stroke \
    --hvg_max_rank 2000 --seed 42 \
    --unassigned_label "unperturbed" --control_label "__NONE__" --min_cells_per_pert 25 \
    --use_action_correction False --use_pad False \
    --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True \
    --maxiter ${MAXITER:-1500} --patience 50 --earlystop_min_delta 0.05 --lr 1e-4 --batch_size 2048 --train_size 0.95 \
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
    --dynamicVAE False --beta_warmup_epochs 20 --beta_target 1.0 --free_bits 0.5 --gp_free_bits 1.0 \
    --use_nngp False --use_paracrine ${PARACRINE:-False} ${GRAPH_ARGS:-} \
    --use_niche_interaction False --use_niche_film True --niche_film_dim 8 --niche_film_k 25 \
    --niche_film_gcap 3.0 --niche_film_l2 1e-3 \
    --use_pert_gate True --gate_hidden 64 \
    --use_f_pert True --f_pert_init_std 0.01 --f_pert_clamp 5.0 \
    --spatial_basal_only True --lfc_lambda ${LFC:-1.0} --lfc_ref_tissue CTX \
    --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0 \
    --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
    --fpert_ls_min ${LS_MIN:-0.05} --fpert_ls_max ${LS_MAX:-0.60} \
    --tile_by_stack ${TILE} --z_stack_obs mouse --z_index_obsm section_index \
    --fpert_n_z_levels ${NZ} --fpert_z_gain ${ZGAIN:-0.25} \
    --ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 \
    --fpert_svgp_inducing ${INDUCING:-16} --fpert_svgp_ls_init 5.0 \
    --resume ${RESUME:-False} \
    --verbosity 1
echo "[$(date)] stroke train arm=${ARM} done -> ${OUT}"
