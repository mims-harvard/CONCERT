#!/bin/bash -l
#SBATCH -J gate_sweep
#SBATCH --mem=64G
#SBATCH --time 0-01:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o gate_%x_%A_%a.out
#SBATCH -e gate_%x_%A_%a.err
# TRANSPLANT + POSITION gate, with an eval-time gamma ablation.
#   sets  : PT1 footprint transplanted to Sham3, plus 4 anatomical regions of Sham3
#   gcap  : 3.0 = as trained;  0.0 = clamps niche-FiLM g_scale to 0 so gamma == 1 exactly,
#           i.e. the SAME weights with the sign-flipping niche modulation switched off.
# If the sign flips vanish at gcap=0, the niche-FiLM scalar is the cause.
# Usage: sbatch sub_gate_sweep.sh <2d|3d>
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
ARM="${1:?need arm 2d|3d}"; case "$ARM" in 3d) NZ=4 ;; *) NZ=1 ;; esac
NS=${CONCERT_WORKDIR}
SL=${NS}/stroke3d/slides
FILES="${SL}/stroke_paste2_PT1.h5ad,${SL}/stroke_paste2_PT2.h5ad,${SL}/stroke_paste2_PT3.h5ad,${SL}/stroke_paste2_PT4.h5ad,${SL}/stroke_paste2_Sham1.h5ad,${SL}/stroke_paste2_Sham2.h5ad,${SL}/stroke_paste2_Sham3.h5ad,${SL}/stroke_paste2_Sham4.h5ad"
SETS=${NS}/outputs/stroke_2dpara/pert_sets
cd ${NS}/CONCERT-spac-niche
for GCAP in 3.0 0.0; do
  TAGG=$(echo $GCAP | tr -d .)
  OUT=${CONCERT_WORKDIR}/scratch/CONCERT_rimtest/gate_${ARM}_g${TAGG}
  mkdir -p $OUT
  for f in model.pt hvg_gene_names.txt pert_map.json; do ln -sf ${NS}/outputs/stroke_${ARM}/$f $OUT/$f; done
  COMMON="--config config.yaml --stage eval --multi_sample_files ${FILES} --outdir ${OUT}
    --model_file model.pt --project_index stroke --hvg_max_rank 2000 --seed 42
    --unassigned_label unperturbed --control_label __NONE__ --min_cells_per_pert 25
    --use_action_correction False --use_pad False
    --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True
    --dynamicVAE False --free_bits 0.5 --gp_free_bits 1.0
    --use_nngp False --use_paracrine False
    --use_niche_interaction False --use_niche_film True --niche_film_dim 8 --niche_film_k 25
    --niche_film_gcap ${GCAP} --niche_film_l2 1e-3
    --use_pert_gate True --gate_hidden 64
    --use_f_pert True --f_pert_clamp 5.0
    --spatial_basal_only True --lfc_lambda 1.0 --lfc_ref_tissue CTX
    --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0
    --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15
    --fpert_ls_min 0.05 --fpert_ls_max 0.60
    --tile_by_stack True --z_stack_obs mouse --z_index_obsm section_index
    --fpert_n_z_levels ${NZ} --fpert_z_gain 0.25
    --ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18
    --fpert_svgp_inducing 16 --fpert_svgp_ls_init 5.0 --verbosity 1"
  for S in transplant_PT1 region_CTX region_TH region_STR region_HY; do
    [ -f ${SETS}/${S}.txt ] || { echo "[skip] ${S}"; continue; }
    echo "[$(date)] ${ARM} gcap=${GCAP} ${S} ON"
    python -u run_concert_spac.py ${COMMON} --sample "g_${S}_ON" --pert_cells "${SETS}/${S}.txt" \
        --target_cell_tissue __KEEP__ --target_cell_perturbation "stroke"
    echo "[$(date)] ${ARM} gcap=${GCAP} ${S} OFF"
    python -u run_concert_spac.py ${COMMON} --sample "g_${S}_OFF" --pert_cells "${SETS}/${S}.txt" \
        --target_cell_tissue __KEEP__ --target_cell_perturbation "__basal__"
  done
done
echo "[$(date)] gate sweep ${ARM} done"
