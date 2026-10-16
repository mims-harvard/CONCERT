#!/bin/bash -l
#SBATCH -J leak_arms
#SBATCH --mem=64G
#SBATCH --time 0-00:40
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o leak_arms_%j.out
#SBATCH -e leak_arms_%j.err
# Same lesion-ON vs lesion-OFF gate, applied to the arms that were trained WITHOUT paracrine
# (stroke_2d, stroke_3d). These are the arms behind the KEEP panels, so we need to know whether
# the inverted f_pert is a 2dpara-only problem or general.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=${CONCERT_WORKDIR}
SL=${NS}/stroke3d/slides
FILES="${SL}/stroke_paste2_PT1.h5ad,${SL}/stroke_paste2_PT2.h5ad,${SL}/stroke_paste2_PT3.h5ad,${SL}/stroke_paste2_PT4.h5ad,${SL}/stroke_paste2_Sham1.h5ad,${SL}/stroke_paste2_Sham2.h5ad,${SL}/stroke_paste2_Sham3.h5ad,${SL}/stroke_paste2_Sham4.h5ad"
CELLS=${NS}/outputs/stroke_2dpara/pert_sets/real_lesion_PT1.txt
cd ${NS}/CONCERT-spac-niche
for ARM in 2d 3d; do
  case "$ARM" in 3d) NZ=4 ;; *) NZ=1 ;; esac
  OUT=${CONCERT_WORKDIR}/scratch/CONCERT_rimtest/leak_${ARM}
  COMMON="--config config.yaml --stage eval --multi_sample_files ${FILES} --outdir ${OUT}
    --model_file model.pt --project_index stroke --hvg_max_rank 2000 --seed 42
    --unassigned_label unperturbed --control_label __NONE__ --min_cells_per_pert 25
    --use_action_correction False --use_pad False
    --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True
    --dynamicVAE False --free_bits 0.5 --gp_free_bits 1.0
    --use_nngp False --use_paracrine False
    --use_niche_interaction False --use_niche_film True --niche_film_dim 8 --niche_film_k 25
    --niche_film_gcap 3.0 --niche_film_l2 1e-3
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
  echo "[$(date)] ${ARM} lesion ON"
  python -u run_concert_spac.py ${COMMON} --sample "leak_ON" --pert_cells "${CELLS}" \
      --target_cell_tissue __KEEP__ --target_cell_perturbation "stroke"
  echo "[$(date)] ${ARM} lesion OFF"
  python -u run_concert_spac.py ${COMMON} --sample "leak_OFF" --pert_cells "${CELLS}" \
      --target_cell_tissue __KEEP__ --target_cell_perturbation "__basal__"
done
echo "[$(date)] done"
