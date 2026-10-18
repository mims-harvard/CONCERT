#!/bin/bash -l
#SBATCH -J stroke_shape2d
#SBATCH --mem=64G
#SBATCH --time 0-02:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o stroke_%x_%j.out
#SBATCH -e stroke_%x_%j.err
#
# The 2D arm never got a shape-matched counterfactual, so its per-gene figures fell back to
# thresholding the model's OWN field at the lesion rate. With a 2D field that is barely
# informative (field r = 0.017) that threshold picks a SCATTER of spots, not a contiguous patch --
# which is why the 2D predicted region looks nothing like the observed lesion. Here the real
# lesion's shape is imposed, exactly as it already is for 3D, so the two arms are comparable.
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
OUT="${NS}/outputs/stroke_2d"
SHAPE="${NS}/outputs/stroke_3d/shape_sets"    # geometry only, arm-independent
cp -r "${SHAPE}" "${OUT}/shape_sets" 2>/dev/null || true
cd ${CODEBASE_ROOT}/models/concert_spac_niche

python -u run_concert_spac.py \
    --config config.yaml --sample "shape_on" --stage eval \
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
    --fpert_n_z_levels 1 --fpert_z_gain 0.25 \
    --ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 \
    --fpert_svgp_inducing 16 --fpert_svgp_ls_init 5.0 \
    --pert_cells "${SHAPE}/shape_lesion.txt" \
    --target_cell_tissue "__KEEP__" --target_cell_perturbation "stroke" \
    --verbosity 1

python -u ${CODEBASE_ROOT}/experiments/07_stroke_case_study/figures/plot_field_by_gene.py \
    --outdir "$OUT" --slides "${SL}" --arm 2d \
    --shape_dir "${OUT}/shape_sets" --on_glob "shape_on_*" --tag "_shape" \
    --genes Gm42418,Scgb1a1,Vim,Lgals1,Timp1,Spp1
echo "[$(date)] shape-matched 2d done -> ${OUT}/figures"
