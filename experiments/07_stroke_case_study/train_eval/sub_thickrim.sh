#!/bin/bash -l
#SBATCH -J thickrim
#SBATCH --mem=64G
#SBATCH --time 0-03:00
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/thickrim_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/thickrim_%j.err
# Dilate the imposed peri-infarct rim and test whether a ring-shaped marker field appears.
# The CORE is unchanged, so the existing oncore/off passes are reused; only the PIA pass is rerun
# per thickness.
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
SL=${NS}/stroke3d/slides_cr
FILES="${SL}/stroke_paste2_PT1.h5ad,${SL}/stroke_paste2_PT2.h5ad,${SL}/stroke_paste2_PT3.h5ad,${SL}/stroke_paste2_PT4.h5ad,${SL}/stroke_paste2_Sham1.h5ad,${SL}/stroke_paste2_Sham2.h5ad,${SL}/stroke_paste2_Sham3.h5ad,${SL}/stroke_paste2_Sham4.h5ad"
SETS=${NS}/outputs/thick_rim_sets

python -u ${NS}/stroke3d/make_thick_rim_sets.py --files "${FILES}" --slides "${SL}" \
    --outdir "${SETS}" --thickness 1,2,3,5

cd ${NS}/CONCERT-spac-niche
BASE="--config config.yaml --stage eval --multi_sample_files ${FILES}
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
  --spatial_basal_only True --lfc_ref_tissue CTX
  --basal_field rff --basal_bayes True --moran_lambda 1.0 --disentangle_radius 4.0
  --fpert_field gp_pp --fpert_coord_scale 20.0 --fpert_rff_features 128 --fpert_rff_bandwidth 0.15
  --fpert_ls_min 0.05 --fpert_ls_max 0.60
  --tile_by_stack True --z_stack_obs mouse --z_index_obsm section_index --fpert_z_gain 0.25
  --ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18
  --fpert_svgp_inducing 16 --fpert_svgp_ls_init 5.0 --verbosity 1"

for ARM in 2d 3d; do
  OUT=${NS}/outputs/stroke_${ARM}_cr
  case "$ARM" in 3d) NZ=4 ;; 2d) NZ=1 ;; esac
  for TAG in obs t1 t2 t3 t5; do
    echo "[$(date)] arm=${ARM} rim=${TAG}"
    python -u run_concert_spac.py ${BASE} --fpert_n_z_levels ${NZ} --outdir "${OUT}" \
        --sample "stroke_${ARM}_cr_onrim_${TAG}" --pert_cells "${SETS}/thick_rim_${TAG}.txt" \
        --target_cell_tissue CTX --target_cell_perturbation "PIA"
  done
done

echo "=== 2D (section 1) ==="
python -u ${NS}/stroke3d/fig_thick_rim.py --arm 2d --out ${NS}/outputs/stroke_2d_cr \
    --sets "${SETS}" --slides "${SL}" --sections 1
echo "=== 3D (all sections) ==="
python -u ${NS}/stroke3d/fig_thick_rim.py --arm 3d --out ${NS}/outputs/stroke_3d_cr \
    --sets "${SETS}" --slides "${SL}"
echo "[$(date)] thickrim done"
