#!/bin/bash -l
#SBATCH -J gate_fix
#SBATCH --mem=64G
#SBATCH --time 0-01:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o gatefix_%x_%j.out
#SBATCH -e gatefix_%x_%j.err
# Acceptance gate for a retrained 2dpara arm.
#   PARA=own : the arm's own trained paracrine setting (what the figures would use)
#   PARA=off : paracrine disabled, isolating what f_pert alone carries
# Sets: real PT1 lesion (at-lesion fit), PT1 footprint transplanted to Sham3, region_TH (deep).
# Usage: sbatch sub_gate_fix.sh <fixA|fixB|fixC> <own|off>
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
ARM="${1:?fixA|fixB|fixC}"; MODE="${2:-own}"
NS=${CONCERT_WORKDIR}
SL=${NS}/stroke3d/slides
FILES="${SL}/stroke_paste2_PT1.h5ad,${SL}/stroke_paste2_PT2.h5ad,${SL}/stroke_paste2_PT3.h5ad,${SL}/stroke_paste2_PT4.h5ad,${SL}/stroke_paste2_Sham1.h5ad,${SL}/stroke_paste2_Sham2.h5ad,${SL}/stroke_paste2_Sham3.h5ad,${SL}/stroke_paste2_Sham4.h5ad"
SETS=${NS}/outputs/stroke_2dpara/pert_sets
SRC=${NS}/outputs/stroke_2dpara_${ARM}
# fixC trained without paracrine; fixA/fixB with the rebuilt r_max=1.0 graph.
if [ "$ARM" = "fixC" ] || [ "$MODE" = "off" ]; then PARA="--use_paracrine False"
else PARA="--use_paracrine True --graph_file ${NS}/outputs/stroke_2dpara/graph_r1.npz --paracrine_r_max 1.0"; fi
OUT=${CONCERT_WORKDIR}/scratch/CONCERT_rimtest/gate_${ARM}_${MODE}
mkdir -p $OUT
for f in model.pt hvg_gene_names.txt pert_map.json; do ln -sf ${SRC}/$f $OUT/$f; done
cd ${NS}/CONCERT-spac-niche
COMMON="--config config.yaml --stage eval --multi_sample_files ${FILES} --outdir ${OUT}
  --model_file model.pt --project_index stroke --hvg_max_rank 2000 --seed 42
  --unassigned_label unperturbed --control_label __NONE__ --min_cells_per_pert 25
  --use_action_correction False --use_pad False
  --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True
  --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True
  --dynamicVAE False --free_bits 0.5 --gp_free_bits 1.0
  --use_nngp False ${PARA}
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
for S in real_lesion_PT1 transplant_PT1 region_TH; do
  for T in ON:stroke OFF:__basal__; do
    N=${T%%:*}; P=${T##*:}
    echo "[$(date)] ${ARM}/${MODE} ${S} ${N}"
    python -u run_concert_spac.py ${COMMON} --sample "g_${S}_${N}" --pert_cells "${SETS}/${S}.txt" \
        --target_cell_tissue __KEEP__ --target_cell_perturbation "${P}"
  done
done
echo "[$(date)] gate ${ARM}/${MODE} done"
