#!/bin/bash -l
#SBATCH -J stroke_fix2
#SBATCH --mem=48G
#SBATCH --time 0-00:30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o stroke_%x_%j.out
#SBATCH -e stroke_%x_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=${CONCERT_WORKDIR}
# 1. shape-matched 2D per-gene figures (eval already on disk; only the glob was wrong)
python -u ${NS}/stroke3d/stroke_field_by_gene.py \
    --outdir "${NS}/outputs/stroke_2d" --slides "${NS}/stroke3d/slides" --arm 2d \
    --shape_dir "${NS}/outputs/stroke_2d/shape_sets" --on_glob "shape_on_*" --tag "_shape" \
    --genes Gm42418,Scgb1a1,Vim,Lgals1,Timp1,Spp1
# 2. four-condition spatial slide
python -u ${NS}/stroke3d/fig_region_conditions.py --outdir "${NS}/outputs/stroke_2dpara" --arm 2dpara
