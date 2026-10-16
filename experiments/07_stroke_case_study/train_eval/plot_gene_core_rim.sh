#!/bin/bash -l
#SBATCH -J stroke_gene2d
#SBATCH --mem=64G
#SBATCH --time 0-00:30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o stroke_%x_%j.out
#SBATCH -e stroke_%x_%j.err
# Per-gene core/rim figure for one arm. Usage: sbatch -p gpu_test plot_gene_core_rim.sh <arm>
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
ARM="${1:?need arm}"
NS=${CONCERT_WORKDIR}
python -u ${CODEBASE_ROOT}/experiments/07_stroke_case_study/figures/plot_field_by_gene.py \
    --outdir "${NS}/outputs/stroke_${ARM}" --slides "${NS}/stroke3d/slides" --arm "${ARM}" \
    --genes Gm42418,Scgb1a1,Vim,Lgals1,Timp1,Spp1
