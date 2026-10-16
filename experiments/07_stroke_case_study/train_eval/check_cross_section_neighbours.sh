#!/bin/bash -l
#SBATCH -J stroke_st0b
#SBATCH --mem=100G
#SBATCH --time 0-01:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o stage0b_%j.out
#SBATCH -e stage0b_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
cd ${CONCERT_WORKDIR}/stroke3d
python -u ${CODEBASE_ROOT}/experiments/07_stroke_case_study/diagnostics/check_cross_section_neighbours.py
