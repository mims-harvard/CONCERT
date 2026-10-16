#!/bin/bash -l
#SBATCH -J lsnow
#SBATCH --mem=32G
#SBATCH --time 0-00:20
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o lsnow_%j.out
#SBATCH -e lsnow_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
python -u ${CODEBASE_ROOT}/experiments/07_stroke_case_study/figures/plot_section_neighbourhood.py
