#!/bin/bash -l
#SBATCH -J stroke_rep
#SBATCH --mem=64G
#SBATCH --time 0-00:30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o stroke_%x_%j.out
#SBATCH -e stroke_%x_%j.err
# Field + proportion comparison for one arm. Usage: sbatch sub_report_stroke.sh <2d|3d>  (TAG env optional)
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
ARM="${1:?need arm: 2d|3d}"
NS=${CONCERT_WORKDIR}
python -u ${NS}/stroke3d/stroke_pattern_report.py \
    --outdir "${NS}/outputs/stroke_${ARM}${TAG:-}" --slides "${NS}/stroke3d/slides" --arm "${ARM}"
