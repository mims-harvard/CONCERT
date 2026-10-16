#!/bin/bash -l
#SBATCH -J diag_rim5
#SBATCH --mem=48G
#SBATCH --time 0-00:30
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o diag_rim5_%j.out
#SBATCH -e diag_rim5_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
python -u ${CONCERT_WORKDIR}/stroke3d/diag_rim5.py
