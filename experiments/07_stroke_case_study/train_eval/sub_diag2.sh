#!/bin/bash -l
#SBATCH -J stroke_cap
#SBATCH --mem=32G
#SBATCH --time 0-00:15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -o stroke_%x_%j.out
#SBATCH -e stroke_%x_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
python -u ${CONCERT_WORKDIR}/stroke3d/diag_fpert_capacity.py
