#!/bin/bash -l
#SBATCH -J check_position_gate
#SBATCH --mem=48G
#SBATCH --time 0-00:30
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH -o diag_gate_%j.out
#SBATCH -e diag_gate_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
python -u ${CODEBASE_ROOT}/experiments/07_stroke_case_study/diagnostics/check_position_gate.py
