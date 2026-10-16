#!/bin/bash -l
#SBATCH -J supptab
#SBATCH -p <partition>
#SBATCH --mem=16G
#SBATCH --time 0-00:15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -o supptab_%j.out
#SBATCH -e supptab_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
python -u ${CODEBASE_ROOT}/experiments/07_stroke_case_study/prep/make_supplementary_tables.py
