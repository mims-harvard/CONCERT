#!/bin/bash -l
#SBATCH -J letter
#SBATCH -p <partition>
#SBATCH --mem=16G
#SBATCH --time 0-00:15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -o letter_%j.out
#SBATCH -e letter_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
python -c "import docx" 2>/dev/null || pip install --quiet python-docx
python -u ${CONCERT_WORKDIR}/stroke3d/annotate_letter.py
