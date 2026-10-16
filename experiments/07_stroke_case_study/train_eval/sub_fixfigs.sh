#!/bin/bash -l
#SBATCH -J fixfigs
#SBATCH --mem=64G
#SBATCH --time 0-01:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o fixfigs_%j.out
#SBATCH -e fixfigs_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=${CONCERT_WORKDIR}
echo "=== how close are the two cortical-depth patches ==="
python -u ${NS}/stroke3d/diag_patch_sep.py
echo "=== condition figures, row 2 now the induction ==="
for G in Lcn2 Vim Timp1 Gm42418; do
  python -u ${NS}/stroke3d/fig_conditions.py --outdir "${NS}/outputs/stroke_2dpara_fixC" \
      --arm "fixC_${G}" --gene "${G}"
done
