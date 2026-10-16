#!/bin/bash -l
#SBATCH -J crfigs
#SBATCH --mem=64G
#SBATCH --time 0-01:00
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o crfigs_%j.out
#SBATCH -e crfigs_%j.err
# The core/rim arm's figures were never generated. ICA and PIA are SEPARATE perturbations here
# (pert_map {__basal__:0, ICA:1, PIA:2}), which is the only arm that reproduces a rim-high gene.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=${CONCERT_WORKDIR}
echo "=== 2D core/rim: section 1 only ==="
python -u ${NS}/stroke3d/fig_cr_patch.py --arm 2d --tag _cr --sections 1 \
    --genes Gm42418,Scgb1a1,Spp1,Vim,Lgals1,Timp1,Lcn2
echo "=== 3D core/rim: all four sections ==="
python -u ${NS}/stroke3d/fig_cr_patch.py --arm 3d --tag _cr \
    --genes Gm42418,Scgb1a1,Spp1,Vim,Lgals1,Timp1,Lcn2
