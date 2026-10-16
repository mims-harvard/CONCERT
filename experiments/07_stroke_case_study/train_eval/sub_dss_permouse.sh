#!/bin/bash -l
#SBATCH -J dsspm
#SBATCH -p <partition>
#SBATCH --mem=64G
#SBATCH --time 0-01:30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o dsspm_%j.out
#SBATCH -e dsspm_%j.err
# One figure pair per mouse (GEX field + inflamed region), rows = every slide of that mouse.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
cd ${CONCERT_WORKDIR}/paper/Outputs/case_study
export DSS_ALL_SLIDES=1
for M in 062221_D9_m3 062921_D0_m3a 082421_D0_m6 082421_D21_m1 082421_D21_m2 092421_D3_m1 092421_D3_m4 100221_D9_m5; do
  echo "[$(date)] mouse $M"
  DSS_MICE=$M DSS_OUT_PREFIX="dss_permouse_${M}" python -u plot_q3_per_mouse.py
done
ls -1 dss_permouse_*.pdf | wc -l
