#!/bin/bash -l
#SBATCH -J fixfigs
#SBATCH --mem=64G
#SBATCH --time 0-01:00
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/fixfigs_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/fixfigs_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
echo "=== how close are the two cortical-depth patches ==="
python -u ${NS}/stroke3d/diag_patch_sep.py
echo "=== condition figures, row 2 now the induction ==="
for G in Lcn2 Vim Timp1 Gm42418; do
  python -u ${NS}/stroke3d/fig_conditions.py --outdir "${NS}/outputs/stroke_2dpara_fixC" \
      --arm "fixC_${G}" --gene "${G}"
done
