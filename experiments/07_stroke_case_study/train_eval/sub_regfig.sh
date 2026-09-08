#!/bin/bash -l
#SBATCH -J stroke_regfig
#SBATCH --mem=32G
#SBATCH --time 0-00:20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/stroke_%x_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/stroke_%x_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
python -u ${NS}/stroke3d/fig_region_conditions.py --outdir "${NS}/outputs/stroke_${1:-2dpara}" --arm "${1:-2dpara}"
