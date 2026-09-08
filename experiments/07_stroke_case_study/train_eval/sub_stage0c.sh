#!/bin/bash -l
#SBATCH -J stroke_st0c
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=100G
#SBATCH --time 0-01:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/stage0c_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/stage0c_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
cd /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d
python -u stage0c_response_3d.py
