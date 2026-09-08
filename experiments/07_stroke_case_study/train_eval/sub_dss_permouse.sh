#!/bin/bash -l
#SBATCH -J dsspm
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=64G
#SBATCH --time 0-01:30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/dsspm_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/dsspm_%j.err
# One figure pair per mouse (GEX field + inflamed region), rows = every slide of that mouse.
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
cd /n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study
export DSS_ALL_SLIDES=1
for M in 062221_D9_m3 062921_D0_m3a 082421_D0_m6 082421_D21_m1 082421_D21_m2 092421_D3_m1 092421_D3_m4 100221_D9_m5; do
  echo "[$(date)] mouse $M"
  DSS_MICE=$M DSS_OUT_PREFIX="dss_permouse_${M}" python -u plot_q3_per_mouse.py
done
ls -1 dss_permouse_*.pdf | wc -l
