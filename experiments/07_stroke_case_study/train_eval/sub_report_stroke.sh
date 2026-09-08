#!/bin/bash -l
#SBATCH -J stroke_rep
#SBATCH --mem=64G
#SBATCH --time 0-00:30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/stroke_%x_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/stroke_%x_%j.err
# Field + proportion comparison for one arm. Usage: sbatch sub_report_stroke.sh <2d|3d>  (TAG env optional)
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
ARM="${1:?need arm: 2d|3d}"
NS=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
python -u ${NS}/stroke3d/stroke_pattern_report.py \
    --outdir "${NS}/outputs/stroke_${ARM}${TAG:-}" --slides "${NS}/stroke3d/slides" --arm "${ARM}"
