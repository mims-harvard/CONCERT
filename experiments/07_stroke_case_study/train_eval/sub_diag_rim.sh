#!/bin/bash -l
#SBATCH -J diag_rim
#SBATCH --mem=48G
#SBATCH --time 0-00:30
#SBATCH -p kempner_requeue
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/diag_rim_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/diag_rim_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
python -u /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/diag_rim3.py
