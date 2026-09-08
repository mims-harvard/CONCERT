#!/bin/bash -l
#SBATCH -J stroke_cap
#SBATCH --mem=32G
#SBATCH --time 0-00:15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/stroke_%x_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/stroke_%x_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
python -u /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/diag_fpert_capacity.py
