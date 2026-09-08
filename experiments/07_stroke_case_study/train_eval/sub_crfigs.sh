#!/bin/bash -l
#SBATCH -J crfigs
#SBATCH --mem=64G
#SBATCH --time 0-01:00
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/crfigs_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/crfigs_%j.err
# The core/rim arm's figures were never generated. ICA and PIA are SEPARATE perturbations here
# (pert_map {__basal__:0, ICA:1, PIA:2}), which is the only arm that reproduces a rim-high gene.
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
echo "=== 2D core/rim: section 1 only ==="
python -u ${NS}/stroke3d/fig_cr_patch.py --arm 2d --tag _cr --sections 1 \
    --genes Gm42418,Scgb1a1,Spp1,Vim,Lgals1,Timp1,Lcn2
echo "=== 3D core/rim: all four sections ==="
python -u ${NS}/stroke3d/fig_cr_patch.py --arm 3d --tag _cr \
    --genes Gm42418,Scgb1a1,Spp1,Vim,Lgals1,Timp1,Lcn2
