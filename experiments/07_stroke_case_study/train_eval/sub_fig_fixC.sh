#!/bin/bash -l
#SBATCH -J fig_fixC
#SBATCH --mem=48G
#SBATCH --time 0-00:40
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/figfixC_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/figfixC_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
O=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/stroke_2dpara_fixC
for G in Gm42418 Lcn2 Vim Timp1; do
  echo "=== ${G}"
  python -u /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/fig_conditions.py \
      --outdir $O --arm fixC_${G} --gene ${G}
done
