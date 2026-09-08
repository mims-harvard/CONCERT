#!/bin/bash -l
#SBATCH -J diag_leakarm
#SBATCH --mem=48G
#SBATCH --time 0-00:30
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/diag_leakarm_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/diag_leakarm_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
B=/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_rimtest
for A in 2d 3d; do
  echo; echo "################ ARM stroke_${A} ################"
  python -u /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/diag_leak_arm.py $B/leak_$A $A
done
