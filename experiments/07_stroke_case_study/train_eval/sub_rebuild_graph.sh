#!/bin/bash -l
#SBATCH -J graph_r1
#SBATCH --mem=48G
#SBATCH --time 0-00:30
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/graph_r1_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/graph_r1_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
O=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/stroke_2dpara
cd /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/CONCERT-spac-niche
python -u build_spatial_graph.py --graph_in_npz ${O}/graph_inputs.npz \
       --out_npz ${O}/graph_r1.npz --r_max 1.0
