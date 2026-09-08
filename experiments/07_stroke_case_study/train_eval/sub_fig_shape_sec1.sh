#!/bin/bash -l
#SBATCH -J fig_shape_sec1
#SBATCH --mem=64G
#SBATCH --time 0-00:40
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/figshapesec1_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/figshapesec1_%j.err
# Figure only: the section-1 shape-matched counterfactual already ran (job 42757995 wrote
# shape1_on_stroke___KEEP___stroke_perturbed_counts.h5ad); only the plotting step failed, on a
# missing wildcard in --on_glob.
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
python -u ${NS}/stroke3d/stroke_field_by_gene.py \
    --outdir "${NS}/outputs/stroke_2d" --slides "${NS}/stroke3d/slides" --arm 2d \
    --shape_dir "${NS}/outputs/stroke_2d/shape_sets_sec1" --on_glob "shape1_on_*" \
    --tag "_shape_sec1" --sections 1 \
    --genes Gm42418,Scgb1a1,Vim,Lgals1,Timp1,Spp1,Lcn2
