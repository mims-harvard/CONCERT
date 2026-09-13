#!/bin/bash -l
#SBATCH -J t3_mu_states
#SBATCH -p kempner_h100
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=200G
#SBATCH --time 0-02:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/logs/task3_mu_states_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/logs/task3_mu_states_%j.err

# Cache CONCERT mu_KO / mu_un / mu_normal for all Task-3 peri (no empirical means).
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
export CUDA_VISIBLE_DEVICES=0

EVAL=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval
OUT=/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/spac_seq/pmap_s0_jak2_concert_amortized
CKPT=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_pmap_niche_seen
mkdir -p "$OUT" "${EVAL}/logs"

python -u "${EVAL}/cache_task3_concert_state_signatures.py" \
    --ckpt_dir "$CKPT" \
    --outdir "$OUT" \
    --n_samples 5 --batch_size 2048 \
    --force

# Rebuild Task-3 figs with CONCERT-only imputed signatures
python -u "${EVAL}/replot_task3_deconv_impute_subspots.py"
python -u "${EVAL}/replot_task3_umap_pure_vs_peri.py"
python -u "${EVAL}/replot_task3_marker_slide.py" --gene Fn1
python -u "${EVAL}/replot_task3_marker_slide.py" --gene Spp1

echo "[$(date)] done -> $OUT"
