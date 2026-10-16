#!/bin/bash -l
#SBATCH -J pmap_jak2_deconv
#SBATCH -p <partition>
#SBATCH --mem=200G
#SBATCH --time 0-06:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH -o pmap_jak2_concert_%j.out
#SBATCH -e pmap_jak2_concert_%j.err

# Caller: sbatch sub_pmap_s0_jak2_concert_amortized.sh
# Paper method: location-matched CONCERT mu + amortized head (same as the SPAC-seq 2-1 deconvolution).
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=name,memory.total,index --format=csv,noheader
python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
x=torch.zeros(1, device="cuda"); print("cuda_ok", x.device); del x
PY

EVAL=${CONCERT_WORKDIR}/scoring
OUT=${CONCERT_WORKDIR}/paper/Outputs/spac_seq/pmap_s0_jak2_concert_amortized
CKPT=${CONCERT_WORKDIR}/outputs/ms_pmap_niche_seen
mkdir -p "$OUT" "${EVAL}/logs"

python -u "${EVAL}/run_pmap_s0_jak2_concert_amortized.py" \
    --ckpt_dir "$CKPT" \
    --outdir "$OUT" \
    --n_synth 4000 --n_synth_test 1000 --epochs 80 \
    --n_samples 5 --batch_size 2048 \
    --device cuda

echo "[$(date)] done -> $OUT"
