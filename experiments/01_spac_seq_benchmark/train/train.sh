#!/bin/bash -l
#SBATCH -J concert_spac_seq
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 2-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_spac_seq_%j.out
#SBATCH -e concert_spac_seq_%j.err
#
# Train CONCERT on the SPAC-seq slides (five sections, 1-1 to 1-5).
#
# Usage: sbatch train.sh <seen|unseen>
#   seen   perturbations held out within a slide; the perturbation is observed on other slides
#   unseen perturbations held out from every slide
#
# The holdout file is produced by ../data_preparation/05_build_holdout_coords.py --dataset spac.
#
# Clonality gate: guide spots of a (slide, perturbation) clone with fewer than 100 training
# spots get weight 0 in the likelihood and in the LFC targets; all NTC spots of a slide form one
# clone. A per-slide guide shift is learned on top of the shared guide shift.

# Locate config/paths.sh. Under sbatch the batch script is copied to a spool directory, so a
# path relative to this file does not resolve there; CODEBASE_ROOT is the fallback. Failing
# loudly matters here: if this source silently fails, activate_env never runs, `set -e` is never
# enabled, and every step below fails while the script still reports success.
_PATHS="$(dirname "${BASH_SOURCE[0]}")/../../../config/paths.sh"
[ -f "$_PATHS" ] || _PATHS="${CODEBASE_ROOT:-}/config/paths.sh"
[ -f "$_PATHS" ] || {
    echo "[FATAL] cannot find config/paths.sh."
    echo "        Submitting with sbatch? Pass the codebase root explicitly:"
    echo "          sbatch --export=ALL,CODEBASE_ROOT=/path/to/codebase $0 ..."
    exit 1
}
source "$_PATHS"
activate_env

REGIME="${1:?usage: sbatch train.sh <seen|unseen>}"
MODEL_DIR="${CODEBASE_ROOT}/models/concert_benchmark"
OUT="${OUTPUT_ROOT}/spac_seq_${REGIME}"
HOLDOUT="${HOLDOUT_COORDS}/holdout_spac_${REGIME}.json"

FILES=$(for i in 1 2 3 4 5; do echo -n "${SPAC_SEQ_DIR}/raw_output_1-${i}.h5ad,"; done | sed 's/,$//')
[ -f "$HOLDOUT" ] || { echo "[FATAL] holdout file missing: $HOLDOUT"; exit 1; }
mkdir -p "$OUT"
cd "$MODEL_DIR"

# Every model setting of the reported runs, including the lengthscale band of the spatial field
# and the clonality gate, is in models/concert_benchmark/configs/spacseq.yaml. Evaluation reads
# the same file through `run_concert_spac.py --stage eval --config`; the benchmark scorer
# (evaluation/evaluate.sh) passes the same band explicitly.
python -u run_concert_spac.py --config configs/spacseq.yaml --stage train \
    --multi_sample_files "${FILES}" --outdir "$OUT" \
    --holdout_file "${HOLDOUT}" --resume "${RESUME:-False}"

echo "[$(date)] SPAC-seq ${REGIME} training finished -> ${OUT}"
