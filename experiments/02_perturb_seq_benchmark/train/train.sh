#!/bin/bash -l
#SBATCH -J concert_perturb_seq
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 0-06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_perturb_seq_%j.out
#SBATCH -e concert_perturb_seq_%j.err
#
# Train CONCERT on the spatial Perturb-seq slides (four sections, s0 to s3).
#
# Usage: sbatch train.sh <seen|unseen>
#
# The model is the same one used for SPAC-seq (models/concert_benchmark). The settings differ
# only where the two assays differ: this panel has no non-targeting control class, spots are
# denser so the spatial field needs more inducing points and a wider lengthscale band, and no
# action correction is applied because guide assignment is already unambiguous here.
#
# The inducing-point count and the patience below are not free choices. Carrying over SPAC-seq's
# 16 inducing points and short patience costs about a third of the rank correlation on this
# assay; 64 points with patience 300 is what the reported numbers were produced with.

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
OUT="${OUTPUT_ROOT}/perturb_seq_${REGIME}"
HOLDOUT="${HOLDOUT_COORDS}/holdout_pmap_${REGIME}.json"

FILES=$(for s in 0 1 2 3; do echo -n "${PERTURB_SEQ_DIR}/perturbmap_s${s}_spacfmt.h5ad,"; done | sed 's/,$//')
[ -f "$HOLDOUT" ] || { echo "[FATAL] holdout file missing: $HOLDOUT"; exit 1; }
mkdir -p "$OUT"
cd "$MODEL_DIR"

# Every model setting of the reported runs (64 inducing points, patience 300, the lengthscale band
# [0.01, 0.60] at coordinate scale 70, no action correction, no control class) is in
# models/concert_benchmark/configs/perturbmap.yaml; `run_concert_spac.py --stage eval --config` reads
# the same file, and evaluation/evaluate.sh passes the same band explicitly.
python -u run_concert_spac.py --config configs/perturbmap.yaml --stage train \
    --multi_sample_files "${FILES}" --outdir "$OUT" \
    --holdout_file "${HOLDOUT}" --resume "${RESUME:-False}"

echo "[$(date)] Perturb-seq ${REGIME} training finished -> ${OUT}"
