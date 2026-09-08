#!/bin/bash -l
#SBATCH -J concert_spac_seq_eval
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 0-08:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_spac_seq_eval_%A_%a.out
#SBATCH -e concert_spac_seq_eval_%A_%a.err
#
# Score a trained SPAC-seq checkpoint on the held-out perturbations of one slide.
# Submit as an array so the five slides are scored in parallel:
#
#   sbatch --array=1-5 eval_spac_seq_benchmark.sh <seen|unseen>
#
# Each task writes per_pert_summary.json for its slide; build_benchmark_v3_summary.py
# (in benchmark_evaluation/) collects the five files into the reported table.
#
# The lengthscale band must be passed here as well as at training. It is a plain attribute of
# the spatial field rather than a saved weight, so the same trained parameters render a
# different field if the band is left at its default.

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

REGIME="${1:?usage: sbatch --array=1-5 eval_spac_seq_benchmark.sh <seen|unseen>}"
SLIDE="1-${SLURM_ARRAY_TASK_ID:?submit with --array=1-5}"
MODEL="${OUTPUT_ROOT}/spac_seq_${REGIME}/model.pt"
PREPARED="${SPAC_SEQ_DIR}/prepared"
OUT="${OUTPUT_ROOT}/spac_seq_${REGIME}_eval"

[ -f "$MODEL" ] || { echo "[FATAL] checkpoint missing: $MODEL"; exit 1; }
cd "${CODEBASE_ROOT}/benchmark_evaluation"
export PYTHONPATH="${CODEBASE_ROOT}/models/concert_benchmark:${PYTHONPATH}"

python -u run_concert_v3_benchmark_eval.py \
    --regime "${REGIME}" \
    --target_slide "${SLIDE}" \
    --prepared_dir "${PREPARED}" \
    --model_file "${MODEL}" \
    --outdir "${OUT}" \
    --n_boot 10 \
    --min_g_cells 20 \
    --max_src_cells_per_niche 500 \
    --ls_prior_min 3 --ls_prior_max 18 \
    --fpert_ls_min 0.15 --fpert_ls_max 0.90 \
    --force

echo "[$(date)] SPAC-seq ${REGIME} evaluation finished for slide ${SLIDE} -> ${OUT}"
