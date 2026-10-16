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
# Score a clonality-gated SPAC-seq checkpoint on the benchmark.
#
#   sbatch evaluate.sh unseen              one job, all 85 KO-target-slide pairs
#   sbatch --array=1-5 evaluate.sh seen    one slide per array task
#
# Scoring rule (both settings): the prediction is one Poisson draw at the measured library size
# of each source spot (--cf_output poisson_depth), and the LFC is taken against the measured
# source spots (--cf_seen_ref observed), the same count-level reference every method receives.
# shared_gene_rule.py applies the shared gene rule: slide HVGs inside the CONCERT 2000-gene panel,
# minus the knocked-out gene. Aggregate with aggregate.sh.
#
# The lengthscale band must be passed here as well as at training. It is a plain attribute of
# the spatial field rather than a saved weight, so the same trained parameters render a
# different field if the band is left at its default.

# Locate config/paths.sh. Under sbatch the batch script is copied to a spool directory, so a
# path relative to this file does not resolve there; CODEBASE_ROOT is the fallback.
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

REGIME="${1:?usage: sbatch evaluate.sh <seen|unseen>}"
# Default: the released checkpoint. MODEL=<your model.pt> scores your own training run.
CKPT_NAME=$([ "$REGIME" = "seen" ] && echo spacseq_within_niche || echo spacseq_cross_slide)
MODEL="${MODEL:-${CONCERT_CHECKPOINT_ROOT}/${CKPT_NAME}/model.pt}"
PREPARED="${SPAC_SEQ_PREPARED}"
OUT="${OUTPUT_ROOT}/spac_seq_${REGIME}_eval"

[ -f "$MODEL" ] || { echo "[FATAL] checkpoint missing: $MODEL"; exit 1; }
cd "${CODEBASE_ROOT}/benchmark_evaluation"
export PYTHONPATH="${CODEBASE_ROOT}/models/concert_benchmark:${PYTHONPATH}"

COMMON="--prepared_dir ${PREPARED} --model_file ${MODEL} --outdir ${OUT} --n_boot 10 \
        --min_g_cells 20 --ls_prior_min 3 --ls_prior_max 18 --fpert_ls_min 0.15 --fpert_ls_max 0.90 \
        --cf_output poisson_depth --cf_seen_ref observed --force"

if [ "$REGIME" = "unseen" ]; then
    python -u shared_gene_rule.py --concert_ctx_check \
        evaluate_concert_spacseq.py --regime unseen \
        --benchmark_dir "${PREPARED}/benchmark" --max_src_cells 5000 ${COMMON}
else
    SLIDE="1-${SLURM_ARRAY_TASK_ID:?submit the seen setting with --array=1-5}"
    python -u shared_gene_rule.py --concert_ctx_check \
        evaluate_concert_spacseq.py --regime seen --target_slide "${SLIDE}" \
        --max_src_cells_per_niche 500 ${COMMON}
fi

echo "[$(date)] SPAC-seq ${REGIME} evaluation finished -> ${OUT}"
