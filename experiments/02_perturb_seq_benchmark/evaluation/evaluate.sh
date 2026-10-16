#!/bin/bash -l
#SBATCH -J concert_perturb_seq_eval
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --time 0-04:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_perturb_seq_eval_%j.out
#SBATCH -e concert_perturb_seq_eval_%j.err
#
# Score a trained Perturb-seq checkpoint on the held-out niche patches of all four sections.
#
# Usage: sbatch evaluate.sh <seen|unseen>
#
# The lengthscale band and coordinate scale are passed again here and MUST match the values the
# checkpoint was trained with: the band defines the squashing function inside the perturbation
# field, so a checkpoint read back with a different band reconstructs different lengthscales.
# The inducing-point count must match too, or the reference grid stored in the checkpoint has
# the wrong shape and loading fails.

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

REGIME="${1:?usage: sbatch evaluate.sh <seen|unseen>}"
EVAL_DIR="${CODEBASE_ROOT}/benchmark_evaluation"
# Default: the released checkpoint. MODEL=<your model.pt> scores your own training run.
CKPT_NAME=$([ "$REGIME" = "seen" ] && echo perturbmap_within_niche || echo perturbmap_cross_slide)
MODEL="${MODEL:-${CONCERT_CHECKPOINT_ROOT}/${CKPT_NAME}/model.pt}"
OUT="${OUTPUT_ROOT}/perturb_seq_${REGIME}_eval"

[ -f "$MODEL" ] || { echo "[FATAL] checkpoint missing: $MODEL"; exit 1; }
mkdir -p "$OUT"
cd "$EVAL_DIR"

# Must match train/train.sh.
GUARD="--ls_prior_min 3 --ls_prior_max 18 --fpert_svgp_inducing 64 \
       --fpert_ls_min 0.01 --fpert_ls_max 0.60 --fpert_coord_scale 70.0"

# In the cross-section setting the held-out perturbations are named in the holdout file, and
# records with too few cells to score are dropped.
UNSEEN_KO=""
if [ "$REGIME" = "unseen" ]; then
    UNSEEN_KO="--unseen_ko_file ${HOLDOUT_COORDS}/holdout_pmap_unseen.json --min_g_cells 20"
fi

python -u evaluate_concert_perturbmap.py \
    --model_file "$MODEL" --outdir "$OUT" --slides ${SLIDES:-0 1 2 3} --n_boot 10 \
    ${GUARD} ${UNSEEN_KO}

echo "[$(date)] Perturb-seq ${REGIME} evaluation finished -> ${OUT}"
