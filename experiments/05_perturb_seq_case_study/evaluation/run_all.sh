#!/bin/bash -l
#SBATCH -J pmap_case_study
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH --time 0-01:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o pmap_case_study_%j.out
#SBATCH -e pmap_case_study_%j.err
#
# Run the whole Perturb-seq case study on the checkpoint from experiment 02, in order.
#
# Usage: sbatch run_all.sh
#
# Each step fixes its own batch size and does not take one from the command line. That is
# deliberate: the niche descriptor is built from each spot's nearest neighbours WITHIN ITS
# BATCH, so a batch smaller than the section silently replaces every spot's neighbourhood with
# a subsample of it, and every niche quantity below is then measured against the wrong
# neighbours. See the comment in case_study.py::build_args.

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

# Derived from CODEBASE_ROOT rather than from this file, for the sbatch reason above.
HERE="${CODEBASE_ROOT}/experiments/05_perturb_seq_case_study/evaluation"
export PMAP_CKPT="${PMAP_CKPT:-${OUTPUT_ROOT}/perturb_seq_seen}"
export PMAP_CASE_OUT="${PMAP_CASE_OUT:-${OUTPUT_ROOT}/perturb_seq_case_study/case}"
export PMAP_INTERPRET_OUT="${PMAP_INTERPRET_OUT:-${OUTPUT_ROOT}/perturb_seq_case_study/interpret}"
export PMAP_SYSTEMATIC_OUT="${PMAP_SYSTEMATIC_OUT:-${OUTPUT_ROOT}/perturb_seq_case_study/systematic}"

[ -f "${PMAP_CKPT}/model.pt" ] || {
    echo "[FATAL] checkpoint missing: ${PMAP_CKPT}/model.pt -- run experiment 02 training first"
    exit 1
}
cd "$HERE"

# 1. Transplantation: move a patch of unperturbed spots into two microenvironments.
python -u case_study.py --batch_size 2048

# 2. Decomposition of the predicted effect, and how far an edit reaches, resolved by how many
#    of each spot's neighbours were actually edited.
python -u interpret_perturbations.py

# 3. Each knockout placed into many sampled neighbourhoods.
python -u transplant_analysis.py

# 4. Which programmes track the neighbourhood, with the overall effect size divided out.
python -u pathway_analysis.py

echo "[$(date)] Perturb-seq case study finished -> ${OUTPUT_ROOT}/perturb_seq_case_study"
