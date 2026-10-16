#!/bin/bash -l
#SBATCH -J concert_pmap_rescore
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time 0-04:00
#SBATCH -o concert_pmap_rescore_%j.out
#SBATCH -e concert_pmap_rescore_%j.err
#
# Perturb-map within-niche benchmark, disjoint reference (the reported Fig. 4j numbers).
#
#   sbatch aggregate.sh               rescore, then tables and figure
#   SKIP_RESCORE=1 sbatch aggregate.sh  tables and figure only
#
# Rescores the existing predictions of every method; nothing is retrained. The LFC reference is
# built from cells outside the held-out and seen patches (role-0 controls of the same niche, the
# slide's role-0 controls when the niche has fewer than 5), so reference and prediction share no
# cells. perturbmap_reference.py holds the rule; score_perturbmap.py applies it to all methods;
# perturbmap_tables.py writes the per-record tables, the significance tests and the bar panel.
# Outputs: ${CONCERT_WORKDIR}/perturbmap_results (PM_RESULTS overrides the directory).

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
export PYTHONHASHSEED=0
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"

BE="${CODEBASE_ROOT}/benchmark_evaluation"
OUT="${PM_RESULTS:-${CONCERT_WORKDIR}/perturbmap_results}"
mkdir -p "${OUT}"
if [ "${SKIP_RESCORE:-0}" != "1" ]; then
    python -u "${BE}/score_perturbmap.py" --outdir "${OUT}"
fi
python -u "${BE}/perturbmap_tables.py" --indir "${OUT}"
echo "[$(date)] Perturb-map rescore finished -> ${OUT}"
