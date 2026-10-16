#!/bin/bash -l
#SBATCH -J concert_pmap_rescore_v4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time 0-04:00
#SBATCH -o concert_pmap_rescore_v4_%j.out
#SBATCH -e concert_pmap_rescore_v4_%j.err
#
# Perturb-map within-niche benchmark, v4 reference (the reported Fig. 4j numbers).
#
#   sbatch rescore_perturb_seq_v4.sh               rescore, then tables and figure
#   SKIP_RESCORE=1 sbatch rescore_perturb_seq_v4.sh  tables and figure only
#
# Rescores the existing predictions of every method; nothing is retrained. The LFC reference is
# built from cells outside the held-out and seen patches (role-0 controls of the same niche, the
# slide's role-0 controls when the niche has fewer than 5), so reference and prediction share no
# cells. perturbmap_v4_scoring.py holds the rule; rescore_pm_v4.py applies it to all methods;
# analyze_pm_v4.py writes the per-record tables, the significance tests and the bar panel.
# Outputs: ${SCORING_LIB}/multi_pm_ho_v4.

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
cd "${SCORING_LIB}"
if [ "${SKIP_RESCORE:-0}" != "1" ]; then
    python -u "${BE}/rescore_pm_v4.py" --outdir multi_pm_ho_v4
fi
python -u "${BE}/analyze_pm_v4.py" --indir multi_pm_ho_v4
echo "[$(date)] Perturb-map v4 rescore finished -> ${SCORING_LIB}/multi_pm_ho_v4"
