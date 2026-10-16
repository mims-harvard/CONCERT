#!/bin/bash -l
#SBATCH -J concert_mean_target_audit
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time 0-12:00
#SBATCH -o concert_mean_target_audit_%j.out
#SBATCH -e concert_mean_target_audit_%j.err
#
# How strong the Mean target reference strategy is as a function of the number of knockout cells
# it averages (the audit behind the 50-cell Mean target of the SPAC-seq benchmark).
#
#   sbatch run_mean_target_audit.sh seen
#   sbatch run_mean_target_audit.sh unseen
#   python summarize_sweep.py            after both, writes mean_target_ncells_summary.csv
#
# audit_mean_target_pool.py (no GPU needed) counts the knockout cells each unit's Mean target
# draws on, against the cells the models train on.

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

ARM="${1:?usage: sbatch run_mean_target_audit.sh <seen|unseen>}"
cd "${SCORING_LIB}"
python -u "${CODEBASE_ROOT}/experiments/01_spac_seq_benchmark/mean_target_audit/mean_target_ncells_sweep.py" \
    --outdir mean_target_audit --arms "${ARM}"
