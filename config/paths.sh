#!/bin/bash
# Central path configuration. Every run script in experiments/ sources this file, so this is
# the only place that needs editing when the codebase is moved to another machine.

# Root of this codebase (the directory containing models/, experiments/, benchmark_evaluation/).
CODEBASE_ROOT="${CODEBASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Inputs: the downloadable data/ and checkpoints/ folders, placed in the repository root
# (layout in benchmark_evaluation/release_paths.py). Each can be moved with its variable.
CONCERT_DATA_DIR="${CONCERT_DATA_DIR:-${CODEBASE_ROOT}/data}"
CONCERT_CHECKPOINT_ROOT="${CONCERT_CHECKPOINT_ROOT:-${CODEBASE_ROOT}/checkpoints}"
export CONCERT_DATA_DIR CONCERT_CHECKPOINT_ROOT
SPAC_SEQ_DIR="${CONCERT_DATA_DIR}/spacseq/raw"
SPAC_SEQ_PREPARED="${CONCERT_DATA_DIR}/spacseq/prepared"
PERTURB_SEQ_DIR="${CONCERT_DATA_DIR}/perturbmap/raw"
HOLDOUT_COORDS="${CONCERT_DATA_DIR}/benchmark_assets/holdout_coords"
SPACSEQ_PREDICTIONS="${CONCERT_DATA_DIR}/benchmark_assets/results/spacseq_predictions"
DSS_DIR="${CONCERT_DATA_DIR}/dss_colitis/sections"
PERTURBFISH_DIR="${CONCERT_DATA_DIR}/perturbfish"

# Where new training runs, evaluations and aggregations are written.
CONCERT_WORKDIR="${CONCERT_WORKDIR:-${CODEBASE_ROOT}/runs}"
export CONCERT_WORKDIR
OUTPUT_ROOT="${OUTPUT_ROOT:-${CONCERT_WORKDIR}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

# Python environment: the conda environment built from env_requirements.txt (main README).
# Set CONDA_SH to your conda.sh when conda is not initialised in batch shells.
CONDA_ENV="${CONDA_ENV:-concert_env}"

activate_env() {
    set +u
    [ -n "${CONDA_SH:-}" ] && source "${CONDA_SH}"   # conda.sh of your installation, if conda is not initialised
    conda activate "${CONDA_ENV}"
    set -eo pipefail
    export NUMBA_CACHE_DIR="/tmp/numba_cache_$$"
    mkdir -p "${NUMBA_CACHE_DIR}"
}

mkdir -p "${LOG_DIR}"
