#!/bin/bash
# Central path configuration. Every run script in experiments/ sources this file, so this is
# the only place that needs editing when the codebase is moved to another machine.

# Root of this codebase (the directory containing models/, experiments/, benchmark_evaluation/).
CODEBASE_ROOT="${CODEBASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Input data. See experiments/*/README.md for the expected contents of each.
DATA_ROOT="${DATA_ROOT:-/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets}"
SPAC_SEQ_DIR="${SPAC_SEQ_DIR:-${DATA_ROOT}/newdata/spac_seq}"
PERTURB_SEQ_DIR="${PERTURB_SEQ_DIR:-${DATA_ROOT}}"
DSS_DIR="${DSS_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/processed/dryad_merfish_dss}"

# Where checkpoints and evaluation results are written.
OUTPUT_ROOT="${OUTPUT_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

# Python environment. Replace with your own conda installation or drop the two lines below
# and activate the environment before submitting.
CONDA_SH="${CONDA_SH:-/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env}"

activate_env() {
    set +u
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    set -eo pipefail
    export NUMBA_CACHE_DIR="/tmp/numba_cache_$$"
    mkdir -p "${NUMBA_CACHE_DIR}"
}

mkdir -p "${LOG_DIR}"
