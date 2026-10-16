#!/bin/bash -l
#SBATCH -J concert_dss
#SBATCH -p gpu
#SBATCH --mem=256G
#SBATCH --time 1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o concert_dss_%j.out
#SBATCH -e concert_dss_%j.err
#
# Train the temporal CONCERT model on the DSS colitis MERFISH time course
# (Healthy, day 3, day 9, day 21; 52 sections listed in the data manifest).
#
# Usage: sbatch train.sh [TAG]
#   HOLDOUT_MICE="m1,m2"  optional; excludes every section of those animals from training,
#                         giving a leave-one-animal-out checkpoint
#
# The response is written as amplitude(t) x spatial field(lengthscale(t)) x perturbation
# effect, so a single model covers all four timepoints and can be asked for a timepoint it was
# not trained on. Three settings below fix the direction and magnitude of the inflammation
# response, which are otherwise not identified by the likelihood alone:
#   - a per-cell inflammation field supervised against the observed marker score, so the
#     magnitude is tied to measured inflammation rather than free
#   - the field's time input is frozen, because each section carries exactly one day and
#     "healthy architecture at day 9" therefore never appears in the data; time still enters
#     through the spatial field, which is where the rise and resolution are learned
#   - the sign of the effect is constrained to be positive on the inflammation marker panel
#     only; the remaining genes stay free, since resolution genuinely lowers some of them

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

TAG="${1:-dss}"
MODEL_DIR="${CODEBASE_ROOT}/models/concert_temporal"
MANIFEST="${DSS_DIR}/manifest.csv"
ARCH_DIR="${CONCERT_DATA_DIR}/dss_colitis/architecture"
OUT="${OUTPUT_ROOT}/dss_${TAG}"

# Manifest columns: filename, ..., mouse id, sample type.
FILELIST=$(awk -F, -v hm="${HOLDOUT_MICE:-}" '
    BEGIN { n = split(hm, a, ","); for (i = 1; i <= n; i++) if (a[i] != "") holdout[a[i]] = 1 }
    NR > 1 { if (!($3 in holdout)) print $1 }' "$MANIFEST")
FILES=$(echo "$FILELIST" | sed "s#^#${DSS_DIR}/#" | paste -sd, -)
echo "[data] $(echo "$FILELIST" | wc -l) sections (held-out animals: ${HOLDOUT_MICE:-none})"

mkdir -p "$OUT"
cd "$MODEL_DIR"

# Every model setting of the reported run (time kernel, supervised inflammation field with its
# time input frozen, marker-sign constraint, niche FiLM) is in models/concert_temporal/configs/dss.yaml.
python -u run_concert_spac.py --config configs/dss.yaml --stage train \
    --sample "dss_${TAG}" \
    --multi_sample_files "${FILES}" --outdir "$OUT" \
    --arch_desc_dir "${ARCH_DIR}" --resume False

echo "[$(date)] DSS training finished -> ${OUT}"
