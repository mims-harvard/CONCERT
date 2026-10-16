#!/bin/bash
# SPAC-seq sensitivity sweep: 65 arms = 5 base seeds + 10 factors x 2 values x 3 seeds.
# ~17 h per arm (16 h train + ~50 min 5-slide eval), 24 non-preemptible slots -> 3 waves, ~2 days.
# ${PART_REQUEUE} is NOT used: a preempted 16 h arm restarts from scratch.
# Usage: WAVE=1 ./submit_spacseq_sweep.sh
# Partition names of your cluster: an H100 partition, an A100 partition and a preemptible one.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

PART_H100="${PART_H100:?set PART_H100}"; PART_A100="${PART_A100:?set PART_A100}"; PART_REQUEUE="${PART_REQUEUE:?set PART_REQUEUE}"
set -euo pipefail
cd ${CODEBASE_ROOT}/models/concert_spac_niche
WAVE="${WAVE:?need WAVE=1|2|3}"

# factor settings, anchored at the REPORTED spac base (inducing 16, band [0.15,0.90], init 0.25)
SETTINGS=(
  "lsb1|LSMIN=0.05 LSMAX=0.30 LSINIT=0.083"
  "lsb3|LSMIN=0.45 LSMAX=2.70 LSINIT=0.75"
  "ind8|INDUCE=8"
  "ind64|INDUCE=64"
  "cs10|CSCALE=10.0"
  "cs40|CSCALE=40.0"
  "mor0|MORAN=0.0"
  "mor2|MORAN=2.0"
  "lfc05|LFCL=0.5"
  "lfc2|LFCL=2.0"
  "dis2|DISR=2.0"
  "dis8|DISR=8.0"
  "nfd4|NFDIM=4"
  "nfd16|NFDIM=16"
  "nfk10|NFK=10"
  "nfk50|NFK=50"
  "brf64|BRFFF=64"
  "brf256|BRFFF=256"
  "brb005|BRFFB=0.05"
  "brb030|BRFFB=0.30"
)
case "$WAVE" in
  1) SEED=42; EXTRA=("base42|SEED=42" "seed1|SEED=1" "seed2|SEED=2" "seed3|SEED=3") ;;
  2) SEED=1;  EXTRA=("seed4|SEED=4") ;;
  3) SEED=2;  EXTRA=() ;;
  *) echo "WAVE must be 1, 2 or 3"; exit 1 ;;
esac

ARMS=()
for e in "${EXTRA[@]}"; do ARMS+=("$e"); done          # base-config runs define the seed band
sfx=""; [ "$SEED" != "42" ] && sfx="_s${SEED}"
for e in "${SETTINGS[@]}"; do ARMS+=("${e%%|*}${sfx}|${e#*|} SEED=${SEED}"); done

count_part () { squeue -u "$USER" -h -p "$1" -o "%i" | wc -l; }
n_h100=$(count_part ${PART_H100}); n_a100=$(count_part ${PART_A100})
for entry in "${ARMS[@]}"; do
  tag="${entry%%|*}"; envs="${entry#*|}"
  if   [ "$n_h100" -lt 12 ]; then part=${PART_H100}; n_h100=$((n_h100+1))
  elif [ "$n_a100" -lt 12 ]; then part=${PART_A100};      n_a100=$((n_a100+1))
  elif [ "${QUEUE:-0}" = "1" ]; then
    # caps limit RUNNING jobs, not submissions: extra arms queue and start as slots free
    part=$([ $(( (n_h100 + n_a100) % 2 )) -eq 0 ] && echo ${PART_H100} || echo ${PART_A100})
    n_h100=$((n_h100+1))
  else echo "SKIP ${tag}: both capped partitions full (rerun this wave later)"; continue
  fi
  jid=$(env TAG="$tag" $envs sbatch --parsable -p "$part" -J "spsens_${tag}" run_spacseq_arm.sh)
  echo "${tag}  ${part}  ${jid}  [${envs}]"
done
echo "h100=${n_h100} ${PART_A100}=${n_a100}"
