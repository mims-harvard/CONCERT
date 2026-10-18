#!/bin/bash
# Fan out the 20 remaining SI sensitivity arms (base42 is submitted separately as the smoke test).
# Design: experiments/06_sensitivity_analysis/README.md
# Caps: 12 concurrent on ${PART_H100}, 12 on ${PART_A100} (lab policy), overflow -> ${PART_REQUEUE}.
# Partition names of your cluster: an H100 partition, an A100 partition and a preemptible one.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

PART_H100="${PART_H100:?set PART_H100}"; PART_A100="${PART_A100:?set PART_A100}"; PART_REQUEUE="${PART_REQUEUE:?set PART_REQUEUE}"
set -euo pipefail
cd ${CODEBASE_ROOT}/models/concert_spac_niche

# Replicate every factor arm at extra seeds: REPS="1 2" ./submit_perturbmap_sweep.sh
# (seed noise turned out to be as large as the factor effects, so one run per arm cannot
#  separate a real effect from initialisation variance.)
REPS="${REPS:-}"
# Launch a subset only:  ONLY="nfd4 nfd16" REPS="42 1 2" ./submit_perturbmap_sweep.sh
# Seed 42 keeps the bare tag (that is what the aggregator expects); other seeds get _s<N>.
ONLY="${ONLY:-}"

# tag|env overrides  -- each arm is base with exactly one factor moved
ARMS=(
  "seed1|SEED=1"
  "seed2|SEED=2"
  "seed3|SEED=3"
  "seed4|SEED=4"
  "lsb1|LSMIN=0.05 LSMAX=0.30 LSINIT=0.083"
  "lsb2|LSMIN=0.15 LSMAX=0.90 LSINIT=0.25"
  "lsb3|LSMIN=0.45 LSMAX=2.70 LSINIT=0.75"
  "ind16|INDUCE=16"
  "ind32|INDUCE=32"
  "ind128|INDUCE=128"
  "cs10|CSCALE=10.0"
  "cs40|CSCALE=40.0"
  "cs70|CSCALE=70.0"
  "mor0|MORAN=0.0"
  "mor05|MORAN=0.5"
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

count_part () { squeue -u "$USER" -h -p "$1" -o "%i" | wc -l; }
n_h100=$(count_part ${PART_H100})
n_a100=$(count_part ${PART_A100})

if [ -n "$ONLY" ]; then
  KEEP=(); for e in "${ARMS[@]}"; do for w in $ONLY; do [ "${e%%|*}" = "$w" ] && KEEP+=("$e"); done; done
  ARMS=("${KEEP[@]}")
fi

if [ -n "$REPS" ]; then      # replication pass: skip the seed arms, they already vary the seed
  KEEP=(); for e in "${ARMS[@]}"; do case "${e%%|*}" in seed*) ;; *) KEEP+=("$e");; esac; done
  ARMS=()
  for s in $REPS; do
    sfx="_s${s}"; [ "$s" = "42" ] && sfx=""
    for e in "${KEEP[@]}"; do ARMS+=("${e%%|*}${sfx}|${e#*|} SEED=${s}"); done
  done
fi

for entry in "${ARMS[@]}"; do
  tag="${entry%%|*}"; envs="${entry#*|}"
  if   [ "$n_h100" -lt 12 ]; then part=${PART_H100}; n_h100=$((n_h100+1))
  elif [ "$n_a100" -lt 12 ]; then part=${PART_A100};      n_a100=$((n_a100+1))
  else                            part=${PART_REQUEUE}
  fi
  jid=$(env TAG="$tag" $envs sbatch --parsable -p "$part" -J "sens_${tag}" run_perturbmap_arm.sh)
  echo "${tag}  ${part}  ${jid}  [${envs}]"
done
echo "h100=${n_h100} ${PART_A100}=${n_a100}"
