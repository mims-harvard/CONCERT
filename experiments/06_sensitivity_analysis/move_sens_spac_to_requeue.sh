#!/bin/bash
# Enforce the lab partition policy: at most 12 jobs on ${PART_A100} and 12 on ${PART_H100}; everything
# else goes to ${PART_REQUEUE}. Cancels the surplus PENDING arms and resubmits them to requeue with
# RESUME=True + --requeue, so a preempted arm continues from its checkpoint instead of restarting
# (run_concert_spac.py:1439 guards resume on the checkpoint existing, so a fresh arm is unaffected).
# Partition names of your cluster: an H100 partition, an A100 partition and a preemptible one.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

PART_H100="${PART_H100:?set PART_H100}"; PART_A100="${PART_A100:?set PART_A100}"; PART_REQUEUE="${PART_REQUEUE:?set PART_REQUEUE}"
set -euo pipefail
cd ${CODEBASE_ROOT}/models/concert_spac_niche
LIST="${1:?need a file of '<jobid> <jobname>' lines}"
count_part () { squeue -u "$USER" -h -p "$1" -o "%i" | wc -l; }
n_h100=$(count_part ${PART_H100}); n_a100=$(count_part ${PART_A100})

env_for () {                       # tag -> knob overrides, mirroring launch_sens_spac.sh
  case "$1" in
    lsb1)   echo "LSMIN=0.05 LSMAX=0.30 LSINIT=0.083" ;;
    lsb3)   echo "LSMIN=0.45 LSMAX=2.70 LSINIT=0.75" ;;
    ind8)   echo "INDUCE=8" ;;      ind64)  echo "INDUCE=64" ;;
    cs10)   echo "CSCALE=10.0" ;;   cs40)   echo "CSCALE=40.0" ;;
    mor0)   echo "MORAN=0.0" ;;     mor2)   echo "MORAN=2.0" ;;
    lfc05)  echo "LFCL=0.5" ;;      lfc2)   echo "LFCL=2.0" ;;
    dis2)   echo "DISR=2.0" ;;      dis8)   echo "DISR=8.0" ;;
    nfd4)   echo "NFDIM=4" ;;       nfd16)  echo "NFDIM=16" ;;
    nfk10)  echo "NFK=10" ;;        nfk50)  echo "NFK=50" ;;
    brf64)  echo "BRFFF=64" ;;      brf256) echo "BRFFF=256" ;;
    brb005) echo "BRFFB=0.05" ;;    brb030) echo "BRFFB=0.30" ;;
    base42|seed*) echo "" ;;                       # base config, seed comes from the tag
    *) echo "UNKNOWN" ;;
  esac
}

while read -r jid name; do
  tag="${name#spsens_}"
  case "$tag" in
    *_s1) base="${tag%_s1}"; seed=1 ;;
    *_s2) base="${tag%_s2}"; seed=2 ;;
    seed1) base=seed1; seed=1 ;;  seed2) base=seed2; seed=2 ;;
    seed3) base=seed3; seed=3 ;;  seed4) base=seed4; seed=4 ;;
    *)    base="$tag";  seed=42 ;;
  esac
  knobs=$(env_for "$base")
  if [ "$knobs" = "UNKNOWN" ]; then echo "!! unknown tag $tag ($jid) - left alone"; continue; fi
  [ "$jid" != "0" ] && scancel "$jid" || true
  # fill the capped partitions to 12 each first, then overflow to the uncapped requeue partition
  if   [ "$n_h100" -lt 12 ]; then part=${PART_H100}; n_h100=$((n_h100+1))
  elif [ "$n_a100" -lt 12 ]; then part=${PART_A100};      n_a100=$((n_a100+1))
  else part=${PART_REQUEUE}
  fi
  new=$(env TAG="$tag" RESUME=True $knobs SEED="$seed" \
        sbatch --parsable --requeue -p "$part" -J "spsens_${tag}" sub_sens_spac_arm.sh)
  echo "${tag}: ${jid} -> ${new} (${part})"
done < "$LIST"
