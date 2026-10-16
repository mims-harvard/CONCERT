#!/bin/bash
# Relaunch SPAC-seq sensitivity arms that died on the RTX PRO 6000 Blackwell nodes.
# 22 of the 65 arms failed ~1 min in with
#   nvrtc: error: invalid value for --gpu-architecture (-arch)
# because the concert_env torch build ships no sm_120 kernels. The Blackwell nodes are the only
# ones carrying the rtx6000pro feature, so we exclude them by name (computed live from sinfo).
# Usage: ./relaunch_sens_spac_failed.sh [tag ...]        (no args = the 22 known-failed arms)
# Partition names of your cluster: an H100 partition, an A100 partition and a preemptible one.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

PART_H100="${PART_H100:?set PART_H100}"; PART_A100="${PART_A100:?set PART_A100}"; PART_REQUEUE="${PART_REQUEUE:?set PART_REQUEUE}"
set -euo pipefail
cd ${CODEBASE_ROOT}/models/concert_spac_niche

EXCL=$(sinfo -N -h -o "%N %f" -p "${PART_REQUEUE},${PART_A100},${PART_H100}" \
       | awk '/rtx6000pro/{print $1}' | sort -u | paste -sd, -)
[ -n "$EXCL" ] || { echo "[FATAL] could not resolve rtx6000pro node list"; exit 1; }
echo "excluding: ${EXCL}"

DEFAULT_TAGS="brf256_s2 brf64_s2 lfc05 lsb1_s1 lsb1_s2 lsb3_s1 lsb3_s2 mor0_s1 mor0_s2 mor2 \
mor2_s1 mor2_s2 nfd16 nfd16_s1 nfd16_s2 nfd4 nfd4_s1 nfd4_s2 nfk10_s1 nfk10_s2 nfk50_s1 nfk50_s2"
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && read -ra TAGS <<< "$DEFAULT_TAGS"

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
    base42|seed*) echo "" ;;
    *) echo "UNKNOWN" ;;
  esac
}

count_part () { squeue -u "$USER" -h -p "$1" -o "%i" | wc -l; }
n_h100=$(count_part ${PART_H100}); n_a100=$(count_part ${PART_A100})
for tag in "${TAGS[@]}"; do
  case "$tag" in
    *_s1) base="${tag%_s1}"; seed=1 ;;
    *_s2) base="${tag%_s2}"; seed=2 ;;
    seed1) base=seed1; seed=1 ;;  seed2) base=seed2; seed=2 ;;
    seed3) base=seed3; seed=3 ;;  seed4) base=seed4; seed=4 ;;
    *)    base="$tag";  seed=42 ;;
  esac
  knobs=$(env_for "$base")
  [ "$knobs" = "UNKNOWN" ] && { echo "!! unknown tag ${tag} - skipped"; continue; }
  if   [ "$n_h100" -lt 12 ]; then part=${PART_H100}; n_h100=$((n_h100+1))
  elif [ "$n_a100" -lt 12 ]; then part=${PART_A100};      n_a100=$((n_a100+1))
  else part=${PART_REQUEUE}
  fi
  jid=$(env TAG="$tag" RESUME=True $knobs SEED="$seed" \
        sbatch --parsable --requeue -x "$EXCL" -p "$part" -J "spsens_${tag}" sub_sens_spac_arm.sh)
  echo "${tag}  ${part}  ${jid}  [${knobs} SEED=${seed}]"
done
