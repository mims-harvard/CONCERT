#!/bin/bash -l
#SBATCH -J fig_fixC
#SBATCH --mem=48G
#SBATCH --time 0-00:40
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH -o figfixC_%j.out
#SBATCH -e figfixC_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
O=${CONCERT_WORKDIR}/outputs/stroke_2dpara_fixC
for G in Gm42418 Lcn2 Vim Timp1; do
  echo "=== ${G}"
  python -u ${CONCERT_WORKDIR}/stroke3d/fig_conditions.py \
      --outdir $O --arm fixC_${G} --gene ${G}
done
