#!/bin/bash
#SBATCH --job-name=spsens_aggplot
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -o aggplot_%j.out
#SBATCH -e aggplot_%j.err
# conda's activate.d hooks reference unbound vars -> keep `set -u` off until after activate
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python -u aggregate_spacseq.py      # -> $CONCERT_WORKDIR/sensitivity/sens_spac_arms.csv
python -u plot_spacseq.py --csv "${CONCERT_WORKDIR}/sensitivity/sens_spac_arms.csv"
echo DONE_AGGPLOT
