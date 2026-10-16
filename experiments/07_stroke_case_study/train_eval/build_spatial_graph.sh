#!/bin/bash -l
#SBATCH -J graph_r1
#SBATCH --mem=48G
#SBATCH --time 0-00:30
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH -o graph_r1_%j.out
#SBATCH -e graph_r1_%j.err
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
O=${CONCERT_WORKDIR}/outputs/stroke_2dpara
cd ${CODEBASE_ROOT}/models/concert_spac_niche
python -u build_spatial_graph.py --graph_in_npz ${O}/graph_inputs.npz \
       --out_npz ${O}/graph_r1.npz --r_max 1.0
