#!/bin/bash -l
#SBATCH -J fig_shape_sec1
#SBATCH --mem=64G
#SBATCH --time 0-00:40
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o figshapesec1_%j.out
#SBATCH -e figshapesec1_%j.err
# Figure only: the section-1 shape-matched counterfactual already ran (job 42757995 wrote
# shape1_on_stroke___KEEP___stroke_perturbed_counts.h5ad); only the plotting step failed, on a
# missing wildcard in --on_glob.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

set +u
conda activate concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
NS=${CONCERT_WORKDIR}
python -u ${NS}/stroke3d/stroke_field_by_gene.py \
    --outdir "${NS}/outputs/stroke_2d" --slides "${NS}/stroke3d/slides" --arm 2d \
    --shape_dir "${NS}/outputs/stroke_2d/shape_sets_sec1" --on_glob "shape1_on_*" \
    --tag "_shape_sec1" --sections 1 \
    --genes Gm42418,Scgb1a1,Vim,Lgals1,Timp1,Spp1,Lcn2
