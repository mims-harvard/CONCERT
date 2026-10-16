#!/bin/bash
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

cd ${CONCERT_WORKDIR}/stroke3d
W () { while [ "$(squeue -u "$USER" -h -j "$1" 2>/dev/null | wc -l)" -ne 0 ]; do sleep 30; done; }
OK () { [ "$(sacct -X -n -j "$1" --format=State 2>/dev/null | tr -d ' ' | head -1)" = "COMPLETED" ]; }
W "42444387"
OK 42444387 && echo "TRAIN-OK C" || { echo "TRAIN-FAILED C"; exit 1; }
E=$(sbatch --parsable -p gpu_test -t 5:00:00 --export=ALL,CRTAG=_crC sub_eval_cr.sh 3d); echo "evalC=$E"
W "$E"
OK "$E" && D=$(sbatch --parsable -p gpu_test sub_diag_cr.sh 3d _crC) && echo "diagC=$D" && W "$D"
echo "C-PIPELINE-DONE"
