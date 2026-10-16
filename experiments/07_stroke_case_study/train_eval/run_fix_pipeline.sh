#!/bin/bash
# gpu_test allows 2 queued jobs, so everything is sequenced here, 2 at a time.
# Repository roots (CODEBASE_ROOT, CONCERT_DATA_DIR, CONCERT_CHECKPOINT_ROOT, CONCERT_WORKDIR)
source "${CODEBASE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel)}/config/paths.sh"

cd ${CONCERT_WORKDIR}/stroke3d
W () { while [ "$(squeue -u "$USER" -h -j "$1" 2>/dev/null | wc -l)" -ne 0 ]; do sleep 30; done; }
OK () { [ "$(sacct -X -n -j "$1" --format=State 2>/dev/null | tr -d ' ' | head -1)" = "COMPLETED" ]; }

W "42436542,42436544"
OK 42436542 && echo "TRAIN-OK A(ls_max=0.20)" || echo "TRAIN-FAILED A"
OK 42436544 && echo "TRAIN-OK B(z_gain=0.08)" || echo "TRAIN-FAILED B"

# figures + Gm42418 check first (cheap, 1 slot), alongside eval A
F=$(sbatch --parsable -p gpu_test sub_fix3.sh); echo "fix3=$F"
EA=""; OK 42436542 && EA=$(sbatch --parsable -p gpu_test -t 5:00:00 --export=ALL,CRTAG=_crA sub_eval_cr.sh 3d) && echo "evalA=$EA"
W "$F,$EA"
EB=""; OK 42436544 && EB=$(sbatch --parsable -p gpu_test -t 5:00:00 --export=ALL,CRTAG=_crB sub_eval_cr.sh 3d) && echo "evalB=$EB"
DA=""; [ -n "$EA" ] && OK "$EA" && DA=$(sbatch --parsable -p gpu_test sub_diag_cr.sh 3d _crA) && echo "diagA=$DA"
W "$EB,$DA"
DB=""; [ -n "$EB" ] && OK "$EB" && DB=$(sbatch --parsable -p gpu_test sub_diag_cr.sh 3d _crB) && echo "diagB=$DB"
W "$DB"
echo "FIX-PIPELINE-DONE"
sacct -X -j "$F,$EA,$EB,$DA,$DB" --format=JobID,JobName%14,State,Elapsed 2>/dev/null
