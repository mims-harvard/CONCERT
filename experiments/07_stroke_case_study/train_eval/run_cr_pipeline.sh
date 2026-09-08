#!/bin/bash
# gpu_test allows only 2 queued jobs per user, so the chain is driven here instead of with
# --dependency: wait for the pair of jobs in flight, submit the next pair, repeat.
cd /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d
wait_for () { while [ "$(squeue -u xianglin226 -h -j "$1" 2>/dev/null | wc -l)" -ne 0 ]; do sleep 30; done; }
ok () { [ "$(sacct -X -n -j "$1" --format=State 2>/dev/null | tr -d ' ' | head -1)" = "COMPLETED" ]; }

wait_for "42417267,42417273"
for a in 3d 2d; do
  j=42417267; [ "$a" = "2d" ] && j=42417273
  ok $j || { echo "TRAIN-FAILED arm=$a job=$j state=$(sacct -X -n -j $j --format=State|head -1)"; continue; }
  echo "TRAIN-OK arm=$a"
done
ok 42417267 && E3=$(sbatch --parsable -p gpu_test -t 5:00:00 sub_eval_cr.sh 3d) && echo "eval3d=$E3"
ok 42417273 && E2=$(sbatch --parsable -p gpu_test -t 5:00:00 sub_eval_cr.sh 2d) && echo "eval2d=$E2"
wait_for "$E3,$E2"
for p in "3d:$E3" "2d:$E2"; do
  a="${p%%:*}"; j="${p##*:}"; [ -z "$j" ] && continue
  ok "$j" || { echo "EVAL-FAILED arm=$a job=$j"; continue; }
  echo "EVAL-OK arm=$a"
done
ok "$E3" && D3=$(sbatch --parsable -p gpu_test sub_diag_cr.sh 3d) && echo "diag3d=$D3"
ok "$E2" && D2=$(sbatch --parsable -p gpu_test sub_diag_cr.sh 2d) && echo "diag2d=$D2"
wait_for "$D3,$D2"
echo "PIPELINE-DONE diag3d=$D3 diag2d=$D2"
sacct -X -j "$D3,$D2" --format=JobID,JobName%14,State,Elapsed 2>/dev/null
