#!/bin/bash -l
#SBATCH -J dsscount
#SBATCH -p kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=32G
#SBATCH --time 0-00:15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/dsscount_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/dsscount_%j.err
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
python - <<'PY'
import numpy as np
D="/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
import glob, re, pathlib
src=pathlib.Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study/plot_q3_gt_vs_pred_slides.py").read_text()
m=re.search(r'DUMP = \((.*?)\)', src, re.S)
dump="".join(re.findall(r'"([^"]+)"', m.group(1)))
print("dump:", dump)
d=np.load(dump, allow_pickle=True)
mouse,sid,day = d["mouse"], d["sample_id"], d["src_day"]
mice=sorted(set(mouse.tolist()))
FEATURED=["082421_D0_m6","092421_D3_m1","100221_D9_m5","082421_D21_m1"]
print(f"total mice: {len(mice)}   featured in main/Ext: {len(FEATURED)}")
tot=0
for M in mice:
    k=mouse==M; ns=len(set(sid[k].tolist())); tot+=ns
    tag="  <-- featured" if M in FEATURED else ""
    print(f"  {M:>14s} slides={ns:2d} day={float(np.unique(day[k])[0]):5.1f} cells={int(k.sum()):7d}{tag}")
print(f"total mouse x slide combinations: {tot}")
PY
