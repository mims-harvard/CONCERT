#!/bin/bash -l
#SBATCH -J spac_sens
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=200G
#SBATCH --time 2-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o /n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_sens/logs/spac_sens_%x_%j.out
#SBATCH -e /n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_sens/logs/spac_sens_%x_%j.err
#
# SPAC-seq sensitivity arm: 5-slide train + the v3 benchmark eval over all 5 slides, one job.
# Base = the REPORTED spac config (sub_train_spac_niche_lsfix.sh -> concert/seen_lsfix), so an arm
# is "base with exactly one factor moved". ~16 h per arm.
# Design: docs/superpowers/specs/2026-08-17-concert-sensitivity-analysis-design.md
#
# Usage: TAG=seed1 SEED=1 sbatch -p kempner_h100 sub_sens_spac_arm.sh
# Knobs (env): SEED INDUCE LSINIT LSMIN LSMAX CSCALE MORAN LFCL DISR NFDIM NFK BRFFF BRFFB
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# kempner_requeue nodes are shared and often hand out a GPU that is already busy; that surfaces as
# "CUDA-capable device(s) is/are busy or unavailable" ~1 min in and burns the whole arm. Probe the
# device first and put the job back in the queue instead of failing, so it retries on another node.
if ! python -c "import torch; torch.zeros(8, device='cuda'); print('CUDA OK')" 2>/dev/null; then
  echo "[$(date)] CUDA unavailable on $(hostname) - requeueing job ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}" || exit 1
  sleep 60
  exit 0
fi

# The concert_env torch build has no sm_120 cubin, so on the RTX PRO 6000 Blackwell nodes
# (holygpu7c*) every JIT-compiled kernel dies ~1 min in with
#   nvrtc: error: invalid value for --gpu-architecture (-arch)
# and the whole arm is lost. Submit with --exclude of those nodes (see relaunch_sens_spac_failed.sh);
# this probe is the backstop if one slips through.
# torch 2.3.0 is built against CUDA 11.8 here, whose ceiling is sm_90 (H100) -- anything newer
# has no cubin and no forward-compatible PTX. get_arch_list() can come back empty, so treat an
# empty list as "unknown" and fall back to the major-version ceiling rather than requeueing blindly.
if ! python -c "
import sys, torch
major, minor = torch.cuda.get_device_capability()
archs = torch.cuda.get_arch_list()
ok = ('sm_%d%d' % (major, minor)) in archs if archs else major <= 9
sys.exit(0 if ok else 3)" 2>/dev/null; then
  echo "[$(date)] GPU arch unsupported by this torch build on $(hostname) - requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}" || exit 1
  sleep 60
  exit 0
fi

TAG="${TAG:?need TAG}"
REGIME="${REGIME:-seen}"
SEED="${SEED:-42}"
INDUCE="${INDUCE:-16}"       # reported spac base = 16 (pmap base was 64)
LSINIT="${LSINIT:-0.25}"
LSMIN="${LSMIN:-0.15}"
LSMAX="${LSMAX:-0.90}"
CSCALE="${CSCALE:-20.0}"
MORAN="${MORAN:-1.0}"
LFCL="${LFCL:-1.0}"
DISR="${DISR:-4.0}"
NFDIM="${NFDIM:-8}"
NFK="${NFK:-25}"
BRFFF="${BRFFF:-128}"
BRFFB="${BRFFB:-0.15}"

NS=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
BASE=/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets
SP=${BASE}/newdata/spac_seq
HOCO=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/v3_holdout_coords
EVAL=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval

FILES="${SP}/raw_output_1-1.h5ad,${SP}/raw_output_1-2.h5ad,${SP}/raw_output_1-3.h5ad,${SP}/raw_output_1-4.h5ad,${SP}/raw_output_1-5.h5ad"
HOLD="${HOCO}/holdout_v3_spac_${REGIME}.json"
[ -f "$HOLD" ] || { echo "[FATAL] holdout file missing: $HOLD"; exit 1; }

SWEEP=/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_sens
OUT="${SWEEP}/outputs/ms_spac1_sens_${REGIME}_${TAG}"
RES="${SWEEP}/results/${REGIME}_${TAG}"
mkdir -p "$OUT" "$RES" "${SWEEP}/logs"

cat > "${OUT}/arm.json" <<EOJ
{"tag":"${TAG}","dataset":"spac1","regime":"${REGIME}","seed":${SEED},
 "fpert_svgp_inducing":${INDUCE},"fpert_svgp_ls_init":${LSINIT},
 "fpert_ls_min":${LSMIN},"fpert_ls_max":${LSMAX},"fpert_coord_scale":${CSCALE},
 "moran_lambda":${MORAN},"lfc_lambda":${LFCL},"disentangle_radius":${DISR},
 "niche_film_dim":${NFDIM},"niche_film_k":${NFK},
 "basal_rff_features":${BRFFF},"basal_rff_bandwidth":${BRFFB},"job":"${SLURM_JOB_ID}"}
EOJ

cd ${NS}/CONCERT-spac-niche
GUARD="--ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 --fpert_svgp_inducing ${INDUCE} --fpert_svgp_ls_init ${LSINIT} --fpert_ls_min ${LSMIN} --fpert_ls_max ${LSMAX}"
NICHE="--use_niche_interaction False --use_niche_film True --niche_film_dim ${NFDIM} --niche_film_k ${NFK} --niche_film_gcap 3.0 --niche_film_l2 1e-3"
BASALRFF="--basal_rff_features ${BRFFF} --basal_rff_bandwidth ${BRFFB}"

echo "[$(date)] SPAC SENS TRAIN tag=${TAG} seed=${SEED} induce=${INDUCE} band=[${LSMIN},${LSMAX}] init=${LSINIT} cscale=${CSCALE} moran=${MORAN} lfc=${LFCL} disr=${DISR} nfdim=${NFDIM} nfk=${NFK} brff=${BRFFF}/${BRFFB} -> ${OUT}"
python -u run_concert_spac.py \
    --config config.yaml --sample "ms_spac1" --stage train \
    --multi_sample_files "${FILES}" --outdir "$OUT" --model_file "model.pt" --project_index spac \
    --holdout_v3_file "${HOLD}" \
    --hvg_max_rank 2000 --seed ${SEED} \
    --inducing_point_steps 6 --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True \
    --maxiter 1500 --patience 50 --earlystop_min_delta 0.05 --lr 1e-4 --batch_size 2048 --train_size 0.95 \
    --use_action_correction True --use_pad False --min_cells_per_pert 25 \
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
    --dynamicVAE False --beta_warmup_epochs 20 --beta_target 1.0 --free_bits 0.5 --gp_free_bits 1.0 \
    --use_nngp False --use_paracrine False ${NICHE} \
    --use_pert_gate True --gate_hidden 64 \
    --use_f_pert True --f_pert_init_std 0.01 --f_pert_clamp 5.0 \
    --spatial_basal_only True --lfc_lambda ${LFCL} \
    --basal_field rff --basal_bayes True --moran_lambda ${MORAN} --disentangle_radius ${DISR} ${BASALRFF} \
    --fpert_field gp_pp --fpert_coord_scale ${CSCALE} --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
    ${GUARD} \
    --resume ${RESUME:-False} \
    --verbosity 1
echo "[$(date)] SPAC SENS TRAIN ${TAG} done -> ${OUT}"

# Eval: same protocol as the reported numbers (concert/seen_lsfix). Arch flags MUST match the
# checkpoint -- inducing and basal RFF width auto-detect, but the ls band, coord scale and
# niche_film_k are plain floats absent from the state_dict and are passed explicitly.
cd "$EVAL"
EGUARD="--ls_prior_min 3 --ls_prior_max 18 --fpert_svgp_inducing ${INDUCE} --fpert_svgp_ls_init ${LSINIT} --fpert_ls_min ${LSMIN} --fpert_ls_max ${LSMAX} --fpert_coord_scale ${CSCALE} --niche_film_k ${NFK}"
if [ "$REGIME" = "seen" ]; then
  for SID in 1-1 1-2 1-3 1-4 1-5; do
    echo "[$(date)] SPAC SENS EVAL ${TAG} slide ${SID}"
    python -u run_concert_v3_benchmark_eval.py --regime seen --target_slide "${SID}" \
      --prepared_dir "${EVAL}/multi_1-5/prepared" --model_file "${OUT}/model.pt" \
      --outdir "$RES" --n_boot 10 --min_g_cells 20 --max_src_cells_per_niche 500 --force ${EGUARD}
  done
else
  python -u run_concert_v3_benchmark_eval.py --regime unseen \
    --prepared_dir "${EVAL}/multi_1-5/prepared" --model_file "${OUT}/model.pt" \
    --outdir "$RES" --n_boot 10 --min_g_cells 20 --max_src_cells_per_niche 500 --force ${EGUARD}
fi
cp "${OUT}/arm.json" "${RES}/arm.json"
echo "[$(date)] SPAC SENS ARM ${TAG} DONE -> ${RES}"
