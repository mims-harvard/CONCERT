#!/bin/bash -l
#SBATCH -J pmap_sens
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=200G
#SBATCH --time 0-03:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/pmap_sens_%x_%j.out
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/pmap_sens_%x_%j.err
#
# SI sensitivity analysis, one arm = train + paired eval in a single job.
# Base = the REPORTED pmap config (sub_train_pmap_niche.sh -> concert_ms_seen); every knob below
# defaults to that base, so an arm is "base with exactly one factor moved".
# Design: docs/superpowers/specs/2026-08-17-concert-sensitivity-analysis-design.md
#
# Usage:  TAG=seed1 SEED=1 sbatch -p kempner sub_sens_pmap_arm.sh
# Knobs (env, all optional): SEED INDUCE LSINIT LSMIN LSMAX CSCALE MORAN LFCL DISR REGIME
#   LSMIN/LSMAX empty => flags omitted => model falls back to ls_prior [3,18] (the base behaviour).
#   Whatever is passed at train time is passed identically at eval time: the band defines the
#   sigmoid squash inside FpertField, and inducing sets the Z shape, so a mismatch silently
#   reconstructs the wrong field.
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
module purge; module load gcc
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$; mkdir -p "${NUMBA_CACHE_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TAG="${TAG:?need TAG}"
REGIME="${REGIME:-seen}"
SEED="${SEED:-42}"
INDUCE="${INDUCE:-64}"
LSINIT="${LSINIT:-5.0}"
LSMIN="${LSMIN:-}"
LSMAX="${LSMAX:-}"
CSCALE="${CSCALE:-20.0}"
MORAN="${MORAN:-1.0}"
LFCL="${LFCL:-1.0}"
DISR="${DISR:-4.0}"
NFDIM="${NFDIM:-8}"      # niche-FiLM code width
NFK="${NFK:-25}"         # niche-FiLM neighbourhood size (NOT in the state_dict -> passed to eval too)
BRFFF="${BRFFF:-128}"    # basal RFF feature count
BRFFB="${BRFFB:-0.15}"   # basal RFF bandwidth

OUTBASE=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT
BASE=/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets
HOCO=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/v3_holdout_coords
EVAL=/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval

FILES="${BASE}/perturbmap_s0_spacfmt.h5ad,${BASE}/perturbmap_s1_spacfmt.h5ad,${BASE}/perturbmap_s2_spacfmt.h5ad,${BASE}/perturbmap_s3_spacfmt.h5ad"
IND=4; MINCELLS=0
LABELS="--excluded_label= --unassigned_label unperturbed --control_label __NONE__"; ACTCORR=False
HOLD="${HOCO}/holdout_v3_pmap_${REGIME}.json"
[ -f "$HOLD" ] || { echo "[FATAL] holdout file missing: $HOLD"; exit 1; }

OUT="${OUTBASE}/outputs/ms_pmap_sens_${REGIME}_${TAG}"
RES="/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/results_panel1053ko_niche/concert_sens_${REGIME}_${TAG}"
mkdir -p "$OUT" "$RES" "${OUTBASE}/logs"

# The checkpoint is a bare state_dict, so the arm config lives only in this manifest.
cat > "${OUT}/arm.json" <<EOF
{"tag":"${TAG}","regime":"${REGIME}","seed":${SEED},"fpert_svgp_inducing":${INDUCE},
 "fpert_svgp_ls_init":${LSINIT},"fpert_ls_min":"${LSMIN}","fpert_ls_max":"${LSMAX}",
 "fpert_coord_scale":${CSCALE},"moran_lambda":${MORAN},"lfc_lambda":${LFCL},
 "disentangle_radius":${DISR},"niche_film_dim":${NFDIM},"niche_film_k":${NFK},
 "basal_rff_features":${BRFFF},"basal_rff_bandwidth":${BRFFB},"job":"${SLURM_JOB_ID}"}
EOF

# Band flags are shared by train and eval; empty => omitted => ls_prior [3,18] fallback.
BANDFLAGS=""
if [ -n "$LSMIN" ]; then BANDFLAGS="--fpert_ls_min ${LSMIN} --fpert_ls_max ${LSMAX}"; fi

cd /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/CONCERT-spac-niche
GUARD="--ls_prior_lambda 1.0 --ls_prior_min 3 --ls_prior_max 18 --fpert_svgp_inducing ${INDUCE} --fpert_svgp_ls_init ${LSINIT} ${BANDFLAGS}"
NICHE="--use_niche_interaction False --use_niche_film True --niche_film_dim ${NFDIM} --niche_film_k ${NFK} --niche_film_gcap 3.0 --niche_film_l2 1e-3"
BASALRFF="--basal_rff_features ${BRFFF} --basal_rff_bandwidth ${BRFFB}"

echo "[$(date)] SENS TRAIN tag=${TAG} regime=${REGIME} seed=${SEED} induce=${INDUCE} lsinit=${LSINIT} band=[${LSMIN:-3},${LSMAX:-18}] cscale=${CSCALE} moran=${MORAN} lfc=${LFCL} disr=${DISR} nfdim=${NFDIM} nfk=${NFK} brff=${BRFFF}/${BRFFB} -> ${OUT}"
python -u run_concert_spac.py \
    --config config.yaml --sample "ms_pmap" --stage train \
    --multi_sample_files "${FILES}" --outdir "$OUT" --model_file "model.pt" --project_index spac \
    --holdout_v3_file "${HOLD}" \
    --hvg_max_rank 2000 --seed ${SEED} \
    --inducing_point_steps ${IND} --grid_inducing_points True --loc_range 20.0 --kernel_scale 10.0 --multi_kernel_mode True \
    --maxiter 1500 --patience 300 --lr 1e-4 --batch_size 2048 --train_size 0.95 \
    --use_action_correction ${ACTCORR} --use_pad False ${LABELS} --min_cells_per_pert ${MINCELLS} \
    --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
    --dynamicVAE False --beta_warmup_epochs 20 --beta_target 1.0 --free_bits 0.5 --gp_free_bits 1.0 \
    --use_nngp False --use_paracrine False ${NICHE} \
    --use_pert_gate True --gate_hidden 64 \
    --use_f_pert True --f_pert_init_std 0.01 --f_pert_clamp 5.0 \
    --spatial_basal_only True --lfc_lambda ${LFCL} \
    --basal_field rff --basal_bayes True --moran_lambda ${MORAN} --disentangle_radius ${DISR} ${BASALRFF} \
    --fpert_field gp_pp --fpert_coord_scale ${CSCALE} --fpert_rff_features 128 --fpert_rff_bandwidth 0.15 \
    ${GUARD} \
    --resume False \
    --verbosity 1
echo "[$(date)] SENS TRAIN ${TAG} done -> ${OUT}"

cd "$EVAL"
V3KO=""
if [ "$REGIME" = "unseen" ]; then
  V3KO="--v3_unseen_ko_file ${HOCO}/holdout_v3_pmap_unseen.json --min_g_cells 20"
fi
EGUARD="--ls_prior_min 3 --ls_prior_max 18 --fpert_svgp_inducing ${INDUCE} --fpert_svgp_ls_init ${LSINIT} --fpert_coord_scale ${CSCALE} --niche_film_k ${NFK} ${BANDFLAGS}"
echo "[$(date)] SENS EVAL ${TAG} -> ${RES}"
python -u run_concert_pm_niche.py --model_file "${OUT}/model.pt" --outdir "$RES" \
    --slides ${SLIDES:-0 1 2 3} --n_boot 10 ${EGUARD} ${V3KO}
cp "${OUT}/arm.json" "${RES}/arm.json"
echo "[$(date)] SENS ARM ${TAG} DONE -> ${RES}"
