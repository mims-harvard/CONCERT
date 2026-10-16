#!/bin/bash -l
#SBATCH -J concert_spac_seq_aggregate_v4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time 0-12:00
#SBATCH -o concert_spac_seq_aggregate_v4_%j.out
#SBATCH -e concert_spac_seq_aggregate_v4_%j.err
#
# Collect the SPAC-seq benchmark v4 results of every method into the reported tables and figures.
#
#   sbatch aggregate_spac_seq_benchmark_v4.sh
#
# Inputs, under ${SCORING_LIB}/multi_1-5_ho_v4_sampled/<method>/{seen,unseen}:
#   concert_clonegate   eval_spac_seq_benchmark_v4.sh (poisson_depth output, observed reference)
#   biolord gears scgen celcomen steamboat cellagentchat
#                       the competitors, each scored with --sampled_output True: one Poisson draw at
#                       the library size of each source spot, the same transform CONCERT is scored under
#   reference_strategies/unseen
#                       the naive reference strategies of the unseen setting
#   spac_{seen,unseen}_state_our_rule.csv in the output directory (STATE, rescore_state_our_rule.py)
#
# Mean target is the mean of 50 randomly drawn knockout cells per unit in both settings
# (rescore_mean_target_capped_v4.py, step 5); the all-cell rows are kept as *.bak_mtall.
#
# Outputs (${SCORING_LIB}/v4_aggregate_sampled): per-record CSVs, the significance tables, the
# supplementary-table workbook, figures/ (sorted by organize_v4_sampled_figures.sh), the summary,
# and the per-knockout panel numbers.

# Locate config/paths.sh. Under sbatch the batch script is copied to a spool directory, so a
# path relative to this file does not resolve there; CODEBASE_ROOT is the fallback.
_PATHS="$(dirname "${BASH_SOURCE[0]}")/../../../config/paths.sh"
[ -f "$_PATHS" ] || _PATHS="${CODEBASE_ROOT:-}/config/paths.sh"
[ -f "$_PATHS" ] || {
    echo "[FATAL] cannot find config/paths.sh."
    echo "        Submitting with sbatch? Pass the codebase root explicitly:"
    echo "          sbatch --export=ALL,CODEBASE_ROOT=/path/to/codebase $0 ..."
    exit 1
}
source "$_PATHS"
activate_env
export PYTHONHASHSEED=0

BE="${CODEBASE_ROOT}/benchmark_evaluation"
GF=multi_1-5_ho_v4_sampled
OUT="${SCORING_LIB}/v4_aggregate_sampled"
MT="${SCORING_LIB}/mean_target_audit/mt50"
# Perturb-map combined results, read by the significance tables for the cross-dataset sheet.
PM_COMBINED_DIR="${PM_COMBINED_DIR:-/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/spac_seq_lsfix}"
cd "${SCORING_LIB}"
mkdir -p "${OUT}"

echo "[1] completeness"
for m in biolord gears scgen celcomen steamboat cellagentchat concert_clonegate; do
    n=$(ls -d $GF/$m/unseen/unseen_pert/*__slide_* | wc -l); echo "$m unseen $n/85"; [ "$n" -eq 85 ] || exit 3
    for s in 1-1 1-2 1-3 1-4 1-5; do
        test -f $GF/$m/seen/patch/slide_$s/lfc_scores.json || { echo "missing $m seen $s"; exit 3; }
    done
done
for f in spac_seen_state_our_rule.csv spac_unseen_state_our_rule.csv; do
    test -s "${OUT}/$f" || { echo "missing ${OUT}/$f (run rescore_state_our_rule.py first)"; exit 3; }
done

echo "[2] per-record LFC rho and RMSE, every method"
cp $GF/reference_strategies/unseen/spac_unseen_naive_baselines_our_rule.csv "${OUT}/"
python -u "${BE}/rescore_spac_lfc_distance_v4_sampled.py" --regime all --outdir "${OUT}"

echo "[3] CONCERT unseen gene bootstrap"
python -u "${BE}/attach_concert_unseen_gene_boot_v4.py" --outdir "${OUT}" --concert-v3 $GF/concert_clonegate

echo "[4] seen reference strategies, sampled"
python -u "${BE}/rescore_collab_naive_baselines_seen_genefix.py" --outdir "${OUT}" --sampled_output True

echo "[5] Mean target from 50 knockout cells per unit, spliced into the reference-strategy tables"
python -u "${BE}/rescore_mean_target_capped_v4.py" --outdir "${MT}" --n_cells 50 --sub_seed 0
python -u - <<PY
import shutil
from pathlib import Path
import pandas as pd
out, mt = Path("${OUT}"), Path("${MT}")
for arm in ("seen", "unseen"):
    f = out / f"spac_{arm}_naive_baselines_our_rule.csv"
    bak = f.with_name(f.name + ".bak_mtall")
    shutil.copy2(f, bak)
    old = pd.read_csv(bak, dtype={"slide": str, "niche_key": str})
    new = pd.read_csv(mt / f"spac_{arm}_mean_target_n50.csv", dtype={"slide": str, "niche_key": str})
    df = pd.concat([old[old.method != "mean_target"], new[old.columns]], ignore_index=True)
    df.to_csv(f, index=False)
    print(arm, df.groupby("method_label")[["lfc_spearman", "lfc_dist"]].mean().round(4).to_string())
PY

echo "[6] significance tables and the supplementary workbook"
python -u "${BE}/build_v4_significance_tables.py" --indir "${OUT}" --outdir "${OUT}" \
    --pm_combined_dir "${PM_COMBINED_DIR}" --xlsx "${OUT}/CONCERT_Supplementary_tables_v4_sampled.xlsx"

echo "[7] figures"
for R in seen unseen; do
    python -u "${BE}/plot_spac_benchmark_v4_sampled.py" bars --regime $R --concert-v3 $GF/concert_clonegate --outdir "${OUT}"
    python -u "${BE}/plot_spac_benchmark_v4_sampled.py" dots --regime $R --concert-v3 $GF/concert_clonegate --outdir "${OUT}"
done
O85="${OUT}/unseen_overall_all85"; mkdir -p "${O85}"
for f in spac_unseen_lfc_dist_per_record.csv spac_unseen_lfc_dist_per_pert.csv \
         spac_unseen_naive_baselines_our_rule.csv spac_unseen_state_our_rule.csv; do
    ln -sf "${OUT}/$f" "${O85}/$f"
done
python -u "${BE}/plot_spac_benchmark_v4_sampled.py" bars --regime unseen --top-n 85 \
    --concert-v3 $GF/concert_clonegate --outdir "${O85}"
python -u "${BE}/plot_spac_overall_rank.py" --indir "${OUT}"
python -u "${BE}/plot_spac_condition_summary.py" --indir "${OUT}"
python -u "${BE}/plot_spac_rho_vs_rmse.py" --indir "${OUT}"
bash "${CODEBASE_ROOT}/experiments/01_spac_seq_benchmark/evaluation/organize_v4_sampled_figures.sh" "${OUT}"

echo "[8] summary and per-knockout panel numbers"
python -u "${BE}/summarize_v4.py" "${OUT}" "${PM_COMBINED_DIR}"
python -u "${BE}/panel_numbers_sampled.py"
echo "[$(date)] aggregation finished -> ${OUT}"
