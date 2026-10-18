#!/bin/bash -l
#SBATCH -J concert_spac_seq_aggregate
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time 0-12:00
#SBATCH -o concert_spac_seq_aggregate_%j.out
#SBATCH -e concert_spac_seq_aggregate_%j.err
#
# Collect the SPAC-seq benchmark results of every method into the reported tables and figures.
#
#   sbatch aggregate.sh
#
# Inputs, under ${CONCERT_DATA_DIR}/benchmark_assets/results/spacseq_predictions/<method>/{seen,unseen}:
#   concert             evaluate.sh (poisson_depth output, observed reference)
#   biolord gears scgen celcomen steamboat cellagentchat
#                       the competitors, each scored with --sampled_output True: one Poisson draw at
#                       the library size of each source spot, the same transform CONCERT is scored under
#   reference_strategies/unseen
#                       the naive reference strategies of the unseen setting
#   spac_{seen,unseen}_state_our_rule.csv, shipped beside the reported tables (STATE, score_state_predictions.py)
#
# Mean target is the mean of 50 randomly drawn knockout cells per unit in both settings
# (score_mean_target.py, step 5); the all-cell rows are kept as *.bak_mtall.
#
# Outputs (${CONCERT_WORKDIR}/spacseq_results, or SPACSEQ_RESULTS): per-record CSVs, the significance tables, the
# supplementary-table workbook, figures/ (sorted by organize_figures.sh), the summary,
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
GF="${SPACSEQ_PREDICTIONS}"
SHIPPED="${CONCERT_DATA_DIR}/benchmark_assets/results/spacseq_results"
OUT="${SPACSEQ_RESULTS:-${CONCERT_WORKDIR}/spacseq_results}"
MT="${MT_DIR:-${OUT}/mean_target_n50}"
# Perturb-map combined results, read by the significance tables for the cross-dataset sheet.
PM_COMBINED_DIR="${PM_COMBINED_DIR:-${CONCERT_DATA_DIR}/benchmark_assets/results/harness_tables/perturbmap_combined}"
mkdir -p "${OUT}"
# STATE enters as two shipped tables (score_state_predictions.py writes them from the STATE predictions).
for f in spac_seen_state_our_rule.csv spac_unseen_state_our_rule.csv; do
    [ -s "${OUT}/$f" ] || cp "${SHIPPED}/$f" "${OUT}/$f"
done

echo "[1] completeness"
for m in biolord gears scgen celcomen steamboat cellagentchat concert; do
    n=$(ls -d "$GF"/$m/unseen/unseen_pert/*__slide_* | wc -l); echo "$m unseen $n/85"; [ "$n" -eq 85 ] || exit 3
    for s in 1-1 1-2 1-3 1-4 1-5; do
        test -f "$GF"/$m/seen/patch/slide_$s/lfc_scores.json || { echo "missing $m seen $s"; exit 3; }
    done
done
for f in spac_seen_state_our_rule.csv spac_unseen_state_our_rule.csv; do
    test -s "${OUT}/$f" || { echo "missing ${OUT}/$f (run score_state_predictions.py first)"; exit 3; }
done

echo "[2] per-record LFC rho and RMSE, every method"
cp "$GF/reference_strategies/unseen/spac_unseen_naive_baselines_our_rule.csv" "${OUT}/"
python -u "${BE}/score_spacseq.py" --regime all --outdir "${OUT}"

echo "[3] CONCERT unseen gene bootstrap"
python -u "${BE}/add_cross_slide_error.py" --outdir "${OUT}" --concert-dir "$GF/concert"

echo "[4] seen reference strategies, sampled"
python -u "${BE}/reference_strategies_seen.py" --outdir "${OUT}" --sampled_output True

echo "[5] Mean target from 50 knockout cells per unit, spliced into the reference-strategy tables"
python -u "${BE}/score_mean_target.py" --outdir "${MT}" --n_cells 50 --sub_seed 0
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
python -u "${BE}/significance_tables.py" --indir "${OUT}" --outdir "${OUT}" \
    --pm_combined_dir "${PM_COMBINED_DIR}" --xlsx "${OUT}/CONCERT_Supplementary_tables.xlsx"

echo "[7] figures"
for R in seen unseen; do
    python -u "${BE}/plot_spacseq_benchmark.py" bars --regime $R --concert-dir "$GF/concert" --outdir "${OUT}"
    python -u "${BE}/plot_spacseq_benchmark.py" dots --regime $R --concert-dir "$GF/concert" --outdir "${OUT}"
done
O85="${OUT}/unseen_overall_all85"; mkdir -p "${O85}"
for f in spac_unseen_lfc_dist_per_record.csv spac_unseen_lfc_dist_per_pert.csv \
         spac_unseen_naive_baselines_our_rule.csv spac_unseen_state_our_rule.csv; do
    ln -sf "${OUT}/$f" "${O85}/$f"
done
python -u "${BE}/plot_spacseq_benchmark.py" bars --regime unseen --top-n 85 \
    --concert-dir "$GF/concert" --outdir "${O85}"
python -u "${BE}/plot_spacseq_rank.py" --indir "${OUT}"
python -u "${BE}/plot_spacseq_condition_summary.py" --indir "${OUT}"
python -u "${BE}/plot_spacseq_rho_vs_rmse.py" --indir "${OUT}"
bash "${CODEBASE_ROOT}/experiments/01_spac_seq_benchmark/evaluation/organize_figures.sh" "${OUT}"

echo "[8] summary and per-knockout panel numbers"
python -u "${BE}/method_summary.py" "${OUT}"
python -u "${BE}/figure_panel_numbers.py" "${OUT}"
echo "[$(date)] aggregation finished -> ${OUT}"
