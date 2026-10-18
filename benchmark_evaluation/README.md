# Benchmark evaluation

The scoring library shared by the SPAC-seq (experiment 01) and Perturb-map (experiment 02)
benchmarks. The repository ships one benchmark protocol, the one reported in the paper, and the
files here are its libraries; they are run through the entry points below.

## What to run

| Goal | Entry point |
|---|---|
| Reproduce a reported number, step by step | `reproduction/01_spac_seq_benchmark.py`, `reproduction/02_perturb_map_benchmark.py`, or notebooks 01 and 02 |
| Score a checkpoint on the SPAC-seq benchmark | `experiments/01_spac_seq_benchmark/evaluation/evaluate.sh` |
| Rebuild the SPAC-seq tables, tests and figures | `experiments/01_spac_seq_benchmark/evaluation/aggregate.sh` |
| Score a checkpoint on the Perturb-map benchmark | `experiments/02_perturb_seq_benchmark/evaluation/evaluate.sh` |
| Rebuild the Perturb-map tables, tests and Fig. 4j | `experiments/02_perturb_seq_benchmark/evaluation/aggregate.sh` |

Every input path comes from `release_paths.py`, which reads `CONCERT_DATA_DIR` and
`CONCERT_CHECKPOINT_ROOT` (the `data/` and `checkpoints/` folders).

## What is scored

For a held-out perturbation, the model is given a cell's location and its unperturbed
expression, and asked what that cell would look like carrying the perturbation. The prediction
is compared with the measured cells that actually carry it, through the log fold change over
matched unperturbed cells:

- **log fold change RMSE** - error in the size of the response, per perturbation and slide;
- **log fold change rank correlation** - agreement in which genes move and in which direction.

Both are computed per perturbation and slide, then averaged, with bootstrap resampling for the
reported spread.

## The scoring rule

`evaluate_concert_spacseq.py` scores one checkpoint on the SPAC-seq benchmark: one slide of
the within-niche (`seen`) setting per call, or every cross-slide (`unseen`) unit in one call. It runs through
`shared_gene_rule.py` with `--cf_output poisson_depth --cf_seen_ref observed`, so the prediction is
one Poisson draw at the measured library size and the LFC reference is the measured source spots
in both settings. `experiments/01_spac_seq_benchmark/evaluation/evaluate.sh`
carries the full flag list. The default output flags (predicted mean, LFC against CONCERT's own
basal prediction) are the ones the sensitivity sweep of experiment 06 uses.

Pass the same spatial-field lengthscale band that training used. The band is a plain attribute
of the field and stays outside the saved weights, so the default band renders a different field
from identical weights.

Every compared method is scored under that same draw. A competitor's predicted profile is
converted to counts at each source cell's observed library size, drawn once from a Poisson, and
returned to log-CP10K before the LFC is taken, so a prediction carries the count sparsity of the
measurement it is scored against. The reference stays the measured source cells for every method.

`cross_slide_error_breakdown.py` reads the predicted LFC each method wrote for the cross-slide
units and reports, beside the benchmark metrics, the RMSE of predicting no change, the predicted
response magnitude relative to the observed one, the share of genes whose predicted LFC sits at the
epsilon floor, and how much of a prediction is shared between different knockouts on one slide.

The reported tables and figures are written by
`experiments/01_spac_seq_benchmark/evaluation/aggregate.sh`, which chains the
rescoring, the 50-cell Mean target, the significance tables and the plotting modules listed below.

### Perturb-map

`evaluate_concert_perturbmap.py` is the Perturb-map evaluation harness. The reported within-niche numbers
rescore the stored predicted LFCs of every method against a reference built from the niche's
control cells outside every held-out patch (`perturbmap_reference.py`), so reference and
prediction share no cells. `experiments/02_perturb_seq_benchmark/evaluation/aggregate.sh`
runs `score_perturbmap.py` and then `perturbmap_tables.py`.

## Modules

| File | Role |
|---|---|
| `release_paths.py` | The layout of `data/` and `checkpoints/`; every path below derives from it |
| `evaluate_concert_spacseq.py` | SPAC-seq scorer of one CONCERT checkpoint: one slide of the within-niche setting, or every cross-slide unit |
| `shared_gene_rule.py` | Runs a scorer under the shared gene rule (slide HVGs inside the CONCERT panel, minus the knockout) |
| `evaluate_concert_perturbmap.py` | Perturb-map scorer of one CONCERT checkpoint (the evaluation harness) |
| `eval_spatial_counterfactual.py`, `make_umap.py` | Load a checkpoint and decode counterfactual predictions |
| `benchmark_holdouts.py` | Evaluation panels, knockout holdouts and training manifests (both datasets) |
| `spacseq_cross_slide_panel.py`, `spacseq_cross_slide_scoring.py` | SPAC-seq cross-slide panel, shared source cells and cross-slide scoring |
| `make_source_cells.py` | The 5,000 shared source spots per target slide every cross-slide scorer predicts from |
| `lfc_scoring.py` | Observed log fold changes, scored-gene masks and the agreement metrics, from stored predictions |
| `score_spacseq.py` | Every method's per-record SPAC-seq rank correlation and RMSE |
| `add_cross_slide_error.py` | Gives CONCERT's cross-slide records the same gene-bootstrap error as every other method |
| `reference_strategies.py`, `reference_strategies_seen.py`, `reference_strategies_unseen.py` | The four reference strategies (Mean target, Mean niche, Local kNN, GEX kNN): shared code, within-niche setting, cross-slide setting |
| `score_mean_target.py` | Mean target from 50 knockout cells per unit, the reported rule |
| `score_state_predictions.py` | STATE's delivered predictions scored under the shared rule |
| `significance_tables.py`, `method_summary.py`, `figure_panel_numbers.py` | Paired Wilcoxon tests with Benjamini-Hochberg adjustment and the supplementary workbook; method summary; per-knockout panel numbers |
| `plot_spacseq_benchmark.py`, `plot_spacseq_bars.py`, `plot_spacseq_paired_dots.py` | Per-knockout bar and paired-dot panels |
| `plot_spacseq_rank.py`, `plot_spacseq_condition_summary.py`, `plot_spacseq_rho_vs_rmse.py` | Overall rank, per-condition summary and rho-versus-RMSE figures |
| `cross_slide_error_breakdown.py` | Decomposes each method's cross-slide error against the do-nothing prediction |
| `perturbmap_methods.py` | Registry of every method's stored Perturb-map predictions and the harness-reference rescore |
| `perturbmap_reference.py`, `score_perturbmap.py`, `perturbmap_tables.py`, `plot_perturbmap_panel.py` | Perturb-map: the disjoint reference, the rescore of every method, the tables and tests, and the Fig. 4j panel |
| `spacseq_common.py`, `spacseq_slides.py`, `perturbmap_common.py` | Data loading, slide handling and matching for each assay |
| `method_inputs.py`, `train_manifests.py` | Assemble the inputs and manifests an evaluation run needs |
| `eval_utils.py` | Metric helpers of the counterfactual decoder |
| `scoring_helpers/` | Helper modules the scorers import (`eval_utils.py`, `spac_edist_align.py`) |
