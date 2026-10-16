# Benchmark evaluation

Scoring code shared by the SPAC-seq (experiment 01) and Perturb-seq (experiment 02)
benchmarks. It is kept in one place because both experiments score the same quantity in the
same way; only the data and the checkpoint differ.

## What is scored

For a held-out perturbation, the model is given a cell's location and its unperturbed
expression, and asked what that cell would look like carrying the perturbation. The prediction
is compared with the measured cells that actually carry it, through the log fold change over
matched unperturbed cells:

- **log fold change RMSE** - error in the size of the response, per perturbation and slide;
- **log fold change rank correlation** - agreement in which genes move and in which direction.

Both are computed per perturbation and slide, then averaged, with bootstrap resampling for the
reported spread.

## Entry point

`run_concert_v3_benchmark_eval.py` scores one slide of one checkpoint and writes
`per_pert_summary.json` under its output directory. The experiment run scripts submit it once
per slide. `build_benchmark_v3_summary.py` collects those files into the reported table.

Pass the same spatial-field lengthscale band that training used. It is a plain attribute of
the field rather than a saved weight, so leaving it at the default renders a different field
from identical weights.

## Benchmark v4

The reported SPAC-seq numbers come from the v4 protocol: `run_concert_v4_benchmark_eval.py`
run through `genefix_wrap.py`, with `--cf_output poisson_depth --cf_seen_ref observed`, so the
prediction is one Poisson draw at the measured library size and the LFC reference is the measured
source spots in both settings. `experiments/01_spac_seq_benchmark/evaluation/eval_spac_seq_benchmark_v4.sh`
carries the full flag list. The v3 scripts remain as the record of the earlier protocol.

Every compared method is scored under that same draw. A competitor's predicted profile is
converted to counts at each source cell's observed library size, drawn once from a Poisson, and
returned to log-CP10K before the LFC is taken, so a prediction carries the count sparsity of the
measurement it is scored against. The reference stays the measured source cells for every method.

`diagnose_spatial_methods_unseen.py` reads the predicted LFC each method wrote for the cross-slide
units and reports, beside the benchmark metrics, the RMSE of predicting no change, the predicted
response magnitude relative to the observed one, the share of genes whose predicted LFC sits at the
epsilon floor, and how much of a prediction is shared between different knockouts on one slide.

The reported tables and figures are written by
`experiments/01_spac_seq_benchmark/evaluation/aggregate_spac_seq_benchmark_v4.sh`, which chains the
rescoring, the 50-cell Mean target, the significance tables and the plotting modules listed below.

### Perturb-map, benchmark v4

The reported Perturb-map within-niche numbers rescore the stored predicted LFCs of every method
against a reference built from the niche's control cells outside every held-out patch
(`perturbmap_v4_scoring.py`), so reference and prediction share no cells.
`experiments/02_perturb_seq_benchmark/evaluation/rescore_perturb_seq_v4.sh` runs
`rescore_pm_v4.py` and then `analyze_pm_v4.py`.

## Modules

| File | Role |
|---|---|
| `run_concert_v3_benchmark_eval.py` | Entry point: loads a checkpoint, predicts held-out perturbations, writes scores |
| `benchmark_v3_lib.py` | Holdout definitions, pairing of perturbed and unperturbed cells |
| `benchmark_v3_eval_lib.py` | The metrics themselves and their bootstrap |
| `build_benchmark_v3_summary.py` | Collects per-slide results into the reported table |
| `eval_spatial_counterfactual.py` | Counterfactual prediction machinery the entry point builds on |
| `eval_cf_una_multi_holdout.py`, `eval_cf_una_only.py` | Variants scoring unperturbed-anchored predictions |
| `eval_utils.py` | Metric helpers shared across the above |
| `run_concert_v4_benchmark_eval.py` | Benchmark v4 entry point (clonality-gated checkpoints): 85 KO-target-slide pairs in the unseen setting, Poisson output at measured depth |
| `benchmark_v4_lib.py`, `benchmark_v4_eval_lib.py` | v4 holdout panel, shared source cells and unseen scoring |
| `build_v4_source_cells.py` | The 5,000 shared source spots per target slide every v4 unseen evaluator predicts from |
| `genefix_wrap.py` | Runs an eval script under the shared gene rule (slide HVGs in the CONCERT panel, minus the KO) |
| `agg_clonegate_v4b.py` | Collects v4 per-unit results into paired summary statistics |
| `rescore_spac_lfc_distance_v4.py`, `rescore_spac_lfc_distance_v4_sampled.py` | Every method's per-record LFC rho and RMSE on the v4 record set (the sampled variant reads `multi_1-5_ho_v4_sampled/`) |
| `attach_concert_unseen_gene_boot_v4.py` | Attaches the cross-slide bootstrap spread of CONCERT to the per-record table |
| `rescore_collab_naive_baselines.py`, `rescore_collab_naive_baselines_seen_genefix.py`, `rescore_naive_baselines_unseen_v4_genefix.py` | The four reference strategies (Mean target, Mean niche, Local kNN, GEX kNN) under the shared rule, with `--sampled_output` |
| `rescore_mean_target_capped_v4.py` | Mean target from a fixed number of knockout cells per unit (50 in the reported tables) |
| `rescore_state_our_rule.py` | STATE predictions scored under the shared rule |
| `agg_v4_sampled_unseen.py` | Paired summary of the sampled cross-slide results |
| `build_v4_significance_tables.py` | Paired Wilcoxon tests with Benjamini-Hochberg adjustment, and the supplementary-table workbook |
| `summarize_v4.py`, `panel_numbers_sampled.py` | Method summary and the per-knockout numbers of the figure panels |
| `plot_spac_benchmark_bars.py`, `plot_spac_benchmark_v4.py`, `plot_spac_benchmark_v4_sampled.py`, `plot_spac_concert_vs_competitor_dots.py` | Per-knockout bar and paired-dot panels |
| `plot_spac_overall_rank.py`, `plot_spac_condition_summary.py`, `plot_spac_rho_vs_rmse.py` | Overall rank, per-condition summary and rho-versus-RMSE figures |
| `perturbmap_v4_scoring.py`, `rescore_pm_v4.py`, `analyze_pm_v4.py`, `plot_pmap_benchmark_v4.py` | Perturb-map benchmark v4: the reference rule, the rescore of every method, the tables and tests, and the Fig. 4j panel |
| `spacseq_eval_common.py`, `spacseq_multi_common.py` | SPAC-seq data loading, slide handling, matching |
| `perturbmap_common.py` | The same for the Perturb-seq sections |
| `method_input_lib.py`, `train_manifest_lib.py` | Assemble the inputs and manifests an evaluation run needs |
| `diagnose_spatial_methods_unseen.py` | Decomposes the cross-slide error of every method against the do-nothing prediction |
| `make_umap.py` | Embedding plots used to inspect a checkpoint |
