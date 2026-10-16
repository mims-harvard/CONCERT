# Experiment 01 - SPAC-seq held-out perturbation benchmark

Five tissue sections carrying a pooled perturbation panel. A set of perturbations is held out;
the model is given each held-out cell's location and unperturbed expression and asked to
predict its perturbed expression, which is then compared with what was measured.

Two holdout settings are reported:

- **seen** - a perturbation is removed from one slide but remains visible on the others, so
  the question is whether its effect transfers to a new tissue context;
- **unseen** - a perturbation is removed from every slide, so the question is whether its
  effect can be inferred from related perturbations.

## Inputs

Everything sits in the downloadable `data/` folder (`CONCERT_DATA_DIR`, see `data/README.md`):

- `spacseq/raw/raw_output_1-1.h5ad` ... `raw_output_1-5.h5ad` - the five measured sections, with
  expression, per-cell perturbation assignment and array coordinates;
- `spacseq/prepared/` - the matched cell sets and raw counts the scoring reads;
- `benchmark_assets/holdout_coords/holdout_spac_{seen,unseen}.json` - which cells are held out on
  which slide, written by `data_preparation/05_build_holdout_coords.py --dataset spac`;
- `benchmark_assets/spacseq_benchmark/` (also reachable as `spacseq/prepared/benchmark/`) - the
  evaluation panel, holdout pools and shared source cells, written by
  `data_preparation/04_build_benchmark_panel.py` and `benchmark_evaluation/make_source_cells.py`;
- `benchmark_assets/results/spacseq_predictions/` - every method's per-unit results.

The scoring runs default to the released checkpoints in `checkpoints/`; `MODEL=<model.pt>`
scores your own training run.

## Order

1. **Filter and select genes.** `data_preparation/01_filter_cells.py` applies the cell and guide
   quality rules; `data_preparation/02_rank_genes.py` ranks genes so training can take the top
   2000. `data_preparation/03_build_spatial_graph.py` writes the neighbour graph.
2. **Define the holdouts.** `data_preparation/05_build_holdout_coords.py --dataset spac` turns the
   holdout pools of `prepared/benchmark/` into the coordinate-keyed files training reads, for
   both settings. Run it after step 3.
   `data_preparation/build_niche_holdout.py` covers the microenvironment-level variant.
3. **Build evaluation assets.** `data_preparation/04_build_benchmark_panel.py` writes the
   evaluation panel (85 knockout-target-slide pairs in the unseen setting) and holdout pools;
   `benchmark_evaluation/make_source_cells.py` draws the 5,000 shared source spots per
   target slide.
4. **Train.** `sbatch train/train.sh seen` (and `unseen`). One GPU; the reported runs
   trained for up to 1500 epochs with early stopping. Both settings use the clonality gate
   (`--clonality_gate clone_size --clone_min_cells 100`) and the per-slide guide shift
   (`--slide_guide_shift True`).
5. **Score.** `sbatch evaluation/evaluate.sh unseen` scores every unseen
   pair in one job; `sbatch --array=1-5 evaluation/evaluate.sh seen` runs one
   slide per array task. Both use the reported scoring rule: one Poisson draw at each source
   spot's measured library size (`--cf_output poisson_depth`), the LFC taken against the
   measured source spots (`--cf_seen_ref observed`), and the shared gene rule of
   `shared_gene_rule.py`. The competitors are scored under the same draw (`--sampled_output True`).
6. **Aggregate.** `sbatch evaluation/aggregate.sh` collects every method's
   per-record results from `data/benchmark_assets/results/spacseq_predictions/`, rebuilds Mean target from
   50 randomly drawn knockout cells per unit (`score_mean_target.py`), and writes the
   reported tables, the paired significance tests, the supplementary-table workbook, every
   benchmark figure (sorted by `evaluation/organize_figures.sh`) and the per-knockout
   panel numbers to `${CONCERT_WORKDIR}/spacseq_results/` (the shipped copy is
   `data/benchmark_assets/results/spacseq_results/`). STATE enters through
   `spac_{seen,unseen}_state_our_rule.csv` (`score_state_predictions.py`).
7. **Mean target audit.** `mean_target_audit/` measures how Mean target depends on the number of
   knockout cells it averages (`sbatch mean_target_audit/run_audit.sh seen`, then
   `unseen`, then `summarize_audit.py`); it is the basis for the 50-cell rule.

## Notes

The lengthscale band of the spatial field is passed at both training and evaluation. It is a
plain attribute rather than a saved weight, so evaluating without it renders a different field
from the same trained parameters.

The trained checkpoint from the `seen` setting is the input to experiment 04.
