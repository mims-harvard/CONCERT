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

`SPAC_SEQ_DIR` (set in `config/paths.sh`) must contain:

- `raw_output_1-1.h5ad` ... `raw_output_1-5.h5ad` - the five measured sections, with expression,
  per-cell perturbation assignment and array coordinates;
- `holdout_v4_spac_seen.json`, `holdout_v4_spac_unseen.json` - which perturbations are held out
  on which slide, written by `data_preparation/build_v4_holdout_coords.py`;
- `prepared/benchmark_v4/` - the evaluation panel, holdout pools and shared source cells the
  evaluation reads, written by `data_preparation/build_benchmark_v4_assets.py` and
  `benchmark_evaluation/build_v4_source_cells.py`.

## Order

1. **Filter and select genes.** `data_preparation/filter_spac.py` applies the cell and guide
   quality rules; `data_preparation/find_hvgs.py` ranks genes so training can take the top
   2000. `data_preparation/build_spatial_graph.py` writes the neighbour graph.
2. **Define the holdouts.** `data_preparation/build_v3_holdout_coords.py` chooses the held-out
   perturbations for both settings and records the affected cells.
   `data_preparation/make_niche_holdout.py` covers the microenvironment-level variant.
3. **Build evaluation assets.** `data_preparation/build_benchmark_v4_assets.py` writes the
   evaluation panel (85 knockout-target-slide pairs in the unseen setting) and holdout pools;
   `benchmark_evaluation/build_v4_source_cells.py` draws the 5,000 shared source spots per
   target slide.
4. **Train.** `sbatch train/train_spac_seq.sh seen` (and `unseen`). One GPU; the reported runs
   trained for up to 1500 epochs with early stopping. Both settings use the clonality gate
   (`--clonality_gate clone_size --clone_min_cells 100`) and the per-slide guide shift
   (`--slide_guide_shift True`).
5. **Score.** `sbatch evaluation/eval_spac_seq_benchmark_v4.sh unseen` scores every unseen
   pair in one job; `sbatch --array=1-5 evaluation/eval_spac_seq_benchmark_v4.sh seen` runs one
   slide per array task. Collect the outputs with `agg_clonegate_v4b.py` in
   `../../benchmark_evaluation/`. The v3 scripts (`build_v3_holdout_coords.py`,
   `build_benchmark_v3_assets.py`, `eval_spac_seq_benchmark.sh`) remain as the record of the
   earlier protocol; `build_v4_holdout_coords.py` asserts it reproduces their holdout files.

## Notes

The lengthscale band of the spatial field is passed at both training and evaluation. It is a
plain attribute rather than a saved weight, so evaluating without it renders a different field
from the same trained parameters.

The trained checkpoint from the `seen` setting is the input to experiment 04.
