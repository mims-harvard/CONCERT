# Experiment 02 - spatial Perturb-seq held-out perturbation benchmark

The same held-out perturbation question as experiment 01, on four sections of a spatial
Perturb-seq assay. Running it on a second assay tests whether the result depends on the
measurement platform.

## Differences from experiment 01

The model is the same (`models/concert_benchmark`); the settings differ where the assays do:

- there is no non-targeting control class, so unperturbed cells are the comparison group and
  the control label is left unset;
- guide assignment is unambiguous here, so the technical correction that separates intended
  from assigned perturbation is switched off;
- spots are denser and sections smaller, so the spatial field uses more inducing points (64
  against 16) and a wider lengthscale band, read at a larger coordinate scale;
- no minimum cell count per perturbation is imposed, since the panel is already balanced.

## Inputs

`PERTURB_SEQ_DIR` (set in `config/paths.sh`) must contain:

- `perturbmap_s0_spacfmt.h5ad` ... `perturbmap_s3_spacfmt.h5ad` - the four sections converted to
  the common format by `data_preparation/prep_perturbmap_slide.py`;
- `holdout_perturb_seq_seen.json`, `holdout_perturb_seq_unseen.json` - holdout definitions,
  written by `build_v3_holdout_coords.py` in experiment 01's data preparation, which handles
  both assays.

## Order

1. **Convert the sections.** `data_preparation/prep_perturbmap_slide.py` writes each section
   in the format the model reads, with expression, perturbation assignment and coordinates in
   the expected fields.
2. **Define the holdouts** using experiment 01's `build_v3_holdout_coords.py`.
3. **Train.** `sbatch train/train_perturb_seq.sh seen` (and `unseen`).
4. **Score.** `sbatch evaluation/eval_perturb_seq.sh seen` runs
   `../../benchmark_evaluation/run_concert_pm_niche.py` on the checkpoint and writes a
   per-perturbation summary per section. The lengthscale band, coordinate scale and inducing
   count are passed again there and must match the training values - the band defines the
   squashing function inside the spatial field, so a checkpoint read back under a different
   band reconstructs different lengthscales rather than failing.
5. **Compare against the published methods.**
   `../../benchmark_evaluation/rescore_pm_lfc_distance.py` rescores every method under one
   rule and writes a per-record CSV; `figures/plot_pmap_benchmark.py` turns that into the
   panel.

   **Scoring must be restricted to the records all methods cover.** The methods do not all
   produce a prediction for every held-out niche patch, and scoring each on whatever it happens
   to cover is not a like-for-like comparison. `plot_pmap_benchmark.py` takes the intersection
   itself; anything else reading the CSV has to do the same.
