# Experiment 03 - DSS colitis time course

A MERFISH time course of mouse colon through DSS-induced colitis and its resolution: healthy,
day 3, day 9 and day 21, across 52 sections from several animals. One temporal model is fitted
over the whole course, then asked about a timepoint that was removed from training.

## Inputs

`DSS_DIR` (set in `config/paths.sh`) must contain:

- one `.h5ad` per section, with expression, coordinates and the day of collection;
- `manifest.csv`, listing for each section its filename, animal identifier and timepoint. The
  run scripts read animal and timepoint from this file to build training and holdout lists, so
  its column order matters: filename first, animal third, timepoint fourth;
- a sibling directory `<DSS_DIR>_arch` holding the per-cell architecture descriptors, written
  by `data_preparation/preprocess_spac_timeseries.py`.

## Order

1. **Prepare the sections.** `data_preparation/preprocess_spac_timeseries.py` normalizes each
   section, writes the day column the model conditions on, and computes the architecture
   descriptors. `data_preparation/build_reactome_inflammation_sets.py` writes the pathway gene
   sets used by the figures.
2. **Train.** `sbatch train/train_dss.sh <tag>`. Setting `HOLDOUT_MICE="m1,m2"` excludes every
   section of those animals, giving a leave-one-animal-out checkpoint. To hold out a timepoint
   instead, train on a manifest subset that omits it.
3. **Evaluate.** `sbatch evaluation/eval_dss_timepoint.sh DSS9 <tag>` predicts the held-out
   timepoint and scores it against the measured sections for that day, alongside a linear
   interpolation between the flanking timepoints. `evaluation/composition_eval_dss.py` scores
   cell-type composition and spatial structure; `evaluation/loto_eval_dss.py` holds the
   held-out-timepoint machinery it builds on.
4. **Calibrate.** `evaluation/compute_calibration.py` and `evaluation/calibrate_heldout.py`
   fit the scale factor that puts predictions on the observed scale.
   `evaluation/compute_observed_per_mouse.py` and `evaluation/compute_pathway_trajectories.py`
   write the per-animal and per-pathway trajectories the figures read.
5. **Plot.** The scripts in `figures/` each write one figure, reading the dumps produced
   above. `figures/paper_style.py` holds the shared style, the day colours and the animal and
   day naming, so every panel labels the same animal the same way.

## Reading the figures

Wherever a measured value exists it is shown next to the prediction rather than the prediction
alone, using a split mark: a whole mark means prediction only, a split mark means the measured
value is shown beside it. Predictions are put on the observed scale by a scale factor fitted
per feature, so the shape of the comparison across conditions stays a real test.
