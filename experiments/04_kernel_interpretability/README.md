# Experiment 04 - are the learned spatial parameters interpretable?

The model gives every perturbation two spatial parameters: a **niche gain**, setting how
strongly its predicted response may be reshaped by the local microenvironment, and a
**footprint lengthscale**, setting how far its spatial effect field extends. Neither is told
anything about where the perturbed cells actually sit. This experiment asks whether they
nevertheless track the observed arrangement of those cells.

Inputs are a trained checkpoint from experiment 01 (the within-slide setting) and the measured
sections it was trained on.

## Order

1. `01_extract_kernel_params.py` reads the checkpoint, extracts both parameters per
   perturbation, and computes descriptors of the observed arrangement of each perturbation's
   cells directly from the coordinates: the fraction of section area the cells reach, the
   nearest-neighbour index, the radius of gyration, and the number, mean size and largest
   share of separate foci found by spatial clustering.
2. `02_merge_with_accuracy.py` joins those against the benchmark-scored perturbations and
   computes the correlations. Both raw and partial correlations are reported: the number of
   perturbed cells drives several descriptors mechanically, so it and the perturbation
   embedding norm are removed before the correlations are read, and the descriptors are
   corrected for multiple testing.
3. `03_plot_interpretability.py` writes the figures and a `figure_stats.json` holding
   every number in them, so the captions cannot drift from the panels.
   `plot_niche_gain.py` writes the reduced single-panel version.

## What it finds

Across the perturbations scored in the within-slide benchmark, the niche gain rises with the
fraction of tissue a perturbation reaches and with the nearest-neighbour index, and the
footprint lengthscale falls with both. The focus-structure descriptors do not survive
correction once cell number is accounted for.

The direction is the one identifiability predicts. A response that depends on the
microenvironment can only be estimated for a perturbation observed in more than one
microenvironment: perturbations scattered across the tissue are sampled across many
neighbourhoods and receive a large gain, while perturbations confined to one region receive a
small one and fall back on a spatially uniform response. Nothing in the training objective
imposes this.

## Reading the exemplar figure

Exemplars are chosen inside a matched band of perturbed-cell number, and ranked on the same
residual quantities the statistics use, so the panels are comparable and are selected on the
axis the claim rests on. The nearest-neighbour index is below one for every perturbation -
cells sharing a perturbation are always more clumped than random placement - so it is a
relative ordering across perturbations, not an absolute test of randomness.

## Repetition on Perturb-FISH

The same readout is repeated on a Perturb-FISH tumour section, trained with the within-slide
configuration used for SPAC-seq.

1. `perturbfish_train_and_extract.sh` runs `perturbfish_prepare_data.py` on
   `data/perturbfish/perturbfish_tumor_count.h5ad`, which moves the perturbation assignment and the
   MERFISH centroids into the layout `run_concert_spac.py` reads and rescales the coordinates to one
   unit per cell spacing; it then trains the model from `models/concert_benchmark` and runs
   `01_extract_kernel_params.py` on the checkpoint. The trained checkpoint ships as
   `checkpoints/perturbfish/` and its readout as `data/perturbfish/kernel_param_analysis/`.
2. `perturbfish_compare.sh` runs `perturbfish_partial_correlations.py`, which computes the partial rank
   correlations with the same two controls (log10 cell number and perturbation embedding norm)
   on the per-perturbation table and prints them beside the SPAC-seq values
   (`data/spacseq/kernel_params/`), and then
3. `perturbfish_plot_consistency.py`, which writes the supplementary figure and its stats JSON:
   panel a places the four pre-specified relations of the two datasets side by side with 95%
   Fisher intervals, panels b and c show the Perturb-FISH rank residuals behind two of them.

On the 31 Perturb-FISH perturbations all four relations take the SPAC-seq sign: the niche gain
rises with the fraction of tissue reached (partial rho +0.42) and with the nearest-neighbour index
(+0.31), and the footprint lengthscale falls with both (-0.21 and -0.22). The Benjamini-Hochberg
q-values at this sample size lie between 0.11 and 0.63, so the section serves as a directional
replication of the SPAC-seq result.
