# Experiment 04 - are the learned spatial parameters interpretable?

The model gives every perturbation two spatial parameters: a **niche gain**, setting how
strongly its predicted response may be reshaped by the local microenvironment, and a
**footprint lengthscale**, setting how far its spatial effect field extends. Neither is told
anything about where the perturbed cells actually sit. This experiment asks whether they
nevertheless track the observed arrangement of those cells.

Inputs are a trained checkpoint from experiment 01 (the within-slide setting) and the measured
sections it was trained on.

## Order

1. `analyze_kernel_params_spac.py` reads the checkpoint, extracts both parameters per
   perturbation, and computes descriptors of the observed arrangement of each perturbation's
   cells directly from the coordinates: the fraction of section area the cells reach, the
   nearest-neighbour index, the radius of gyration, and the number, mean size and largest
   share of separate foci found by spatial clustering.
2. `merge_kernel_params_eval.py` joins those against the benchmark-scored perturbations and
   computes the correlations. Both raw and partial correlations are reported: the number of
   perturbed cells drives several descriptors mechanically, so it and the perturbation
   embedding norm are removed before the correlations are read, and the descriptors are
   corrected for multiple testing.
3. `make_interpret_figures.py` writes the figures and a `figure_stats.json` holding
   every number in them, so the captions cannot drift from the panels.
   `make_fig_niche_gain_simple.py` writes the reduced single-panel version.

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
