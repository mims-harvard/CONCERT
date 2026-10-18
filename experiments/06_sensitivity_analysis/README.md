# Experiment 06 - how much do the reported numbers depend on the choices behind them?

This experiment measures how the benchmark metrics respond to changes in the model's
hyper-parameters. Every setting is retrained at three random seeds, and the whole family is read
against a band built from repeated runs of the base configuration itself, which separates a real
effect from initialisation noise.

It is run twice, once per benchmark dataset, and the two datasets give different answers.

## Design

Ten hyper-parameters, each moved to values on either side of the base setting, one at a time,
everything else held fixed. The SPAC-seq arms (`submit_spacseq_sweep.sh`) are:

| Tag | Knob | Base | Arms |
|---|---|---|---|
| `lsb1` / `lsb3` | footprint lengthscale band | [0.15, 0.90] | [0.05, 0.30] / [0.45, 2.70] |
| `ind8` / `ind64` | inducing points | 16 | 8 / 64 |
| `cs10` / `cs40` | coordinate scale | 20 | 10 / 40 |
| `mor0` / `mor2` | Moran smoothness weight | 1 | 0 / 2 |
| `lfc05` / `lfc2` | log-fold-change loss weight | 1 | 0.5 / 2 |
| `dis2` / `dis8` | disentangle radius | 4 | 2 / 8 |
| `nfd4` / `nfd16` | niche code width | 8 | 4 / 16 |
| `nfk10` / `nfk50` | niche neighbourhood size | 25 | 10 / 50 |
| `brf64` / `brf256` | basal random features | 128 | 64 / 256 |
| `brb005` / `brb030` | basal feature bandwidth | 0.15 | 0.05 / 0.30 |

The Perturb-map arms (`submit_perturbmap_sweep.sh`) move the same ten knobs around the
Perturb-map base (64 inducing points, so its inducing arms are 16, 32 and 128; its coordinate
scale arms are 10, 40 and 70; its Moran arms are 0, 0.5 and 2; three lengthscale bands).

Each sweep also has a **seed band**: five runs of the base configuration (seed 42 and seeds 1 to
4), differing only in seed. An effect counts as real when it leaves that band. Metrics are the two
benchmark metrics, LFC Spearman rank correlation and LFC RMSE.

The base configurations are the Perturb-map within-niche model and, for SPAC-seq, the
within-niche model trained without the clonality gate and the per-slide guide shift (seed 42),
that is, the configuration before those two options were added. Arms are scored with the default
output flags of `benchmark_evaluation/evaluate_concert_spacseq.py` (predicted mean, LFC against
the model's own basal prediction).

The radius knob selects a scale-aware variant of the Moran term. Two further knobs, the
random-feature count and bandwidth of the *perturbation* field, apply only when that field is
built from random features; the base model builds it as a per-perturbation Gaussian process, so
those two knobs stay outside the sweep.

## Order

1. **Train and score the arms.** `submit_perturbmap_sweep.sh` and `submit_spacseq_sweep.sh` expand
   the arm lists into one job per arm and submit them. Each arm is one `run_perturbmap_arm.sh` or
   `run_spacseq_arm.sh` job that trains *and* scores in a single allocation. Arms are identified
   by tag; `_s1` / `_s2` suffixes are the seed replicates. Every arm writes an `arm.json` manifest
   beside its checkpoint and results, recording the configuration that produced it.
2. **Aggregate.** `aggregate_perturbmap.py` / `aggregate_spacseq.py` walk the result trees and
   emit one tidy row per arm x slide x perturbation. The SPAC-seq aggregator scans the two batch
   directories under the sensitivity results root and takes each arm from the first directory
   that holds a complete copy of it.
3. **Plot.** `plot_perturbmap.py` / `plot_spacseq.py` write one figure per metric, plus a
   `figure_stats*.json` holding every number in the panels and a caption generated from that same
   JSON. The SPAC-seq plotter also marks any point that rests on a single seed.

`aggregate_and_plot_spacseq.sh` runs steps 2 and 3 for SPAC-seq as a batch job and writes
`${CONCERT_WORKDIR}/sensitivity/sens_spac_arms.csv` and the figures.

`time_spacseq_single_slide.sh` is a timing probe: the base recipe on a single section, used to
estimate the run time of one SPAC-seq arm.

## What it finds

On Perturb-map the seed band is wide: five seeds of the base configuration span rank correlation
0.443-0.566. Every one of the ten hyper-parameters moves the mean by less than that, so at one run
per arm, effects smaller than about 0.12 rank-correlation units lie within seed variation on this
dataset. The Perturb-map seed 42 run is the best of the five drawn.

On SPAC-seq the same band is about twenty times narrower, 0.399-0.405, so smaller effects are
resolvable. Three leave the band: the niche code width peaks at the base value and loses 0.056 at
either 4 or 16; doubling the smoothness weight costs 0.043; widening the niche neighbourhood to 50
costs 0.027. The other seven factors stay within 0.005. The SPAC-seq seed 42 run sits mid-band.

The Perturb-map scorer draws posterior and bootstrap samples with an unfixed random seed, so
re-scoring one checkpoint moves rank correlation by about 0.003, far below the training seed band.

## Cost

A Perturb-map arm is about ten minutes of training plus a minute of scoring, so the whole sweep
fits in an hour. A SPAC-seq arm is about sixteen hours of training plus the five-slide scoring.
Early-epoch timing underestimates the full run, so plan from the full figure.
