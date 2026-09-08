# Experiment 06 — how much do the reported numbers depend on the choices behind them?

A reviewer asked whether the results survive changes to the hyper-parameters. This experiment
answers it the only way that separates a real effect from initialisation noise: every setting is
retrained at three random seeds, and the whole family is read against a band built from repeated
runs of the published configuration itself.

It is run twice, once per benchmark dataset, and the two datasets do **not** give the same answer.
Read `design_spec.md` for the full design; the short version is below.

## Design

Ten hyper-parameters, each moved to two values on either side of the published setting, one at a
time, everything else held fixed:

| Tag | Knob | Published | Arms |
|---|---|---|---|
| `lsb1` / `lsb3` | footprint lengthscale band | see per-dataset launcher | narrow / wide |
| `ind8` / `ind64` | inducing points | 16 | 8 / 64 |
| `cs10` / `cs40` | coordinate scale | 20 | 10 / 40 |
| `mor0` / `mor2` | Moran smoothness weight | 1 | 0 / 2 |
| `lfc05` / `lfc2` | log-fold-change loss weight | 1 | 0.5 / 2 |
| `dis2` / `dis8` | disentangle radius | 4 | 2 / 8 |
| `nfd4` / `nfd16` | niche code width | 8 | 4 / 16 |
| `nfk10` / `nfk50` | niche neighbourhood size | 25 | 10 / 50 |
| `brf64` / `brf256` | basal random features | 128 | 64 / 256 |
| `brb005` / `brb030` | basal feature bandwidth | 0.15 | 0.05 / 0.30 |

Plus a **seed band**: five runs of the published configuration, differing only in seed. Nothing
can be called an effect unless it leaves that band. Metrics are the two benchmark metrics, LFC
rank correlation and LFC RMSE — no E-distance, no R², no PCC.

The radius knob selects a scale-aware variant of the Moran term; it is *not* a disentanglement
strength, and should not be described as one in reader-facing text. Two further knobs that look
sweepable are **inert** under the published configuration — the random-feature count and bandwidth
of the *perturbation* field apply only when that field is built from random features, and the
published model builds it as a per-perturbation Gaussian process. Sweeping them would produce a
panel of pure noise.

## Order

1. **Train and score the arms.** `launch_sens_pmap.sh` and `launch_sens_spac.sh` expand the table
   above into one job per arm and submit them, spreading across partitions and respecting the
   running-job caps. Each arm is one `sub_sens_{pmap,spac}_arm.sh` job that trains *and* scores in
   a single allocation — a train/eval dependency pair would hold two slots for the same arm. Arms
   are identified by tag; `_s1` / `_s2` suffixes are the seed replicates. Every arm writes an
   `arm.json` manifest beside its checkpoint, because the checkpoints themselves are bare state
   dicts with no record of the configuration that produced them.
2. **Aggregate.** `agg_sens_pmap.py` / `agg_sens_spac.py` walk the result trees and emit one tidy
   row per arm × slide × perturbation. The SPAC-seq sweep straddles two storage roots because it
   was moved mid-flight; the aggregator scans both and the first root holding a *complete* arm
   wins, so a half-written duplicate never shadows a finished one.
3. **Plot.** `plot_sens_pmap.py` / `plot_sens_spac.py` write one figure per metric, plus a
   `figure_stats*.json` holding every number in the panels and a generated caption drawn from that
   same JSON — the caption cannot drift from the figure. The SPAC plotter also flags any point
   that rests on a single seed.

`sub_agg_plot_spac.sh` runs steps 2 and 3 for SPAC-seq as a batch job and copies the two PDFs into
the manuscript figure directory.

## Two helpers that are not part of the sweep

- `relaunch_sens_spac_failed.sh` restarts arms from their checkpoints. Two arms died at epoch ~410
  within nine minutes of each other with no traceback — a node or filesystem event, not a bug in
  the arm.
- `move_sens_spac_to_requeue.sh` moves queued arms onto the preemptible partition. Note that the
  main launcher deliberately avoids that partition: a preempted sixteen-hour arm restarts from
  scratch. Use it only for arms that resume from a checkpoint.
- `sub_sens_spac_ss.sh` is a **timing probe, not an arm** — the published recipe on a single
  section, used to measure how long one SPAC-seq arm would take before committing to the sweep.

## What it finds

**They tell opposite stories, and the reason is the seed band.**

On Perturb-map the band is wide: five seeds of the published configuration span rank correlation
0.443–0.566. Every one of the ten hyper-parameters moves the mean by less than that, so the
robustness claim holds, but its converse is the real finding — at one run per arm, effects smaller
than about 0.12 rank-correlation units are simply not resolvable on this dataset.

On SPAC-seq the same band is about twenty times narrower, 0.399–0.405. Real effects therefore
*are* resolvable, and three leave the band: the niche code width is sharply peaked at the published
value and loses 0.056 at either 4 or 16; doubling the smoothness weight costs 0.043; widening the
niche neighbourhood to 50 costs 0.027. The other seven factors are flat to within 0.005. Do not
carry the Perturb-map sentence about seed noise into the SPAC-seq text.

One consequence worth keeping in view: the Perturb-map run reported in the paper is seed 42, which
is the best of the five drawn. The SPAC-seq seed 42 sits mid-band.

## Two evaluator faults this sweep exposed, both fixed

The Perturb-map scorer had the niche neighbourhood size hard-coded, so any arm trained with a
different value was silently mis-scored; it is now a flag. The basal random-feature count was not
detected from the checkpoint, so off-default arms failed to load; it is now read from the weights.

A third, still open: the Perturb-map scorer sets no random seed and draws posterior and bootstrap
samples, so re-scoring one checkpoint twice moves rank correlation by about 0.003. That is far
below the training seed band and no conclusion depends on it, but differences that small between
two scorings are not real.

## Cost

A Perturb-map arm is about ten minutes of training plus a minute of scoring, so the whole sweep
fits in an hour. A SPAC-seq arm is sixteen hours. Do not estimate the latter from early epochs: at
epoch 95 the timing probe projected under three hours and then ran four times longer.
