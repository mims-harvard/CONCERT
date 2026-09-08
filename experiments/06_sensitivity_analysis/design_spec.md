# CONCERT sensitivity analysis for the SI (R1) - design

**Date:** 2026-08-17 (designed 2026-08-16, interrupted by the holylfs06 outage)
**Approved scope:** B (user, 2026-08-16), with a concurrency cap of 12 `kempner_h100` + 12 `kempner`.

## Why

Reviewers asked for a sensitivity analysis. One already exists - SI Note 9
(`/n/holylabs/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/manuscript/SI/SI-notes_R1.tex:205`,
figures at `SI/SI-figures_R1.tex:477,486`) - but it covers the **old architecture** (encoder/decoder
depth, GP dim, Normal dim, kernel scale, target KL, `loc_range`) and says nothing about any component
the revision actually argues for: the `f_pert` spatial footprint field, the RFF basal field, the
Moran penalty, the disentangle radius, the LFC penalty, niche-FiLM.

This analysis is a **companion** to Note 9, not a replacement.

## Vehicle

**Perturb-MAP, 4 slides, `seen` regime, current niche-FiLM architecture.** Not a single slide:
a full 4-slide Perturb-MAP arm trains in **7-11 min** (measured from
`/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/logs/pmap_band_*.out`, Aug 4-5) and evaluates
in **2-3 min**, so the multi-slide setting is already cheap. Single-slide would only save time on
SPAC-seq, where the multi-slide `lsfix` run took 15h52m.

## Base configuration

The arm the paper reports for Perturb-MAP: `concert_ms_seen`, produced by
`/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/CONCERT-spac-niche/sub_train_pmap_niche.sh`
(checkpoint `outputs/ms_pmap_niche_seen/model.pt`, 2026-07-08; it is the CONCERT entry in
`spac_seq_eval/plot_pm_benchmark_bars.py:32`).

| knob | base value |
|---|---|
| `seed` | 42 |
| `fpert_svgp_inducing` | 64 |
| `fpert_svgp_ls_init` | 5.0 |
| `fpert_ls_min` / `fpert_ls_max` | *unset* -> falls back to `ls_prior_min/max` = **3 / 18** (`concert_spac.py:824-825`), in normalized units |
| `fpert_coord_scale` | 20.0 |
| `moran_lambda` | 1.0 |
| `lfc_lambda` | 1.0 |
| `disentangle_radius` | 4.0 |

Note the footprint band on the base arm is `[3, 18]` **normalized** = `[60, 360]` loc units, i.e.
effectively unconstrained relative to the tissue extent. That is what is reported, so that is the
anchor; the lengthscale ladder below tests exactly this.

Checkpoints are bare `state_dict`s with no stored config, so every arm writes an `arm.json`
manifest next to its model.

## The seed band is the yardstick

Every existing arm is one seed (`--seed 42` hard-coded). Without a noise scale, a table of numbers
that wobble slightly proves nothing. So: **5 seeds at the base config** (42, 1, 2, 3, 4), drawn as a
grey band across every panel. "Insensitive to X" then becomes checkable - the X-arm points sit
inside the band.

## Factors (one-at-a-time from base)

| # | Factor | Values (base **bold**) | New arms |
|---|---|---|---|
| 1 | footprint lengthscale band (normalized; ×20 = loc units) | [0.05,0.30] / [0.15,0.90] / [0.45,2.7] / **[3,18]** - ls_max = 6 / 18 / 54 / **360** loc | 3 |
| 2 | `fpert_svgp_inducing` | 16 / 32 / **64** / 128 | 3 |
| 3 | `fpert_coord_scale` | 10 / **20** / 40 / 70 | 3 |
| 4 | `moran_lambda` | 0 / 0.5 / **1** / 2 | 3 |
| 5 | `lfc_lambda` | 0.5 / **1** / 2 | 2 |
| 6 | `disentangle_radius` | 2 / **4** / 8 | 2 |
| 9 | `seed` | **42** / 1 / 2 / 3 / 4 | 4 |
| - | base re-run at seed 42 under current code | - | 1 |

**21 arms**, ~15 min each (train + eval in one job) -> ~5 GPU-h, well under an hour wall-clock at
the 12+12 cap. `fpert_ls_init` moves with the band (proportional: 0.083 / 0.25 / 0.75 / 5.0) so the
init always sits inside the squash band.

The base arm is re-trained rather than reused: the July 8 checkpoint predates ~6 weeks of commits,
and a code-drift confound would be invisible.

## Reused, as a secondary panel - DROPPED in the final analysis

`bandE` / `bandF` / `bandI` (`outputs/ms_pmap_niche_seen_band{E,F,I}`) are a genuine one-factor
`fpert_ls_max` ladder, but at `coord_scale 70`, `inducing 64`, `ls_init 0.04` - a *different* base
than the reported config, and one seed each. Once every factor was re-run at the reported base with
three seeds, this ladder added nothing except a second operating point a reader would have to hold
in mind, so it is **not** in the final figures. The other bands (A, B, D, G) vary 3-4 knobs at once
and were never presentable as one-factor curves.

## Metrics

Whatever the existing scorer already emits per slide × per perturbation with bootstrap SDs
(`lfc_agg.json`): `spearman`, `lfc_dist`, `e_dist`, `sign_agreement`, `topk_overlap`. Headline
panels use LFC Spearman rho and LFC distance, to stay commensurable with the main-text figure;
`e_dist` secondary.

## Deliverables

1. `sens_pmap_arms.csv` - one row per arm × slide × perturbation × metric.
2. Small-multiples figure: one panel per factor, x = factor value, y = metric, grey band = seed
   spread, two rows (rho, LFC distance). House style per the `plotting-figures` skill.
3. An SI note that cross-references Note 9 rather than replacing it.

## SPAC-seq confirmation - recommendation SUPERSEDED (the full sweep was run 2026-08-19..21; see the SPAC-seq results section at the end of this document)

Scope B included a 3-arm single-slide SPAC-seq confirmation, conditional on a timing measurement.
Measured: job 39868589, slide 1-1, base config, early-stopped at **epoch 552 after 9h22m**
(33,325 s). An interim read at epoch 95 suggested ~2.7h; that was wrong - the validation ELBO was
flat over three epochs but kept improving afterwards, so it ran 4x longer.

At 9.3h per arm a minimal band (3 seeds) is an overnight run, and the single-slide eval adapter
still has to be written. More importantly the *scientific* value is weaker than it looks: the
configuration the paper reports for SPAC-seq is the **five-slide** model, so a single-slide sweep
would characterise a model that appears nowhere in the paper. The defensible SPAC-seq version is
the five-slide config at 15h52m per arm - roughly two days for a 3-seed band plus factor arms.

**Recommendation: leave it out.** The SI note already states the analysis was not repeated on
SPAC-seq. The trained single-slide model is kept at
`/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_spac1_sens_ss_seen_base/` in case
the decision changes.

## Outcome (2026-08-17, all arms COMPLETED)

**Final: 77 models, ten factors, two figures** (revised after review - see "Revision" below).
First pass was 53 models: 21 first-wave arms + 32 replicates (every factor arm re-run at seeds 1 and 2 once the
seed band turned out to be as wide as the factor effects - one run per arm could not separate a
real effect from initialisation variance, which is exactly the question a reviewer is asking).

- **Base re-run reproduces the reported arm** within ≤0.02 rho on all 12 slide × perturbation
  cells -> no code drift since 2026-07-08.
- **Five-seed band:** rho 0.443-0.566 (mean 0.507); LFC RMSE 1.93-2.19. The reported
  configuration used seed 42, the **best of the five**.
- **Every hyper-parameter setting lies inside that band on both metrics.** Largest deviation from
  base: 0.051 rho (32 inducing points) and 0.14 RMSE (niche code width, footprint lengthscale
  ceiling); the three penalty weights move rho by ≤0.005.
- `moran_lambda = 0` (penalty fully off) is indistinguishable from the reported weight -> the
  spatial smoothness penalty shapes what the latents represent, not prediction accuracy.

Correction to the pre-outage design: `--disentangle_radius` is **not** a separate HSIC term. It
selects the scale-aware Moran variant (`concert_spac.py:1379-1380`); the HSIC term is
`--disentangle_lambda`, which is **0** in the reported configuration. Logs confirm the radius
changes the objective (`MORAN 0.444` at radius 2 vs `0.189` at radius 4). The panel is valid; only
its label was wrong.

Artifacts (all paths absolute):
- Figures: `/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/sensitivity/fig_sens_pmap_{rho,rmse}.{pdf,png}`
- Numbers of record: `.../Outputs/sensitivity/figure_stats.json`, caption generated from it as
  `.../Outputs/sensitivity/figure_legend_sensitivity.md`
- Per-arm table: `/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/results/sens_pmap_arms.csv`
- Manuscript: `manuscript/SI/SI-figures_R1.tex` (`figS: sensi_spatial`, `figS: sensi_spatial_rmse`) and four `\xl{}` paragraphs
  appended to Note 9 in `manuscript/SI/SI-notes_R1.tex`
- Code: `CONCERT-spac-niche/sub_sens_pmap_arm.sh`, `CONCERT-spac-niche/launch_sens_pmap.sh`,
  `spac_seq_eval/agg_sens_pmap.py`, `spac_seq_eval/plot_sens_pmap.py`

**Open decision for Xianglin:** the main text reports the seed-42 Perturb-MAP run (rho 0.566)
while the five-seed mean is 0.507. The SI note states this plainly. Either leave the main text as
a single reported run, or switch the Perturb-MAP numbers to seed-averaged mean ± SD.

## Honest limitations to state in the note

- One-factor-at-a-time; no interactions are probed.
- Perturb-MAP only unless the SPAC-seq arm lands.
- The `seen` regime only.


## Revision (2026-08-17, after Xianglin's review)

Five changes, all applied:

1. **Four factors added** - niche code width (`niche_film_dim` 4/8/16), niche neighbourhood size
   (`niche_film_k` 10/25/50), basal field random features (64/128/256) and basal field bandwidth
   (0.05/0.15/0.30). These are revision components that had no coverage at all. 24 more arms -> 77.
   **Deliberately excluded:** `fpert_rff_features` / `fpert_rff_bandwidth`, which only take effect
   when the footprint field is `rff`; the reported config uses `gp_pp` (`concert_spac.py:159-172`),
   so those panels would have been pure noise.
2. **Metrics switched to the benchmark pair** - LFC rank correlation and LFC RMSE (`lfc_dist`,
   `rescore_pm_lfc_distance.py:2`), replacing energy distance. Already present in every arm's
   `lfc_agg.json`, so no re-runs were needed.
3. **Error bars are now standard error** across the three seeds (was SD).
4. **Y-axis headroom** widened to 1.9x the data span (rho clipped to [0,1]) so an insensitive
   result does not read as volatile on a tight axis.
5. **Layout** - two separate 3x4 figures, one per metric, legend in the spare twelfth cell.

Two evaluator bugs this surfaced, both fixed:

- `niche_film_k` was **hard-coded to 25** at eval (`spac_seq_eval/run_concert_pm_niche.py:180`),
  with a comment saying it must match training or gamma renders differently. Sweeping it without a
  flag would have silently evaluated every arm at k=25. Added `--niche_film_k`, default 25, so
  every existing checkpoint is unaffected.
- Basal RFF feature count had no auto-detect and is a hard architecture dim, so `brf64`/`brf256`
  failed to load (`Size mismatch for basal_rff.rff_W: checkpoint (2, 64) vs model (2, 128)`).
  Added detection from the checkpoint, mirroring the existing `fpert_field.Z` logic, plus a
  `getattr`-guarded passthrough in `CONCERT-spac-niche/eval_spatial_counterfactual.py`
  (`_add_fpert_kwargs`). That failure is itself evidence the factor is real, not decorative.

Final numbers: seed band rho 0.443-0.566 and RMSE 1.93-2.19; largest factor movement 0.051 rho
(32 inducing points) and 0.14 RMSE (niche code width, footprint lengthscale ceiling). Every
setting lies inside the seed band on both metrics.

## Evaluator-patch regression check (2026-08-18)

The two evaluator fixes touch shared benchmark code, so they were verified rather than assumed:
`base42` was re-scored with the patched evaluator (job 40010456) and compared to its original
result, with a second identical run (job 40011700) as the stochasticity control.

| metric | patched vs pre-patch (max abs) | same code twice (max abs) |
|---|---|---|
| LFC rank correlation | 0.00372 | 0.00307 |
| LFC RMSE | 0.00613 | 0.00422 |
| energy distance | 0.04355 | 0.08395 |

Identical code differs by the same amount, so the patches are no-ops for default checkpoints.

**Incidental finding: `run_concert_pm_niche.py` never sets a random seed** and draws
`--n_samples 10` posterior samples plus 10 bootstrap replicates, so every score it produces -
including the numbers in the benchmark figures - carries roughly 0.003 rank correlation of
run-to-run jitter. That is ~40x smaller than the five-seed training band (0.123), so no conclusion
in this analysis or the benchmark depends on it, but seeding the evaluator would be a cheap
improvement.

Scratch result dirs `concert_sens_seen_base42_REGRESS{,2}` were deleted after the comparison;
regenerate with `scratchpad/regress_eval.sh` if needed.


## SPAC-seq sweep - RUN 2026-08-19..21, and the conclusion INVERTS

The "recommend dropping" call above was overturned. The sweep was run on the configuration the
paper actually reports for SPAC-seq - the **five-slide** model - in the **seen regime only**
(the regime is fixed at train time by `--holdout_v3_file`, `sub_sens_spac_arm.sh:79`, so an
unseen sweep is 65 fresh trainings, not a re-scoring; Xianglin decided seen-only 2026-08-21).

Design: 65 arms = 5 base seeds + 10 factors x 2 off-base levels x 3 seeds (42, 1, 2). Every arm
scores the identical panel of 5 slides x 100 (level L2, perturbation) cells - verified, zero
panel drift. Results straddle two roots because the sweep moved mid-flight:
`spac_seq_eval/multi_1-5_ho_v3/concert_sens/seen_<tag>/patch/slide_*/` (wave 1) and
`/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_sens/results/seen_<tag>/patch/slide_*/` (relaunches).

**The seed band is ~20x TIGHTER than Perturb-MAP's**: rho 0.3988-0.4053 (base seed 42 = 0.4013,
mid-band, not the best draw) and RMSE 3.769-3.788. So unlike Perturb-MAP, real hyper-parameter
effects are resolvable, and three leave the band:

| factor | worst level | rho | dev from base |
|---|---|---|---|
| `niche_film_dim` | 4 (and 16) | 0.3439 / 0.3468 | -0.056 / -0.053 |
| `moran_lambda` | 2 | 0.3566 | -0.043 |
| `niche_film_k` | 50 | 0.3732 | -0.027 |

`moran_lambda=0` (0.3868) and `niche_film_k=10` (0.3866) sit at the lower edge but do not clear
the conservative mean+SEM test the plot script applies (`plot_sens_spac.py:194`). The remaining
seven factors are flat to <=0.005 rho. Both metrics agree on the ranking.

Code (mirrors the pmap pair): `CONCERT-spac-niche/sub_sens_spac_arm.sh` (env knobs),
`launch_sens_spac.sh`, `relaunch_sens_spac_failed.sh`, `spac_seq_eval/agg_sens_spac.py` ->
`CONCERT-main/results/sens_spac_arms.csv`, `spac_seq_eval/plot_sens_spac.py` ->
`CONCERT-main/Outputs/sensitivity/fig_sens_spac_{rho,rmse}.{pdf,png}` + `figure_stats_spac.json`
+ `figure_legend_sensitivity_spac.md`.

Two infrastructure lessons worth keeping:

- **22 of 65 arms died ~1 min in** with `nvrtc: error: invalid value for --gpu-architecture`:
  the `concert_env` torch build ships no sm_120 cubin, and the RTX PRO 6000 Blackwell nodes are
  the only ones carrying the `rtx6000pro` feature. `relaunch_sens_spac_failed.sh` computes that
  node list live from `sinfo` and passes it to `sbatch -x`.
- **`kempner` and `kempner_h100` share `QoS=kempner_base`**, which caps the *account* at 96 GPUs
  (`MaxTRESPA gres/gpu=96`). When the lab is over that cap, moving a job between those two
  partitions is a no-op - both report `MaxGRESPerAccount`. `kempner_requeue` has `QoS=N/A` and is
  not subject to it. `relaunch_sens_spac_failed.sh:53` picks the partition by counting the user's
  own jobs (`< 12`), which is the wrong cap and parks jobs behind that wall.
