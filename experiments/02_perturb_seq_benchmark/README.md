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
5. **Compare against the published methods (benchmark v4, the reported numbers).**
   `sbatch evaluation/rescore_perturb_seq_v4.sh` rescores the stored predicted LFCs of every
   method, the four reference strategies and STATE against the v4 reference: the knockout cells
   of the niche against the niche's control cells outside every held-out patch (patch role 0),
   with the slide's role-0 controls as the fallback below 5 cells. The held-out source spots carry
   the `control` label on this assay, and v4 asserts that the reference is disjoint from them.
   `perturbmap_v4_scoring.py` holds the rule, `rescore_pm_v4.py` applies it, and
   `analyze_pm_v4.py` writes the per-record tables, the paired tests and the Fig. 4j panel
   (`plot_pmap_benchmark_v4.py`) to `${SCORING_LIB}/multi_pm_ho_v4/`. The predictions are the
   ones on disk; nothing is retrained or re-run.

   The earlier v3 rescore (`../../benchmark_evaluation/rescore_pm_lfc_distance.py`,
   `figures/plot_pmap_benchmark.py`) stays as the record of the previous reference;
   `rescore_pm_v4.py` reproduces its numbers exactly as a check.

   **The comparison uses the records every method covers.** Under v4 the model methods cover 31
   niche records and the reference strategies and STATE 27; the reported comparison is the
   27-record intersection (Tgfbr2 22, Ifngr2 3, Jak2 2), which `analyze_pm_v4.py` and
   `plot_pmap_benchmark.py` take themselves. Any other reader of the per-record CSVs applies the
   same intersection.
