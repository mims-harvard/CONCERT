# Experiment 02 - Perturb-map perturbation prediction benchmark

The same two prediction tasks as experiment 01 (within-niche and cross-slide prediction), on four
Perturb-map lung tumour sections. Running them on a second assay tests whether the result holds
across measurement platforms.

## Differences from experiment 01

The model is the same (`models/concert_benchmark`); the settings differ where the assays do:

- the assay has no non-targeting control class, so unperturbed cells are the comparison group
  and the control label is left unset;
- guide assignment is unambiguous here, so the technical correction that separates intended
  from assigned perturbation is switched off;
- spots are denser and sections smaller, so the spatial field uses more inducing points (64
  against 16) and a wider lengthscale band, read at a larger coordinate scale;
- the minimum cell count per perturbation is left at zero, since the panel is already balanced.

## Inputs

Everything sits in the `data/` folder (`CONCERT_DATA_DIR`, see `datasets/README.md`):

- `perturbmap/raw/perturbmap_s0_spacfmt.h5ad` ... `perturbmap_s3_spacfmt.h5ad` - the four sections
  converted to the common format by `data_preparation/01_convert_sections.py`;
- `perturbmap/prepared/`, `perturbmap/patches/` - the prepared cell sets and the niche patches;
- `benchmark_assets/holdout_coords/holdout_pmap_{seen,unseen}.json` - holdout definitions,
  written by `05_build_holdout_coords.py --dataset pmap` in experiment 01's data preparation, which
  handles both assays.

## Order

1. **Convert the sections.** `data_preparation/01_convert_sections.py` writes each section
   in the format the model reads, with expression, perturbation assignment and coordinates in
   the expected fields.
2. **Define the holdouts.** `data_preparation/02_build_benchmark_panel.py` writes the
   evaluation panel, holdout pools and training manifests to `prepared_panel1053ko/benchmark/`;
   experiment 01's `05_build_holdout_coords.py --dataset pmap` turns them into the coordinate-keyed
   files training reads.
3. **Train.** `sbatch train/train.sh seen` (and `unseen`).
4. **Score.** `sbatch evaluation/evaluate.sh seen` runs
   `../../benchmark_evaluation/evaluate_concert_perturbmap.py` on the checkpoint and writes a
   per-perturbation summary per section. The lengthscale band, coordinate scale and inducing
   count are passed again there and must match the training values: the band defines the
   squashing function inside the spatial field, so a checkpoint read back under a different
   band silently reconstructs different lengthscales.
5. **Compare against the published methods (the reported numbers).**
   `sbatch evaluation/aggregate.sh` rescores the stored predicted LFCs of every
   method, the four reference strategies and STATE against the disjoint reference: the knockout cells
   of the niche against the niche's control cells outside every held-out patch (patch role 0),
   with the slide's role-0 controls as the fallback below 5 cells. The held-out source spots carry
   the `control` label on this assay, and the rescore asserts that the reference is disjoint from them.
   `perturbmap_reference.py` holds the rule, `score_perturbmap.py` applies it, and
   `perturbmap_tables.py` writes the per-record tables, the paired tests and the Fig. 4j panel
   (`plot_perturbmap_panel.py`) to `${CONCERT_WORKDIR}/perturbmap_results/` (the shipped
   copy is `data/benchmark_assets/results/perturbmap_results/`). The predictions are the
   ones on disk; the aggregation step only rescores them.

   The harness reference (every control cell of the niche, `lfc_scoring.observed_lfc_pm_niche`)
   fixes which records are scored and the shared record set; `score_perturbmap.py` writes both
   scores side by side (`*_harness` and the unsuffixed reported columns).

   **The comparison uses the records every method covers.** The model methods cover 31
   niche records and the reference strategies and STATE 27; the reported comparison is the
   27-record intersection (Tgfbr2 22, Ifngr2 3, Jak2 2), which `perturbmap_tables.py` and
   `plot_perturbmap_benchmark.py` take themselves. Any other reader of the per-record CSVs applies the
   same intersection.
