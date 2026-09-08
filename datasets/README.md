# Datasets

No data is stored in this repository. Every dataset CONCERT was trained and evaluated on is
public, or available from the authors of the original study, and is linked below. Each row
gives the study the sections come from, where the raw data lives, and a processed copy in the
`.h5ad` layout the code reads.

| Dataset | Used for | Raw data | Processed data |
|---|---|---|---|
| [**SPAC-seq** CRISPR screen](https://doi.org/10.1016/j.cell.2026.04.049) (Zhang et al., *Cell* 2026) | Benchmark, `experiments/01_spac_seq_benchmark` | *pending — see note below* | *pending — see note below* |
| [**Perturb-map** lung tumour](https://www.cell.com/cell/fulltext/S0092-8674(22)00195-7) (Dhainaut et al., *Cell* 2022) | Benchmark, `experiments/02_perturb_seq_benchmark` | [GEO GSE193460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460) | [Figshare](https://figshare.com/articles/dataset/Datasets_-_Perturb-Map/29198468) |
| [**Colitis MERFISH** time course](https://doi.org/10.1016/j.cell.2024.03.013) (Cadinu et al., *Cell* 2024) | Time-course case study, `experiments/03_dss_gut_case_study` | *pending — see note below* | *pending — see note below* |
| [**Mouse gut DSS** Visium](https://www.nature.com/articles/s41586-024-08216-z) (Mayassi et al., *Nature* 2024) | Colon inflammation | [Broad SCP2771](https://singlecell.broadinstitute.org/single_cell?type=study&page=1&terms=SCP2771) | [Figshare](https://figshare.com/articles/dataset/Mouse_gut_inflammation_dataset/29882873) |
| [**Mouse brain stroke**](https://www.science.org/doi/abs/10.1126/scitranslmed.adg1323) (Han et al., *Sci. Transl. Med.* 2024) | Stroke case study, `experiments/07_stroke_case_study` | From the authors of the original paper on request | [Figshare](https://figshare.com/articles/dataset/Mouse_brain_stroke_dataset/29882900) |

> **Pending links.** The accession and Figshare links for the SPAC-seq screen and for the
> colitis MERFISH time course are not yet public and are marked as placeholders in the
> manuscript's data-availability statement as well. They will be filled in here when the
> deposits are released; until then, contact the authors.

## Where the code looks for the data

The scripts and notebooks resolve every root from an environment variable, with the paths the
published results were produced on as the fallback. Set these once and nothing else needs
editing:

```bash
export CONCERT_DATA_ROOT=/path/to/datasets          # the measured sections
export CONCERT_OUTPUT_ROOT=/path/to/outputs         # trained checkpoints, one dir per run
export CONCERT_SCORING_LIB=/path/to/spac_seq_eval   # prepared cell sets + shipped results
export CONCERT_PAPER_NUMBERS=/path/to/paper_numbers # reported per-record tables
export CONCERT_WORKDIR=/path/to/scratch             # where the scripts write
```

`config/paths.sh` carries the same roots for the SLURM run scripts under `experiments/`.
`python scripts/01_spac_seq_benchmark.py --steps weights` prints an existence table for every
input it needs and stops on the first missing one, which is the quickest way to check a new
machine is wired up.

## What each experiment expects

| Experiment | Input |
|---|---|
| `01_spac_seq_benchmark` | five sections `raw_output_1-{1..5}.h5ad` under `$CONCERT_DATA_ROOT/newdata/spac_seq`, plus the prepared matched cell sets and the evaluation panel |
| `02_perturb_seq_benchmark` | four sections `perturbmap_s{0..3}_spacfmt.h5ad` under `$CONCERT_DATA_ROOT`, plus the niche patches and the assay panel mask `in_panel_1053` |
| `03_dss_gut_case_study` | the MERFISH sections listed in `manifest.csv`, plus the tissue-architecture descriptors in the sibling `_arch` directory |
| `07_stroke_case_study` | the PASTE2-aligned stacks `stroke_paste2_{PT,Sham}{1..4}.h5ad` and the ICA/PIA region labels of the measured animal |

The per-experiment `README.md` under `experiments/` lists the derived files each pipeline
builds from these, and the order to build them in.

## Trained checkpoints

Checkpoints are not included. Each result can be retrained from the scripts under
`experiments/`, and the tutorials in `notebooks/` and `scripts/` load a checkpoint directory
that must contain **both** `model.pt` and `pert_map.json` — the perturbation vocabulary is read
rather than rebuilt, so that the integer code of every perturbation matches the one the weights
were trained with.
