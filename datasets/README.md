# Datasets

No data is stored in this repository. Every dataset CONCERT was trained and evaluated on is
public, or available from the authors of the original study, and is linked below. Each row
gives the study the sections come from, where the raw data lives, and a processed copy in the
`.h5ad` layout the code reads.

| Dataset | Used for | Raw data | Processed data |
|---|---|---|---|
| [**SPAC-seq** CRISPR screen](https://doi.org/10.1016/j.cell.2026.04.049) (Zhang et al., *Cell* 2026) | Benchmark, `experiments/01_spac_seq_benchmark` | [SPAC-seq portal](https://spac.pku-genomics.org) | [preprocessing scripts](../experiments/01_spac_seq_benchmark/data_preparation/) |
| [**Perturb-map** lung tumour](https://www.cell.com/cell/fulltext/S0092-8674(22)00195-7) (Dhainaut et al., *Cell* 2022) | Benchmark, `experiments/02_perturb_seq_benchmark` | [GEO GSE193460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460) | [Figshare](https://figshare.com/articles/dataset/Datasets_-_Perturb-Map/29198468) |
| [**Colitis MERFISH** time course](https://doi.org/10.1016/j.cell.2024.03.013) (Cadinu et al., *Cell* 2024) | Time-course case study, `experiments/03_dss_gut_case_study` | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.rjdfn2zh3) | [preprocessing script](../experiments/03_dss_gut_case_study/data_preparation/convert_sections.py) |
| [**Mouse gut DSS** Visium](https://www.nature.com/articles/s41586-024-08216-z) (Mayassi et al., *Nature* 2024) | Colon inflammation | [Broad SCP2771](https://singlecell.broadinstitute.org/single_cell?type=study&page=1&terms=SCP2771) | [Figshare](https://figshare.com/articles/dataset/Mouse_gut_inflammation_dataset/29882873) |
| [**Mouse brain stroke**](https://www.science.org/doi/abs/10.1126/scitranslmed.adg1323) (Han et al., *Sci. Transl. Med.* 2024) | Stroke case study, `experiments/07_stroke_case_study` | From the authors of the original paper on request | [Figshare](https://figshare.com/articles/dataset/Mouse_brain_stroke_dataset/29882900) |
| [**Perturb-FISH** tumour section](https://doi.org/10.1016/j.cell.2025.02.012) (Binan et al., *Cell* 2025) | Kernel interpretability, `experiments/04_kernel_interpretability` | [Original data](https://doi.org/10.35077/ace-gem-get) | [preprocessing script](../experiments/04_kernel_interpretability/perturbfish_prepare_data.py) |

## The two input folders

Every input the notebooks, the `reproduction/` scripts and the experiment run scripts read sits in two
folders, `data/` and `checkpoints/`, available from the authors upon request. Place both in the repository
root; reruns write to `runs/` there. Each location can be moved with an environment variable:
`CONCERT_DATA_DIR`, `CONCERT_CHECKPOINT_ROOT` and `CONCERT_WORKDIR`.

`data/README.md` lists every file, its size and the notebook or script that reads it. The layout
is defined once, in `benchmark_evaluation/release_paths.py`; the scripts, the notebooks and
`config/paths.sh` (for the SLURM run scripts under `experiments/`) all derive their paths from it.
`python reproduction/01_spac_seq_benchmark.py --steps weights` prints an existence table for every
input it needs and stops on the first missing one, which is the quickest way to check a new
machine is wired up.

## What each experiment reads

| Experiment | Input in `data/` |
|---|---|
| `01_spac_seq_benchmark` | `spacseq/raw/` (five sections), `spacseq/prepared/` (matched cell sets, raw counts), `benchmark_assets/spacseq_benchmark/`, `benchmark_assets/holdout_coords/`, `benchmark_assets/results/spacseq_*` |
| `02_perturb_seq_benchmark` | `perturbmap/raw/` (four sections), `perturbmap/prepared/`, `perturbmap/patches/`, `benchmark_assets/perturbmap_benchmark/`, `benchmark_assets/results/perturbmap_*` |
| `03_dss_gut_case_study` | `dss_colitis/sections/` (MERFISH sections and `manifest.csv`), `dss_colitis/architecture/`, `dss_colitis/case_study/`, `dss_colitis/heldout_eval/` |
| `04_kernel_interpretability` | `spacseq/kernel_params/`, `perturbfish/` |
| `06_sensitivity_analysis` | `benchmark_assets/results/spacseq_sensitivity/`, `benchmark_assets/results/sensitivity_tables/` |
| `07_stroke_case_study` | `stroke/slides/` (the PASTE2-aligned stacks), `stroke/Mouse_brain_stroke_all_data.h5` (ICA/PIA labels), `stroke/decodes_{2d,3d}/` (decodes, gene axis `hvg_gene_names.txt`, lesion geometry `shape_sets/`) |
| `08_deconvolution_imputation` | `deconvolution/spacseq_2-1/` (task 1) |

## Trained checkpoints

The trained checkpoints form one folder, `checkpoints/`, available from the authors upon request,
with one subdirectory per model:

| Directory | Model code (`models/`) | Result | Notebook / script |
|---|---|---|---|
| `spacseq_within_niche` | `concert_benchmark` | SPAC-seq within-niche benchmark (experiment 01) | `01_spac_seq_benchmark` |
| `spacseq_cross_slide` | `concert_benchmark` | SPAC-seq cross-slide benchmark (experiment 01) | `01_spac_seq_benchmark` |
| `perturbmap_within_niche` | `concert_benchmark` | Perturb-map within-niche benchmark (experiment 02) | `02_perturb_map_benchmark` |
| `perturbmap_cross_slide` | `concert_benchmark` | Perturb-map cross-slide benchmark (experiment 02) | `02_perturb_map_benchmark` |
| `dss_colitis` | `concert_temporal` | DSS colitis time course (experiment 03) | `03_dss_colitis_per_mouse_marker_trend` |
| `perturbfish` | `concert_benchmark` | Perturb-FISH repetition of the interpretability readout (experiment 04) | `experiments/04_kernel_interpretability/perturbfish_train_and_extract.sh` (step 3) |
| `spacseq_kernel_analysis` | `concert_benchmark` | SPAC-seq kernel-parameter readout (experiment 04) | `experiments/04_kernel_interpretability/01_extract_kernel_params.py` |
| `stroke_2d_core_rim` | `concert_spac_niche` | Stroke case study, 2D arm (experiment 07) | `04_stroke_2d_and_3d_prediction` |
| `stroke_3d_core_rim` | `concert_spac_niche` | Stroke case study, 3D arm (experiment 07) | `04_stroke_2d_and_3d_prediction` |

Each directory holds `model.pt` and the files needed to load it, at least `pert_map.json` (the
perturbation vocabulary, read from this file so that the integer code of every perturbation
matches the one the weights were trained with) and `hvg_gene_names.txt` (the model's gene axis).
The folder is about 5.9 GB. Every notebook and
script reads its checkpoint from `$CONCERT_CHECKPOINT_ROOT`. Every result can also be retrained
from the scripts under `experiments/`.
