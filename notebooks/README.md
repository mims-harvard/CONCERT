# Tutorial notebooks

Four notebooks that reproduce the paper's results from the **trained checkpoints**. Nothing here
retrains anything: every notebook loads a `model.pt` and runs inference.

| Notebook | What it reproduces | Checkpoints |
|---|---|---|
| [`01_spac_seq_benchmark.ipynb`](01_spac_seq_benchmark.ipynb) | SPAC-seq held-out perturbation benchmark, both settings, and the comparison against the published methods and the reference strategies | `spacseq_within_niche`, `spacseq_cross_slide` |
| [`02_perturb_map_benchmark.ipynb`](02_perturb_map_benchmark.ipynb) | The same benchmark on the spatial Perturb-map assay, scored with the disjoint reference | `perturbmap_within_niche`, `perturbmap_cross_slide` |
| [`03_dss_colitis_per_mouse_marker_trend.ipynb`](03_dss_colitis_per_mouse_marker_trend.ipynb) | The inflammation marker trajectory of each animal over the DSS time course - a longitudinal curve for animals measured once | `dss_colitis` |
| [`04_stroke_2d_and_3d_prediction.ipynb`](04_stroke_2d_and_3d_prediction.ipynb) | A stroke lesion imposed on a healthy brain and predicted, in a 2D and a 3D arm, against the measured lesion | `stroke_2d_core_rim`, `stroke_3d_core_rim` |

Read 01 first if you are reading more than one: the scoring protocol, the two benchmark settings
and the two metrics are explained there and only referred to afterwards.

The same four results are also available as command-line scripts in [`../reproduction/`](../reproduction/)
- same steps, same order, same numbers - for when you want a result in a batch job rather than
in a browser.

## Running them

All four expect the `concert_env` environment. Start Jupyter **inside a GPU allocation** - the
notebooks that predict are unusably slow on CPU, and the environment is pinned to CUDA 11.8 and
cannot use Blackwell GPUs:

```bash
# inside a one-GPU allocation from your scheduler (H100 or A100, about 200 GB host memory)
conda activate concert_env
cd notebooks && jupyter lab --no-browser --ip=0.0.0.0
```

Start Jupyter from anywhere inside the repository: the first cell walks up from the working
directory to find the repository root. Every input comes from the two input folders, `data/`
and `checkpoints/`, named by `CONCERT_DATA_DIR` and `CONCERT_CHECKPOINT_ROOT`; see
[`../datasets/README.md`](../datasets/README.md).

Notebooks 3 and 4 are mostly CPU work - they read result tables and draw - and only need the GPU
for the one cell that re-runs the decode, which is off by default.

Each notebook writes under `CONCERT/notebook_runs/<name>/`, or wherever the environment variable
`CONCERT_NB_WORKDIR` points. The inputs under `data/` and `checkpoints/` stay read-only, so a
failed write, which can truncate its target file to zero, leaves them intact.

## Expensive cells

Every long-running cell is behind a flag at the top of the cell, so a first pass can be made
without any of them. Defaults:

| Notebook | Flag | Default | Cost |
|---|---|---|---|
| 01 | `RUN_SLIDE_EVAL` | `True` | 15-40 min for one slide |
| 01 | `RUN_ONE_UNSEEN_UNIT` | `True` | a few minutes |
| 02 | `RUN_SECTION_EVAL` | `True` | 5-15 min for one section |
| 03 | `RUN_DECODE` | `False` | 30-45 min and ~200 GB for all 15 animals |
| 03 | `RECOMPUTE_OBSERVED` | `False` | a few minutes of I/O, no GPU |
| 04 | `RUN_DECODE` | `False` | ~20 min per arm |

Turning one off makes the notebook read the shipped result instead, so the numbers and figures
still come out.

## What the notebooks depend on

* the released code in `../models/` and `../benchmark_evaluation/`;
* the trained checkpoints: the `checkpoints/` folder at `$CONCERT_CHECKPOINT_ROOT` - each
  checkpoint directory holds `model.pt` **and** `pert_map.json`, because the perturbation
  vocabulary is read from that file;
* the `data/` folder at `$CONCERT_DATA_DIR`: the measured sections, the benchmark assets, the
  shipped decodes of the case studies and the reported tables the notebooks check their numbers
  against (`data/README.md` lists every file).

Every path is resolved and existence-checked in the first code cell of each notebook, so a
missing input is reported there rather than three cells later.

## Naming

The code paths, output directories and CSV columns use `seen` and `unseen`. In the manuscript
those two settings are **within-niche prediction** and **cross-slide prediction**. The notebooks
use the manuscript names in prose and the directory names when naming files.
