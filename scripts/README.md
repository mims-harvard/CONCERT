# Reproduction scripts

One script per paper result, reproduced from a **trained checkpoint**. Nothing here retrains:
every script loads `model.pt`, runs inference, prints the numbers and writes the tables and
figures. They are the command-line form of the notebooks in `../notebooks/` - same steps, same
order, same numbers - for when you want a result in a batch job rather than a browser.

| Script | Notebook | Reproduces | Checkpoints |
|---|---|---|---|
| [`01_spac_seq_benchmark.py`](01_spac_seq_benchmark.py) | [`01`](../notebooks/01_spac_seq_benchmark.ipynb) | SPAC-seq held-out perturbation benchmark, both settings, and the comparison against the published methods | `spacseq_within_niche`, `spacseq_cross_slide` |
| [`02_perturb_map_benchmark.py`](02_perturb_map_benchmark.py) | [`02`](../notebooks/02_perturb_map_benchmark.ipynb) | The same benchmark on the spatial Perturb-map assay | `perturbmap_within_niche`, `perturbmap_cross_slide` |
| [`03_dss_colitis_marker_trend.py`](03_dss_colitis_marker_trend.py) | [`03`](../notebooks/03_dss_colitis_per_mouse_marker_trend.ipynb) | The inflammation marker trajectory of each animal over the colitis time course | `dss_colitis` |
| [`04_stroke_2d_3d_prediction.py`](04_stroke_2d_3d_prediction.py) | [`04`](../notebooks/04_stroke_2d_and_3d_prediction.ipynb) | A stroke lesion imposed on a healthy brain and predicted, 2D and 3D, against the measured lesion | `stroke_2d_core_rim`, `stroke_3d_core_rim` |

Read 01 first if you are reading more than one: the scoring protocol, the two benchmark
settings and the two metrics are explained there and only referred to afterwards.

## Running them

```bash
conda activate concert_env
cd scripts

# every script prints what it needs and stops on the first missing input
python 01_spac_seq_benchmark.py --help

# the cheap steps: read the shipped results and reproduce the reported numbers (seconds)
python 01_spac_seq_benchmark.py --steps aggregate compare
python 02_perturb_map_benchmark.py --steps aggregate compare
python 03_dss_colitis_marker_trend.py
python 04_stroke_2d_3d_prediction.py

# the steps that predict (GPU)
python 01_spac_seq_benchmark.py --steps weights predict slide --slide 1-1
python 02_perturb_map_benchmark.py --steps weights predict section --section 0
python 03_dss_colitis_marker_trend.py --run-decode --mice all
python 04_stroke_2d_3d_prediction.py --run-decode --arms 2d 3d
```

Each script takes `--steps` and runs only those, in the fixed order the result is built in;
`--all` runs everything. `--outdir` chooses where it writes, defaulting to
`$CONCERT_WORKDIR/<name>`.

## Cost

Every step that predicts is opt-in. The defaults read the results that came with the paper, so
a first pass costs seconds and still produces the reported numbers and every figure.

| Script | Expensive step | Cost |
|---|---|---|
| 01 | `slide` - score one slide over the full panel | 15-40 min on one H100 |
| 01 | `crossslide --run-unit` - one evaluation unit end to end | a few minutes |
| 02 | `section` - score one section | 5-15 min |
| 03 | `--run-decode --mice all` - re-decode all 15 animals at four virtual days | 30-45 min, ~200 GB RAM |
| 04 | `--run-decode` - the three decode passes | ~20 min per arm |

## Environment

The prediction steps need a GPU: on CPU they are unusably slow. The environment is pinned to
CUDA 11.8 and **cannot use Blackwell GPUs** - a run that dies with an `nvrtc -arch` error a few
minutes in was scheduled on one. Ask for an H100 or A100.

```bash
# inside a one-GPU allocation from your scheduler (about 200 GB of host memory for script 01)
conda activate concert_env
python scripts/01_spac_seq_benchmark.py --all
```

## Paths

All four scripts resolve their roots through [`concert_paths.py`](concert_paths.py), which reads
two environment variables, `CONCERT_DATA_DIR` (the downloadable `data/` folder) and
`CONCERT_CHECKPOINT_ROOT` (the `checkpoints/` folder), plus the optional `CONCERT_WORKDIR`. See
[`../datasets/README.md`](../datasets/README.md).

Nothing is written inside the repository, and nothing is written to the checkpoint directories:
where a tool insists on loading `model.pt` from its own output directory, the script builds a
symlink shadow of the checkpoint directory under the work directory instead.

## Naming

The code paths, output directories and CSV columns use `seen` and `unseen`. In the manuscript
those two settings are **within-niche prediction** and **cross-slide prediction**. The scripts
print the manuscript names and keep the directory names on disk; do not carry `seen`/`unseen`
into manuscript text.

## Two different comparison rules

The two benchmarks compare against the published methods differently, and the difference is
not cosmetic:

* **Perturb-map** - the methods cover enough of the same records that the reported comparison
  is the intersection of the records **all** methods cover (27 of 31).
* **SPAC-seq** - the all-method intersection is **empty** (CelComen covers 15 of 129 records,
  CellAgentChat 11). The reported comparison is **paired, one competitor at a time**, on the
  records that competitor and CONCERT share. The `n_pairs` column of the shipped statistics
  table is that count, and `--steps compare` checks itself against it.
