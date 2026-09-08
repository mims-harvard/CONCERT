# CONCERT predicts niche-aware perturbation responses in spatial transcriptomics

[![ProjectPage](https://img.shields.io/badge/Project-CONCERT-red)](https://zitniklab.hms.harvard.edu/projects/CONCERT/) [![PaperPage](https://img.shields.io/badge/Paper-bioRxiv-orange)](https://www.biorxiv.org/content/early/2025/11/10/2025.11.08.686890) [![CodePage](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/mims-harvard/CONCERT/) [![Data](https://img.shields.io/badge/Data-Links-purple)](./datasets/) [![Tutorials](https://img.shields.io/badge/Tutorials-Notebooks-yellow)](./notebooks/) ![License](https://img.shields.io/badge/license-MIT-blue)

[Xiang Lin](https://xianglin226.github.io/), [Zhenglun Kong](https://scholar.google.com/citations?hl=en&user=XYa4NVYAAAAJ), [Soumya Ghosh](https://scholar.google.com/citations?user=GEYQenQAAAAJ&hl=en), [Manolis Kellis](https://scholar.google.com/citations?user=lsYXBx8AAAAJ&hl=en), [Marinka Zitnik](https://scholar.google.com/citations?user=YtUDgPIAAAAJ&hl=en)

Spatial perturbation transcriptomics measures how genetic or chemical edits alter gene
expression while preserving tissue context. Perturbation outcomes depend on a cell's intrinsic
state and also on how effects propagate across cellular microenvironments. **CONCERT** is a
niche-aware generative model that embeds perturbation context and learns spatial kernels with a
Gaussian-process variational autoencoder to predict perturbation effects across tissue. Each
perturbation is given a spatial effect field over the section and a gain that controls how
strongly the local microenvironment may reshape its response, so the same perturbation can
produce different responses in different neighbourhoods. A temporal variant extends the same
construction over time, so one model covers a whole time course and can be asked for a
timepoint it was never trained on.

We formalise two prediction tasks: **within-niche prediction**, where a cell's response to a
perturbation carried by cells of its own local niche is predicted, and **cross-slide
prediction**, where the response must be transferred to a slide on which that perturbation was
never observed. CONCERT is evaluated on a SPAC-seq screen of hundreds of knockouts and on
Perturb-map lung tumour sections, and outperforms dissociated counterfactual models, spatial
perturbation models and neighbour-averaging strategies on both the magnitude and the gene
ranking of the predicted response. Two case studies go beyond benchmarking: in DSS-induced
colitis CONCERT reconstructs expression at unmeasured timepoints and recovers the non-monotone
course of the injury, rising to a day-9 peak and resolving by day 21, which interpolation
between observed timepoints cannot produce; in ischemic stroke it predicts responses under
lesion sizes and 3D geometries that photothrombosis cannot produce experimentally.

## ⚡ Challenges from existing perturbation models

Ignoring cells' tissue context when predicting perturbations

<p align="center">
  <img src="./issue.jpg" alt="issue" width="400"/>
</p>

## 🔥 Challenges from existing spatial perturbation sequencing technologies

<p align="center">
  <img src="./issue2.jpg" alt="issue2" width="400"/>
</p>

## 🛠️ Functions of CONCERT

1. Predict perturbations on cells considering their niches (2D or 3D).
2. Predict perturbations across slides, including perturbations never seen on the target slide.
3. Predict perturbations at timepoints that were never measured.
4. Impute missing cells with perturbation prediction.
5. Learn the scope of a perturbation's effect on tissue space.
6. Disentangle perturbation effects — intra- or inter-cellular.

<p align="center">
  <img src="./function.jpg" alt="model" width="400"/>
</p>

## 🗂️ Repository layout

| Path | Contents |
|---|---|
| [`models/concert_benchmark/`](./models/concert_benchmark/) | Model used for the SPAC-seq and Perturb-map benchmarks |
| [`models/concert_temporal/`](./models/concert_temporal/) | Model used for the colitis time-course case study |
| [`models/concert_spac_niche/`](./models/concert_spac_niche/) | Model used for the stroke case study |
| [`benchmark_evaluation/`](./benchmark_evaluation/) | Scoring code shared by the two benchmark experiments |
| [`experiments/`](./experiments/) | One directory per experiment: data preparation, training, evaluation, figures |
| [`scripts/`](./scripts/) | One command-line script per paper result, run from a trained checkpoint |
| [`notebooks/`](./notebooks/) | The same four results as step-by-step tutorials |
| [`datasets/`](./datasets/) | Where every dataset comes from, and how to point the code at it |
| [`config/paths.sh`](./config/paths.sh) | Data, output and environment paths — the one file the SLURM scripts read |

The three model directories are deliberate copies rather than one shared package. They diverged
in the model source itself, not only in configuration, and keeping them separate means a change
made for the time course cannot silently alter a benchmark number.

## ⚙️ Dependencies

Python 3.11.13 and one CUDA GPU for training and for the prediction steps; the figure and
table steps run on CPU. The reported results were produced in this environment:

```
torch==2.3.0          # CUDA 11.8 build
numpy==1.26.4
scipy==1.12.0
pandas==2.3.3
scikit-learn==1.8.0
scanpy==1.10.1
anndata==0.12.2
h5py==3.13.0
matplotlib==3.10.1
scvi-tools==1.3.3
wandb==0.22.1
```

```bash
conda create -n concert_env python=3.11 && conda activate concert_env
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r env_requirements.txt
```

CUDA 11.8 supports up to `sm_90`, so this environment **cannot run on Blackwell GPUs** — a run
that dies with an `nvrtc -arch` error a few minutes in was scheduled on one. Ask for an H100 or
an A100.

## 💻 Run CONCERT

### Getting started

1. Create the environment from `env_requirements.txt`.
2. Point the code at your data. Both the scripts and the SLURM runners read the same roots:

   ```bash
   export CONCERT_DATA_ROOT=/path/to/datasets      # the measured sections
   export CONCERT_OUTPUT_ROOT=/path/to/outputs     # where checkpoints are written
   export CONCERT_WORKDIR=/path/to/scratch         # where the scripts write results
   ```

   `config/paths.sh` carries the same values for the `sbatch` scripts under `experiments/`, and
   is the only file that needs editing when the code moves to another machine.
3. Check the wiring. Every script prints an existence table for each input it needs and stops
   on the first missing one:

   ```bash
   python scripts/01_spac_seq_benchmark.py --steps aggregate
   ```

### Example 1 — train on a spatial perturbation dataset

```bash
cd models/concert_benchmark
python run_concert_spac.py \
  --config config.yaml --stage train \
  --sample perturbmap --project_index map \
  --data_file $CONCERT_DATA_ROOT/perturbmap_s0_spacfmt.h5ad \
  --outdir $CONCERT_OUTPUT_ROOT/perturbmap_s0 --model_file model.pt \
  --hvg_max_rank 2000 --seed 42 \
  --multi_kernel_mode True --loc_range 20.0 --inducing_point_steps 4 \
  --use_expr_encoder True --expr_hidden_dims 512 256 --use_basal_encoder_only True \
  --basal_field rff --basal_bayes True --spatial_basal_only True \
  --fpert_field gp_pp --fpert_coord_scale 70.0 \
  --fpert_svgp_inducing 64 --fpert_ls_min 0.01 --fpert_ls_max 0.60 \
  --use_niche_film True --niche_film_dim 8 --niche_film_k 25 --niche_film_gcap 3.0
```

Training writes `model.pt` **and** `pert_map.json` into `--outdir`. Both are needed later: the
perturbation vocabulary is read from `pert_map.json` rather than rebuilt, so that the integer
code of every perturbation matches the one the weights were trained with.

The lengthscale band (`--fpert_ls_min` / `--fpert_ls_max`) is a plain attribute of the spatial
field, not a saved weight. **It has to be passed at evaluation exactly as at training** — leave
it out and the same weights render a different spatial field, and the run completes with
plausible, wrong numbers.

The reported runs used `experiments/*/train/*.sh`, which pass the full settings:

```bash
sbatch experiments/01_spac_seq_benchmark/train/train_spac_seq.sh seen
sbatch experiments/02_perturb_seq_benchmark/train/train_perturb_seq.sh seen
sbatch experiments/03_dss_gut_case_study/train/train_dss.sh
```

### Example 2 — counterfactual prediction on chosen spots

Give a file of spot indices, the tissue type to place them in and the perturbation to apply.
Spots can be selected interactively with the Shiny app in
[`select_cells/`](./select_cells/).

```bash
cd models/concert_benchmark
python run_concert_spac.py \
  --config config.yaml --stage eval \
  --sample perturbmap --project_index map \
  --data_file $CONCERT_DATA_ROOT/perturbmap_s0_spacfmt.h5ad \
  --outdir $CONCERT_OUTPUT_ROOT/perturbmap_s0 --model_file model.pt \
  --pert_cells spots.txt \
  --target_cell_tissue tumor --target_cell_perturbation Jak2 \
  --fpert_field gp_pp --fpert_coord_scale 70.0 \
  --fpert_ls_min 0.01 --fpert_ls_max 0.60          # same band as training
```

The decoder is

```
log mu = f_basal(x, expression) + 1[has_guide] * mu_action + 1[is_real_KO] * f_pert(g, x)
```

so a decode to perturbation `g` and a decode to the basal slot differ by exactly the
perturbation terms — the basal reconstruction cancels. Every reported score is computed on that
difference of two decodes, not on one decode alone.

### Example 3 — reproduce a paper result from a trained checkpoint

Four scripts, one per result. They print the numbers and write the tables and figures; the
steps that predict are opt-in, so a first pass reads the shipped results and still reproduces
every reported number.

```bash
python scripts/01_spac_seq_benchmark.py --steps aggregate compare   # SPAC-seq benchmark
python scripts/02_perturb_map_benchmark.py --steps aggregate compare # Perturb-map benchmark
python scripts/03_dss_colitis_marker_trend.py                        # colitis time course
python scripts/04_stroke_2d_3d_prediction.py                         # stroke, 2D and 3D
```

Add the prediction steps on a GPU node:

```bash
python scripts/01_spac_seq_benchmark.py --steps weights predict slide --slide 1-1
python scripts/04_stroke_2d_3d_prediction.py --run-decode --arms 2d 3d
```

See [`scripts/README.md`](./scripts/README.md) for every step, its cost and its output.

### 📓 Tutorials

The same four results, explained step by step, are in [`notebooks/`](./notebooks/) — start
there if you are reading rather than running:

| Notebook | What it reproduces |
|---|---|
| [`01_spac_seq_benchmark.ipynb`](./notebooks/01_spac_seq_benchmark.ipynb) | SPAC-seq held-out perturbation benchmark, both settings, and the comparison against the published methods |
| [`02_perturb_map_benchmark.ipynb`](./notebooks/02_perturb_map_benchmark.ipynb) | The same benchmark on the spatial Perturb-map assay |
| [`03_dss_colitis_per_mouse_marker_trend.ipynb`](./notebooks/03_dss_colitis_per_mouse_marker_trend.ipynb) | A longitudinal inflammation trajectory for animals each measured once |
| [`04_stroke_2d_and_3d_prediction.ipynb`](./notebooks/04_stroke_2d_and_3d_prediction.ipynb) | A stroke lesion imposed on a healthy brain and predicted, 2D and 3D, against the measured lesion |

Read 01 first: the scoring protocol, the two settings and the two metrics are explained there
and only referred to afterwards.

### Reproducing an experiment end to end

Each directory under `experiments/` is self-contained — its own data preparation, training,
evaluation and figures — and carries a `README.md` listing the inputs it expects, the order to
run things in, and what it writes.

| Directory | Question |
|---|---|
| [`01_spac_seq_benchmark`](./experiments/01_spac_seq_benchmark/) | Can the response of a perturbation held out from a slide be predicted from location and unperturbed expression? |
| [`02_perturb_seq_benchmark`](./experiments/02_perturb_seq_benchmark/) | The same question on spatial Perturb-map, a different assay and perturbation set |
| [`03_dss_gut_case_study`](./experiments/03_dss_gut_case_study/) | Does one model over a colitis time course reproduce the tissue at a timepoint it was not trained on? |
| [`04_kernel_interpretability`](./experiments/04_kernel_interpretability/) | Do the spatial parameters the model learns track the observed arrangement of the perturbed cells? |
| [`05_perturb_seq_case_study`](./experiments/05_perturb_seq_case_study/) | Does the same knockout, placed in different parts of a section, produce a different predicted response? |
| [`06_sensitivity_analysis`](./experiments/06_sensitivity_analysis/) | Do the reported numbers survive changes to the hyper-parameters, once initialisation noise is measured? |
| [`07_stroke_case_study`](./experiments/07_stroke_case_study/) | Can a lesion of a chosen size and position be imposed on a healthy brain and its response predicted, in 2D and in 3D? |

The run scripts are SLURM batch scripts whose `#SBATCH` headers carry generic partition names;
set the partition and account your cluster uses before submitting. To run without a scheduler,
execute a script directly with `bash` — nothing outside the headers depends on SLURM except the
SPAC-seq evaluation, which reads `SLURM_ARRAY_TASK_ID` to pick a slide and can be given that
variable by hand.

### A note on paths

`scripts/`, `notebooks/` and `config/paths.sh` resolve every root from an environment variable
and need nothing edited beyond the exports above. The pipeline scripts under `experiments/`
are the record of the runs that produced the reported results, and many of them carry the
absolute paths those runs used inline rather than through `config/paths.sh`. They are kept as
they ran; edit the paths at the top of a script before submitting it on another machine.

### Naming

The code paths, output directories and CSV columns use `seen` and `unseen`. In the manuscript
those two settings are **within-niche prediction** and **cross-slide prediction**.

## 🗃️ Datasets

No data is included in this repository. Every dataset is linked, with the processed copies the
code reads, in [`datasets/README.md`](./datasets/README.md).

| Dataset | Raw data | Processed data |
|---|---|---|
| [SPAC-seq CRISPR screen](https://doi.org/10.1016/j.cell.2026.04.049) | *pending release* | *pending release* |
| [Perturb-Map](https://www.cell.com/cell/fulltext/S0092-8674(22)00195-7) | [GEO GSE193460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460) | [Figshare](https://figshare.com/articles/dataset/Datasets_-_Perturb-Map/29198468) |
| [Colitis MERFISH time course](https://doi.org/10.1016/j.cell.2024.03.013) | *pending release* | *pending release* |
| [Colon inflammation](https://www.nature.com/articles/s41586-024-08216-z) | [Broad SCP2771](https://singlecell.broadinstitute.org/single_cell?type=study&page=1&terms=SCP2771) | [Figshare](https://figshare.com/articles/dataset/Mouse_gut_inflammation_dataset/29882873) |
| [Brain stroke](https://www.science.org/doi/abs/10.1126/scitranslmed.adg1323) | From the authors of the original paper on request | [Figshare](https://figshare.com/articles/dataset/Mouse_brain_stroke_dataset/29882900) |

Trained checkpoints are not included; every result can be regenerated from the scripts.

## ⚖️ License

The code in this package is licensed under the MIT License. See [`LICENSE`](./LICENSE).

## 📝 Reference

```bibtex
@article{Lin2025.11.08.686890,
  author  = {Lin, Xiang and Kong, Zhenglun and Ghosh, Soumya and Kellis, Manolis and Zitnik, Marinka},
  title   = {CONCERT predicts niche-aware perturbation responses in spatial transcriptomics},
  elocation-id = {2025.11.08.686890},
  year    = {2025},
  doi     = {10.1101/2025.11.08.686890},
  publisher = {Cold Spring Harbor Laboratory},
  URL     = {https://www.biorxiv.org/content/early/2025/11/10/2025.11.08.686890},
  journal = {bioRxiv}
}
```

## 💬 Questions

Please leave a GitHub issue or contact [Xiang Lin](mailto:xianglin226@gmail.com) or
[Marinka Zitnik](mailto:marinka@zitnik.si).
