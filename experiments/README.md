# Experiments

Each subdirectory is one experiment and holds everything specific to it: data preparation, the
run scripts that produced the reported results, and any analysis or figure code. Shared model
code lives in `../models/`, shared scoring code in `../benchmark_evaluation/`.

| Directory | Question | Model |
|---|---|---|
| `01_spac_seq_benchmark/` | Can the response of a perturbation held out from a slide be predicted from location and unperturbed expression? | `models/concert_benchmark` |
| `02_perturb_seq_benchmark/` | The same question on spatial Perturb-map, a different assay with a different perturbation set | `models/concert_benchmark` |
| `03_dss_gut_case_study/` | Does one model over a colitis time course reproduce the tissue at a timepoint it was not trained on? | `models/concert_temporal` |
| `04_kernel_interpretability/` | Do the spatial parameters the model learns track the observed arrangement of the perturbed cells? | reads checkpoints from 01 |
| `05_perturb_seq_case_study/` | Does the same knockout, placed in different parts of a section, produce a different predicted response? | reads checkpoints from 02 |
| `06_sensitivity_analysis/` | Do the reported numbers survive changes to the hyper-parameters, once initialisation noise is measured? | retrains both benchmark models |
| `07_stroke_case_study/` | Can a lesion of a chosen size and position be imposed on a healthy brain and its response predicted, in 2D and in 3D? | `models/concert_spac_niche` |
| `08_deconvolution_imputation/` | Can CONCERT unmix spots that pool several perturbation states, and fill in a patch of tissue removed from a slide? | `models/concert_spac_niche` (`deconv_impute.py`) |

## Running one

Every run script sources `../../config/paths.sh` and takes its arguments on the command line,
so from inside an experiment directory:

    sbatch train/train_spac_seq.sh seen
    sbatch --array=1-5 evaluation/eval_spac_seq_benchmark.sh seen

Scripts read from the `data/` and `checkpoints/` folders and write under `CONCERT_WORKDIR` from `config/paths.sh` and never write back into this
repository. Run data preparation before training: the training scripts fail immediately with a
clear message if a required input is missing, rather than training on a partial set.

Experiments 01 and 02 are independent of each other and can run in either order. Experiment 04
needs a trained checkpoint from experiment 01 and experiment 05 one from experiment 02.
Experiments 03 and 07 are self-contained. Experiment 08 reads a Perturb-map within-niche
checkpoint (the alternative niche checkpoint, whose results ship as `data/benchmark_assets/results/perturbmap_concert/within_niche_alternative`) for Supplementary Fig. 10 and single-slide SPAC-seq
models of slides 2-1 and 2-2 for Figs. 9 and 11.

## Reproducing a result without rerunning the pipeline

Four of these experiments have a tutorial that starts from the trained checkpoint instead:
`../scripts/` as command-line scripts, `../notebooks/` as step-by-step notebooks.

| Experiment | Script | Notebook |
|---|---|---|
| `01_spac_seq_benchmark` | `../scripts/01_spac_seq_benchmark.py` | `../notebooks/01_spac_seq_benchmark.ipynb` |
| `02_perturb_seq_benchmark` | `../scripts/02_perturb_map_benchmark.py` | `../notebooks/02_perturb_map_benchmark.ipynb` |
| `03_dss_gut_case_study` | `../scripts/03_dss_colitis_marker_trend.py` | `../notebooks/03_dss_colitis_per_mouse_marker_trend.ipynb` |
| `07_stroke_case_study` | `../scripts/04_stroke_2d_3d_prediction.py` | `../notebooks/04_stroke_2d_and_3d_prediction.ipynb` |
