# Benchmark evaluation

Scoring code shared by the SPAC-seq (experiment 01) and Perturb-seq (experiment 02)
benchmarks. It is kept in one place because both experiments score the same quantity in the
same way; only the data and the checkpoint differ.

## What is scored

For a held-out perturbation, the model is given a cell's location and its unperturbed
expression, and asked what that cell would look like carrying the perturbation. The prediction
is compared with the measured cells that actually carry it, through the log fold change over
matched unperturbed cells:

- **log fold change RMSE** — error in the size of the response, per perturbation and slide;
- **log fold change rank correlation** — agreement in which genes move and in which direction.

Both are computed per perturbation and slide, then averaged, with bootstrap resampling for the
reported spread.

## Entry point

`run_concert_v3_benchmark_eval.py` scores one slide of one checkpoint and writes
`per_pert_summary.json` under its output directory. The experiment run scripts submit it once
per slide. `build_benchmark_v3_summary.py` collects those files into the reported table.

Pass the same spatial-field lengthscale band that training used. It is a plain attribute of
the field rather than a saved weight, so leaving it at the default renders a different field
from identical weights.

## Modules

| File | Role |
|---|---|
| `run_concert_v3_benchmark_eval.py` | Entry point: loads a checkpoint, predicts held-out perturbations, writes scores |
| `benchmark_v3_lib.py` | Holdout definitions, pairing of perturbed and unperturbed cells |
| `benchmark_v3_eval_lib.py` | The metrics themselves and their bootstrap |
| `build_benchmark_v3_summary.py` | Collects per-slide results into the reported table |
| `eval_spatial_counterfactual.py` | Counterfactual prediction machinery the entry point builds on |
| `eval_cf_una_multi_holdout.py`, `eval_cf_una_only.py` | Variants scoring unperturbed-anchored predictions |
| `eval_utils.py` | Metric helpers shared across the above |
| `spacseq_eval_common.py`, `spacseq_multi_common.py` | SPAC-seq data loading, slide handling, matching |
| `perturbmap_common.py` | The same for the Perturb-seq sections |
| `method_input_lib.py`, `train_manifest_lib.py` | Assemble the inputs and manifests an evaluation run needs |
| `make_umap.py` | Embedding plots used to inspect a checkpoint |
