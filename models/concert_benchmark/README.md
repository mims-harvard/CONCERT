# CONCERT — benchmark model

The model used for both held-out perturbation benchmarks: SPAC-seq (experiment 01) and spatial
Perturb-seq (experiment 02). The two experiments differ in their settings, not in this code.

## Entry point

`run_concert_spac.py` is the only script meant to be called directly. It parses the settings,
builds the model, and runs one stage:

    python run_concert_spac.py --config config.yaml --stage train  ...

`--stage train` fits a model and writes a checkpoint; the evaluation stages are driven from
`benchmark_evaluation/`. Defaults live in `config.yaml` and every one can be overridden on the
command line. The experiment run scripts pass the settings used for the reported results.

## Modules

| File | Role |
|---|---|
| `concert_spac.py` | The model: encoders, latent structure, perturbation effect, likelihood, training loop |
| `kernel.py` | Covariance functions and the spatial fields built on them |
| `SVGP_Batch.py` | Sparse variational Gaussian process with inducing points, in minibatches |
| `VAE_utils.py` | Encoder and decoder building blocks |
| `lord_batch.py` | Per-cell latent table and its optimizer |
| `nngp.py` | Neighbour-graph smoothing of the latent field |
| `spatial_graph.py` | Neighbour graph construction from coordinates |
| `pert_action_denoiser.py` | Separates the intended perturbation from technical guide-assignment noise |
| `deconv_impute.py` | Spot deconvolution and imputation used by the spot-level readouts |
| `preprocess.py` | Loading, gene selection and normalization shared by all stages |
| `I_PID.py` | Controller that adapts the weight on the latent regularizer during training |

## How a prediction is formed

A cell's expression is decoded from a basal part and a perturbation part. The basal part is
amortized from the cell's own expression and a smooth spatial component, so tissue structure
does not leak into the perturbation term. The perturbation part is a learned per-perturbation
effect multiplied by a spatial field evaluated at the cell's coordinates, then passed through a
head conditioned on the cell's local microenvironment. That head carries a per-perturbation
gain, which is the quantity experiment 04 examines: it can shrink to a spatially uniform
response, or grow so the same perturbation acts differently in different neighbourhoods.

The spatial field's lengthscale is bounded to an interpretable band. Note that this band is a
plain attribute rather than a saved weight, so evaluation has to be given the same band that
training used — otherwise identical weights render a different field. The run scripts in
`experiments/` pass it in both places.
