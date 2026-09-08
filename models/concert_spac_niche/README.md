# CONCERT - stroke case-study model

The model used for the stroke case study (experiment 07): a lesion of a chosen size and
position imposed on a healthy brain, predicted in a 2D arm and in a 3D arm that couples four
z levels of the section stack.

## Entry point

`run_concert_spac.py` is the only script meant to be called directly:

    python run_concert_spac.py --config config.yaml --stage train ...
    python run_concert_spac.py --config config.yaml --stage eval  --pert_cells <cells.txt> \
           --target_cell_tissue CTX --target_cell_perturbation ICA ...

`--stage eval` loads `model.pt` from `--outdir`, so it writes into the checkpoint directory
unless you give it a symlink shadow of one; `scripts/04_stroke_2d_3d_prediction.py` builds that
shadow for you and carries the full flag list the reported decodes used.

The 3D arm differs from the 2D arm only in `--fpert_n_z_levels` (4 rather than 1) and the z
gain; everything else is shared.

## Contents

Unlike `../concert_benchmark/` and `../concert_temporal/`, this directory is the **complete
working tree** of the stroke model, not a trimmed copy. Alongside the model itself
(`concert_spac.py`, `kernel.py`, `SVGP_Batch.py`, `nngp.py`, `VAE_utils.py`, `lord_batch.py`,
`I_PID.py`, `pert_action_denoiser.py`) and the preparation and evaluation entry points
(`preprocess.py`, `spatial_graph.py`, `eval_spatial_counterfactual.py`, `eval_utils.py`,
`deconv_impute.py`) it holds the diagnostics that were run against it - the `diag_*.py`,
`analyze_*.py`, `eval_*_patches.py` and gate scripts. Those are kept because they are the
record of how the model was checked, not because the case study needs them.

The three model directories are deliberate copies rather than one shared package. They diverged
in the model source itself, not only in configuration, and keeping them separate means a change
made for one result cannot silently alter another.
