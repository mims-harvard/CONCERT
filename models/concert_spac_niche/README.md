# CONCERT - stroke case-study model

The model used for the stroke case study (experiment 07) and the deconvolution and imputation
analyses (experiment 08). In the stroke case study a lesion of a chosen size and
position imposed on a healthy brain, predicted in a 2D arm and in a 3D arm that couples four
z levels of the section stack.

## Entry point

`run_concert_spac.py` is the only script meant to be called directly:

    python run_concert_spac.py --config config.yaml --stage train ...
    python run_concert_spac.py --config config.yaml --stage eval  --pert_cells <cells.txt> \
           --target_cell_tissue CTX --target_cell_perturbation ICA ...

`--stage eval` loads `model.pt` from `--outdir`, so it writes into the checkpoint directory
unless you give it a symlink shadow of one; `reproduction/04_stroke_2d_3d_prediction.py` builds that
shadow for you and carries the full flag list the reported decodes used.

The 3D arm (`--fpert_n_z_levels 4`) and the 2D arm (`--fpert_n_z_levels 1`) share every other
setting.

## Contents

This directory holds the **complete working tree** of the stroke model, while
`../concert_benchmark/` and `../concert_temporal/` hold trimmed copies. Alongside the model itself
(`concert_spac.py`, `kernel.py`, `SVGP_Batch.py`, `nngp.py`, `VAE_utils.py`, `lord_batch.py`,
`I_PID.py`, `pert_action_denoiser.py`) and the preparation and evaluation entry points
(`preprocess.py`, `spatial_graph.py`, `eval_utils.py`, `deconv_impute.py`) it holds two
helpers that `deconv_impute.py` imports (`decompose_levels.py`, `step0_leakage_gate.py`); the case
study itself runs from `run_concert_spac.py` alone.

The three model directories are deliberate copies. Their model sources differ as well as their
configurations, and keeping them separate means a change made for one result leaves the others
unchanged.

## Niche grid for the log-fold-change targets

**Per-slide grid.** With `--lfc_lambda > 0`, training
precomputes pseudo-bulk knockout log-fold-change targets per niche, binned on a 6x6 grid. The grid
is built within each slide: tiled multi-slide coordinates are folded back to the per-slide
`[0, loc_range]` frame, and each bin id is offset by the slide index, so every slide contributes
its own 36 niches. This matches the per-slide grid that evaluation uses (`--niche_grid 6`). The
same change is applied in `../concert_benchmark/run_concert_spac.py` and
`../concert_temporal/run_concert_spac.py`.
