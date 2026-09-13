# Experiment 08 - deconvolution and imputation

Analysis code for Supplementary Figs. 9-11: CONCERT used to unmix spots that pool several
perturbation states, and to fill in a contiguous patch of tissue removed from a slide. The
CONCERT model code these scripts call is `models/concert_spac_niche/deconv_impute.py`, reached
through `run_concert_spac.py --stage deconv` and `--stage impute_patch`.

The scripts keep the absolute lab paths of the runs that produced the figures, as described in
the main README. Each one states its inputs and outputs as path constants or `argparse` defaults
near the top of the file.

## Supplementary Fig. 9 - SPAC-seq deconvolution of pseudo-bulk spots

Slide 2-1 is aggregated into pseudo-bulk spots of about nine neighbouring cells, so every spot
carries a known mixture of perturbation identities. Four estimators recover the knockout fraction
of the mixed spots: NNLS and a learned unmixing head, each run once with empirical signatures
(means of pure spots) and once with CONCERT signatures (location-matched counterfactual profiles).

Run order:

1. `spot_aggregate_plots.py --mode plot1` partitions slide 2-1 into spots with a greedy
   non-overlapping kNN partition and writes `spot_plots/pseudobulk_2-1_wholeslide.h5ad` and
   `spot_plots/spot_composition_2-1_wholeslide.csv`.
2. `build_spot_only_assets.py --outdir spot_only_2-1` splits those spots into pure and mixture
   spots and writes `pure_spots.h5ad`, `mixture_spots.h5ad`, `spot_roles.csv`,
   `membership.npz` and `manifest.json`.
3. `concert_deconv_methods_v2.py --assets spot_only_2-1 --concert_npz <CONCERT signatures>`
   runs the four estimators on the mixture spots and writes `per_spot_v2.csv`,
   `summary_v2.json` and `run_meta_v2.json` into
   `Outputs/spac_seq/spot_only_2-1/deconv_methods_v2`.
4. `replot_task1_paper_figs.py` draws the bar, win-rate, calibration and slide panels from
   `per_spot_v2.csv` and `summary_v2.json`, and copies them to `Outputs/ext_exp`.
5. `replot_task1_paper_figs_4arm.py` rebuilds panel b with the four estimators side by side and
   writes `resp_deconv_spacseq.png`, `resp_deconv_spacseq.pdf` and
   `resp_deconv_spacseq_stats.json`.

## Supplementary Fig. 10 - Perturb-map periphery-spot deconvolution and sub-spots

On Perturb-map section 0, spots at the edge of the Jak2 knockout niche mix knockout, unperturbed
tumour and normal tissue. Each periphery spot is unmixed with CONCERT signatures queried at its
own location, split into typed sub-spots placed along its spatial neighbours, and every sub-spot
receives the CONCERT counterfactual profile of its type. The sub-spot profiles are scored against
the observed profiles of nearby spots of the same type.

Run order (checkpoint: the Perturb-map within-niche model, `outputs/ms_pmap_niche_seen`):

1. `sub_pmap_s0_jak2_concert_amortized.sh` runs `run_pmap_s0_jak2_concert_amortized.py`, which
   queries the checkpoint for the location-matched signatures, trains the unmixing head on
   synthetic mixtures and writes `location_signatures.npz`, `amortized_head.pt` and
   `summary.json` into `Outputs/spac_seq/pmap_s0_jak2_concert_amortized`.
2. `sub_cache_task3_concert_states.sh` runs `cache_task3_concert_state_signatures.py`, which
   adds the normal-tissue profile `mu_normal` for every periphery spot to
   `location_signatures.npz`, and then the three figure scripts in order:
   - `replot_task3_deconv_impute_subspots.py` writes `peri_deconv_impute_scores.csv`,
     `jak2_niche_subspots.csv`, `summary_deconv_impute.json`, the bar panel and the slide of
     sub-spots;
   - `replot_task3_umap_pure_vs_peri.py` writes the UMAP of observed spots, periphery spots and
     imputed sub-spot profiles;
   - `replot_task3_marker_slide.py --gene Fn1` and `--gene Spp1` write the marker slides.
3. `replot_task3_deconv_impute_subspots_panelc.py` recomputes panel c from
   `peri_deconv_impute_scores.csv`, checks the means against `summary_deconv_impute.json`, and
   writes `resp_deconv_perturbmap.png`, `resp_deconv_perturbmap.pdf` and
   `resp_deconv_perturbmap_stats.json`.

## Supplementary Fig. 11 - SPAC-seq patch imputation

On slide 2-2 a contiguous patch of cells is removed for each of five knockouts (Ccn1, Cttn,
H2-Ea, Iqgap1, Phgr1). CONCERT, Tangram, SpatialScope and SpaIM each predict the removed cells.
Every method is scored with the same observed reference: the log-fold change of a patch is taken
against the 50 nearest retained unassigned cells, for the true and for the predicted expression
alike.

Run order:

1. `spot_aggregate_plots.py --mode plot2` selects the patches and writes
   `spot_plots/patch_removal_2-2_top5_cellindex.npz` (cell index, knockout group and role of
   every experiment cell). All methods read this one mask.
2. `rebuild_masked_2-2_eval5.py` writes the CONCERT training file
   `data_mask/raw_output_2-2_masked_v2_eval5.h5ad`, which keeps the unassigned cells and the five
   knockouts and holds out the masked patches.
3. CONCERT is trained on that file and run with `--stage impute_patch --mask_index_file
   patch_removal_2-2_top5_cellindex.npz`, writing `task2_2-2_eval5_tune_lfc10/impute_results.npz`.
4. The three baselines, in `baselines/`, run from the same mask:
   - `prepare_spac_reference.py` builds the single-cell reference `Ref_SPAC_2-2.h5ad`, the masked
     input `sp_adata.h5ad`, `ground_truth.h5ad` and `mask_info.npz`;
   - `build_context_pool.py` writes `context_pool.h5ad`, the retained unassigned tissue shared by
     the Tangram and SpaIM neighbour pools;
   - SpatialScope: `Train_scRef.py` from the SpatialScope package trains the score model on
     `Ref_SPAC_2-2.h5ad`, then `run_spac_imputation.py` samples the masked cells;
   - Tangram: `run_tangram_imputation.py --use_ref --device cuda:0 --n_context 0`;
   - SpaIM: `run_spaim_imputation.py --device cuda:0 --epochs 300 --n_context 0`;
   - `compute_full_metrics.py --method <tangram|spaim>` recomputes the extended metrics with
     `perturbation_metrics.py`.
5. `make_task2_paper_figs.py` draws the patch-removal map (panel a) and the gene panels.
   `make_task2_paper_figs_sharedref.py` recomputes the true-versus-predicted log-fold changes with
   the shared reference and writes `task2_spac22_eval5_lfc10_lfc_scatter_sharedref_points.npz`.
6. `compose_resp_imputation_patch_sharedref.py` draws panel b from
   `task2_spac22_impute_rescored_our_rule.csv` (per-cell Spearman and log-fold-change Spearman for
   all four methods under the shared rule, error bars as s.e.m. across the five knockouts) and
   panel c from the scatter points, and writes `resp_imputation_patch.png`,
   `resp_imputation_patch.pdf` and `resp_imputation_patch_stats.json`.
