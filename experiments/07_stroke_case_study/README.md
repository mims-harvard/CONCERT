# Experiment 07 - mouse stroke case study (photothrombosis)

A photothrombotic lesion is imposed on the healthy (sham) brain, and CONCERT predicts the
response of the sham tissue. The lesion geometry comes from the stroke animal, with a core
(ischemic area, ICA) and a rim (peri-infarct area, PIA). Two arms are reported, a 2D arm that
reads each section on its own and a 3D arm whose perturbation field couples the four sections of
the stack. The model code is `models/concert_spac_niche`.

```
prep/         builds the CONCERT slides and the perturbation spot sets
train_eval/   batch scripts for slide preparation, training, evaluation and figures
figures/      figure scripts and their statistics
diagnostics/  supporting checks of the model and the data, run through the check_*.sh scripts
```

## Reported arms

| Training output | Checkpoint | What it is |
|---|---|---|
| `stroke_2d_cr` | `checkpoints/stroke_2d_core_rim` | 2D arm: ICA and PIA as two separate perturbations, sections read independently (`--fpert_n_z_levels 1`) |
| `stroke_3d_cr` | `checkpoints/stroke_3d_core_rim` | 3D arm: the same labels, with the section index as a third coordinate of the perturbation field (`--fpert_n_z_levels 4`) |

`reproduction/04_stroke_2d_3d_prediction.py` and `notebooks/04_stroke_2d_and_3d_prediction.ipynb`
start from these two checkpoints.

## Scripts behind the reported results

1. **Slides.** `train_eval/prepare_slides.sh` runs `prep/prepare_slides.py --pert_by_region`,
   which writes one `.h5ad` per section (PT1-4, Sham1-4) to `${CONCERT_WORKDIR}/stroke3d/slides_cr`.
   The lesion spots carry their compartment label (ICA or PIA) as the perturbation, the
   section index is stored as the z coordinate, and the coordinates are the PASTE2-aligned ones
   read from `${CONCERT_DATA_DIR}/external/CONCERT/delivery/stroke-registration-benchmark/`.
2. **Train.** `sbatch train_eval/train.sh 2d` (and `3d`) trains on the compartment-labelled
   slides and writes `${CONCERT_WORKDIR}/outputs/stroke_{2d,3d}_cr`.
3. **Imposed lesion.** `prep/make_shape_matched_set.py` transfers the lesion mask of the stroke
   animal onto the sham sections, section by section, and writes the imposed core and rim spot
   sets (`shape_core.txt`, `shape_rim.txt`) to `${CONCERT_WORKDIR}/outputs/stroke_3d/shape_sets`.
4. **Predict.** `sbatch train_eval/evaluate_core_rim.sh 2d` (and `3d`) runs three decode passes
   over the sham sections: ICA on the imposed core, PIA on the imposed rim, and every sham spot
   left unperturbed as the matched baseline.
5. **Figures.** `sbatch train_eval/plot_core_rim.sh` runs `figures/plot_core_rim_patch.py` for
   both arms: observed and predicted gene maps per section, the rim-to-core depth profile and the
   core and rim bar panels, with a `figure_stats_patch_*.json` holding every plotted number.

## Notes on the model

- The perturbation term is applied to the spots that carry a perturbation
  (`models/concert_spac_niche/concert_spac.py`, the `is_real_KO.unsqueeze(1) * shift` line), so
  the predicted response lives on the imposed lesion spots. In the 3D arm the section index is a
  third coordinate of the perturbation field.
- The core and rim each have their own perturbation embedding, lengthscale and gene loadings,
  which is what lets the two compartments respond differently.
- The niche code is built from the expression latent and coordinates of the neighbours, so the
  core pass and the rim pass can be decoded separately and assembled per compartment.

## Other scripts

The remaining scripts in `train_eval/`, `prep/` and `figures/` run additional counterfactual
scenarios on the stroke model (lesion size, cortical depth, multiple lesions, lesion placement
on one section) and the supporting checks in `diagnostics/`.
