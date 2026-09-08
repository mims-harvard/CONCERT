# 07 — Mouse stroke case study (photothrombosis)

Added 2026-08-30. Everything behind the stroke case study, including the diagnostics that
overturned three earlier readings. Read the traps below before re-running anything.

```
prep/         building the CONCERT slides and every counterfactual perturbation set
figures/      the manuscript figures and their statistics
diagnostics/  the investigations; several are the record of a retraction
train_eval/   SLURM submitters for training, evaluation and the figure jobs
```

## Which arms are which

| output dir | what it is |
|---|---|
| `stroke_2d` / `stroke_3d` | one perturbation label for the whole lesion. **Cannot express core vs rim** |
| `stroke_2d_cr` / `stroke_3d_cr` | ICA and PIA as SEPARATE perturbations (`prep_stroke_h5ad.py --pert_by_region`). This is the arm that works |
| `stroke_2dpara_fixC` | the condition demos (lesion size, region). Paracrine REMOVED |
| `*_i64`, `*_i256` | inducing-grid retrains, 2026-08-30. Informative, not adopted |

## Traps, all of them measured

1. **Paracrine spillover was the entire injury signal** in the original condition panels. With
   `use_paracrine=False` the response went to zero (Vim +2.91 -> -0.06). Bounding and normalising
   it did not help; only removing it did (`stroke_2dpara_fixC`). The lesion-size dose-response
   disappeared with it, and that is correct: per-spot induction is close to size-invariant, and
   only the INTEGRATED response scales with lesion size.
2. **`f_pert` is gated by `is_real_KO`** (`concert_spac.py`, the `is_real_KO.unsqueeze(1) * shift`
   line). An unperturbed spot receives EXACTLY zero. Any test of whether a lesion propagates to
   unperturbed tissue or to an unperturbed section returns 0.0000 for both the 2D and the 3D model
   and always will. The 3D kernel is z-as-a-third-coordinate, not propagation.
3. **The core/rim contrast comes from the perturbation LABEL, not from the spatial field.** The
   single-label arm is flat inside an imposed lesion (Lcn2 core 0.801 / rim 0.789). The `_cr` arm
   gets 6/7 signs right including Lcn2 rim-high. Three mechanisms were proposed for the flatness
   and all three were refuted by measurement: niche-descriptor dilution (the descriptor reads
   neighbour latent states, never perturbation labels), too-long lengthscale (PIA's is 0.0878,
   0.28x the lesion), and vanishing RBF features (max feature 0.60-0.98, never near zero).
   A finer inducing grid raises within-lesion field variation 19x but does not create a core/rim
   contrast, and it makes the 2D sign table worse.
4. **Rim thickness is not a lever.** The imposed rim already matches the real one (6.87 vs 6.63
   spot-rings). Dilating it changes nothing per spot; the apparent decay of the observed-shaped
   rim is a coverage artifact, not a weaker response.
5. **Coordinates are PASTE2-aligned upstream.** `prep_stroke_h5ad.py` reads
   `benchmark_artifacts/arms/paste2/coordinates.parquet` from the registration delivery; no
   alignment happens inside CONCERT. Overlap fractions differ per pair (0.3 to 0.8).

## Open

The 3D core/rim result is unresolved. `stroke_2d_cr` reaches 6/7 signs; the best 3D arm
(`stroke_3d_cr_i64`) reached 5/6 in a single run, and across nine arms the score only moves
between 3/6 and 5/6 with which genes pass shuffling between them. Replicate across seeds before
making any 3D core/rim claim.
