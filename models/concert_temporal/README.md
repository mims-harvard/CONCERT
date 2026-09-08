# CONCERT — temporal model

The model used for the DSS colitis time-course case study (experiment 03). It shares its
structure with `../concert_benchmark/` and adds time as an explicit input, so one model spans
the whole course instead of one model per timepoint.

## What is different

The response is written as three factors: an amplitude that varies with time, a spatial field
whose lengthscale also varies with time, and the perturbation effect itself. Time enters
through a Fourier embedding of the day, so the amplitude and the extent of the response can
rise and then resolve, and the model can be evaluated at a day it never saw.

Because each measured section carries exactly one day, the direction and size of the
inflammation response are not pinned down by the likelihood alone. Three parts of the model
fix them:

- a per-cell inflammation field, predicted from the local architecture and coordinates and
  tied to the observed marker score, so the size of the response follows measured inflammation
  rather than floating free;
- the field's own time input is held fixed, because a combination like "healthy architecture
  at day 9" is absent from the data and the field would otherwise be asked to extrapolate into
  it. Time still reaches the prediction through the spatial field;
- the sign of the effect is constrained to be positive on the inflammation marker panel, which
  is an up-regulated set by definition. The remaining genes stay unconstrained, since
  resolution genuinely lowers some of them.

A shared basal component is fitted across all sections so that differences between timepoints
are carried by the response rather than absorbed into per-section baselines.

## Entry point

`run_concert_spac.py`, as in the benchmark model. Two stages matter here:

- `--stage train` fits the model over the sections listed on the command line;
- `--stage loto_eval` predicts a held-out timepoint and scores it against the measured
  sections for that timepoint, alongside a linear-interpolation baseline.

Modules are the same as in `../concert_benchmark/README.md`; `concert_spac.py` and
`run_concert_spac.py` carry the additional time-conditioning code described above.
