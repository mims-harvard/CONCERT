# Experiment 05 - does a predicted response depend on where the cell sits?

Experiment 02 asks whether held-out responses can be predicted. This one asks what the model
has actually learned about place: whether the same knockout, put into different parts of the
same section, is predicted to do different things, and whether the parameter that is supposed
to control that dependence really does.

Everything here runs on one Perturb-map section and on the within-niche checkpoint from
experiment 02; the checkpoint is used as trained.

## Inputs

- the checkpoint from `experiments/02_perturb_seq_benchmark/train/train.sh seen`;
- the measured section (`GSM5808054_data.h5`) and the spot selections that define the patches.

Paths are read from the environment (`PMAP_CKPT`, `PMAP_RAW`, `PMAP_SELECTIONS`, and the three
output directories); `evaluation/run_all.sh` sets them from `config/paths.sh` and runs
the four steps in order.

## Order

1. **`evaluation/case_study.py` - transplantation.** Take a patch of unperturbed spots,
   apply the same knockout to it, and read the predicted profile; then do it again with a patch
   in a different microenvironment. The comparison is between the two predictions and against
   the spots that carry that knockout for real.
2. **`evaluation/interpret_perturbations.py` - what the effect is made of, and how far it reaches.**
   Splits the predicted shift into a part that is the same everywhere, a part that varies with
   position, and a part that depends on the neighbourhood, and reports their relative sizes per
   knockout. It then measures how much an unedited spot moves when some of its neighbours are
   edited, **resolved by how many of its neighbours were edited**. That resolution is the whole
   experiment: a spot without edited neighbours stays fixed, and inside a dense patch of edited
   spots every neighbour is edited, so a before/after comparison across such a patch reads as a
   null whatever the model does. Resolving by neighbour count measures the effect where it can
   exist.
3. **`evaluation/transplant_analysis.py` - replication.** Places each knockout into many
   sampled neighbourhoods, so how much a response varies with its surroundings is measured
   across neighbourhoods as well as for a single pair.
4. **`evaluation/pathway_analysis.py` - which programmes track the neighbourhood.**
   Correlates the predicted per-gene response against the composition of the neighbourhood it
   was placed in, then tests gene sets on that ranking.

   **The amplitude and the shape must be separated or the result is an artefact.** The overall
   size of a knockout's effect is itself strongly related to the composition of the
   neighbourhood, and that single fact pushes every gene's correlation off zero in the same
   direction, after which gene-set testing reports a long list of sets that are all restating
   it. Each neighbourhood's response vector is therefore scaled to unit length and the per-gene
   correlations are centred before ranking, so the question asked is whether the *shape* of the
   response tracks the neighbourhood.

## Figures

`figures/plot_final_panels.py` writes the assembled panels; `figures/plot_panels.py`
writes the transplantation panels on their own. Both take their inputs from the directories the
evaluation step wrote and emit a `figure_stats.json` next to the panels, so a caption can be
checked against the numbers actually plotted.

The benchmark panel that appears in the same figure is produced by experiment 02
(`experiments/02_perturb_seq_benchmark/figures/plot_perturbmap_benchmark.py`).

## Reading the results

Two things are worth stating plainly when these numbers are used.

The response a knockout is predicted to produce is damped in neighbourhoods rich in tumour
content, and epithelial-to-mesenchymal transition is the programme whose response depends most
on the neighbourhood in every knockout tested. The first of these is a statement about effect
size and the second about effect shape; keeping them apart is what step 4 is for.

A perturbed spot does influence its neighbours, but the effect is graded and small - it rises
monotonically with the number of edited neighbours and reaches roughly one percent of the
expression change at full exposure. It is exactly zero for spots without edited neighbours,
which makes the gradient interpretable. It is best described as a graded, small effect.
