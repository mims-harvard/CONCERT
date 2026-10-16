#!/bin/bash
# Sort the figures of a sampled-benchmark run into <aggregation dir>/figures/, so the
# aggregation directory keeps the CSVs and the logs and the figures stay findable.
# Safe to re-run: it moves whatever is currently at the top level and leaves the rest alone.
set -eo pipefail
OUT=${1:?usage: organize_v4_sampled_figures.sh <aggregation dir>}
F=$OUT/figures
mkdir -p $F/per_knockout_bars $F/per_knockout_dots $F/unseen_all85_bars $F/overall_rank
for r in seen unseen; do
  for m in lfc_bars lfc_dist_bars; do
    for e in pdf png; do
      [ -f $OUT/spacseq_${r}_${m}.$e ] && mv -f $OUT/spacseq_${r}_${m}.$e $F/per_knockout_bars/
    done
  done
  for m in lfc_rho lfc_rmse; do
    for e in pdf png; do
      [ -f $OUT/spacseq_${r}_concert_vs_competitor_${m}_dots.$e ] && \
        mv -f $OUT/spacseq_${r}_concert_vs_competitor_${m}_dots.$e $F/per_knockout_dots/
    done
  done
done
for e in pdf png; do
  for m in lfc_bars lfc_dist_bars; do
    [ -f $OUT/unseen_overall_all85/spacseq_unseen_${m}.$e ] && \
      mv -f $OUT/unseen_overall_all85/spacseq_unseen_${m}.$e $F/unseen_all85_bars/
  done
done
for f in $OUT/spacseq_*_overall_rank_*.pdf $OUT/spacseq_*_overall_rank_*.png \
         $OUT/overall_rank_figure_stats.json $OUT/overall_rank_captions.md; do
  [ -f "$f" ] && mv -f "$f" $F/overall_rank/
done
echo "figures under $F"
ls -1 $F
