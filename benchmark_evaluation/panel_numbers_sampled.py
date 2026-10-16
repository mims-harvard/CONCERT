#!/usr/bin/env python3
"""Headline numbers for the manuscript text from the sampled aggregation.

Prints, for both arms: the knockouts shown as panels in the bar figures, the panel-b macro
average per method over those knockouts, and the knockout-level win counts CONCERT reports.
"""
import sys
from pathlib import Path

import pandas as pd

EVAL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval")
OUT = EVAL / "v4_aggregate_sampled"
GF = EVAL / "multi_1-5_ho_v4_sampled"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_spac_benchmark_bars as pb  # noqa: E402

pb.SPAC_V3 = GF
pb.METHOD_SOURCES[:] = [(label, GF, sub) for label, _, sub in pb.METHOD_SOURCES]
# Same panel file the figures use (plot_spac_benchmark_v4.py sets this before selecting genes),
# so the knockouts here are the ones the bar panels show.
pb.SPAC_PANEL = EVAL / "multi_1-5" / "prepared" / "benchmark_v4" / "eval_panel.json"

concert_seen = pb.load_concert_by_gene(GF / "concert_clonegate", "seen")
seen_panel = pb.select_top_genes(concert_seen, GF, 8)
eval_units = pb.load_eval_units()
concert_unseen = pb.load_concert_by_gene(GF / "concert_clonegate", "unseen", eval_units)
unseen_panel = pb.select_unseen_genes(concert_unseen, eval_units, GF)[:8]


def load(regime):
    parts = [pd.read_csv(OUT / f"spac_{regime}_lfc_dist_per_record.csv"),
             pd.read_csv(OUT / f"spac_{regime}_naive_baselines_our_rule.csv")]
    state = OUT / f"spac_{regime}_state_our_rule.csv"
    if state.is_file():
        parts.append(pd.read_csv(state))
    return pd.concat(parts, ignore_index=True)


def report(regime, panel):
    df = load(regime)
    per_gene = df.groupby(["method_label", "gene"])[["lfc_spearman", "lfc_dist"]].mean().reset_index()
    print(f"\n===== {regime}: panel knockouts ({len(panel)}) = {panel}")
    sub = per_gene[per_gene.gene.isin(panel)]
    macro = sub.groupby("method_label")[["lfc_spearman", "lfc_dist"]].agg(["mean", "count"])
    macro.columns = ["rho", "n_rho", "rmse", "n_rmse"]
    print("--- panel-b macro over the displayed knockouts")
    print(macro.sort_values("rmse").round(4).to_string())

    print("--- CONCERT rank within each displayed knockout (1 = best)")
    for g in panel:
        row = sub[sub.gene == g]
        r_rho = int((row.lfc_spearman > float(row[row.method_label == "CONCERT"].lfc_spearman.iloc[0])).sum()) + 1
        r_rmse = int((row.lfc_dist < float(row[row.method_label == "CONCERT"].lfc_dist.iloc[0])).sum()) + 1
        print(f"  {g:<10} rho rank {r_rho}/{len(row)}   rmse rank {r_rmse}/{len(row)}")

    print(f"\n--- all-knockout win counts ({regime})")
    wide_r = per_gene.pivot(index="gene", columns="method_label", values="lfc_spearman")
    wide_d = per_gene.pivot(index="gene", columns="method_label", values="lfc_dist")
    for name, wide, better in (("rho", wide_r, "max"), ("rmse", wide_d, "min")):
        c = wide["CONCERT"]
        others = wide.drop(columns=["CONCERT"])
        # A knockout counts as a CONCERT win where the comparator returned a value; a
        # method that cannot predict that knockout leaves a NaN and is passed over.
        if better == "max":
            wins = (c.values[:, None] > others.values).sum()
            best = ((others.le(c, axis=0)) | others.isna()).all(axis=1).sum()
        else:
            wins = (c.values[:, None] < others.values).sum()
            best = ((others.ge(c, axis=0)) | others.isna()).all(axis=1).sum()
        pairs = others.notna().sum().sum()
        print(f"{name}: CONCERT best in {best}/{len(wide)} knockouts; "
              f"wins {wins} of {pairs} knockout-by-method comparisons")


report("seen", seen_panel)
report("unseen", unseen_panel)
