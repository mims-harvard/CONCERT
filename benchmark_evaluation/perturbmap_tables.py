#!/usr/bin/env python3
"""Perturb-map seen benchmark: per-record tables, significance, per-knockout ranks and the panel.

Input : perturbmap_results/pm_seen_rescored_per_record.csv (score_perturbmap.py)
The shared record set (records every plotted method scored) is fixed by the harness reference
columns (*_harness); every reported value uses the disjoint reference (unsuffixed columns).
Writes the per-record CSVs, the significance tables and the ranks to --indir, and the Fig. 4j
panel to --figdir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import release_paths as rp  # noqa: E402

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import significance_tables as bst  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "02_perturb_map_benchmark" / "figures"))
import plot_perturbmap_benchmark as ppb  # noqa: E402
import plot_perturbmap_panel as ppbp  # noqa: E402

KEY = ["gene", "slide", "niche_key"]
FIG_METHODS = [m for m, _l, _c in ppb.METHODS]           # concert (bandI) + 5 published methods
MODEL_METHODS = ["concert", "concert_niche_seen", "scgen", "biolord", "gears", "steamboat", "celcomen"]
COMPETITORS = ["biolord", "celcomen", "gears", "scgen", "steamboat",
               "mean_target", "mean_niche", "local_knn", "gex_knn", "state"]
LABEL = {"concert": "CONCERT", "concert_niche_seen": "CONCERT (niche ckpt)", "scgen": "scGEN",
         "biolord": "Biolord", "gears": "GEARS", "steamboat": "Steamboat", "celcomen": "CelComen",
         "mean_target": "Mean target", "mean_niche": "Mean niche", "local_knn": "Local kNN",
         "gex_knn": "GEX kNN", "state": "STATE"}


def per_record_frame(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    x = df[["method", "method_label"] + KEY + ["n_scored_genes"]].copy()
    for c in ("lfc_spearman", "lfc_dist", "lfc_spearman_boot", "lfc_dist_boot"):
        x[c] = df[f"{c}{suffix}"]
    return x


def summary(rec: pd.DataFrame, shared: set, methods) -> pd.DataFrame:
    rows = []
    r = rec[[(g, s, k) in shared for g, s, k in zip(rec.gene, rec.slide, rec.niche_key)]]
    for m in methods:
        sub = r[r.method == m]
        for gene, gs in list(sub.groupby("gene")) + [("ALL", sub)]:
            n = len(gs)
            rows.append(dict(method=m, method_label=LABEL[m], gene=gene, n_records=n,
                             lfc_spearman=gs.lfc_spearman.mean(), lfc_dist=gs.lfc_dist.mean(),
                             lfc_spearman_sem=gs.lfc_spearman.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                             lfc_dist_sem=gs.lfc_dist.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan))
    return pd.DataFrame(rows)


def sig(rec: pd.DataFrame, concert_method: str, shared: set) -> pd.DataFrame:
    r = rec[[(g, s, k) in shared for g, s, k in zip(rec.gene, rec.slide, rec.niche_key)]]
    r = r[r.method.isin([concert_method] + COMPETITORS)].copy()
    r["method_label"] = r["method"].map(LABEL)
    st = bst.regime_stats(r, "perturbmap_seen")   # same procedure as significance_tables.py
    st["benchmark"] = "perturbmap_seen"
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=rp.WORKDIR / "perturbmap_results")
    ap.add_argument("--figdir", type=Path, default=None, help="Default: <indir>/figures")
    args = ap.parse_args()
    out = args.indir
    figdir = args.figdir or (out / "figures")
    figdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out / "pm_seen_rescored_per_record.csv", dtype={"slide": str, "niche_key": str})
    harness = per_record_frame(df[df.scored_harness], "_harness")
    rec = per_record_frame(df[df.scored], "")

    # 1. shared record set: the records every plotted method scored under the harness reference
    rows_h = harness.astype({"lfc_spearman": str, "lfc_dist": str}).to_dict("records")
    shared = ppb.restrict_to_shared_records(rows_h, FIG_METHODS)
    shared_set = {(g, s, k) for g in shared for (s, k) in shared[g]}
    print("shared records:", {p: len(shared[p]) for p in ppb.PERTS})

    # 2. per-record CSVs
    rec[rec.method.isin(MODEL_METHODS)].to_csv(out / "pm_seen_lfc_dist_per_record.csv", index=False)
    rec[rec.method.isin(["mean_target", "mean_niche", "local_knn", "gex_knn"])].to_csv(
        out / "pm_seen_naive_baselines_our_rule.csv", index=False)
    rec[rec.method == "state"].to_csv(out / "pm_seen_state_our_rule.csv", index=False)
    summ = summary(rec, shared_set, ["concert", "concert_niche_seen"] + COMPETITORS)
    summ.to_csv(out / "pm_seen_per_pert.csv", index=False)

    # 3. significance + wins, for the reported checkpoint and for the niche checkpoint
    pd.set_option("display.width", 250)
    sig_out = {}
    for suffix, cm in (("", "concert"), ("_nicheckpt", "concert_niche_seen")):
        st = sig(rec, cm, shared_set)
        st["concert_label"] = "CONCERT"
        st.to_csv(out / f"perturbmap_seen_concert_vs_competitor_stats{suffix}.csv", index=False)
        sig_out[suffix] = st
        print(f"\n=== SIGNIFICANCE (CONCERT={cm}) ===")
        print(st[["metric", "competitor_label", "n_pairs", "concert_mean", "competitor_mean", "n_concert_better",
                  "n_concert_worse", "stat", "p", "p_adj", "p_concert_better_adj"]].to_string(index=False))
        for metric in ("lfc_spearman", "lfc_dist"):
            s = st[st.metric == metric]
            better = np.sign(s.mean_delta_concert_minus_comp) == (1 if metric == "lfc_spearman" else -1)
            win = s[(s.p_adj < 0.05) & better]
            print(f"  {metric}: significantly better (two-sided BH p<0.05, favourable mean) vs "
                  f"{len(win)}/{len(s)}; not: {s[~((s.p_adj < 0.05) & better)].competitor_label.tolist()}")
        for metric, _n, _b, mtag in bst.METRICS:
            s = st[st.metric == metric].rename(columns={"competitor_label": "method"})
            comb = pd.DataFrame({"method": bst.ORDER}).merge(
                s[["method", "n_pairs", "p", "p_adj"]].rename(columns={"p": "perturbmap_seen_p", "p_adj": "perturbmap_seen_p_adj"}),
                on="method", how="left")
            comb.to_csv(out / f"concert_vs_competitor_{mtag}_stats_perturbmap_seen{suffix}.csv", index=False)

    # 4. per-knockout ranks on the panel means (figure methods and all methods)
    print("\n=== PER-KNOCKOUT RANKS ===")
    wins = []
    for gene in ppb.PERTS + ["ALL"]:
        for metric, asc in (("lfc_spearman", False), ("lfc_dist", True)):
            for scope, ms in (("figure", FIG_METHODS), ("all", ["concert"] + COMPETITORS)):
                s = summ[(summ.gene == gene) & summ.method.isin(ms)].sort_values(metric, ascending=asc)
                order = s.method_label.tolist()
                wins.append(dict(gene=gene, metric=metric, scope=scope, concert_rank=order.index("CONCERT") + 1,
                                 ranking=" > ".join(order)))
    wins = pd.DataFrame(wins)
    wins.to_csv(out / "pm_seen_ranks.csv", index=False)
    print(wins.to_string(index=False))

    # 5. Fig. 4j panel
    rows = rec[rec.method.isin(MODEL_METHODS)].astype({"lfc_spearman": str, "lfc_dist": str}).to_dict("records")
    stats = ppb.aggregate(rows, shared)
    for p in ppb.PERTS:
        for m in FIG_METHODS:
            assert stats[(p, m)]["n_records"] == len(shared[p]), (p, m)
    ppbp.make_figure(stats, shared, figdir, stem="pmap_benchmark")
    payload = {"reference": "role-0 control cells in the niche bin (slide-wide role-0 control if < 5)",
               "perturbations": ppb.PERTS, "n_shared_records": {p: len(shared[p]) for p in ppb.PERTS},
               "methods": [lab for _m, lab, _c in ppb.METHODS],
               "values": {p: {lab: stats[(p, m)] for m, lab, _c in ppb.METHODS} for p in ppb.PERTS}}
    (figdir / "figure_stats.json").write_text(json.dumps(payload, indent=2))
    lines = ["Perturb-map benchmark panel (pmap_benchmark.pdf).", "",
             "Observed log-fold changes use as reference the unperturbed tumor spots of the same niche that lie "
             "outside every held-out and seen patch (slide-wide such spots when a niche holds fewer than five); "
             "the predicted spots are therefore never part of the reference.", "", "Values (mean +/- s.e.m.):"]
    for p in ppb.PERTS:
        lines.append(f"  {p} (n = {len(shared[p])} records)")
        for m, lab, _c in ppb.METHODS:
            s = stats[(p, m)]
            lines.append(f"    {lab}: rank correlation {s['rho']:.3f} +/- {s['rho_sem']:.3f}, "
                         f"RMSE {s['rmse']:.3f} +/- {s['rmse_sem']:.3f}")
    (figdir / "figure_legend.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", figdir)


if __name__ == "__main__":
    main()
