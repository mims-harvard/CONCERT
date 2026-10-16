#!/usr/bin/env python3
"""Benchmark v4, Perturb-map seen: reproduction check, before/after tables, significance, wins, panel.

Input : multi_pm_ho_v4/pm_seen_v4_vs_v3_per_record.csv (rescore_pm_v4.py)
Checks: v3 scores against the reported files
  Outputs/pm_bench_bandI/pm_seen_lfc_dist_per_record.csv   (Fig. 4j source, sub_pm_rescore.sh)
  Outputs/fig5_panels/figure_stats.json                     (plot_pmap_benchmark.py)
  Outputs/spac_seq_lsfix/pm_seen_lfc_dist_per_record.csv, pm_seen_naive_baselines_our_rule.csv,
  pm_seen_state_our_rule.csv, perturbmap_seen_concert_vs_competitor_stats.csv (Supp. Tables 1-2)
Writes only new files, in multi_pm_ho_v4/ and Outputs/fig5_panels_v4/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v4_significance_tables as bst  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "02_perturb_seq_benchmark" / "figures"))
import plot_pmap_benchmark as ppb  # noqa: E402
import plot_pmap_benchmark_v4 as ppb4  # noqa: E402

CM = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main")
REP_BAND = CM / "Outputs/pm_bench_bandI/pm_seen_lfc_dist_per_record.csv"
REP_FIG = CM / "Outputs/fig5_panels/figure_stats.json"
LSFIX = CM / "Outputs/spac_seq_lsfix"
KEY = ["gene", "slide", "niche_key"]
FIG_METHODS = [m for m, _l, _c in ppb.METHODS]           # concert (bandI) + 5 published methods
COMPETITORS = ["biolord", "celcomen", "gears", "scgen", "steamboat",
               "mean_target", "mean_niche", "local_knn", "gex_knn", "state"]
LABEL = {"concert": "CONCERT", "concert_niche_seen": "CONCERT (niche ckpt)", "scgen": "scGEN",
         "biolord": "Biolord", "gears": "GEARS", "steamboat": "Steamboat", "celcomen": "CelComen",
         "mean_target": "Mean target", "mean_niche": "Mean niche", "local_knn": "Local kNN",
         "gex_knn": "GEX kNN", "state": "STATE"}


def maxdiff(a: pd.DataFrame, b: pd.DataFrame, cols_a, cols_b, name):
    j = a.merge(b, on=["method"] + KEY, how="outer", indicator=True)
    both = j[j["_merge"] == "both"]
    out = {"check": name, "n_reported": int((j["_merge"] != "left_only").sum()),
           "n_matched": int(len(both)), "n_reported_missing_here": int((j["_merge"] == "right_only").sum())}
    for ca, cb in zip(cols_a, cols_b):
        out[f"maxabs_{cb}"] = float(np.nanmax(np.abs(both[ca].astype(float) - both[cb].astype(float)))) if len(both) else np.nan
    return out


def per_record_frame(df: pd.DataFrame, ver: str) -> pd.DataFrame:
    x = df[["method", "method_label"] + KEY + ["n_scored_genes"]].copy()
    x["lfc_spearman"] = df[f"lfc_spearman_{ver}"]
    x["lfc_dist"] = df[f"lfc_dist_{ver}"]
    x["lfc_spearman_boot"] = df[f"lfc_spearman_boot_{ver}"]
    x["lfc_dist_boot"] = df[f"lfc_dist_boot_{ver}"]
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
    st = bst.regime_stats(r, "perturbmap_seen")   # same procedure as build_v4_significance_tables.py:66-111
    st["benchmark"] = "perturbmap_seen"
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=EVAL / "multi_pm_ho_v4")
    ap.add_argument("--figdir", type=Path, default=CM / "Outputs/fig5_panels_v4")
    args = ap.parse_args()
    out, figdir = args.indir, args.figdir
    figdir.mkdir(parents=True, exist_ok=True)
    assert "fig5_panels_v4" in str(figdir) and "multi_pm_ho_v4" in str(out)

    df = pd.read_csv(out / "pm_seen_v4_vs_v3_per_record.csv", dtype={"slide": str, "niche_key": str})
    v3 = per_record_frame(df[df.scored_v3], "v3")
    v4r = per_record_frame(df[df.scored_v4], "v4")

    # 1. reproduction of reported numbers with the v3 reference
    band = pd.read_csv(REP_BAND, dtype={"slide": str, "niche_key": str})
    b2 =band.rename(columns={c: c + "_r" for c in ["lfc_spearman", "lfc_dist", "lfc_spearman_boot", "lfc_dist_boot",
                                                    "method_label", "n_scored_genes"]})
    checks = [maxdiff(v3, b2, ["lfc_spearman", "lfc_dist", "lfc_spearman_boot", "lfc_dist_boot", "n_scored_genes"],
                      ["lfc_spearman_r", "lfc_dist_r", "lfc_spearman_boot_r", "lfc_dist_boot_r", "n_scored_genes_r"],
                      "Outputs/pm_bench_bandI/pm_seen_lfc_dist_per_record.csv")]
    for fname in ("pm_seen_lfc_dist_per_record.csv", "pm_seen_naive_baselines_our_rule.csv", "pm_seen_state_our_rule.csv"):
        rep = pd.read_csv(LSFIX / fname, dtype={"slide": str, "niche_key": str})
        rep = rep.rename(columns={c: c + "_r" for c in ["lfc_spearman", "lfc_dist", "method_label", "n_scored_genes"]})
        checks.append(maxdiff(v3, rep, ["lfc_spearman", "lfc_dist", "n_scored_genes"],
                              ["lfc_spearman_r", "lfc_dist_r", "n_scored_genes_r"], f"Outputs/spac_seq_lsfix/{fname}"))
    rows_v3 = v3.astype({"lfc_spearman": str, "lfc_dist": str}).to_dict("records")
    shared = ppb.restrict_to_shared_records(rows_v3, FIG_METHODS)
    stats_v3 = ppb.aggregate(rows_v3, shared)
    fig_rep = json.loads(REP_FIG.read_text())
    fig_diff = 0.0
    for p in ppb.PERTS:
        assert len(shared[p]) == fig_rep["n_shared_records"][p], (p, len(shared[p]))
        for m, lab, _c in ppb.METHODS:
            for k in ("rho", "rho_sem", "rmse", "rmse_sem", "n_records"):
                fig_diff = max(fig_diff, abs(stats_v3[(p, m)][k] - fig_rep["values"][p][lab][k]))
    checks.append({"check": "Outputs/fig5_panels/figure_stats.json", "maxabs_all_values": fig_diff})
    shared_set = {(g, s, k) for g in shared for (s, k) in shared[g]}
    rep_sig = pd.read_csv(LSFIX / "perturbmap_seen_concert_vs_competitor_stats.csv")
    st3n = sig(v3, "concert_niche_seen", shared_set)
    j = st3n.merge(rep_sig, on=["metric", "competitor_method"], suffixes=("", "_r"))
    sig_diff = {c: float(np.nanmax(np.abs(j[c] - j[c + "_r"]) / np.maximum(np.abs(j[c + "_r"]), 1e-300)))
                for c in ("concert_mean", "competitor_mean", "stat", "p", "p_adj")}
    checks.append({"check": "Outputs/spac_seq_lsfix/perturbmap_seen_concert_vs_competitor_stats.csv "
                            "(max relative diff)", "n_rows_matched": len(j), "n_pairs": sorted(set(j.n_pairs)), **sig_diff})
    lsfix_units = set(map(tuple, pd.read_csv(LSFIX / "pm_seen_naive_baselines_our_rule.csv",
                                             dtype={"slide": str, "niche_key": str})[KEY].drop_duplicates().to_numpy()))
    checks.append({"check": "Fig. 4j shared records == Supp. Table record units", "equal": shared_set == lsfix_units,
                   "n_fig": len(shared_set), "n_table": len(lsfix_units)})
    (out / "reproduction_check_v3.json").write_text(json.dumps(checks, indent=2, default=str))
    print("=== REPRODUCTION ===")
    for c in checks:
        print(c)

    # 2. status changes and overlap
    all_rec = df.copy()
    all_rec["in_shared"] = [(g, s, k) in shared_set for g, s, k in zip(all_rec.gene, all_rec.slide, all_rec.niche_key)]
    status = all_rec[(all_rec.scored_v3 != all_rec.scored_v4) | (all_rec.ref_scope_v4 != "niche")]
    status.to_csv(out / "pm_seen_v4_status_changes.csv", index=False)
    ov = (all_rec[all_rec.method == "concert"][KEY + ["n_src", "n_ref_v3", "ref_scope_v3", "n_src_in_ref_v3",
                                                      "n_role12_in_ref_v3", "frac_ref_v3_from_src",
                                                      "n_ref_v4", "ref_scope_v4", "n_ko", "ko_scope", "in_shared"]])
    ov.to_csv(out / "pm_seen_v4_reference_per_record.csv", index=False)
    print("\n=== STATUS CHANGES (scored v3 != v4, or v4 reference fell back to slide) ===")
    print(status[["method"] + KEY + ["scored_v3", "scored_v4", "n_ref_v3", "n_ref_v4", "ref_scope_v4", "in_shared"]]
          .to_string(index=False) if len(status) else "none")
    sh = ov[ov.in_shared]
    print(f"\nshared records: {len(sh)}; with source cells in v3 reference: {(sh.n_src_in_ref_v3 > 0).sum()}; "
          f"max share {sh.frac_ref_v3_from_src.max():.3f}; with role-1/2 cells in v3 reference: "
          f"{(sh.n_role12_in_ref_v3 > 0).sum()}; v4 scope counts {sh.ref_scope_v4.value_counts().to_dict()}")
    print(sh.to_string(index=False))

    # 3. per-record CSVs in the reported format
    model_ms = ["concert", "concert_niche_seen", "concert_v3_seen", "scgen", "biolord", "gears", "steamboat", "celcomen"]
    v4r[v4r.method.isin(model_ms)].to_csv(out / "pm_seen_lfc_dist_per_record_v4.csv", index=False)
    v4r[v4r.method.isin(["mean_target", "mean_niche", "local_knn", "gex_knn"])].to_csv(
        out / "pm_seen_naive_baselines_our_rule_v4.csv", index=False)
    v4r[v4r.method == "state"].to_csv(out / "pm_seen_state_our_rule_v4.csv", index=False)

    # 4. before/after summaries on the reported record set
    methods_all = ["concert", "concert_niche_seen"] + COMPETITORS
    s3 = summary(v3, shared_set, methods_all)
    s4 = summary(v4r, shared_set, methods_all)
    ba = s3.merge(s4, on=["method", "method_label", "gene"], suffixes=("_v3", "_v4"))
    ba["delta_rho"] = ba.lfc_spearman_v4 - ba.lfc_spearman_v3
    ba["delta_rmse"] = ba.lfc_dist_v4 - ba.lfc_dist_v3
    ba.to_csv(out / "pm_seen_v4_vs_v3_per_pert.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== BEFORE / AFTER (shared records) ===")
    print(ba[["method_label", "gene", "n_records_v3", "n_records_v4", "lfc_spearman_v3", "lfc_spearman_v4",
              "delta_rho", "lfc_dist_v3", "lfc_dist_v4", "delta_rmse", "lfc_spearman_sem_v4", "lfc_dist_sem_v4"]]
          .round(4).to_string(index=False))

    # 5. significance + wins
    sig_out = {}
    for tag, rec, cm in (("v4_bandI", v4r, "concert"), ("v4_nicheckpt", v4r, "concert_niche_seen"),
                         ("v3_bandI", v3, "concert"), ("v3_nicheckpt", v3, "concert_niche_seen")):
        st = sig(rec, cm, shared_set)
        st["concert_label"] = "CONCERT"
        st.to_csv(out / f"perturbmap_seen_concert_vs_competitor_stats_{tag}.csv", index=False)
        sig_out[tag] = st
        print(f"\n=== SIGNIFICANCE {tag} (CONCERT={cm}) ===")
        print(st[["metric", "competitor_label", "n_pairs", "concert_mean", "competitor_mean", "n_concert_better",
                  "n_concert_worse", "stat", "p", "p_adj", "p_concert_better_adj"]].to_string(index=False))
        for metric in ("lfc_spearman", "lfc_dist"):
            s = st[st.metric == metric]
            better = np.sign(s.mean_delta_concert_minus_comp) == (1 if metric == "lfc_spearman" else -1)
            win = s[(s.p_adj < 0.05) & better]
            print(f"  {metric}: significantly better (two-sided BH p<0.05, favourable mean) vs "
                  f"{len(win)}/{len(s)}; not: {s[~((s.p_adj < 0.05) & better)].competitor_label.tolist()}")
    for tag in ("v4_bandI", "v4_nicheckpt"):
        for metric, _n, _b, mtag in bst.METRICS:
            s = sig_out[tag]
            s = s[s.metric == metric].rename(columns={"competitor_label": "method"})
            comb = pd.DataFrame({"method": bst.ORDER}).merge(
                s[["method", "n_pairs", "p", "p_adj"]].rename(columns={"p": "perturbmap_seen_p", "p_adj": "perturbmap_seen_p_adj"}),
                on="method", how="left")
            comb.to_csv(out / f"concert_vs_competitor_{mtag}_stats_perturbmap_seen_{tag}.csv", index=False)

    # per-knockout wins on the panel means (figure methods and all methods)
    print("\n=== PER-KNOCKOUT RANKS (v4) ===")
    wins = []
    for gene in ppb.PERTS + ["ALL"]:
        for metric, asc in (("lfc_spearman", False), ("lfc_dist", True)):
            for scope, ms in (("figure", FIG_METHODS), ("all", ["concert"] + COMPETITORS)):
                s = s4[(s4.gene == gene) & s4.method.isin(ms)].sort_values(metric, ascending=asc)
                order = s.method_label.tolist()
                wins.append(dict(gene=gene, metric=metric, scope=scope, concert_rank=order.index("CONCERT") + 1,
                                 ranking=" > ".join(order)))
    wins = pd.DataFrame(wins)
    wins.to_csv(out / "pm_seen_v4_ranks.csv", index=False)
    print(wins.to_string(index=False))

    # 6. Fig. 4j panel with v4 numbers
    # Same method rows as the reported panel input (pm_bench_bandI holds model methods only), so the
    # y-limit rule in make_figure sees the same set of methods.
    rows_v4 = v4r[v4r.method.isin(set(band.method))].astype({"lfc_spearman": str, "lfc_dist": str}).to_dict("records")
    stats_v4 = ppb.aggregate(rows_v4, shared)
    for p in ppb.PERTS:
        for m in FIG_METHODS:
            assert stats_v4[(p, m)]["n_records"] == len(shared[p]), (p, m)
    ppb4.make_figure(stats_v4, shared, figdir, stem="pmap_benchmark_v4")
    payload = {"reference": "v4: role-0 control cells in the niche bin (slide-wide role-0 control if < 5)",
               "perturbations": ppb.PERTS, "n_shared_records": {p: len(shared[p]) for p in ppb.PERTS},
               "methods": [lab for _m, lab, _c in ppb.METHODS],
               "values": {p: {lab: stats_v4[(p, m)] for m, lab, _c in ppb.METHODS} for p in ppb.PERTS},
               "values_v3_reported": fig_rep["values"]}
    (figdir / "figure_stats_v4.json").write_text(json.dumps(payload, indent=2))
    lines = ["Perturb-map benchmark panel, benchmark v4 reference (pmap_benchmark_v4.pdf).", "",
             "Observed log-fold changes use as reference the unperturbed tumor spots of the same niche that lie "
             "outside every held-out and seen patch (slide-wide such spots when a niche holds fewer than five); "
             "the predicted spots are therefore never part of the reference.", "", "Values (mean +/- s.e.m.):"]
    for p in ppb.PERTS:
        lines.append(f"  {p} (n = {len(shared[p])} records)")
        for m, lab, _c in ppb.METHODS:
            s = stats_v4[(p, m)]
            o = fig_rep["values"][p][lab]
            lines.append(f"    {lab}: rank correlation {s['rho']:.3f} +/- {s['rho_sem']:.3f} (was {o['rho']:.3f}), "
                         f"RMSE {s['rmse']:.3f} +/- {s['rmse_sem']:.3f} (was {o['rmse']:.3f})")
    (figdir / "figure_legend_v4.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", figdir)


if __name__ == "__main__":
    main()
