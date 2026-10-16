#!/usr/bin/env python3
"""SPAC-seq benchmark significance tables: paired Wilcoxon signed-rank (CONCERT vs each competitor) + BH FDR.

Reconstructs the reported procedure (the procedure is
documented in CONCERT-main/Outputs/eval_benchmark_memory_20260810/docs/00_CHAT_MEMORY.md section 3
and in the columns of Outputs/spac_seq_lsfix/*_concert_vs_competitor_stats.csv):
  * pairing unit = (gene, slide, niche_key) shared by CONCERT and that competitor (paired per
    competitor, so n_pairs varies by method); gene-level means when no cell is shared (STATE unseen);
  * fewer than 3 pairs -> no test (p NaN, note "n<3");
  * two-sided p and one-sided p (CONCERT better), scipy.stats.wilcoxon defaults;
  * BH within each full per-regime stats file (all competitors x both metrics);
  * combined tables: SPAC seen / SPAC unseen / Perturb-map seen two-sided p and p_adj.
Perturb-map seen columns are copied from --pm_combined_dir.

Inputs in --indir: spac_{seen,unseen}_lfc_dist_per_record.csv (score_spacseq.py),
spac_{seen,unseen}_naive_baselines_our_rule.csv, spac_seen_state_our_rule.csv (optional).
Outputs in --outdir (never the manuscript folder unless --xlsx is pointed there explicitly).
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
from pathlib import Path

import release_paths as rp  # noqa: E402

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUTPUTS = Path(f"{_rp.WORKDIR}/paper/Outputs")
RESULTS = rp.WORKDIR / "spacseq_results"
METRICS = [("lfc_spearman", "LFC Spearman rho", "higher", "lfc_rho"), ("lfc_dist", "LFC RMSE", "lower", "lfc_rmse")]
MIN_PAIRS = 3  # shipped tables: n_pairs < 3 -> p NaN, note "n<3"
ORDER = ["Biolord", "CelComen", "CellAgentChat", "GEARS", "scGEN", "Steamboat",
         "Mean target", "Mean niche", "Local kNN", "GEX kNN", "STATE"]


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    n = len(q)
    order = np.argsort(q)
    ranked = q[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(ranked, 1.0)
    out[ok] = res
    return out


def load_records(indir: Path, regime: str) -> pd.DataFrame:
    frames = [pd.read_csv(indir / f"spac_{regime}_lfc_dist_per_record.csv")]
    for extra in (f"spac_{regime}_naive_baselines_our_rule.csv", f"spac_{regime}_state_our_rule.csv"):
        p = indir / extra
        if p.is_file():
            frames.append(pd.read_csv(p))
        else:
            print(f"[warn] missing {p}")
    df = pd.concat(frames, ignore_index=True)
    df["slide"] = df["slide"].astype(str)
    df["niche_key"] = df["niche_key"].astype(str)
    return df


def regime_stats(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    is_c = df["method"].astype(str).str.startswith("concert")
    concert = df[is_c]
    cmethod = str(concert["method"].iloc[0])
    key = ["gene", "slide", "niche_key"]
    rows = []
    comps = df[~is_c][["method", "method_label"]].drop_duplicates()
    for _, (m, label) in comps.iterrows():
        comp = df[(df["method"] == m)]
        for metric, name, better, _ in METRICS:
            unit = "gene_slide_niche"
            a = concert.groupby(key)[metric].mean()
            b = comp.groupby(key)[metric].mean()
            j = pd.concat([a.rename("c"), b.rename("k")], axis=1, join="inner").dropna()
            if len(j) == 0:
                # no shared (gene, slide, niche) cell, e.g. STATE unseen: pair on gene means
                unit = "gene"
                a = concert.groupby("gene")[metric].mean()
                b = comp.groupby("gene")[metric].mean()
                j = pd.concat([a.rename("c"), b.rename("k")], axis=1, join="inner").dropna()
            d = (j["c"] - j["k"]).to_numpy()
            row = dict(benchmark=f"spacseq_{regime}", metric=metric, metric_name=name, direction_better=better,
                       concert_method=cmethod, concert_label="CONCERT", competitor_method=m, competitor_label=label,
                       pair_unit=unit, n_pairs=len(j),
                       concert_mean=j["c"].mean() if len(j) else np.nan,
                       competitor_mean=j["k"].mean() if len(j) else np.nan,
                       mean_delta_concert_minus_comp=d.mean() if len(d) else np.nan,
                       median_delta_concert_minus_comp=np.median(d) if len(d) else np.nan)
            sgn = 1 if better == "higher" else -1
            row.update(n_concert_better=int((sgn * d > 0).sum()), n_concert_worse=int((sgn * d < 0).sum()),
                       n_tie=int((d == 0).sum()), test="wilcoxon_signed_rank")
            if len(d) < MIN_PAIRS:
                row.update(stat=np.nan, p=np.nan, p_concert_better=np.nan, note=f"n<{MIN_PAIRS}")
            elif np.any(d != 0):
                st = wilcoxon(j["c"], j["k"])
                alt = "greater" if better == "higher" else "less"
                row.update(stat=float(st.statistic), p=float(st.pvalue),
                           p_concert_better=float(wilcoxon(j["c"], j["k"], alternative=alt).pvalue), note="ok")
            else:
                row.update(stat=np.nan, p=np.nan, p_concert_better=np.nan, note="no_nonzero_pairs")
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_adj"] = bh(out["p"].to_numpy())
    out["p_concert_better_adj"] = bh(out["p_concert_better"].to_numpy())
    out["concert_significantly_better_fdr05"] = out["p_concert_better_adj"] < 0.05
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=RESULTS)
    ap.add_argument("--outdir", type=Path, default=RESULTS)
    ap.add_argument("--pm_combined_dir", type=Path, default=rp.HARNESS_TABLES / "perturbmap_combined")
    ap.add_argument("--xlsx", type=Path, default=RESULTS / "CONCERT_Supplementary_tables.xlsx")
    args = ap.parse_args()
    assert "manuscript" not in str(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    full = {}
    for regime in ("seen", "unseen"):
        st = regime_stats(load_records(args.indir, regime), regime)
        st.to_csv(args.outdir / f"spacseq_{regime}_concert_vs_competitor_stats.csv", index=False)
        full[regime] = st
        for metric, _, _, tag in METRICS:
            s = st[st["metric"] == metric][["competitor_label", "n_pairs", "stat", "p", "p_adj"]]
            s.rename(columns={"competitor_label": "method"}).to_csv(
                args.outdir / f"spacseq_{regime}_concert_vs_competitor_{tag}_stats_summary.csv", index=False)

    sheets = {}
    for metric, _, _, tag in METRICS:
        pm = pd.read_csv(args.pm_combined_dir / f"concert_vs_competitor_{tag}_stats_combined.csv")[
            ["method", "perturbmap_seen_p", "perturbmap_seen_p_adj"]]
        comb = pd.DataFrame({"method": ORDER})
        for regime in ("seen", "unseen"):
            s = full[regime][full[regime]["metric"] == metric].rename(columns={"competitor_label": "method"})
            comb = comb.merge(s[["method", "p", "p_adj"]].rename(columns={
                "p": f"spacseq_{regime}_p", "p_adj": f"spacseq_{regime}_p_adj"}), on="method", how="left")
        comb = comb.merge(pm, on="method", how="left")
        comb.to_csv(args.outdir / f"concert_vs_competitor_{tag}_stats_combined.csv", index=False)
        sheets[tag] = comb
        print(comb.to_string(index=False))

    cols = {"method": "Method", "spacseq_seen_p": "SPAC-seq seen: p", "spacseq_seen_p_adj": "SPAC-seq seen: p (BH)",
            "spacseq_unseen_p": "SPAC-seq unseen: p", "spacseq_unseen_p_adj": "SPAC-seq unseen: p (BH)",
            "perturbmap_seen_p": "Perturb-map seen: p", "perturbmap_seen_p_adj": "Perturb-map seen: p (BH)"}
    with pd.ExcelWriter(args.xlsx, engine="openpyxl") as xl:
        sheets["lfc_rho"].rename(columns=cols).to_excel(xl, sheet_name="Table 1 - LFC rho", index=False)
        sheets["lfc_rmse"].rename(columns=cols).to_excel(xl, sheet_name="Table 2 - LFC RMSE", index=False)
    print("wrote", args.xlsx)


if __name__ == "__main__":
    main()
