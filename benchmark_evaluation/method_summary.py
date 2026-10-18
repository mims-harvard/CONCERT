#!/usr/bin/env python3
"""Per-method mean +- SEM (across scored records) of LFC Spearman and LFC RMSE for the SPAC-seq benchmark,
plus CONCERT rank and CONCERT-vs-competitor stats merge."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
def load(d, regime, state=True):
    fr = [pd.read_csv(d / f"spac_{regime}_lfc_dist_per_record.csv")]
    for x in ("naive_baselines", "state"):
        p = d / f"spac_{regime}_{x}_our_rule.csv"
        if p.is_file() and (state or x != "state"):
            fr.append(pd.read_csv(p))
    return pd.concat(fr, ignore_index=True)
def sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def main(res: Path) -> None:
    RES = res
    out = []
    for regime in ("seen", "unseen"):
        df = load(RES, regime)
        st = pd.read_csv(RES / f"spacseq_{regime}_concert_vs_competitor_stats.csv")
        g = df.groupby("method_label")
        t = pd.DataFrame({
            "n_records": g["lfc_spearman"].apply(lambda s: int(np.isfinite(s).sum())),
            "n_genes": g["gene"].nunique(),
            "rho_mean": g["lfc_spearman"].mean(), "rho_sem": g["lfc_spearman"].apply(sem),
            "rmse_mean": g["lfc_dist"].mean(), "rmse_sem": g["lfc_dist"].apply(sem)}).reset_index()
        t["rank_rho"] = t["rho_mean"].rank(ascending=False, method="min").astype(int)
        t["rank_rmse"] = t["rmse_mean"].rank(ascending=True, method="min").astype(int)
        for metric, tag in (("lfc_spearman", "rho"), ("lfc_dist", "rmse")):
            s = st[st.metric == metric].set_index("competitor_label")
            t[f"{tag}_n_pairs"] = t["method_label"].map(s["n_pairs"])
            t[f"{tag}_q_two_sided"] = t["method_label"].map(s["p_adj"])
            t[f"{tag}_q_concert_better"] = t["method_label"].map(s["p_concert_better_adj"])
        t.insert(0, "regime", regime)
        out.append(t.sort_values("rho_mean", ascending=False))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(RES / "spacseq_method_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="aggregation directory (spac_*_lfc_dist_per_record.csv)")
    a = ap.parse_args()
    main(a.results_dir)
