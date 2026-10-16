#!/usr/bin/env python3
"""Paired aggregate of the v4 UNSEEN arm under the sampled-output rule.

CONCERT: clonegate ckpt, --cf_output poisson_depth (5 Poisson draws: poisson, poisson_s1..s4).
Competitors: multi_1-5_ho_v4_sampled/<m>/unseen (--sampled_output True, one draw, n_boot 10).
Deterministic competitors (genefix) are listed for reference.
Naive baselines: reference_strategies CSV (rho and lfc_dist only).
All comparisons are paired over the units every compared source shares.
"""
import json, glob, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

E = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval"
CG = f"{E}/multi_1-5_ho_v4_clonegate"
SAMP = f"{E}/multi_1-5_ho_v4_sampled"
GF = f"{E}/multi_1-5_ho_v4_genefix"
COMP = ["biolord", "gears", "scgen", "celcomen", "steamboat", "cellagentchat"]
CONCERT_DRAWS = ["poisson", "poisson_s1", "poisson_s2", "poisson_s3", "poisson_s4"]


def load_dir(root):
    out = {}
    for f in sorted(glob.glob(f"{root}/unseen_pert/*/lfc_scores.json")):
        unit = os.path.basename(os.path.dirname(f))
        for g, recs in json.load(open(f))["L2"].items():
            for r in recs:
                out[f"{unit}|{g}|{r.get('niche', -1)}"] = r
    return out


def to_df(d, name):
    return pd.DataFrame({k: {"rho": r["spearman"], "lfc_dist": r["lfc_dist"],
                             "e_dist": r.get("e_dist", np.nan)} for k, r in d.items()}).T.assign(src=name)


def main():
    srcs = {}
    draws = [to_df(load_dir(f"{CG}/eval_{t}/concert_clonegate/unseen"), t) for t in CONCERT_DRAWS]
    keys = sorted(set.intersection(*[set(x.index) for x in draws]))
    srcs["CONCERT"] = sum(x.loc[keys, ["rho", "lfc_dist", "e_dist"]] for x in draws) / len(draws)
    srcs["CONCERT draw0"] = draws[0].loc[keys, ["rho", "lfc_dist", "e_dist"]]
    for m in COMP:
        srcs[f"{m} sampled"] = to_df(load_dir(f"{SAMP}/{m}/unseen"), m)[["rho", "lfc_dist", "e_dist"]]
        gf = f"{GF}/{m}/unseen"
        if os.path.isdir(gf):
            srcs[f"{m} determ"] = to_df(load_dir(gf), m)[["rho", "lfc_dist", "e_dist"]]
    nv = pd.read_csv(f"{SAMP}/reference_strategies/unseen/spac_unseen_naive_baselines_our_rule.csv")
    nv["key"] = nv.gene + "__slide_" + nv.slide.astype(str) + "|" + nv.gene + "|" + nv.niche_key.astype(str)
    for lab, sub in nv.groupby("method_label"):
        srcs[lab] = pd.DataFrame({"rho": sub.lfc_spearman.values, "lfc_dist": sub.lfc_dist.values,
                                  "e_dist": np.nan}, index=sub.key.values)

    ref = srcs["CONCERT"]
    rows = []
    for name, df in srcs.items():
        common = sorted(set(df.index) & set(ref.index))
        a, b = df.loc[common].astype(float), ref.loc[common].astype(float)
        row = {"method": name, "n": len(common)}
        for met in ["rho", "lfc_dist", "e_dist"]:
            row[met] = a[met].mean()
            if name == "CONCERT" or a[met].isna().all():
                continue
            d = (b[met] - a[met]).values if met == "rho" else (a[met] - b[met]).values  # >0 = CONCERT better
            nz = d[d != 0]
            row[f"{met}_CONCERT_wins"] = f"{int((d > 0).sum())}/{len(d)}"
            row[f"{met}_p"] = wilcoxon(nz).pvalue if len(nz) else np.nan
            row[f"{met}_CONCERT_ref"] = b[met].mean()
        rows.append(row)
    res = pd.DataFrame(rows)
    for met in ["rho", "lfc_dist", "e_dist"]:
        m = res[f"{met}_p"].notna()
        res.loc[m, f"{met}_padj"] = multipletests(res.loc[m, f"{met}_p"], method="fdr_bh")[1]
    out = f"{SAMP}/SAMPLED_UNSEEN_AGG.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cols = ["method", "n", "rho", "rho_CONCERT_wins", "rho_padj", "lfc_dist", "lfc_dist_CONCERT_wins",
            "lfc_dist_padj", "e_dist", "e_dist_CONCERT_wins", "e_dist_padj"]
    print(res[cols].round(4).to_string(index=False))
    print(f"\nCONCERT = mean of {len(draws)} Poisson draws over {len(keys)} units; wrote {out}")


if __name__ == "__main__":
    main()
