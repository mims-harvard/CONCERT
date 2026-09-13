"""Partial correlations for the perturb-FISH experiment-04 readout.

analyze_kernel_params_spac.py only writes raw Spearman correlations. The published
SPAC-seq claim rests on the partial correlations that merge_kernel_params_eval.py
computes, controlling for log10 cell number and the perturbation embedding norm.
perturb-FISH has no benchmark scores, so the merge step cannot be run on it; this
reproduces just the partial-correlation part on the per-perturbation table, using the
same rank-residualisation, and prints it next to the SPAC-seq numbers.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--params_csv", required=True)
ap.add_argument("--spac_eval_csv", default="")
ap.add_argument("--out_csv", required=True)
args = ap.parse_args()


def partial_spearman(x, y, controls):
    x = rankdata(x); y = rankdata(y)
    Cm = np.column_stack([rankdata(c) for c in controls] + [np.ones(len(x))])
    bx, *_ = np.linalg.lstsq(Cm, x, rcond=None)
    by, *_ = np.linalg.lstsq(Cm, y, rcond=None)
    rx, ry = x - Cm @ bx, y - Cm @ by
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan, np.nan
    return pearsonr(rx, ry)


TARGETS = ["ls_footprint", "g_scale_niche"]
SPATIAL = ["n_cells_total", "n_train_cells", "nnd_mean", "nni", "local_density",
           "n_patches", "n_patches_per_slide", "mean_patch_size", "max_patch_frac",
           "singleton_frac", "radius_gyration", "occupancy_bins", "occupancy_frac",
           "slide_frac", "emb_l2"]

df = pd.read_csv(args.params_csv)
df = df[df["n_cells_total"].notna()].copy()
df["log_n_cells"] = np.log10(df["n_cells_total"].astype(float))
print("perturbations with spatial covariates: %d" % len(df))

rows = []
for t in TARGETS:
    for c in SPATIAL:
        if c not in df.columns or c == t:
            continue
        keep = list(dict.fromkeys([t, c, "log_n_cells", "emb_l2"]))
        d = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 10 or d[c].nunique() < 3 or d[t].nunique() < 3:
            continue
        rho, pr_ = spearmanr(d[t], d[c])
        r, pp_ = pearsonr(d[t], d[c])
        ctrl = [] if c == "log_n_cells" else [d["log_n_cells"].to_numpy()]
        prho_n, pp_n = partial_spearman(d[t], d[c], ctrl) if ctrl else (np.nan, np.nan)
        ctrl2 = list(ctrl)
        if c != "emb_l2":
            ctrl2.append(d["emb_l2"].to_numpy())
        prho_ne, pp_ne = partial_spearman(d[t], d[c], ctrl2) if ctrl2 else (np.nan, np.nan)
        rows.append(dict(target=t, covariate=c, n=len(d), spearman_rho=rho, spearman_p=pr_,
                         pearson_r=r, pearson_p=pp_,
                         partial_rho_ctrl_ncells=prho_n, partial_p_ctrl_ncells=pp_n,
                         partial_rho_ctrl_ncells_embl2=prho_ne,
                         partial_p_ctrl_ncells_embl2=pp_ne))

cor = pd.DataFrame(rows)
out = []
for t, gg in cor.groupby("target"):
    gg = gg.sort_values("spearman_p").copy()
    m = len(gg)
    gg["q_bh"] = np.minimum.accumulate(
        (gg["spearman_p"].to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1.0)
    gg["q_bh_partial"] = np.nan
    v = gg["partial_p_ctrl_ncells_embl2"].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() > 0:
        o = np.argsort(v[ok]); mm = int(ok.sum())
        q = np.minimum.accumulate((v[ok][o] * mm / np.arange(1, mm + 1))[::-1])[::-1].clip(max=1.0)
        qq = np.empty(mm); qq[o] = q
        gg.loc[ok, "q_bh_partial"] = qq
    out.append(gg)
cor = pd.concat(out).sort_values(["target", "partial_p_ctrl_ncells_embl2"])
cor.to_csv(args.out_csv, index=False)

pd.set_option("display.width", 200)
print("\n=== perturb-FISH: raw vs partial Spearman (control: log10 n_cells, embedding L2) ===")
print(cor[["target", "covariate", "n", "spearman_rho", "q_bh",
           "partial_rho_ctrl_ncells", "partial_rho_ctrl_ncells_embl2",
           "q_bh_partial"]].to_string(index=False, float_format=lambda v: "%+.3f" % v))

if args.spac_eval_csv:
    sp = pd.read_csv(args.spac_eval_csv)
    key = ["target", "covariate"]
    for grp in ["spatial_bench87", "spatial_all547"]:
        s = sp[sp["group"] == grp][key + ["n", "spearman_rho",
                                          "partial_rho_ctrl_ncells_embl2", "q_bh_partial"]]
        m = cor[key + ["n", "spearman_rho", "partial_rho_ctrl_ncells_embl2",
                       "q_bh_partial"]].merge(s, on=key, suffixes=("_pfish", "_spac"))
        m["partial_sign_agrees"] = (np.sign(m["partial_rho_ctrl_ncells_embl2_pfish"]) ==
                                    np.sign(m["partial_rho_ctrl_ncells_embl2_spac"]))
        print("\n=== perturb-FISH vs SPAC-seq %s (partial rho) ===" % grp)
        print(m.to_string(index=False, float_format=lambda v: "%+.3f" % v))
        for t in TARGETS:
            mt = m[m["target"] == t]
            print("  %s: partial-rho sign agreement %d/%d" %
                  (t, int(mt["partial_sign_agrees"].sum()), len(mt)))
