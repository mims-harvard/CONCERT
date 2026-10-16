"""Stage 2 for the per-perturbation kernel-parameter analysis.

Adds three things to analyze_kernel_params_spac.py's output:

 1. EFFECTIVE footprint statistics, computed from the actual learned field rather than from
    the raw lengthscale. The gene-averaged f_pert residual is evaluated on a grid over one
    slide's normalized coordinate box:
        feat(c)  = exp(-||c - Z||^2 / (2 ls(p)^2))          (Z = fpert_field.Z, 16 anchors)
        h(c)     = relu(W0 @ [e_p ; feat(c)] + b0)
        fbar(c)  = mean_g W2[g,:] @ h(c) + mean_g b2        (gene-mean, exact and cheap)
    -> field_sd / field_range (how much the perturbation response actually varies in space)
    -> eff_lengthscale: half-decay distance of the field's spatial autocovariance, in
       loc units (slide width = 20), i.e. the footprint the model really uses.
 2. Per-perturbation within-slide ("seen") benchmark accuracy joined on, so the learned
    parameters can be tested against LFC rho / LFC RMSE for the 87 scored perturbations.
 3. PARTIAL Spearman correlations controlling for training exposure (log n_cells and the
    embedding L2 norm), because ls(p) and g(p) are both monotone functions of a single linear
    projection of the perturbation embedding and embedding norm grows with cell count.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOC_RANGE = 20.0
LS_MIN, LS_MAX = 3.0, 18.0
G_CAP = 3.0

ap = argparse.ArgumentParser()
ap.add_argument("--params_csv", required=True)
ap.add_argument("--model_file", required=True)
ap.add_argument("--eval_dir", required=True)
ap.add_argument("--level", default="L2")
ap.add_argument("--grid", type=int, default=60)
ap.add_argument("--outdir", required=True)
# Only used to draw the band guides on the ls histogram; see analyze_kernel_params_spac.py.
ap.add_argument("--ls_min", type=float, default=3.0)
ap.add_argument("--ls_max", type=float, default=18.0)
args = ap.parse_args()
LS_MIN, LS_MAX = args.ls_min, args.ls_max
os.makedirs(args.outdir, exist_ok=True)

params = pd.read_csv(args.params_csv)

# ------------------------------------------------- 1. effective footprint stats
sd = torch.load(args.model_file, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
emb = sd["lord_encoder.s_encoder.perturbation.weight"].float()      # (K, E)
Z = sd["fpert_field.Z"].float()                                     # (M, 2)
W0 = sd["fpert_field.mlp.0.weight"].float(); b0 = sd["fpert_field.mlp.0.bias"].float()
W2 = sd["fpert_field.mlp.2.weight"].float(); b2 = sd["fpert_field.mlp.2.bias"].float()
w2bar = W2.mean(0)                                                  # (H,) gene-mean readout
b2bar = float(b2.mean())
K, E = emb.shape

g = torch.linspace(0, 1, args.grid)
gx, gy = torch.meshgrid(g, g, indexing="ij")
C = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)            # (P,2) unit box
d2 = torch.cdist(C, Z) ** 2                                         # (P,M)
D = torch.cdist(C, C)                                               # (P,P) normalized units
dr = D.reshape(-1)
NB = 30
edges = torch.linspace(0, float(dr.max()) + 1e-6, NB + 1)
bin_sel = [((dr >= edges[k]) & (dr < edges[k + 1])) for k in range(NB)]

ls_arr = torch.tensor(params.sort_values("pert_code")["ls_footprint"].to_numpy(), dtype=torch.float32)
order = params.sort_values("pert_code").index
rows = []
with torch.no_grad():
    for i, idx in enumerate(order):
        ls_i = float(ls_arr[i])
        feat = torch.exp(-d2 / (2.0 * ls_i * ls_i))                 # (P,M)
        e = emb[int(params.loc[idx, "pert_code"])].unsqueeze(0).expand(C.shape[0], -1)
        h = torch.relu(torch.cat([e, feat], dim=-1) @ W0.t() + b0)  # (P,H)
        fbar = h @ w2bar + b2bar                                    # (P,) gene-mean field
        v = fbar - fbar.mean()
        cov0 = float((v * v).mean())
        eff = float("nan")
        if cov0 > 1e-14:
            cc = ((v.unsqueeze(1) * v.unsqueeze(0)).reshape(-1) / cov0)
            for k in range(NB):
                s = bin_sel[k]
                if bool(s.any()) and float(cc[s].mean()) < 0.5:
                    eff = 0.5 * float(edges[k] + edges[k + 1]) * LOC_RANGE
                    break
            else:
                eff = float(dr.max()) * LOC_RANGE
        rows.append(dict(
            pert_code=int(params.loc[idx, "pert_code"]),
            field_mean=float(fbar.mean()), field_sd=float(fbar.std()),
            field_range=float(fbar.max() - fbar.min()),
            field_cv=float(fbar.std() / (fbar.abs().mean() + 1e-12)),
            eff_lengthscale=eff,
        ))
eff_df = pd.DataFrame(rows)
params = params.merge(eff_df, on="pert_code", how="left")
print(f"[field] gene-mean field on {args.grid}x{args.grid} grid for {len(eff_df)} perturbations")
print(params[["field_sd", "field_range", "field_cv", "eff_lengthscale"]].describe().to_string())

# 2. within-slide benchmark accuracy
recs = []
for f in sorted(glob.glob(os.path.join(args.eval_dir, "slide_1-*", "per_pert_summary.json"))):
    sid = os.path.basename(os.path.dirname(f)).replace("slide_", "")
    for pert, lv in json.load(open(f)).items():
        if args.level not in lv:
            continue
        s = lv[args.level]
        recs.append(dict(perturbation=pert, sid=sid,
                         lfc_rho=s.get("spearman_mean"), lfc_rmse=s.get("lfc_dist_mean"),
                         sign_agreement=s.get("sign_agreement_mean"),
                         topk_overlap=s.get("topk_overlap_mean"),
                         abs_rho=s.get("abs_spearman_mean"), n_sig=s.get("n_sig_mean"),
                         n_niches=s.get("n_niches"), n_scored_genes=s.get("n_scored_genes")))
ev = pd.DataFrame(recs)
print(f"[eval] {len(ev)} (pert, slide) records; {ev['perturbation'].nunique()} perturbations")
agg = ev.groupby("perturbation").agg(
    lfc_rho=("lfc_rho", "mean"), lfc_rmse=("lfc_rmse", "mean"),
    sign_agreement=("sign_agreement", "mean"), topk_overlap=("topk_overlap", "mean"),
    abs_rho=("abs_rho", "mean"), n_sig=("n_sig", "mean"),
    n_niches=("n_niches", "mean"), n_scored_genes=("n_scored_genes", "mean"),
    n_eval_slides=("sid", "nunique")).reset_index()

df = params.merge(agg, on="perturbation", how="left")
df["in_benchmark"] = df["perturbation"].isin(agg["perturbation"]).astype(int)
df["log_n_cells"] = np.log10(df["n_cells_total"].astype(float))
df.to_csv(os.path.join(args.outdir, "per_perturbation_kernel_params_eval.csv"), index=False)


# --------------------------------------------------- 3. correlations + partials
def partial_spearman(x, y, controls):
    """Spearman partial correlation: rank-transform all, OLS-residualise x and y on the
    controls (with intercept), then Pearson on the residuals."""
    x = rankdata(x); y = rankdata(y)
    Cm = np.column_stack([rankdata(c) for c in controls] + [np.ones(len(x))])
    bx, *_ = np.linalg.lstsq(Cm, x, rcond=None)
    by, *_ = np.linalg.lstsq(Cm, y, rcond=None)
    rx, ry = x - Cm @ bx, y - Cm @ by
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan, np.nan
    return pearsonr(rx, ry)


TARGETS = ["ls_footprint", "g_scale_niche", "eff_lengthscale", "field_sd"]
SPATIAL = ["n_cells_total", "n_train_cells", "n_slides_present", "nnd_mean", "nni",
           "local_density", "n_patches", "n_patches_per_slide", "mean_patch_size",
           "max_patch_frac", "singleton_frac", "radius_gyration", "occupancy_bins",
           "occupancy_frac", "slide_frac", "emb_l2"]
ACC = ["lfc_rho", "lfc_rmse", "sign_agreement", "topk_overlap", "abs_rho", "n_sig",
       "n_niches", "n_scored_genes"]

rows = []
for t in TARGETS:
    for grp, cols, sub in (("spatial_all547", SPATIAL, df),
                           ("spatial_bench87", SPATIAL, df[df.in_benchmark == 1]),
                           ("accuracy_bench87", ACC, df[df.in_benchmark == 1])):
        for c in cols:
            if c not in sub.columns or c == t:
                continue
            keep = list(dict.fromkeys([t, c, "log_n_cells", "emb_l2"]))
            d = sub[keep].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 10 or d[c].nunique() < 3 or d[t].nunique() < 3:
                continue
            rho, pr_ = spearmanr(d[t], d[c])
            r, pp_ = pearsonr(d[t], d[c])
            ctrl = []
            if c != "log_n_cells":
                ctrl.append(d["log_n_cells"].to_numpy())
            prho_n, pp_n = partial_spearman(d[t], d[c], ctrl) if ctrl else (np.nan, np.nan)
            ctrl2 = list(ctrl)
            if c != "emb_l2":
                ctrl2.append(d["emb_l2"].to_numpy())
            prho_ne, pp_ne = partial_spearman(d[t], d[c], ctrl2) if ctrl2 else (np.nan, np.nan)
            rows.append(dict(group=grp, target=t, covariate=c, n=len(d),
                             spearman_rho=rho, spearman_p=pr_, pearson_r=r, pearson_p=pp_,
                             partial_rho_ctrl_ncells=prho_n, partial_p_ctrl_ncells=pp_n,
                             partial_rho_ctrl_ncells_embl2=prho_ne,
                             partial_p_ctrl_ncells_embl2=pp_ne))
cor = pd.DataFrame(rows)
out = []
for (grp_, t), gg in cor.groupby(["group", "target"]):
    gg = gg.sort_values("spearman_p").copy()
    m = len(gg)
    gg["q_bh"] = np.minimum.accumulate(
        (gg["spearman_p"].to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1.0)
    gg["q_bh_partial"] = np.nan
    v = gg["partial_p_ctrl_ncells_embl2"].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() > 0:
        o = np.argsort(v[ok]); mm = ok.sum()
        q = np.minimum.accumulate((v[ok][o] * mm / np.arange(1, mm + 1))[::-1])[::-1].clip(max=1.0)
        qq = np.empty(mm); qq[o] = q
        gg.loc[ok, "q_bh_partial"] = qq
    out.append(gg)
cor = pd.concat(out).sort_values(["group", "target", "spearman_p"])
cor.to_csv(os.path.join(args.outdir, "kernel_param_correlations_eval.csv"), index=False)

pd.set_option("display.width", 250)
show = ["group", "target", "covariate", "n", "spearman_rho", "q_bh",
        "partial_rho_ctrl_ncells", "partial_rho_ctrl_ncells_embl2", "q_bh_partial"]
print("\n=== raw vs partial Spearman (control: log10 n_cells, embedding L2) ===")
for grp_ in ["spatial_all547", "spatial_bench87", "accuracy_bench87"]:
    s = cor[cor.group == grp_]
    if not len(s):
        continue
    print(f"\n--- {grp_} ---")
    print(s[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# figure
fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2))
ax = axes[0, 0]
ax.hist(df["ls_footprint"], bins=45, color="#2563eb", alpha=.85)
ax.axvline(LS_MIN, ls="--", c="k", lw=1); ax.axvline(LS_MAX, ls="--", c="k", lw=1)
ax.set_xlabel("$\\ell(p)$  (normalized slide units)"); ax.set_ylabel("# perturbations")
ax.set_title(f"learned footprint lengthscale\nn={len(df)}, guard band [3, 18]", fontsize=9)

ax = axes[0, 1]
ax.hist(df["g_scale_niche"], bins=45, color="#dc2626", alpha=.85)
ax.axvline(G_CAP, ls="--", c="k", lw=1)
ax.set_xlabel("$g(p)$"); ax.set_ylabel("# perturbations")
ax.set_title(f"learned niche gain cap\nn={len(df)}, cap {G_CAP}", fontsize=9)

ax = axes[0, 2]
ax.hist(df["eff_lengthscale"].dropna(), bins=45, color="#7c3aed", alpha=.85)
ax.set_xlabel("effective footprint (loc units, slide = 20)")
ax.set_ylabel("# perturbations")
ax.set_title("half-decay of the actual field\nautocovariance", fontsize=9)

ax = axes[0, 3]
ax.scatter(Z[:, 0], Z[:, 1], s=45, c="#b91c1c", marker="x", label="learned $Z$ anchors")
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="#0f766e", lw=2,
                           label="data box $c\\in[0,1]^2$"))
ax.set_xlabel("$c_x$"); ax.set_ylabel("$c_y$"); ax.legend(fontsize=7)
ax.set_title("inducing anchors migrated OFF the\ndata box -> smooth gradient field", fontsize=9)

panels = [("g_scale_niche", "log_n_cells", df, "all 547"),
          ("g_scale_niche", "nnd_mean", df, "all 547"),
          ("ls_footprint", "log_n_cells", df, "all 547"),
          ("g_scale_niche", "lfc_rho", df[df.in_benchmark == 1], "benchmark 87")]
for j, (t, c, sub, tag) in enumerate(panels):
    ax = axes[1, j]
    d = sub[[t, c]].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(d[c], d[t], s=11, alpha=.6, c="#0f766e")
    rho, p = spearmanr(d[t], d[c])
    ax.set_xlabel(c); ax.set_ylabel(t)
    ax.set_title(f"{tag}: $\\rho$={rho:.3f}, p={p:.2g}, n={len(d)}", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(args.outdir, "fig_kernel_params.png"), dpi=180)
print(f"\nwrote -> {args.outdir}")
