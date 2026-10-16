"""Supplementary figure: the direction of the kernel-parameter relations is the same on
perturb-FISH as on SPAC-seq.

Panel a compares, for the four pre-specified relations, the partial rank correlation
measured on the SPAC-seq within-slide model against the one measured on a perturb-FISH
model trained with the same configuration. Panels b and c show the perturb-FISH data
behind two of those points, as rank residuals after the same two controls.

Reads only saved CSVs; no model is loaded.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--pfish_corr", required=True)
ap.add_argument("--pfish_params", required=True)
ap.add_argument("--spac_corr", required=True)
ap.add_argument("--spac_group", default="spatial_bench87")
ap.add_argument("--out_pdf", required=True)
ap.add_argument("--out_stats", required=True)
args = ap.parse_args()

BLUE = "#2166ac"
RED = "#b2182b"
N_CONTROLS = 2

# The four relations the SPAC-seq result is stated on, in the order they are discussed.
RELATIONS = [
    ("g_scale_niche", "occupancy_frac", "Niche gain vs\ntissue area reached"),
    ("g_scale_niche", "nni", "Niche gain vs\nnearest-neighbour index"),
    ("ls_footprint", "occupancy_frac", "Footprint lengthscale vs\ntissue area reached"),
    ("ls_footprint", "nni", "Footprint lengthscale vs\nnearest-neighbour index"),
]

pf = pd.read_csv(args.pfish_corr)
sp = pd.read_csv(args.spac_corr)
sp = sp[sp["group"] == args.spac_group]


def fisher_ci(rho, n):
    """95% interval for a partial correlation, on the Fisher z scale."""
    if not np.isfinite(rho) or n - N_CONTROLS - 3 <= 0:
        return np.nan, np.nan
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - N_CONTROLS - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def pick(tab, target, cov):
    r = tab[(tab["target"] == target) & (tab["covariate"] == cov)]
    if len(r) != 1:
        raise SystemExit("expected one row for %s vs %s, found %d" % (target, cov, len(r)))
    r = r.iloc[0]
    return (float(r["partial_rho_ctrl_ncells_embl2"]),
            float(r["partial_p_ctrl_ncells_embl2"]),
            float(r["q_bh_partial"]), int(r["n"]))


stats = {"relations": [], "spac_group": args.spac_group, "n_controls": N_CONTROLS}
for target, cov, _ in RELATIONS:
    s_rho, s_p, s_q, s_n = pick(sp, target, cov)
    p_rho, p_p, p_q, p_n = pick(pf, target, cov)
    stats["relations"].append(dict(
        target=target, covariate=cov,
        spac=dict(rho=s_rho, p=s_p, q=s_q, n=s_n, ci=fisher_ci(s_rho, s_n)),
        perturbfish=dict(rho=p_rho, p=p_p, q=p_q, n=p_n, ci=fisher_ci(p_rho, p_n)),
        sign_agrees=bool(np.sign(s_rho) == np.sign(p_rho))))
stats["n_sign_agree"] = int(sum(r["sign_agrees"] for r in stats["relations"]))

# ---------------------------------------------------------------- residuals for b, c
prm = pd.read_csv(args.pfish_params)
prm = prm[prm["n_cells_total"].notna()].copy()
prm["log_n_cells"] = np.log10(prm["n_cells_total"].astype(float))


def rank_residual(v, controls):
    y = rankdata(v)
    Cm = np.column_stack([rankdata(c) for c in controls] + [np.ones(len(y))])
    b, *_ = np.linalg.lstsq(Cm, y, rcond=None)
    return y - Cm @ b


ctrl = [prm["log_n_cells"].to_numpy(), prm["emb_l2"].to_numpy()]
res_occ = rank_residual(prm["occupancy_frac"].to_numpy(), ctrl)
res_gain = rank_residual(prm["g_scale_niche"].to_numpy(), ctrl)
res_ls = rank_residual(prm["ls_footprint"].to_numpy(), ctrl)
stats["n_perturbations_perturbfish"] = int(len(prm))

# ---------------------------------------------------------------- draw
plt.rcParams.update({"font.size": 16, "axes.linewidth": 1.1,
                     "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

fig = plt.figure(figsize=(13.0, 7.6))
# Panel a takes the full width on its own row; b and c share the row below it, so each is
# wider and shorter. The two rows carry different left margins because a's row labels are
# long and b's axis label is not.
gs_top = fig.add_gridspec(1, 1, left=0.215, right=0.985, top=0.955, bottom=0.615)
gs_bot = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.475, bottom=0.115,
                          wspace=0.28)

ax = fig.add_subplot(gs_top[0, 0])
y = np.arange(len(RELATIONS))[::-1]
h = 0.34
for off, key, colour, lab in ((+h / 2, "spac", BLUE, "SPAC-seq"),
                              (-h / 2, "perturbfish", RED, "Perturb-FISH")):
    rho = [r[key]["rho"] for r in stats["relations"]]
    lo = [r[key]["ci"][0] for r in stats["relations"]]
    hi = [r[key]["ci"][1] for r in stats["relations"]]
    ax.barh(y + off, rho, height=h, color=colour, edgecolor="none", label=lab, zorder=2)
    ax.errorbar(rho, y + off, xerr=[np.array(rho) - np.array(lo), np.array(hi) - np.array(rho)],
                fmt="none", ecolor="0.25", elinewidth=1.4, capsize=4, zorder=3)
ax.axvline(0, color="0.35", lw=1.1, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([lab for _, _, lab in RELATIONS], fontsize=15)
ax.set_xlabel("Partial rank correlation", fontsize=17)
ax.set_xlim(-0.75, 0.85)
ax.set_ylim(-0.6, len(RELATIONS) - 0.4)
ax.legend(fontsize=14, loc="lower right", frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

for k, (resid, ylab) in enumerate(((res_gain, "Learned niche gain"),
                                   (res_ls, "Footprint lengthscale"))):
    axk = fig.add_subplot(gs_bot[0, k])
    axk.axhline(0, color="0.8", lw=1.0, zorder=1)
    axk.axvline(0, color="0.8", lw=1.0, zorder=1)
    axk.scatter(res_occ, resid, s=70, color=RED, alpha=0.85, edgecolor="white",
                linewidth=0.6, zorder=3)
    b = np.polyfit(res_occ, resid, 1)
    xs = np.linspace(res_occ.min(), res_occ.max(), 50)
    axk.plot(xs, np.polyval(b, xs), color="0.25", lw=1.8, zorder=2)
    axk.set_xlabel("Tissue area reached\n(rank residual)", fontsize=17)
    axk.set_ylabel("%s\n(rank residual)" % ylab, fontsize=17)
    for side in ("top", "right"):
        axk.spines[side].set_visible(False)

for ax_, lab, dx in zip(fig.axes, ["a", "b", "c"], [0.185, 0.070, 0.070]):
    pos = ax_.get_position()
    fig.text(pos.x0 - dx, pos.y1 + 0.012, lab, fontsize=21, fontweight="bold",
             va="bottom", ha="left")

os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
fig.savefig(args.out_pdf, dpi=300)
with open(args.out_stats, "w") as fh:
    json.dump(stats, fh, indent=2)
print("wrote %s" % args.out_pdf)
print(json.dumps(stats, indent=2))
