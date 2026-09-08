"""ONE simple, clear figure for the biological interpretability claim.

Single message: **the model opens its niche modulation for perturbations that are spread
through many microenvironments, and closes it for perturbations confined to one clump.**

Deliberately scoped:
  - Niche gain ONLY. The footprint lengthscale is left out -- on this checkpoint its
    strongest correlate is the perturbation-embedding norm (+0.356, q=0.011), stronger
    than either spatial descriptor (-0.299 / -0.276, both just past BH), and it is squeezed
    into 0.150-0.272 of a [0.15, 0.90] band. There is no clean per-perturbation pattern to
    show, so it is not shown. It belongs in the methods, not in a biology figure.
  - Two panels only: what it looks like on tissue, and whether it holds across all 87.

Left  : four tissue maps, two spread out (red) vs two clumped (blue), matched on cell number
Right : all 87 benchmark perturbations, occupancy vs niche gain, both adjusted for cell
        number and embedding norm -- the SAME quantities the reported partial correlation
        uses, so the panel shows the actual statistic rather than a raw view of it.

Run in concert_env.
"""
import json, os
import numpy as np
import pandas as pd
import h5py
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Checkpoint whose learned kernel parameters are analysed, and where the figures go.
# Both are overridable so the shipped code carries no machine-specific assumption.
CKPT_DIR = os.environ.get(
    "CONCERT_CKPT_DIR",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_spac1_niche_seen_lsfix")
ANA = os.path.join(CKPT_DIR, "kernel_param_analysis")
OUT = os.environ.get(
    "CONCERT_FIGURE_DIR",
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/"
    "case_study/kernel_interpretability")
SP = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq"
PATHS = [f"{SP}/raw_output_1-{i}.h5ad" for i in range(1, 6)]
CACHE = os.path.join(OUT, "_exemplar_coords.npz")

HI, LO, GREY, TEAL = "#b91c1c", "#1d4ed8", "#ebebee", "#0f766e"

df = pd.read_csv(os.path.join(ANA, "per_perturbation_kernel_params_eval.csv"))
cor = pd.read_csv(os.path.join(ANA, "kernel_param_correlations_eval.csv"))
b = df[df.in_benchmark == 1].copy()

plt.rcParams.update({"font.size": 15, "axes.labelsize": 19, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "legend.fontsize": 15})


def rank_resid(y, X):
    R = lambda v: rankdata(v) / len(v)
    Y = R(y)
    A = np.column_stack([np.ones(len(Y))] + [R(X[c]) for c in X])
    return Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]


b["log_n"] = np.log10(b["n_cells_total"])
ctrl = b[["log_n", "emb_l2"]]
b["g_res"] = rank_resid(b["g_scale_niche"], ctrl)
b["occ_res"] = rank_resid(b["occupancy_frac"], ctrl)

_ref = cor[(cor.group == "spatial_bench87") & (cor.target == "g_scale_niche")
           & (cor.covariate == "occupancy_frac")].iloc[0]
rho_ref, q_ref = float(_ref.partial_rho_ctrl_ncells_embl2), float(_ref.q_bh_partial)
rho_plot = float(spearmanr(b.occ_res, b.g_res).statistic)
assert abs(rho_plot - rho_ref) < 0.02, f"plotted {rho_plot:.3f} != reported {rho_ref:.3f}"

band = b[(b.n_cells_total >= b.n_cells_total.quantile(.30)) &
         (b.n_cells_total <= b.n_cells_total.quantile(.85))]
hi = band[(band.g_res > 0) & (band.occ_res > 0)].nlargest(14, "g_res").nsmallest(5, "max_patch_frac")
lo = band[(band.g_res < 0) & (band.occ_res < 0)].nsmallest(14, "g_res").nlargest(5, "max_patch_frac")
exemplars = list(hi.perturbation) + list(lo.perturbation)
print("[exemplars]", exemplars, f"| plotted rho {rho_plot:.3f} vs reported {rho_ref:.3f}")


def read_obs_str(f_obs, col):
    o = f_obs[col]
    if isinstance(o, h5py.Group) and "codes" in o and "categories" in o:
        return np.array(o["categories"][:]).astype(str)[np.array(o["codes"][:], dtype=int)]
    raw = np.array(o[:])
    if raw.dtype.kind in "SO":
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in raw])
    return raw.astype(str)


if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    coords, perts, sids = z["coords"], z["perts"], z["sids"]
    assert set(exemplars).issubset(set(np.unique(perts))), f"delete {CACHE} and rerun"
else:
    cc, pp, ss = [], [], []
    for k, path in enumerate(PATHS):
        with h5py.File(path, "r") as f:
            pr = read_obs_str(f["obs"], "perturbation")
            pos = np.array(f["obsm"]["array_coords"][:], dtype=np.float32)
            keep = np.ones(len(pr), bool)
            if "filter_pass" in f["obs"]:
                fp = read_obs_str(f["obs"], "filter_pass")
                keep = np.array([s.lower() in {"1", "true", "pass", "yes"} for s in fp])
        pos, pr = pos[keep], pr[keep]
        sub = np.isin(pr, exemplars) | (np.arange(len(pr)) % 4 == 0)
        cc.append(pos[sub]); pp.append(pr[sub]); ss.append(np.full(int(sub.sum()), k))
    coords, perts, sids = np.concatenate(cc), np.concatenate(pp), np.concatenate(ss)
    np.savez_compressed(CACHE, coords=coords, perts=perts, sids=sids)

# ------------------------------------------------------------------ figure -----
# Scatter on top across the left 2/3 (double height), legend in the right 1/3 beside it,
# then 2 rows x 5 map columns below (row 1 spread out, row 2 clumped). 15 grid columns so
# each map spans 3 and the scatter spans 10 = two thirds.
fig = plt.figure(figsize=(21.0, 12.4))
gs = fig.add_gridspec(4, 15, height_ratios=[1.30, 0.12, 1, 1], wspace=0.05, hspace=0.18)

panels = []
for grow, (tab, col) in enumerate(((hi, HI), (lo, LO))):
    for j, (_, r) in enumerate(tab.iterrows()):
        ax = fig.add_subplot(gs[grow + 2, j * 3:(j + 1) * 3])
        m_all = perts == r.perturbation
        best = int(pd.Series(sids[m_all]).value_counts().idxmax())
        bg = sids == best
        mm = m_all & bg
        sc = MinMaxScaler().fit(coords[bg])
        p_bg, p_fg = sc.transform(coords[bg]), sc.transform(coords[mm])
        ax.scatter(p_bg[:, 0], p_bg[:, 1], s=.6, c=GREY, linewidths=0, rasterized=True)
        ax.scatter(p_fg[:, 0], p_fg[:, 1], s=9, c=col, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(r.perturbation, fontsize=22, color=col, pad=6)
        panels.append(dict(perturbation=r.perturbation,
                           group=("spread through the tissue" if grow == 0 else "concentrated in a few spots"),
                           niche_gain=round(float(r.g_scale_niche), 3),
                           occupancy_fraction=round(float(r.occupancy_frac), 4),
                           n_cells=int(r.n_cells_total), slide=f"1-{best+1}"))

ax = fig.add_subplot(gs[0, 0:10])
ax.scatter(b.occ_res, b.g_res, s=54, alpha=.75, c=TEAL, linewidths=0)
for tab, col in ((hi, HI), (lo, LO)):
    ax.scatter(tab.occ_res, tab.g_res, s=215, c=col, edgecolor="k", linewidth=1.0, zorder=5)
z = np.polyfit(b.occ_res, b.g_res, 1)
xs = np.linspace(b.occ_res.min(), b.occ_res.max(), 50)
ax.plot(xs, np.polyval(z, xs), color="#374151", lw=2.0, ls="--", zorder=4)
# Formal single terms. Both axes carry the cell-number-corrected rank, which is signed, so
# the numeric ticks are suppressed rather than printing a negative "tissue occupancy";
# the correction and the correlation live in the caption.
ax.set_xlabel("Tissue occupancy", labelpad=10, fontsize=24)
ax.set_ylabel("Niche gain", labelpad=10, fontsize=24)
ax.tick_params(labelleft=False, labelbottom=False)
ax.spines[["top", "right"]].set_visible(False)

lax = fig.add_subplot(gs[0, 10:15]); lax.axis("off")
lax.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc=TEAL, mec="none", ms=15,
           label="87 benchmark\nperturbations"),
    Line2D([], [], marker="o", ls="", mfc=HI, mec="k", ms=18, label="Spread through\nthe tissue"),
    Line2D([], [], marker="o", ls="", mfc=LO, mec="k", ms=18, label="Concentrated in\na few spots"),
], frameon=False, loc="center left", fontsize=22, labelspacing=1.5, handletextpad=1.0)

fig.savefig(os.path.join(OUT, "fig_niche_gain_simple.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

json.dump({
    "message": ("Learned niche gain rises with how widely a perturbation is spread through "
                "the tissue; it closes toward zero for perturbations confined to one clump."),
    "checkpoint": "CONCERT model, spatial footprint band [0.15, 0.90] in normalized units",
    "n_benchmark": int(len(b)),
    "partial_spearman_rho": round(rho_ref, 4), "q_bh": round(q_ref, 4),
    "controls": "log10(number of perturbed cells) and perturbation-embedding norm, rank-based",
    "exemplar_cell_band": [int(band.n_cells_total.min()), int(band.n_cells_total.max())],
    "panels": panels,
    "lengthscale_excluded_because": (
        "on this checkpoint the footprint lengthscale correlates more strongly with the "
        "perturbation-embedding norm (+0.356, q=0.011) than with either spatial descriptor "
        "(-0.299 q=0.037 occupancy, -0.276 q=0.049 nearest-neighbour index), and spans only "
        "0.150-0.272 of its [0.15,0.90] band (median 0.158). No clean per-perturbation "
        "pattern exists, so it is not plotted as biology."),
}, open(os.path.join(OUT, "fig_niche_gain_simple_stats.json"), "w"), indent=2)
print("->", os.path.join(OUT, "fig_niche_gain_simple.png"))
