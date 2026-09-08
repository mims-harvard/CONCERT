"""Kernel-parameter interpretability figures -- LSFIX checkpoint.

Rebuilt from make_interpret_figures.py (old ckpt) because the units fix changes the
RESULT, not just the numbers: on the CONCERT model BOTH learned per-perturbation kernel
scalars carry spatial signal, and they track the SAME two descriptors with OPPOSITE
signs. So the figure set is organised around that two-knob agreement instead of around
one knob against six mutually-correlated dispersion descriptors.

  fig1_two_knob_dispersion.png  both knobs vs the two descriptors that survive BH
  fig2_exemplar_maps.png        dispersed vs compact tissue maps + gain / occupancy bars
  fig3_confound_control.png     raw vs exposure-corrected, BOTH knobs

House style: no titles/suptitles (panel identity above the axes is the one exception, for
the per-perturbation map grid); no internal symbols in any reader-facing string; every
reader-facing string starts with a capital; large text. All descriptive text lives in
figure_legends_interpret.txt, driven by figure_stats.json so it cannot drift.

Run in concert_env.
"""
import json, os
import numpy as np
import pandas as pd
import h5py
from scipy.stats import spearmanr, rankdata
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
os.makedirs(OUT, exist_ok=True)

HI = "#b91c1c"      # dispersed / high niche gain
LO = "#1d4ed8"      # compact / low niche gain
GREY = "#ebebee"

df = pd.read_csv(os.path.join(ANA, "per_perturbation_kernel_params_eval.csv"))
cor = pd.read_csv(os.path.join(ANA, "kernel_param_correlations_eval.csv"))
b = df[df.in_benchmark == 1].copy()
print(f"[data] {len(df)} perturbations, {len(b)} benchmark-scored")

LABEL = {
    "nni": "Nearest-neighbour index",
    "occupancy_frac": "Tissue occupancy fraction",
    "radius_gyration": "Radius of gyration",
    "mean_patch_size": "Mean patch size (cells)",
    "singleton_frac": "Isolated-cell fraction",
    "n_patches_per_slide": "Patches per slide",
    "local_density": "Local same-perturbation density",
    "max_patch_frac": "Largest-patch fraction",
    "n_cells_total": "Number of perturbed cells",
    "emb_l2": "Perturbation-embedding norm",
    "nnd_mean": "Mean nearest-neighbour distance",
    "occupancy_bins": "Occupied tissue bins",
}
GAIN = "Learned niche gain"
LSF = "Footprint lengthscale"
KNOB = {"g_scale_niche": GAIN, "ls_footprint": LSF}
# The two descriptors that survive BH correction for BOTH knobs on this checkpoint.
COVS = ["occupancy_frac", "nni"]

plt.rcParams.update({"font.size": 13, "axes.labelsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 18})


def stat(target, cov, group="spatial_bench87"):
    r = cor[(cor.group == group) & (cor.target == target) & (cor.covariate == cov)]
    if not len(r):
        return dict(raw=np.nan, partial=np.nan, q=np.nan)
    r = r.iloc[0]
    return dict(raw=float(r.spearman_rho), partial=float(r.partial_rho_ctrl_ncells_embl2),
                q=float(r.q_bh_partial))


def _rank_resid(y, X):
    R = lambda v: rankdata(v) / len(v)
    Y = R(y)
    A = np.column_stack([np.ones(len(Y))] + [R(X[c]) for c in X])
    return Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]


b["log_n"] = np.log10(b["n_cells_total"])
_ctrl = b[["log_n", "emb_l2"]]
b["g_res"] = _rank_resid(b["g_scale_niche"], _ctrl)
b["occ_res"] = _rank_resid(b["occupancy_frac"], _ctrl)
# Guard: the residuals must reproduce the reported partial correlation, else the exemplars
# would be selected on a different quantity than the one the figure claims.
_chk = float(np.corrcoef(b["g_res"], b["occ_res"])[0, 1])
_ref = stat("g_scale_niche", "occupancy_frac")["partial"]
assert abs(_chk - _ref) < 0.02, f"residual corr {_chk:.3f} != reported partial {_ref:.3f}"

_band = b[(b.n_cells_total >= b.n_cells_total.quantile(.30)) &
          (b.n_cells_total <= b.n_cells_total.quantile(.85))]
hi = _band[(_band.g_res > 0) & (_band.occ_res > 0)].nlargest(14, "g_res").nsmallest(5, "max_patch_frac")
lo = _band[(_band.g_res < 0) & (_band.occ_res < 0)].nsmallest(14, "g_res").nlargest(5, "max_patch_frac")
exemplars = list(hi["perturbation"]) + list(lo["perturbation"])
print(f"[exemplars] residual rule, cell band [{_band.n_cells_total.min():.0f}, "
      f"{_band.n_cells_total.max():.0f}], residual-vs-partial check {_chk:.3f} vs {_ref:.3f}")
print("[exemplars] dispersed:", list(hi["perturbation"]), " compact:", list(lo["perturbation"]))


# ------------------------------------------------------------------- fig 1 -----
# 2 rows (one per learned knob) x 3 cols (distribution, then the two surviving descriptors).
fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.0))
stats1 = {}
for i, knob in enumerate(["g_scale_niche", "ls_footprint"]):
    ax = axes[i, 0]
    ax.hist(df[knob], bins=45, color="#9ca3af", alpha=.9, label="All 547 perturbations")
    ax.hist(b[knob], bins=45, color="#0f766e", alpha=.85, label="87 benchmark-scored perturbations")
    for tab, col in ((hi, HI), (lo, LO)):
        for _, r in tab.iterrows():
            ax.axvline(r[knob], color=col, lw=1.4)
    ax.set_xlabel(KNOB[knob]); ax.set_ylabel("Number of perturbations")
    ax.legend(frameon=False, fontsize=16)

    for j, cov in enumerate(COVS):
        ax = axes[i, j + 1]
        d = b[[knob, cov]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(d[cov], d[knob], s=18, alpha=.7, c="#0f766e")
        for tab, col in ((hi, HI), (lo, LO)):
            m = d.index.intersection(tab.index)
            ax.scatter(d.loc[m, cov], d.loc[m, knob], s=56, c=col,
                       edgecolor="k", linewidth=.6, zorder=5)
        z = np.polyfit(d[cov], d[knob], 1)
        xs = np.linspace(d[cov].min(), d[cov].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="#374151", lw=1.3, ls="--")
        ax.set_xlabel(LABEL[cov]); ax.set_ylabel(KNOB[knob])
        stats1[f"{knob}__{cov}"] = stat(knob, cov)

handles = [Line2D([], [], marker="o", ls="", mfc=HI, mec="k", ms=9, label="Spread through the tissue"),
           Line2D([], [], marker="o", ls="", mfc=LO, mec="k", ms=9, label="Concentrated in a few spots")]
axes[0, 1].legend(handles=handles, frameon=False, fontsize=17, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_two_knob_dispersion.png"), dpi=200)
plt.close(fig)
rho_cells, p_cells = spearmanr(b["g_scale_niche"], b["n_cells_total"])
rho_cells_ls, p_cells_ls = spearmanr(b["ls_footprint"], b["n_cells_total"])
print("[fig1] done")


# ------------------------------------------------------------------- fig 3 -----
# Raw vs exposure-corrected, for BOTH knobs. Shows the correction is what creates the
# occupancy/NNI result for the gain, and that it does not manufacture the lengthscale one.
F3 = ["occupancy_frac", "nni", "radius_gyration", "mean_patch_size", "singleton_frac",
      "n_patches_per_slide"]
fig, axs = plt.subplots(1, 2, figsize=(15.0, 5.6), sharey=True)
fig3_stats = {}
for ax, knob in zip(axs, ["g_scale_niche", "ls_footprint"]):
    F = pd.DataFrame([dict(cov=c, **stat(knob, c)) for c in F3]).set_index("cov").loc[F3]
    fig3_stats[knob] = {c: stat(knob, c) for c in F3}
    y = np.arange(len(F))
    ax.axvline(0, color="#9ca3af", lw=1)
    for i, (r_, p_) in enumerate(zip(F.raw, F.partial)):
        ax.plot([r_, p_], [i + .15, i - .15], color="#d4d4d8", lw=1, zorder=0)
    ax.scatter(F.raw, y + .15, s=60, facecolor="none", edgecolor="#6b7280",
               label="Uncorrected correlation")
    cols = ["#b91c1c" if (not np.isnan(q) and q < .05) else "#9ca3af" for q in F.q]
    ax.scatter(F.partial, y - .15, s=70, c=cols,
               label="Corrected for cell number and embedding norm")
    ax.set_yticks(y); ax.set_yticklabels([LABEL[c] for c in F.index])
    ax.set_xlabel(f"Rank correlation with the {KNOB[knob].lower()}")
    ax.invert_yaxis()
axs[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.17), ncol=2, fontsize=18)
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig(os.path.join(OUT, "fig3_confound_control.png"), dpi=200)
plt.close(fig)
print("[fig3] done")


# ------------------------------------------------------------------- fig 2 -----
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
    if not set(exemplars).issubset(set(np.unique(perts))):
        raise SystemExit(f"cache missing exemplars; delete {CACHE} and rerun")
    print("[fig2] loaded cached coords")
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
        isex = np.isin(pr, exemplars)
        sub = isex | (np.arange(len(pr)) % 4 == 0)
        cc.append(pos[sub]); pp.append(pr[sub]); ss.append(np.full(int(sub.sum()), k))
        print(f"[fig2] {os.path.basename(path)}: kept {sub.sum()}")
    coords = np.concatenate(cc); perts = np.concatenate(pp); sids = np.concatenate(ss)
    np.savez_compressed(CACHE, coords=coords, perts=perts, sids=sids)

meta2 = []
# THE TWO-KNOB PANEL: both learned scalars on the same exemplars, plus the descriptor
# they jointly track. The point of the figure is that bars 1 and 2 move in OPPOSITE
# directions across the same four perturbations.
# NOTE -- the footprint lengthscale is deliberately NOT a bar here. Its bench87 range is
# 0.150-0.272 against a floor of 0.150 (median 0.167), so individual perturbations do not
# separate on it: the compact exemplars interleave with the dispersed ones. The lengthscale
# result is real but only as a 87-point aggregate (partial rho -0.30, q=0.037) and is shown
# that way in fig1/fig3. Putting it on four bars would invite the reader to watch it fail.
# One grouped panel, three measures per perturbation, one colour per measure. The three
# span very different ranges (gain 0.44-1.61, occupancy 0.029-0.214, index 0.21-0.36), so
# bar HEIGHT is each measure scaled by its own largest exemplar value and the true value is
# printed on every bar. Red/blue no longer encodes the group on the bars themselves -- it
# moves to the perturbation names, which is where the two rows of maps are already keyed.
BAR = [("Learned niche gain", "g_scale_niche", "#0f766e", "{:.2f}"),
       ("Tissue occupancy fraction", "occupancy_frac", "#d97706", "{:.3f}"),
       ("Nearest-neighbour index", "nni", "#6d28d9", "{:.2f}")]

fig = plt.figure(figsize=(21.0, 15.6))
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.30], wspace=0.05, hspace=0.22)

for grow, (tab, col) in enumerate(((hi, HI), (lo, LO))):
    for j, (_, r) in enumerate(tab.iterrows()):
        ax = fig.add_subplot(gs[grow, j])
        m_all = perts == r.perturbation
        best = int(pd.Series(sids[m_all]).value_counts().idxmax())
        bg = sids == best
        mm = m_all & bg
        sc = MinMaxScaler().fit(coords[bg])
        p_bg, p_fg = sc.transform(coords[bg]), sc.transform(coords[mm])
        ax.scatter(p_bg[:, 0], p_bg[:, 1], s=.6, c=GREY, linewidths=0, rasterized=True)
        ax.scatter(p_fg[:, 0], p_fg[:, 1], s=9, c=col, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(r.perturbation, fontsize=22, color=col, pad=8)
        meta2.append(dict(row=("dispersed" if grow == 0 else "compact"), position=j + 1,
                          perturbation=r.perturbation,
                          niche_gain=round(float(r.g_scale_niche), 3),
                          footprint_lengthscale=round(float(r.ls_footprint), 4),
                          occupancy_fraction=round(float(r.occupancy_frac), 4),
                          slide=f"1-{best+1}", cells_shown=int(mm.sum()),
                          nearest_neighbour_index=round(float(r.nni), 3),
                          largest_patch_fraction=round(float(r.max_patch_frac), 3)))

ex = pd.concat([hi, lo])
names = list(ex["perturbation"])
name_cols = [HI] * len(hi) + [LO] * len(lo)
ax = fig.add_subplot(gs[2, :])
x = np.arange(len(names))
w = 0.26
bar_scale = {}
for k, (title, colname, bcol, fmt) in enumerate(BAR):
    vals = ex[colname].to_numpy(dtype=float)
    ref = float(vals.max())
    bar_scale[colname] = ref
    ax.bar(x + (k - 1) * w, vals / ref, width=w, color=bcol, label=title)
    for i, v in enumerate(vals):
        ax.text(x[i] + (k - 1) * w, v / ref + .012, fmt.format(v), ha="center",
                va="bottom", fontsize=15, rotation=90, color=bcol)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=21, rotation=45, ha="right")
for t, c in zip(ax.get_xticklabels(), name_cols):
    t.set_color(c)
ax.set_ylabel("Scaled to the largest\nvalue in each measure", fontsize=20, linespacing=1.3)
ax.tick_params(axis="y", labelsize=17)
ax.set_ylim(0, 1.30)
ax.set_xlim(-0.6, len(names) - 0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=20, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.16), handlelength=1.6, columnspacing=2.2)

fig.savefig(os.path.join(OUT, "fig2_exemplar_maps.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("[fig2] done")

# ------------------------------------------------------------ companions -------
summ = b[["perturbation", "g_scale_niche", "ls_footprint", "eff_lengthscale", "n_cells_total",
          "nni", "occupancy_frac", "radius_gyration", "mean_patch_size", "singleton_frac",
          "n_patches_per_slide", "local_density", "max_patch_frac", "emb_l2"]]
summ.sort_values("g_scale_niche", ascending=False).to_csv(
    os.path.join(OUT, "kernel_params_benchmark87.csv"), index=False)

ls_desc = dict(min=float(b.ls_footprint.min()), median=float(b.ls_footprint.median()),
               q75=float(b.ls_footprint.quantile(.75)), max=float(b.ls_footprint.max()))
stats_out = {
    "checkpoint": "CONCERT model, spatial footprint band [0.15, 0.90] in normalized units",
    "n_perturbations_total": int(len(df)), "n_benchmark": int(len(b)),
    "fig1_stats": stats1,
    "gain_vs_cell_number": {"rho": round(float(rho_cells), 4), "p": round(float(p_cells), 4)},
    "lengthscale_vs_cell_number": {"rho": round(float(rho_cells_ls), 4),
                                   "p": round(float(p_cells_ls), 4)},
    "fig2_panels": meta2,
    "fig2_bar_scaling": {"note": ("bar height = value / largest exemplar value of that "
                                  "measure; printed labels are the true values"),
                         "denominators": {k: round(v, 4) for k, v in bar_scale.items()}},
    "fig3_stats": fig3_stats,
    "lengthscale_distribution_bench87": ls_desc,
    "floor_caveat": ("gp_pp lengthscale floor is 0.15; bench87 median %.4f, 75th pct %.4f, "
                     "max %.4f -- the floor is still binding, report relatively"
                     % (ls_desc["median"], ls_desc["q75"], ls_desc["max"])),
}
json.dump(stats_out, open(os.path.join(OUT, "figure_stats.json"), "w"), indent=2)
print("all figures ->", OUT)
