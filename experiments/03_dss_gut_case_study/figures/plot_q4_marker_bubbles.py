#!/usr/bin/env python
"""Q4: Marker-gene bubble/dot plots (color = mean expression, size = % expressing).

Built from the SAME imputation dump so observed and imputed use identical cells.
Only markers with a real TEMPORAL TREND are shown (ranked by observed peak/baseline
fold; flat markers like Mmp2/Isg15 are dropped -- they add rows without signal).

Q4a  Observed dotplot     -- cells grouped by TRUE measured day (ground truth).
Q4c  Imputed dotplot      -- all cells re-decoded to each virtual day (model course).
Q4b  Observed-vs-imputed  -- obs|imp side by side per day (agreement check).

color = mean log1p expression (blue = low -> red = high, coolwarm).
size  = fraction of cells with expression > 0.

Styling (see the plotting-figures skill): NO title, large text, dense dots,
minimal whitespace.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import paper_style
paper_style.use()

DUMP = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/dss_timekernel_markersign_timefield_fixedt/heldout_eval/percell_dump_sizefactor.npz"
OUTDIR = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study"
FEATURED = ["092421_D3_m4", "092421_D3_m1", "062221_D9_m3"]   # per-mouse imputed bubbles
DAYS = [0.0, 3.0, 9.0, 21.0]
DAY_LBL = {0.0: "Day 0", 3.0: "Day 3", 9.0: "Day 9", 21.0: "Day 21"}
CMAP = "coolwarm"
MAX_MARKERS = 8          # keep only the strongest-trending markers
MIN_FOLD = 1.5           # peak(d9)/baseline(d0) must exceed this to count as "trending"
MK_FS, DAY_FS, CBAR_FS, LEG_FS = 16, 15, 15, 14

d = np.load(DUMP, allow_pickle=True)
markers_all = [str(x) for x in d["marker_names"]]
obs_expr_all = np.log1p(d["obs_marker_expr"])    # log1p to match the imputed (log1p) scale
src_day = d["src_day"]
mouse_all = d["mouse"]

_rng = np.random.default_rng(0)
def _realize(log1p_mu):
    return np.log1p(_rng.poisson(np.expm1(np.asarray(log1p_mu, np.float64)))).astype(np.float32)
pred_all = {0.0: _realize(d["marker_pred_t0"]), 3.0: _realize(d["marker_pred_t3"]),
            9.0: _realize(d["marker_pred_t9"]), 21.0: _realize(d["marker_pred_t21"])}

# --- keep only markers with a temporal trend (observed peak/baseline fold) ---
_obs_day = np.array([[obs_expr_all[np.isclose(src_day, day, atol=1e-3), j].mean()
                      for j in range(len(markers_all))] for day in DAYS])   # (4,G)
_fold = _obs_day[2] / np.clip(_obs_day[0], 1e-3, None)      # d9 / d0
_order = np.argsort(-_fold)
_keep = [j for j in _order if _fold[j] >= MIN_FOLD][:MAX_MARKERS]
markers = [markers_all[j] for j in _keep]
G = len(markers)
obs_expr = obs_expr_all[:, _keep]
pred = {k: v[:, _keep] for k, v in pred_all.items()}
print(f"[q4] {G} trending markers (d9/d0 fold): " +
      ", ".join(f"{markers_all[j]} {_fold[j]:.1f}x" for j in _keep))


def group_stats(rows):
    if rows.shape[0] == 0:
        return np.full(G, np.nan), np.zeros(G)
    return rows.mean(0), (rows > 0).mean(0)


def sizes(frac):                       # big dots -> minimal whitespace between them
    return 120 + 900 * np.clip(frac, 0, 1)


def _dotplot(col_mu, col_fr, col_lbl, out, sep_pairs=False):
    ncol = col_mu.shape[0]
    fig, ax = plt.subplots(figsize=(1.02 * ncol + 2.6, 0.55 * G + 0.9))
    vmax = float(np.nanpercentile(col_mu, 98))
    q25 = float(np.nanpercentile(col_mu, 25))
    vcenter = min(max(q25, vmax * 1e-3), vmax * 0.98)     # blue->red transition at 25th pctile
    norm = TwoSlopeNorm(vmin=0.0, vcenter=vcenter, vmax=vmax)
    for ci in range(ncol):
        ax.scatter(np.full(G, ci), np.arange(G), s=sizes(col_fr[ci]), c=col_mu[ci],
                   cmap=CMAP, norm=norm, edgecolors="0.25", linewidths=0.6)
    if sep_pairs:
        for x in np.arange(1.5, ncol - 1, 2):
            ax.axvline(x, color="0.85", lw=0.8)
    ax.set_xticks(range(ncol)); ax.set_xticklabels(col_lbl, fontsize=DAY_FS)
    ax.set_yticks(range(G)); ax.set_yticklabels(markers, fontsize=MK_FS)
    ax.set_ylim(-0.55, G - 0.45); ax.set_xlim(-0.55, ncol - 0.45); ax.invert_yaxis()
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    ticks = sorted({0.0, vcenter, (vcenter + vmax) / 2, vmax})   # mark the 25th-pctile transition
    cb.set_ticks(ticks); cb.set_ticklabels([f"{v:.2f}" for v in ticks])
    cb.set_label("mean log1p expr", fontsize=CBAR_FS); cb.ax.tick_params(labelsize=CBAR_FS - 3)
    handles = [Line2D([0], [0], marker='o', ls='', mfc='0.6', mec='0.25',
                      ms=np.sqrt(sizes(f)), label=f"{int(f*100)}%") for f in (0.25, 0.5, 1.0)]
    ax.legend(handles=handles, title="% expressing", loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
              columnspacing=2.4, handletextpad=0.5, borderpad=0.8,
              fontsize=LEG_FS, title_fontsize=LEG_FS)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


# Q4a observed
mu, fr = zip(*[group_stats(obs_expr[np.isclose(src_day, day, atol=1e-3)]) for day in DAYS])
_dotplot(np.array(mu), np.array(fr), [DAY_LBL[x] for x in DAYS],
         f"{OUTDIR}/fig_q4a_observed_dotplot.png")

# Q4c imputed-only over time
mu, fr = zip(*[group_stats(pred[day]) for day in DAYS])
_dotplot(np.array(mu), np.array(fr), [DAY_LBL[x] for x in DAYS],
         f"{OUTDIR}/fig_q4c_imputed_dotplot.png")

# Q4b observed vs imputed
col_lbl, col_mu, col_fr = [], [], []
for day in DAYS:
    mu_o, fr_o = group_stats(obs_expr[np.isclose(src_day, day, atol=1e-3)])
    mu_i, fr_i = group_stats(pred[day])
    col_lbl += [f"{DAY_LBL[day]}\nobs", f"{DAY_LBL[day]}\nimp"]
    col_mu += [mu_o, mu_i]; col_fr += [fr_o, fr_i]
_dotplot(np.array(col_mu), np.array(col_fr), col_lbl,
         f"{OUTDIR}/fig_q4b_obs_vs_imputed_dotplot.png", sep_pairs=True)

# Q4d per-mouse IMPUTED dotplots for the featured (slide-plot) mice; each on its OWN scale
for M in FEATURED:
    mm = mouse_all == M
    if not mm.any():
        print(f"  [skip] {M} not in dump"); continue
    mu, fr = zip(*[group_stats(pred[day][mm]) for day in DAYS])
    _dotplot(np.array(mu), np.array(fr), [DAY_LBL[x] for x in DAYS],
             f"{OUTDIR}/fig_q4_imputed_{M}.png")
