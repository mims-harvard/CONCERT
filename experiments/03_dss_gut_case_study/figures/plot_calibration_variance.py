#!/usr/bin/env python
"""Calibration / population trajectory WITH replicate-mouse variance bands (15 mice).

The old fig_calibration.png drew one smooth observed line and one smooth predicted
line, hiding the large per-mouse heterogeneity the per-mouse panel shows. Here, at
each timepoint, the spread ACROSS REPLICATE MICE is drawn as a shaded +/-SD band plus
the individual mouse points, so the population curve is visibly consistent with the
per-mouse plot.

Both curves are mean log1p DSS-marker expression (17 core markers, count space):
  observed (GT): per_mouse_observed.csv  — each mouse measured at ONE day; per-day
      line = cell-weighted pooled mean (reproduces the committed obs_by_day exactly),
      band = +/-SD across the replicate mice sampled at that day, dots = those mice.
  predicted:     per_mouse_trajectories.csv — every mouse imputed at t=0/3/9/21; line
      = cell-weighted pooled mean (= C1_population_pred), band = +/-SD across all 15
      mice, thin lines = per-mouse imputed trajectories (the 3 flat d9 mice are the
      documented under-fit).

Ratios (pred/obs) reproduce the committed honest calibration (~1.9/1.8/3.5/2.9x).
Pure numpy/matplotlib, head-node safe.
"""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style
paper_style.use()

CS = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/dss_timekernel_markersign_timefield_fixedt/case_study"
HERE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study"
OUT = f"{HERE}/fig_calibration_variance.png"
DAYS = [0, 3, 9, 21]
OBS_C, PRED_C = paper_style.INK, paper_style.DAY_COLORS[9.0]


def _wmean_wsd(vals, wts):
    v, w = np.asarray(vals, float), np.asarray(wts, float)
    m = float((v * w).sum() / w.sum())
    return m, float(np.sqrt((w * (v - v.mean()) ** 2).sum() / w.sum()) if len(v) > 1 else 0.0)


# observed (15 mice)
obs = list(csv.DictReader(open(f"{HERE}/per_mouse_observed.csv")))
obs_pts = {t: [(float(r["observed_marker_log1p"]), float(r["n_cells"]))
               for r in obs if int(float(r["day"])) == t] for t in DAYS}
obs_line = np.array([np.average([v for v, _ in obs_pts[t]], weights=[w for _, w in obs_pts[t]]) for t in DAYS])
obs_sd = np.array([np.std([v for v, _ in obs_pts[t]]) for t in DAYS])   # across-mouse SD

traj = list(csv.DictReader(open(f"{CS}/per_mouse_trajectories.csv")))
n_cell = np.array([float(r["n_cells"]) for r in traj])
P = paper_style.calibrate(np.array([[float(r[f"pred_t{t}"]) for t in DAYS] for r in traj]))
pred_line = np.array([np.average(P[:, i], weights=n_cell) for i in range(4)])
pred_sd = P.std(0)                                                       # across-mouse SD

# Evenly-spaced categorical x. On a true linear time axis the "Day 0" and "Day 3" tick
# labels overlap at readable font sizes; the same categorical spacing is used by the
# velocity and pathway curve figures, so the set stays consistent.
X = np.arange(len(DAYS), dtype=float)

fig, ax = plt.subplots(figsize=(7.4, 5.0))
# predicted: band + per-mouse thin lines + mean
ax.fill_between(X, pred_line - pred_sd, pred_line + pred_sd, color=PRED_C, alpha=0.16, lw=0)
for row in P:
    ax.plot(X, row, "-", color=PRED_C, alpha=0.25, lw=1.0, zorder=1)
ax.plot(X, pred_line, "s-", color=PRED_C, lw=3.0, ms=10, zorder=5, label="CONCERT")
# observed: band + replicate dots + mean
ax.fill_between(X, obs_line - obs_sd, obs_line + obs_sd, color=OBS_C, alpha=0.18, lw=0)
ax.plot(X, obs_line, "o-", color=OBS_C, lw=3.0, ms=10, zorder=5, label="Observed")
for i, t in enumerate(DAYS):
    ax.scatter([X[i]] * len(obs_pts[t]), [v for v, _ in obs_pts[t]], s=70,
               facecolor="white", edgecolor=OBS_C, lw=1.8, zorder=6)


ax.set_xticks(X); ax.set_xticklabels([f"Day {t}" for t in DAYS], fontsize=20)
ax.set_ylabel("Mean log1p DSS-marker expression", fontsize=19)
ax.tick_params(axis="y", labelsize=17)
ax.set_xlim(-0.14, len(DAYS) - 0.86)
top = max(P.max(), max(v for t in DAYS for v, _ in obs_pts[t]))
ax.set_ylim(0, top * 1.03)                     # covers every per-mouse line, nothing clipped
ax.legend(fontsize=18, loc="upper left", frameon=False, handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT)
print("observed  line:", np.round(obs_line, 4), " sd:", np.round(obs_sd, 4),
      " n_mice/day:", {t: len(obs_pts[t]) for t in DAYS})
print("predicted line:", np.round(pred_line, 4), " sd:", np.round(pred_sd, 4))
print("ratios pred/obs:", np.round(pred_line / obs_line, 2))
