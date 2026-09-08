#!/usr/bin/env python
"""Velocity (d/dt) of the inflammation-marker score over the DSS course.

Single panel `fig_velocity_panels.png`, styled like fig_calibration_variance.png: orange
+/-SD band across the 15 mice, faint per-mouse lines, square-marker population mean, and
the observed ground-truth velocity in black.

The claim this figure carries is the SIGN FLIP -- rise into Day 9, fall after it. Linear
interpolation between two static timepoints produces a constant velocity and therefore
cannot reproduce it, which is why the velocity view (not the level view) is the one that
answers the interpolation critique.

Velocity of each segment s = (y[t_{s+1}] - y[t_s]) / (day_{s+1} - day_s), plotted on an
evenly-spaced categorical axis (linear time crowds Day 0 and Day 3 together).
Predictions are put on the observed scale by the ONE shared readout calibration
(paper_style.calibrate). Pure numpy/matplotlib, head-node safe.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as ps
ps.use()

HERE = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study")
CS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
          "dss_timekernel_markersign_timefield_fixedt/case_study")
DAYS = np.array([0, 3, 9, 21], float)
DDAY = np.diff(DAYS)                                   # [3,6,12]
SEG = np.arange(3, dtype=float)                        # categorical segment x
SEGLAB = ["Day 0 -> 3", "Day 3 -> 9", "Day 9 -> 21"]
PRED = ps.DAY_COLORS[9.0]                              # light-orange prediction colour
GT = ps.INK


def vel(y):                                            # y (...,4) -> (...,3) per-day velocity
    return np.diff(y, axis=-1) / DDAY


traj = list(csv.DictReader(open(f"{CS}/per_mouse_trajectories.csv")))
ncell = np.array([float(r["n_cells"]) for r in traj])
P = ps.calibrate(np.array([[float(r[f"pred_t{int(t)}"]) for t in DAYS] for r in traj]))  # (15,4)
V = vel(P)                                             # (15,3)
obs = list(csv.DictReader(open(f"{HERE}/per_mouse_observed.csv")))
obs_line = np.array([np.average([float(r["observed_marker_log1p"]) for r in obs if int(float(r["day"])) == int(t)],
                                weights=[float(r["n_cells"]) for r in obs if int(float(r["day"])) == int(t)])
                     for t in DAYS])
Vgt = vel(obs_line)                                    # (3,)

fig, ax = plt.subplots(figsize=(7.4, 5.0))
m, sd = np.average(V, axis=0, weights=ncell), V.std(0)
ax.fill_between(SEG, m - sd, m + sd, color=PRED, alpha=0.16, lw=0)
for row in V:
    ax.plot(SEG, row, "-", color=PRED, alpha=0.22, lw=1.0, zorder=1)
ax.axhline(0, color=ps.MUTED, lw=1.0, ls="--", zorder=0)
ax.plot(SEG, m, "s-", color=PRED, lw=3.0, ms=10, zorder=5, label="CONCERT")
ax.plot(SEG, Vgt, "o-", color=GT, lw=3.0, ms=10, zorder=6, label="Observed")

ax.set_ylabel("Change in marker score per day", fontsize=19)
ax.set_xticks(SEG); ax.set_xticklabels(SEGLAB, fontsize=20)
ax.set_xlim(-0.14, 2.14)
ax.tick_params(axis="y", labelsize=17)
ax.legend(fontsize=18, loc="upper right", frameon=False, handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig(HERE / "fig_velocity_panels.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("[fig] fig_velocity_panels.png")
print("  CONCERT :", np.round(m, 5))
print("  Observed:", np.round(Vgt, 5))
