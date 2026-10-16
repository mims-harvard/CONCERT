#!/usr/bin/env python
"""Trajectory + velocity figures, summarized per observed-day group.

  fig_per_mouse_trajectories.png : one thin line per mouse, colored by measured day. NO GT.
  fig_per_day_trajectories.png   : mice grouped by measured day -> ONE mean curve per day
                                    (+/-SD band), 4 curves total, nice per-day colors. NO GT.
  fig_velocity.png               : d(inflammation)/dt per day group (4 curves) PLUS the
                                    observed-GT velocity (black), showing the rise->fall sign
                                    flip that linear interpolation cannot produce.

Predicted = mean log1p DSS-marker (imputed at t=0/3/9/21). Observed-GT for velocity is the
cell-weighted population level per day (per_mouse_observed.csv). Head-node safe.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import csv, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style
paper_style.use()

CS = f"{_rp.DATA}/dss_colitis/case_study"
HERE = f"{_rp.WORKDIR}/paper/Outputs/case_study"
DAYS = [0, 3, 9, 21]
DC = paper_style.DAY_COLORS
MID = [(DAYS[i] + DAYS[i + 1]) / 2 for i in range(3)]   # segment midpoints for velocity

traj = list(csv.DictReader(open(f"{CS}/per_mouse_trajectories.csv")))
obs = list(csv.DictReader(open(f"{HERE}/per_mouse_observed.csv")))

# the ONE shared readout calibration -- same rule as every other case-study figure
def cpred(r):
    return paper_style.calibrate([float(r[f"pred_t{t}"]) for t in DAYS])

# per-mouse predicted rows grouped by observed day
by_day = defaultdict(list)          # day -> list of (pred[4], n_cells)
for r in traj:
    od = int(float(r["observed_day"]))
    by_day[od].append((cpred(r), float(r["n_cells"])))

# observed population level per day (cell-weighted) -> obs_by_day
obs_by_day = {}
tmp = defaultdict(list)
for r in obs:
    tmp[int(float(r["day"]))].append((float(r["observed_marker_log1p"]), float(r["n_cells"])))
for d, vs in tmp.items():
    v = np.array([a for a, _ in vs]); w = np.array([b for _, b in vs])
    obs_by_day[d] = float((v * w).sum() / w.sum())
obs_curve = np.array([obs_by_day[d] for d in DAYS])
obs_sd = np.array([np.std([a for a, _ in tmp[d]]) for d in DAYS])   # across-mouse GT SD


def _day_mean_sd(d):
    P = np.array([p for p, _ in by_day[d]]); W = np.array([w for _, w in by_day[d]])
    mean = np.array([np.average(P[:, i], weights=W) for i in range(4)])
    sd = P.std(0)
    return mean, sd, len(P)


# 1. per-mouse trajectories (no GT)
fig, ax = plt.subplots(figsize=(6.8, 5.3))
for r in traj:
    od = float(r["observed_day"])
    y = list(cpred(r))
    ax.plot(DAYS, y, "-", color=DC[od], alpha=0.55, lw=1.4)
    ax.scatter([od], [y[DAYS.index(int(od))]], color=DC[od], s=38, edgecolor="k", lw=0.5, zorder=3)
ax.fill_between(DAYS, obs_curve - obs_sd, obs_curve + obs_sd, color=paper_style.INK, alpha=0.15, lw=0, zorder=1)
ax.plot(DAYS, obs_curve, "o-", color=paper_style.INK, lw=2.8, ms=8, zorder=6)
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color=DC[d], lw=2.4, label=f"sampled Day {int(d)}") for d in DAYS]
h.append(Line2D([0], [0], color=paper_style.INK, lw=2.8, marker="o", label="observed GT (±SD)"))
ax.legend(handles=h, fontsize=12, loc="upper right", frameon=False)
ax.set_xticks(DAYS); ax.set_xticklabels([f"Day {t}" for t in DAYS], fontsize=16)
ax.set_ylabel("predicted inflammation (log1p marker)", fontsize=14)
ax.tick_params(axis="y", labelsize=14); ax.set_ylim(0, None)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_per_mouse_trajectories.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 2. per-day mean trajectories (no GT)
fig, ax = plt.subplots(figsize=(6.8, 5.3))
for d in DAYS:
    mean, sd, n = _day_mean_sd(d)
    ax.fill_between(DAYS, mean - sd, mean + sd, color=DC[float(d)], alpha=0.13, lw=0)
    ax.plot(DAYS, mean, "o-", color=DC[float(d)], lw=3.0, ms=8,
            label=f"sampled Day {d}  (n={n})")
ax.fill_between(DAYS, obs_curve - obs_sd, obs_curve + obs_sd, color=paper_style.INK, alpha=0.15, lw=0, zorder=1)
ax.plot(DAYS, obs_curve, "o-", color=paper_style.INK, lw=3.0, ms=8, zorder=6, label="observed GT")
ax.legend(fontsize=12, loc="upper right", frameon=False)
ax.set_xticks(DAYS); ax.set_xticklabels([f"Day {t}" for t in DAYS], fontsize=16)
ax.set_ylabel("predicted inflammation (log1p marker)", fontsize=14)
ax.tick_params(axis="y", labelsize=14); ax.set_ylim(0, None)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_per_day_trajectories.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 3. velocity: per-day predicted + GT
def _vel(curve):
    return [(curve[i + 1] - curve[i]) / (DAYS[i + 1] - DAYS[i]) for i in range(3)]

fig, ax = plt.subplots(figsize=(6.8, 5.3))
ax.axhline(0, color="0.6", lw=1.0, zorder=0)
for d in DAYS:
    mean, _, n = _day_mean_sd(d)
    ax.plot(MID, _vel(mean), "o-", color=DC[float(d)], lw=2.6, ms=7, label=f"sampled Day {d}")
ax.plot(MID, _vel(obs_curve), "s--", color=paper_style.INK, lw=3.0, ms=9, label="observed GT")
ax.legend(fontsize=12, loc="upper right", frameon=False)
ax.set_xticks(MID); ax.set_xticklabels([f"{a}->{b}" for a, b in zip(DAYS[:-1], DAYS[1:])], fontsize=14)
ax.set_xlabel("day interval", fontsize=15)
ax.set_ylabel("d(inflammation)/dt  (log1p marker / day)", fontsize=15)
ax.tick_params(axis="y", labelsize=14)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_velocity.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("wrote per-mouse, per-day, velocity figures")
for d in DAYS:
    m, s, n = _day_mean_sd(d)
    print(f"  Day{d:2d} (n={n}): pred {np.round(m,3)}")
print("  obs_by_day:", np.round(obs_curve, 4), " GT velocity:", np.round(_vel(obs_curve), 5))
