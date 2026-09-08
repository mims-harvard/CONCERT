#!/usr/bin/env python
"""The interpolation-rebuttal headline figure.

The reviewer's claim is that predicting an unseen individual is "just interpolation --
the mean between two observed timepoints". For an unseen individual, ANY interpolation
collapses to a CONSTANT per cell, whose per-cell correlation with the truth is 0 by
construction. So the rebuttal is not about beating a curve: it is about carrying per-cell
spatial structure that no constant can carry.

Four panels, all from the leave-one-mouse-out imputation dump (count space):
  A  per-cell Pearson(pred, obs) per mouse -- the interpolation baseline is the 0 line
  B  per-cell RMSE, calibrated model vs the best constant, paired per mouse
  C  the calibration ladder: how many mice beat the constant under each calibration
  D  one mouse's per-cell agreement (hexbin), the structure panel A summarises

Every calibration coefficient is fit LEAVE-ONE-MOUSE-OUT (on the other 7 mice), so no
mouse is ever scored with a scale fitted to itself.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps

DUMP = ("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
        "dss_timekernel_markersign_timefield_fixedt/heldout_eval/percell_dump_sizefactor.npz")
OUTDIR = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study"
FEATURED = "062221_D9_m3"          # peak-day mouse for the per-cell panel

ps.use()
d = np.load(DUMP, allow_pickle=True)
mouse, src_day = d["mouse"], d["src_day"]
obs, pred = d["obs_score"], d["pred_diag"]
mice = sorted(set(mouse.tolist()), key=lambda m: (float(src_day[mouse == m][0]), m))
day_of = {m: float(src_day[mouse == m][0]) for m in mice}
print(f"[dump] {obs.size} cells, {len(mice)} mice")


def _fit_scale(mask):
    return float((pred[mask] @ obs[mask]) / (pred[mask] @ pred[mask] + 1e-12))


def _fit_affine(mask):
    A = np.stack([pred[mask], np.ones(int(mask.sum()), np.float32)], 1)
    return np.linalg.lstsq(A, obs[mask], rcond=None)[0]


rows = []
for mid in mice:
    mm = mouse == mid
    others = ~mm
    same_day = others & np.isclose(src_day, day_of[mid], atol=1e-3)
    o, p = obs[mm], pred[mm]
    a_g = _fit_scale(others)
    a_d = _fit_scale(same_day) if same_day.any() else a_g
    a_f, b_f = _fit_affine(others)
    rows.append({
        "mouse": mid, "day": day_of[mid], "n": int(mm.sum()),
        "pearson": float(np.corrcoef(p, o)[0, 1]),
        "rmse_const": float(o.std()),
        "rmse_none": float(np.sqrt(np.mean((p - o) ** 2))),
        "rmse_global": float(np.sqrt(np.mean((a_g * p - o) ** 2))),
        "rmse_per_day": float(np.sqrt(np.mean((a_d * p - o) ** 2))),
        "rmse_affine": float(np.sqrt(np.mean((a_f * p + b_f - o) ** 2))),
        "cal_a": float(a_f), "cal_b": float(b_f),
    })
    print(f"  {mid:<16} day={day_of[mid]:>4.0f} r={rows[-1]['pearson']:.3f} "
          f"rmse {rows[-1]['rmse_affine']:.4f} vs const {rows[-1]['rmse_const']:.4f}")

LADDER = [("none", "rmse_none"), ("global\nscale", "rmse_global"),
          ("per-day\nscale", "rmse_per_day"), ("affine\n$a\\cdot p+b$", "rmse_affine")]
wins = {key: sum(r[key] < r["rmse_const"] for r in rows) for _, key in LADDER}
n_mice = len(rows)
mean_r = float(np.mean([r["pearson"] for r in rows]))
print(f"[pooled] mean Pearson {mean_r:.3f}   ladder {wins}")

short = [m.split("_", 1)[1] for m in mice]          # drop the batch-date prefix
cols = [ps.day_color(day_of[m]) for m in mice]
x = np.arange(n_mice)

fig = plt.figure(figsize=(15.0, 9.2))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.3, 1.0], height_ratios=[1.0, 1.15],
                      hspace=0.42, wspace=0.30, left=0.06, right=0.97, top=0.90, bottom=0.08)

# ── A: per-cell Pearson, per mouse ────────────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
axA.bar(x, [r["pearson"] for r in rows], color=cols, edgecolor="white", lw=0.8, zorder=2)
axA.axhline(0, color=ps.INK, lw=1.4, zorder=3)
axA.axhline(mean_r, color=ps.MUTED, ls="--", lw=1.1, zorder=3)
axA.text(n_mice - 0.45, mean_r + 0.02, f"mean $r$ = {mean_r:.3f}", ha="right",
         fontsize=9, color=ps.MUTED)
axA.text(-0.45, 0.035, "any interpolation for an unseen individual  →  $r$ = 0",
         fontsize=8.5, color=ps.INK)
axA.set_xticks(x); axA.set_xticklabels(short, rotation=35, ha="right")
axA.set_ylabel("per-cell Pearson $r$ (pred vs obs)")
axA.set_title("Per-individual spatial structure is recovered")
axA.set_ylim(-0.04, 0.78); axA.set_xlim(-0.7, n_mice - 0.3)
axA.xaxis.grid(False)
ps.panel_label(axA, "A")

# ── B: calibrated RMSE vs the best constant, paired ───────────────────────────
axB = fig.add_subplot(gs[0, 1])
for i, r in enumerate(rows):
    win = r["rmse_affine"] < r["rmse_const"]
    axB.plot([i, i], sorted((r["rmse_affine"], r["rmse_const"])),
             color=ps.GRID, lw=2.0, zorder=1, solid_capstyle="round")
    axB.scatter(i, r["rmse_const"], marker="_", s=210, color=ps.INK, lw=1.8, zorder=2)
    axB.scatter(i, r["rmse_affine"], s=60, color=ps.day_color(r["day"]), zorder=3,
                edgecolor="white", lw=0.9, marker="o" if win else "X")
axB.set_xticks(x); axB.set_xticklabels(short, rotation=35, ha="right")
axB.set_ylabel("per-cell RMSE (marker score)")
axB.set_title(f"Calibrated model beats the constant on {wins['rmse_affine']}/{n_mice} mice")
axB.set_xlim(-0.7, n_mice - 0.3)
axB.xaxis.grid(False)
axB.legend(handles=[
    plt.Line2D([0], [0], marker="_", ls="", color=ps.INK, ms=11, mew=1.8,
               label="best constant = SD(obs)"),
    plt.Line2D([0], [0], marker="o", ls="", color=ps.MUTED, ms=7, label="CONCERT — wins"),
    plt.Line2D([0], [0], marker="X", ls="", color=ps.MUTED, ms=7, label="CONCERT — loses")],
    loc="upper left", fontsize=8.5)
ps.panel_label(axB, "B")

# ── C: the calibration ladder ─────────────────────────────────────────────────
axC = fig.add_subplot(gs[0, 2])
lbls = [l for l, _ in LADDER]
vals = [wins[k] for _, k in LADDER]
bars = axC.bar(np.arange(len(lbls)), vals, width=0.68,
               color=["#cfcfcf", "#b0b0b0", "#8f8f8f", ps.ACCENT],
               edgecolor="white", lw=0.8, zorder=2)
for b, v in zip(bars, vals):
    axC.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v}/{n_mice}",
             ha="center", fontsize=10, fontweight="bold")
axC.set_xticks(np.arange(len(lbls))); axC.set_xticklabels(lbls, fontsize=8.5)
axC.set_ylabel(f"mice beating the constant"); axC.set_ylim(0, n_mice + 0.9)
axC.set_title("Calibration ladder\n(every fit leave-one-mouse-out)")
axC.xaxis.grid(False)
ps.panel_label(axC, "C", dx=-0.16)

# ── D: one mouse, per-cell ────────────────────────────────────────────────────
axD = fig.add_subplot(gs[1, :2])
r = next(r for r in rows if r["mouse"] == FEATURED)
mm = mouse == FEATURED
o, p = obs[mm], r["cal_a"] * pred[mm] + r["cal_b"]
lim = [0.0, float(np.percentile(np.concatenate([o, p]), 99.5))]
hb = axD.hexbin(p, o, gridsize=95, bins="log", cmap=ps.DENSITY_CMAP, mincnt=1,
                extent=(lim[0], lim[1], lim[0], lim[1]), zorder=2)
axD.plot(lim, lim, ls="--", color=ps.INK, lw=1.2, zorder=3, label="identity")
axD.axhline(o.mean(), color=ps.day_color(9.0), lw=1.8, zorder=3,
            label="best constant — a flat line, $r$ = 0")
axD.set_xlim(lim); axD.set_ylim(lim)
axD.set_xlabel("predicted DSS-marker score  (calibrated on the other 7 mice)")
axD.set_ylabel("observed DSS-marker score")
axD.set_title(f"{FEATURED} — day {r['day']:.0f}, {r['n']:,} cells, $r$ = {r['pearson']:.3f}")
axD.legend(loc="upper left")
axD.grid(False)
cb = fig.colorbar(hb, ax=axD, shrink=0.85, pad=0.015)
cb.set_label("cells (log scale)", fontsize=9)
cb.outline.set_visible(False)
ps.panel_label(axD, "D", dx=-0.065, dy=1.05)

# ── caption block ─────────────────────────────────────────────────────────────
axT = fig.add_subplot(gs[1, 2]); axT.axis("off"); axT.grid(False)
axT.text(0.0, 1.0,
         f"$\\bf{{Leave\\ one\\ mouse\\ out\\ imputation}}$\n"
         f"{n_mice} mice · {obs.size:,} cells\n\n"
         f"mean per-cell $r$ = {mean_r:.3f}\n"
         f"{wins['rmse_affine']}/{n_mice} mice beat the best constant\n"
         f"after an out-of-sample affine\ncalibration\n\n"
         "Each mouse is measured at ONE\ntimepoint, so no interpolation\n"
         "exists for it: any interpolant is a\nconstant per cell, and a constant\n"
         "has per-cell correlation 0.\n\n"
         f"$\\bf{{Limitation}}$\nThe uncalibrated level is over-\npredicted, increasingly at late\n"
         "timepoints — the field reads tissue\narchitecture, which lags molecular\nresolution.",
         va="top", ha="left", fontsize=9.3, linespacing=1.5)

fig.suptitle("Reconstructing an unseen individual's spatial inflammation field", x=0.06, ha="left")
out = f"{OUTDIR}/fig_rebuttal_headline.png"
fig.savefig(out); plt.close(fig)
print("wrote", out)

Path(f"{OUTDIR}/fig_rebuttal_headline_stats.json").write_text(json.dumps(
    {"per_mouse": rows, "mean_pearson": mean_r,
     "mice_beating_constant_by_calibration": {l.replace("\n", " "): wins[k] for l, k in LADDER},
     "n_mice": n_mice}, indent=2, default=float))
print("wrote", f"{OUTDIR}/fig_rebuttal_headline_stats.json")
