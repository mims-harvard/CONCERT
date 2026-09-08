#!/usr/bin/env python
"""Case-study result figures for the DSS temporal CONCERT model (deliverable C1-C3).

Pure plotting from the eval CSVs already written by composition_eval (run_case_study_deliverables)
— NO model, NO torch, NO GPU. Safe to run on the head node.

Inputs (in RUNDIR/case_study/ unless overridden):
  per_mouse_trajectories.csv, heterogeneity.csv, case_study_summary.json
  observed_by_day: the DATA-property baseline (mean log1p DSS-marker expr per day), read from
  the sibling *_full run's observed_marker_by_day.json (falls back to the hardcoded values).

Outputs (RUNDIR/case_study/figures/):
  fig_case_study.png            — 4-panel summary (trajectories / calibration / heterogeneity / Moran)
  fig_per_mouse_trajectories.png, fig_calibration.png, fig_heterogeneity.png, fig_velocity.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAYS = [0.0, 3.0, 9.0, 21.0]
# fixed color per OBSERVED day (which timepoint the mouse was actually sampled at)
DAY_COLORS = {0.0: "#1f77b4", 3.0: "#2ca02c", 9.0: "#ff7f0e", 21.0: "#d62728"}
OBS_BY_DAY_DEFAULT = {0.0: 0.04269, 3.0: 0.04832, 9.0: 0.09538, 21.0: 0.06245}


def _read_csv(path):
    import csv
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load(rundir, obs_json):
    cs = Path(rundir) / "case_study"
    traj = _read_csv(cs / "per_mouse_trajectories.csv")
    het = _read_csv(cs / "heterogeneity.csv")
    summ = json.loads((cs / "case_study_summary.json").read_text())
    obs_by_day = dict(OBS_BY_DAY_DEFAULT)
    if obs_json and Path(obs_json).exists():
        j = json.loads(Path(obs_json).read_text())
        obs_by_day = {float(k): float(v) for k, v in j["observed_by_day"].items()}
    return traj, het, summ, obs_by_day


def _mouse_traj(row):
    return [float(row[f"pred_t{int(d) if d.is_integer() else d}"]) for d in DAYS]


def panel_trajectories(ax, traj):
    for r in traj:
        od = float(r["observed_day"])
        y = [float(r["pred_t0"]), float(r["pred_t3"]), float(r["pred_t9"]), float(r["pred_t21"])]
        ax.plot(DAYS, y, "-", color=DAY_COLORS[od], alpha=0.55, lw=1.3, zorder=1)
        # anchor dot at the mouse's OWN observed day (where ground truth exists)
        ax.scatter([od], [y[DAYS.index(od)]], color=DAY_COLORS[od], s=42,
                   edgecolor="k", lw=0.6, zorder=3)
    # population mean trajectory (bold black)
    ys = np.array([[float(r["pred_t0"]), float(r["pred_t3"]),
                    float(r["pred_t9"]), float(r["pred_t21"])] for r in traj])
    ax.plot(DAYS, ys.mean(0), "k-", lw=2.6, zorder=4, label="population mean")
    ax.set(xlabel="day (virtual decode time t)", ylabel="pred inflammation (mean log1p marker)",
           title="A  Per-mouse virtual longitudinal trajectories")
    ax.set_xticks(DAYS)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=DAY_COLORS[d], lw=2, label=f"sampled d{int(d)}") for d in DAYS]
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                          markeredgecolor="k", markersize=7, label="ground-truth anchor"))
    ax.legend(handles=handles, fontsize=7, loc="upper right")


def panel_calibration(ax, summ, obs_by_day):
    pred = summ["C1_population_pred_trajectory"]
    pred_y = [float(pred[str(int(d))]) for d in DAYS]
    obs_y = [obs_by_day[d] for d in DAYS]
    ax.plot(DAYS, obs_y, "o-", color="k", lw=2, label="observed (data)")
    ax.plot(DAYS, pred_y, "s--", color="#ff7f0e", lw=2, label="predicted (population)")
    for d, po, pr in zip(DAYS, obs_y, pred_y):
        ax.annotate(f"{pr/po:.2f}×", (d, pr), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color="#ff7f0e")
    ax.set(xlabel="day", ylabel="mean log1p DSS-marker expr",
           title="B  Calibration & shape (both peak at d9)")
    ax.set_xticks(DAYS); ax.legend(fontsize=8)


def panel_heterogeneity(ax, het):
    rows = sorted(het, key=lambda r: float(r["pred_marker_score_at_t9"]))
    vals = [float(r["pred_marker_score_at_t9"]) for r in rows]
    cols = [DAY_COLORS[float(r["observed_day"])] for r in rows]
    labels = [r["mouse"].replace("_", " ") for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, edgecolor="k", lw=0.4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
    ax.axvline(np.mean(vals), color="gray", ls=":", lw=1)
    ax.set(xlabel="pred inflammation @ t9 (log1p marker)",
           title="C  Per-mouse heterogeneity at d9 (interp = 0 spread)")


def panel_moran(ax, summ):
    m = summ["C2_moran_by_day"]
    pred = [m[str(int(d))]["moran_pred_diag"] for d in DAYS]
    obs = [m[str(int(d))]["moran_obs_score"] for d in DAYS]
    x = np.arange(len(DAYS)); w = 0.38
    ax.bar(x - w/2, obs, w, label="observed inflammation", color="#1f77b4", edgecolor="k", lw=0.4)
    ax.bar(x + w/2, pred, w, label="predicted field", color="#ff7f0e", edgecolor="k", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"d{int(d)}" for d in DAYS])
    ax.set(ylabel="Moran's I (spatial autocorrelation)",
           title="D  Spatial structure by day (obs peaks at d9)")
    ax.legend(fontsize=8)


def fig_velocity(outdir, traj):
    """dInflammation/dt per mouse — non-constant velocity is impossible for linear interp."""
    fig, ax = plt.subplots(figsize=(6, 4))
    seg_mid = [(DAYS[i] + DAYS[i+1]) / 2 for i in range(3)]
    for r in traj:
        od = float(r["observed_day"])
        y = [float(r["pred_t0"]), float(r["pred_t3"]), float(r["pred_t9"]), float(r["pred_t21"])]
        vel = [(y[i+1] - y[i]) / (DAYS[i+1] - DAYS[i]) for i in range(3)]
        ax.plot(seg_mid, vel, "-o", color=DAY_COLORS[od], alpha=0.55, ms=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="day (segment midpoint)", ylabel="d(inflammation)/dt",
           title="Inflammation velocity per mouse (sign flips at d9 → non-linear)")
    fig.tight_layout(); fig.savefig(outdir / "fig_velocity.png", dpi=150); plt.close(fig)


def fig_trajectories_normalized(outdir, traj):
    """Per-mouse trajectory normalized to each mouse's OWN healthy (t0) level.

    The count-space per-mouse curves differ in absolute amplitude, which can read as
    "some mice barely change." Dividing each curve by its own t0 removes the per-mouse
    amplitude (library depth + architecture-inferred scale) and shows that MOST mice
    share the same rise -> peak@d9 -> fall SHAPE. Mice whose fold-change stays near 1
    are the ones the model UNDER-fits (small architecture-inferred amplitude), not mice
    that are biologically flat — the residual the LOMO affine calibration corrects.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    folds9 = []
    for r in traj:
        od = float(r["observed_day"])
        y = np.array([float(r["pred_t0"]), float(r["pred_t3"]),
                      float(r["pred_t9"]), float(r["pred_t21"])])
        f = y / max(y[0], 1e-9)
        folds9.append((r["mouse"], f[2]))
        flat = f[2] < 2.0                      # <2x peak-over-baseline = under-fit
        ax.plot(DAYS, f, "-", color=("0.6" if flat else DAY_COLORS[od]),
                alpha=(0.5 if flat else 0.8), lw=(1.2 if flat else 1.8),
                zorder=(1 if flat else 2))
        if flat:
            ax.annotate(r["mouse"].split("_", 1)[-1], (DAYS[2], f[2]), fontsize=7,
                        color="0.35", ha="left", va="center")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set(xlabel="day", ylabel="imputed inflammation / own healthy (t0)",
           title="Per-mouse trajectory shape (fold over each mouse's own t0)")
    ax.text(0.02, 0.97, "grey = under-fit mice (peak < 2x baseline)", transform=ax.transAxes,
            fontsize=8, color="0.35", va="top")
    fig.tight_layout(); fig.savefig(outdir / "fig_per_mouse_trajectories_normalized.png", dpi=150)
    plt.close(fig)
    print("  fold@d9 (peak/baseline) per mouse:",
          ", ".join(f"{m.split('_',1)[-1]}={v:.1f}" for m, v in sorted(folds9, key=lambda t: t[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/"
                    "outputs/dss_timekernel_markersign_timefield_fixedt")
    ap.add_argument("--obs_json", default="/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/"
                    "outputs/dss_timekernel_markersign_timefield_full/observed_marker_by_day.json")
    args = ap.parse_args()

    traj, het, summ, obs_by_day = load(args.rundir, args.obs_json)
    outdir = Path(args.rundir) / "case_study" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_trajectories(axes[0, 0], traj)
    panel_calibration(axes[0, 1], summ, obs_by_day)
    panel_heterogeneity(axes[1, 0], het)
    panel_moran(axes[1, 1], summ)
    fig.suptitle("CONCERT DSS temporal case study — descriptive deliverables (C1-C3)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outdir / "fig_case_study.png", dpi=150)
    plt.close(fig)

    # standalone versions
    for name, fn in [("fig_per_mouse_trajectories", panel_trajectories),
                     ("fig_heterogeneity", panel_heterogeneity)]:
        f, a = plt.subplots(figsize=(7, 5.5))
        fn(a, traj if name == "fig_per_mouse_trajectories" else het)
        f.tight_layout(); f.savefig(outdir / f"{name}.png", dpi=150); plt.close(f)
    f, a = plt.subplots(figsize=(6.5, 5)); panel_calibration(a, summ, obs_by_day)
    f.tight_layout(); f.savefig(outdir / "fig_calibration.png", dpi=150); plt.close(f)
    fig_velocity(outdir, traj)
    fig_trajectories_normalized(outdir, traj)

    print("wrote figures ->", outdir)
    for p in sorted(outdir.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
