#!/usr/bin/env python
"""ONE readout calibration for every case-study figure.

    calibrated_f(t)  =  observed_f(Day 0)  +  a_f * ( predicted_f(t) - predicted_f(Day 0) )

Same functional form at every level of aggregation. Exactly ONE fitted number per
displayed quantity (the amplitude a_f); the baseline is the MEASURED observed Day-0
population mean, not a fitted intercept.

Why this form and not the alternatives (all three were fitted and compared here):

  scale-only  a*pred(t)          one parameter, but forced through the origin. The
      counterfactual decode over-predicts healthy tissue ~1.9x, so a through-origin fit
      lands the calibrated baseline at 0.61x observed -- it manufactures a baseline
      failure the model does not actually have.
  affine      a*pred(t) + b      fits well on the aggregate, but the intercept is not
      identifiable per marker: b=0.028 alone exceeds most single-marker baselines
      (~0.005) and floors every row. Leave-one-day-out it blows up on sparse markers.
  anchor      this one           best on the aggregate both in-sample and leave-one-day-
      out, best in-sample per marker, and spends no parameter on the baseline at all.

What it does NOT do: absorb the model's exaggerated response amplitude. Raw predicted
Day-9/Day-0 fold is 4.1x against an observed 2.2x, and one amplitude factor per quantity
cannot fix a shape error -- which is the point. The residual per-marker disagreement
stays visible in the figures and is reported as a model limitation.

Writes calibration.json, read by every plotting script through paper_style.calibration().
Pure numpy, head-node safe.
"""
import csv
import json
import numpy as np

CS = ("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
      "dss_timekernel_markersign_timefield_fixedt/case_study")
HERE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study"
DUMP = ("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
        "dss_timekernel_markersign_timefield_fixedt/heldout_eval/percell_dump_sizefactor.npz")
DAYS = [0, 3, 9, 21]


def fit_amplitude(obs_days, pred_days):
    """Single least-squares amplitude on the response above each side's own Day-0."""
    dx, dy = pred_days - pred_days[0], obs_days - obs_days[0]
    return float((dx * dy).sum() / max(float((dx ** 2).sum()), 1e-30))


def apply(pred, baseline_obs, a, pred_baseline):
    return baseline_obs + a * (np.asarray(pred) - pred_baseline)


def diagnostics(obs_days, pred_days, baseline_obs, a, pred_baseline):
    cal = apply(pred_days, baseline_obs, a, pred_baseline)
    return {"calibrated_day_means": [round(float(v), 5) for v in cal],
            "ratio_to_observed": [round(float(v), 3) for v in cal / obs_days],
            "mae": round(float(np.abs(cal - obs_days).mean()), 5)}


# aggregate 17-marker DSS score (trajectories, velocity, spatial maps)
obs_rows = list(csv.DictReader(open(f"{HERE}/per_mouse_observed.csv")))
obs_agg = np.array([np.average(
    [float(r["observed_marker_log1p"]) for r in obs_rows if int(float(r["day"])) == t],
    weights=[float(r["n_cells"]) for r in obs_rows if int(float(r["day"])) == t]) for t in DAYS])
traj = list(csv.DictReader(open(f"{CS}/per_mouse_trajectories.csv")))
w = np.array([float(r["n_cells"]) for r in traj])
pred_agg = np.array([np.average([float(r[f"pred_t{t}"]) for r in traj], weights=w) for t in DAYS])

a_agg = fit_amplitude(obs_agg, pred_agg)
out = {
    "rule": "calibrated(t) = observed_baseline + a * (pred(t) - pred_baseline)",
    "note": ("Baseline is the measured observed Day-0 population mean (not fitted). "
             "One fitted amplitude per displayed quantity. Apply at the level displayed: "
             "'aggregate' for the 17-marker score, 'per_marker' for single markers."),
    "days": DAYS,
    "aggregate": {
        "a": round(a_agg, 6),
        "observed_baseline": round(float(obs_agg[0]), 6),
        "pred_baseline": round(float(pred_agg[0]), 6),
        "observed_day_means": [round(float(v), 5) for v in obs_agg],
        **diagnostics(obs_agg, pred_agg, obs_agg[0], a_agg, pred_agg[0]),
    },
    "per_marker": {},
}
print("AGGREGATE 17-marker score")
print("  observed  ", np.round(obs_agg, 4))
print("  calibrated", np.array(out["aggregate"]["calibrated_day_means"]),
      " ratio", np.array(out["aggregate"]["ratio_to_observed"]), f" a={a_agg:.4f}")

# per marker (dot plots)
d = np.load(DUMP, allow_pickle=True)
markers = [str(x) for x in d["marker_names"]]
obs_cell = np.log1p(d["obs_marker_expr"])
src_day = d["src_day"]
obs_m = np.array([[obs_cell[np.isclose(src_day, t, atol=1e-3), j].mean()
                   for j in range(len(markers))] for t in DAYS])
pred_m = np.array([[d[f"marker_pred_t{int(t)}"][:, j].mean()
                    for j in range(len(markers))] for t in DAYS])

print("\nPER MARKER")
for j, g in enumerate(markers):
    a = fit_amplitude(obs_m[:, j], pred_m[:, j])
    out["per_marker"][g] = {
        "a": round(a, 6),
        "observed_baseline": round(float(obs_m[0, j]), 6),
        "pred_baseline": round(float(pred_m[0, j]), 6),
        "observed_day_means": [round(float(v), 5) for v in obs_m[:, j]],
        **diagnostics(obs_m[:, j], pred_m[:, j], obs_m[0, j], a, pred_m[0, j]),
    }
    r = out["per_marker"][g]
    print(f"  {g:9s} a={a:8.4f}  cal={np.array(r['calibrated_day_means'])}"
          f"  ratio={np.array(r['ratio_to_observed'])}")

with open(f"{HERE}/calibration.json", "w") as fh:
    json.dump(out, fh, indent=1)
print(f"\nwrote {HERE}/calibration.json")
