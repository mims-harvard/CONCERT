"""How much does a CALIBRATION PASS buy? Leave-one-mouse-out, CPU-only, no retrain.

The held-out reconstruction has the right day-profile but the wrong amplitude (per-mouse
pseudobulk 1.5-4.4x). Before paying for a retrain, measure the ceiling of the cheap fix:
fit a calibration on the OTHER mice and apply it to the held-out one, so nothing is
self-fitted. Three nested calibrations, increasing in what they're allowed to know:

  global  : one scalar a, pred -> a*pred            (fit on the 7 other mice)
  per-day : one scalar a_d per day                  (fit on the other mice AT THAT DAY)
  affine  : pred -> a*pred + b                      (fit on the 7 other mice)

Reported against the same best-constant baseline the eval uses (RMSE = SD of observed),
whose per-cell correlation is 0 by construction. If per-day calibration lands most mice
below the constant, the amplitude is a readout-scale problem and a retrain buys little;
if it doesn't, the miscalibration is cell-level and a retrain is the only route.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import anndata as ad

OUT = Path(f"{_rp.WORKDIR}/outputs/"
           "dss_timekernel_markersign_timefield_fixedt")
DATADIR = Path(f"{_rp.WORKDIR}/concert_data/processed/"
               "dryad_merfish_dss")
EV = OUT / "heldout_eval"

d = np.load(EV / "percell_dump.npz", allow_pickle=True)
sid, mouse, src_day = d["sample_id"], d["mouse"], d["src_day"]
obs_marker = d["obs_marker_expr"]
N = sid.shape[0]

held = set(mouse.tolist())
rows = [l.split(",") for l in (DATADIR / "manifest.csv").read_text().splitlines()[1:] if l.strip()]
files = [DATADIR / r[0] for r in rows if r[2] in held]
n_counts = np.zeros(N, np.float64)
for i, p in enumerate(files):
    m = sid == i
    X = ad.read_h5ad(p).X
    n_counts[m] = np.asarray(X.sum(1)).ravel() if sp.issparse(X) else np.asarray(X).sum(1)
sf = (n_counts / np.median(n_counts)).astype(np.float32)

obs = np.log1p(np.expm1(obs_marker)).mean(1)
pred = np.zeros(N, np.float32)
for t, key in ((0.0, "marker_pred_t0"), (3.0, "marker_pred_t3"),
               (9.0, "marker_pred_t9"), (21.0, "marker_pred_t21")):
    m = np.isclose(src_day, t, atol=1e-3)
    if m.any():
        pred[m] = np.log1p(np.expm1(d[key][m]) * sf[m, None]).mean(1)

mice = sorted(held, key=lambda m: (float(src_day[mouse == m][0]), m))
day_of = {m: float(src_day[mouse == m][0]) for m in mice}


def _fit_scale(mask):        # least-squares a for obs ~ a*pred
    return float((pred[mask] @ obs[mask]) / (pred[mask] @ pred[mask] + 1e-12))


def _fit_affine(mask):       # least-squares (a, b) for obs ~ a*pred + b
    A = np.stack([pred[mask], np.ones(mask.sum(), np.float32)], 1)
    return np.linalg.lstsq(A, obs[mask], rcond=None)[0]


print(f"{'mouse':<16}{'day':>4}{'rmse_c':>9}{'raw':>9}{'glob':>9}{'perday':>9}{'affine':>9}"
      f"{'pearson':>9}")
res, wins = [], {"raw": 0, "global": 0, "per_day": 0, "affine": 0}
for mid in mice:
    mm = mouse == mid
    others = ~mm
    same_day_others = others & np.isclose(src_day, day_of[mid], atol=1e-3)
    o, p = obs[mm], pred[mm]
    rmse_c = float(o.std())

    a_g = _fit_scale(others)
    a_d = _fit_scale(same_day_others) if same_day_others.any() else a_g
    a_f, b_f = _fit_affine(others)

    r = {"mouse": mid, "day": day_of[mid], "n_cells": int(mm.sum()),
         "rmse_best_constant": rmse_c,
         "rmse_raw": float(np.sqrt(np.mean((p - o) ** 2))),
         "rmse_global_scale": float(np.sqrt(np.mean((a_g * p - o) ** 2))),
         "rmse_per_day_scale": float(np.sqrt(np.mean((a_d * p - o) ** 2))),
         "rmse_affine": float(np.sqrt(np.mean((a_f * p + b_f - o) ** 2))),
         "a_global": a_g, "a_per_day": a_d, "a_affine": float(a_f), "b_affine": float(b_f),
         "pearson": float(np.corrcoef(p, o)[0, 1])}
    for k, key in (("raw", "rmse_raw"), ("global", "rmse_global_scale"),
                   ("per_day", "rmse_per_day_scale"), ("affine", "rmse_affine")):
        r[f"beats_constant_{k}"] = bool(r[key] < rmse_c)
        wins[k] += r[f"beats_constant_{k}"]
    res.append(r)
    print(f"{mid:<16}{r['day']:>4.0f}{rmse_c:>9.4f}{r['rmse_raw']:>9.4f}"
          f"{r['rmse_global_scale']:>9.4f}{r['rmse_per_day_scale']:>9.4f}"
          f"{r['rmse_affine']:>9.4f}{r['pearson']:>9.3f}")

n = len(res)
pooled = {f"frac_beats_constant_{k}": wins[k] / n for k in wins}
pooled["mean_pearson"] = float(np.mean([r["pearson"] for r in res]))
pooled["note"] = ("LOMO calibration: every scale/affine is fit on the OTHER mice only. "
                  "beats_constant compares against RMSE = SD(observed), the best constant "
                  "predictor (per-cell corr 0 by construction).")
print("\n[pooled]", json.dumps(pooled, indent=2))
(EV / "heldout_calibration_lomo.json").write_text(
    json.dumps({"per_mouse": res, "pooled": pooled}, indent=2))
print(f"wrote {EV / 'heldout_calibration_lomo.json'}")
