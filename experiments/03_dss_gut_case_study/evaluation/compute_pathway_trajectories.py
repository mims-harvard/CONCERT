#!/usr/bin/env python
"""Per-mouse COUNTERFACTUAL pathway trajectories from the CONCERT imputation decode.

Reads pathway_dump.npz (per-cell per-pathway score at virtual t0/t3/t9/t21 + observed
diagonal, all 15 mice) and produces, per mouse and per pathway:
  - pred_t{0,3,9,21}  = cell-mean predicted pathway score at each virtual day (log1p count)
  - obs_diag          = cell-mean OBSERVED pathway score at the mouse's OWN sampled day
plus a per-pathway global affine readout calibration (obs_pop ~= a*pred_pop + b) fit on the
4 population day-means on the DIAGONAL -- identical recipe to the DSS-marker calibration
(compute_affine_calibration.py) -- so the predicted trajectory sits on the observed scale.

Outputs (into this dir):
  pathway_trajectories.csv        one row per (mouse, pathway)
  pathway_calibration.json        per-pathway {a,b,r2,loo_mae, pred_pop[4], obs_pop[4]}
No torch/GPU -- runs on the head node.
"""
import csv, json
from pathlib import Path
import numpy as np

DUMP = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/"
            "dss_timekernel_markersign_timefield_fixedt/heldout_eval/pathway_dump.npz")
HERE = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study")
DAYS = [0.0, 3.0, 9.0, 21.0]

d = np.load(DUMP, allow_pickle=True)
mouse = d["mouse"]; src_day = d["src_day"]
pathways = [str(x) for x in d["pathway_names"]]
vdays = [float(x) for x in d["virtual_days"]]      # [0,3,9,21]
assert vdays == DAYS, f"virtual_days {vdays} != {DAYS}"
mice = sorted(set(mouse.tolist()))
print(f"[dump] N={mouse.shape[0]}  mice={len(mice)}  pathways={len(pathways)}")

# per-mouse cell-means
rows = []
per = {}   # (mouse,pathway) -> dict
for m in mice:
    mm = mouse == m
    day = float(np.round(src_day[mm][0], 3))
    for pw in pathways:
        pred = d[f"pred__{pw}"][mm]         # (n,4)
        obs = d[f"obs__{pw}"][mm]           # (n,)
        rec = {"mouse": m, "pathway": pw, "day": day, "n_cells": int(mm.sum()),
               "obs_diag": float(obs.mean())}
        for di, dd in enumerate(DAYS):
            rec[f"pred_t{int(dd)}"] = float(pred[:, di].mean())
        # predicted at the mouse's OWN (seen) day = the diagonal, comparable to obs_diag
        rec["pred_diag"] = rec[f"pred_t{int(day)}"]
        per[(m, pw)] = rec
        rows.append(rec)

fields = ["mouse", "pathway", "day", "n_cells", "obs_diag", "pred_diag",
          "pred_t0", "pred_t3", "pred_t9", "pred_t21"]
with open(HERE / "pathway_trajectories.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"[write] {HERE/'pathway_trajectories.csv'}  ({len(rows)} rows)")

# per-pathway affine calibration on the diagonal (4 population day-means)
calib = {}
for pw in pathways:
    recs = [per[(m, pw)] for m in mice]
    ncell = np.array([r["n_cells"] for r in recs], float)
    # predicted population trajectory: all mice decoded at each virtual day
    pred_pop = np.array([np.average([r[f"pred_t{int(dd)}"] for r in recs], weights=ncell)
                         for dd in DAYS])
    # observed population per day: mice SAMPLED at that day, their diagonal obs
    obs_pop = []
    for dd in DAYS:
        sel = [r for r in recs if abs(r["day"] - dd) < 1e-3]
        if sel:
            wv = np.array([r["n_cells"] for r in sel], float)
            obs_pop.append(np.average([r["obs_diag"] for r in sel], weights=wv))
        else:
            obs_pop.append(np.nan)
    obs_pop = np.array(obs_pop)
    ok = ~np.isnan(obs_pop)
    A = np.vstack([pred_pop[ok], np.ones(ok.sum())]).T
    (a, b), *_ = np.linalg.lstsq(A, obs_pop[ok], rcond=None)
    a, b = float(a), float(b)
    resid = obs_pop[ok] - (a * pred_pop[ok] + b)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((obs_pop[ok] - obs_pop[ok].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    # leave-one-day-out
    loo = []
    idx = np.where(ok)[0]
    for i in idx:
        mask = ok.copy(); mask[i] = False
        Ai = np.vstack([pred_pop[mask], np.ones(mask.sum())]).T
        (ai, bi), *_ = np.linalg.lstsq(Ai, obs_pop[mask], rcond=None)
        loo.append(abs((ai * pred_pop[i] + bi) - obs_pop[i]))
    # cross-mouse diagonal validation: across the 15 mice, does the predicted score at
    # each mouse's OWN day track that mouse's observed score? (stronger than the 4 pop-means)
    pd_ = np.array([r["pred_diag"] for r in recs]); od_ = np.array([r["obs_diag"] for r in recs])
    diag_pearson = (float(np.corrcoef(pd_, od_)[0, 1])
                    if pd_.std() > 1e-9 and od_.std() > 1e-9 else float("nan"))
    calib[pw] = {"a": a, "b": b, "r2": r2,
                 "leave_one_day_out_mae": float(np.mean(loo)) if loo else None,
                 "diag_pearson_across_mice": diag_pearson,
                 "validated": bool(a > 0 and r2 >= 0.8),
                 "pred_pop": pred_pop.tolist(), "obs_pop": obs_pop.tolist(),
                 "calib_pop": (a * pred_pop + b).tolist(),
                 "n_genes": int(d["n_genes_per_pathway"][pathways.index(pw)])}

json.dump(calib, open(HERE / "pathway_calibration.json", "w"), indent=1)
print(f"[write] {HERE/'pathway_calibration.json'}")
print(f"\n{'pathway':32s} {'a':>7s} {'r2':>6s} {'diagR':>6s} {'valid':>6s}   pred_pop(0/3/9/21) -> peak")
for pw in pathways:
    c = calib[pw]
    pp = c["pred_pop"]; peak = DAYS[int(np.argmax(pp))]
    print(f"{pw:32s} {c['a']:7.3f} {c['r2']:6.2f} {c['diag_pearson_across_mice']:6.2f} "
          f"{str(c['validated']):>6s}   "
          f"{pp[0]:.3f}/{pp[1]:.3f}/{pp[2]:.3f}/{pp[3]:.3f} -> t{int(peak)}")
