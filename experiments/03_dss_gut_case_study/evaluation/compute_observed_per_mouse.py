#!/usr/bin/env python
"""Per-mouse OBSERVED DSS-marker score (mean log1p over the 17 core markers) for the
15 case-study mice, computed directly from the raw MERFISH slices -- NO model, NO GPU.

Matches the eval's obs_score = log1p(Y_expr[:, marker]).mean(1) (composition_eval_dss.py
L473) so the per-day pooled mean reproduces the committed obs_by_day, and the per-mouse
values give a real replicate-mouse variance band in the SAME scale as the predicted curve.
"""
import csv
from collections import defaultdict
import numpy as np, anndata as ad, scipy.sparse as sp

DD = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/processed/dryad_merfish_dss"
OUT = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/case_study/per_mouse_observed.csv"
MK = ['Isg15','Ifit1','Tnf','Cxcl1','Nos2','Il11','Il1b','Mmp3','Mmp10','Mmp13',
      'Plau','Igfbp5','Col18a1','Mmp2','Osm','Cxcl10','Cxcl5']
MICE = ['062221_D9_m3','062921_D0_m3a','062921_D9_m2a','062921_D9_m5','082421_D0_m6',
        '082421_D0_m7','082421_D21_m1','082421_D21_m2','092421_D3_m1','092421_D3_m2',
        '092421_D3_m3','092421_D3_m4','100221_D9_m2','100221_D9_m3','100221_D9_m5']

man = list(csv.DictReader(open(f"{DD}/manifest.csv")))
files_by_mouse = defaultdict(list)
day_by_mouse = {}
for r in man:
    if r["Mouse_ID"] in MICE:
        files_by_mouse[r["Mouse_ID"]].append(r["file"])
        day_by_mouse[r["Mouse_ID"]] = float(r["day"])

rows = []
for M in MICE:
    ssum, ncell = 0.0, 0
    for f in files_by_mouse[M]:
        a = ad.read_h5ad(f"{DD}/{f}")
        idx = [list(a.var_names).index(g) for g in MK]        # all 17 present (checked)
        X = a.X[:, idx]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        cell_score = np.log1p(X).mean(1)                       # per-cell mean log1p marker
        ssum += float(cell_score.sum()); ncell += cell_score.shape[0]
    rows.append({"mouse": M, "day": day_by_mouse[M], "n_cells": ncell,
                 "observed_marker_log1p": ssum / ncell})
    print(f"  {M:14s} day{int(day_by_mouse[M]):2d} n={ncell:7d} obs={rows[-1]['observed_marker_log1p']:.4f}")

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["mouse", "day", "n_cells", "observed_marker_log1p"])
    w.writeheader(); w.writerows(rows)

# validate: per-day pooled mean vs committed obs_by_day
OBS_COMMITTED = {0: 0.04269, 3: 0.04832, 9: 0.09538, 21: 0.06245}
print("\n=== per-day: cell-weighted pooled mean (vs committed obs_by_day) + across-mouse SD ===")
for dday in [0, 3, 9, 21]:
    rs = [r for r in rows if int(r["day"]) == dday]
    w = np.array([r["n_cells"] for r in rs], float)
    v = np.array([r["observed_marker_log1p"] for r in rs], float)
    pooled = float((v * w).sum() / w.sum())
    print(f"  day{dday:2d}: pooled={pooled:.4f}  committed={OBS_COMMITTED[dday]:.4f}  "
          f"mean_across_mice={v.mean():.4f} sd={v.std():.4f}  n_mice={len(rs)}")
print("wrote", OUT)
