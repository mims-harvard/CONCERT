#!/usr/bin/env python3
"""Give CONCERT UNSEEN the same per-knockout error as every other method.

score_spacseq.py:103-108 stores CONCERT unseen's eval-native bootstrap (n_boot=10,
source-cell procedure). Every other method and CONCERT seen carry the gene bootstrap of
lfc_scoring.score_lfc_agreement (200 gene resamples, seed 0). This script recomputes the
gene bootstrap for CONCERT unseen through the identical competitor path
(rescore_competitor_at_cell, unseen=True: stored model_lfc.json vs the observed reference),
asserts the recomputed point rho and RMSE equal the stored ones (tol 1e-3), and writes back ONLY
lfc_spearman_boot / lfc_dist_boot. Point estimates are never changed. The eval-native values are kept
in <prefix>_concert_unseen_boot_evalnative_vs_gene.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import score_spacseq as rs

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", type=Path, required=True)
ap.add_argument("--concert-dir", type=Path, default=rs.CONCERT_EVAL_ROOT)
ap.add_argument("--tol", type=float, default=1e-3)
args = ap.parse_args()

path = args.outdir / "spac_unseen_lfc_dist_per_record.csv"
rows = list(csv.DictReader(path.open()))
fields = list(rows[0].keys())
side, n_ok, n_mis, n_none = [], 0, 0, 0
for r in rows:
    if not r["method"].startswith("concert"):
        continue
    rec = rs.rescore_competitor_at_cell(
        rs.PREP, args.concert_dir, r["gene"], r["slide"], int(r["niche_key"]), r["method"],
        r["method_label"], 5, unseen=True, benchmark_dir=rs.BENCHMARK_DIR)
    old_rb, old_db = r["lfc_spearman_boot"], r["lfc_dist_boot"]
    if rec is None:
        n_none += 1
        r["lfc_spearman_boot"] = r["lfc_dist_boot"] = "nan"
        side.append(dict(gene=r["gene"], slide=r["slide"], status="no_recompute"))
        continue
    drho = abs(rec.lfc_spearman - float(r["lfc_spearman"]))
    ddist = abs(rec.lfc_dist - float(r["lfc_dist"]))
    ok = drho <= args.tol and ddist <= args.tol
    if ok:
        n_ok += 1
        r["lfc_spearman_boot"], r["lfc_dist_boot"] = rec.lfc_spearman_boot, rec.lfc_dist_boot
    else:
        n_mis += 1
        r["lfc_spearman_boot"] = r["lfc_dist_boot"] = "nan"
    side.append(dict(gene=r["gene"], slide=r["slide"], status="ok" if ok else "point_mismatch",
                     d_rho=drho, d_rmse=ddist, rho_boot_evalnative=old_rb, rho_boot_gene=rec.lfc_spearman_boot,
                     rmse_boot_evalnative=old_db, rmse_boot_gene=rec.lfc_dist_boot))
with path.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
sp = args.outdir / "spac_unseen_concert_unseen_boot_evalnative_vs_gene.csv"
keys = sorted({k for s in side for k in s})
with sp.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(side)
ok = [s for s in side if s["status"] == "ok"]
med = lambda k: float(np.median([float(s[k]) for s in ok])) if ok else float("nan")
print(f"[concert-unseen-gene-boot] ok={n_ok} point_mismatch={n_mis} no_recompute={n_none}")
print(f"median rho boot: eval-native={med('rho_boot_evalnative'):.4f} gene={med('rho_boot_gene'):.4f}; "
      f"median RMSE boot: eval-native={med('rmse_boot_evalnative'):.4f} gene={med('rmse_boot_gene'):.4f}")
if n_mis or n_none:
    raise SystemExit("CONCERT unseen gene bootstrap: point mismatch or missing recompute; see " + str(sp))
