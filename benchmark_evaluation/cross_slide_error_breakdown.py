#!/usr/bin/env python3
"""Why the spatial models score badly on the SPAC-seq cross-slide arm.

For every unit of the sampled benchmark tree this reads the predicted LFC each method wrote
(model_lfc.json, already restricted to the unit's scored genes and in the scoring order) and the
observed LFC recomputed exactly as unseen_spac does, then reports per method:

  rmse            RMSE against the observed LFC (the benchmark metric)
  rmse_zero       RMSE of predicting no change at all, ||o|| / sqrt(G) - the do-nothing floor
  |m| / |o|       predicted response magnitude relative to the observed one
  pearson(m, o)   linear agreement, next to the Spearman the benchmark reports
  floor_frac      share of scored genes whose predicted LFC sits at the log(1e-6) floor, i.e. the
                  method predicts zero counts for a gene the source cells express
  within_slide_r  mean pairwise Pearson between the predicted LFC of different knockouts on the
                  same slide - how much of a prediction is the knockout and how much is the slide

Usage: cross_slide_error_breakdown.py [--tree DIR] [--out CSV]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import release_paths as rp  # noqa: E402

import numpy as np


import spacseq_common as sec  # noqa: E402
from spacseq_cross_slide_scoring import (  # noqa: E402
    ko_holdout_indices,
    load_eval_panel,
    load_source_cells,
    unseen_reference_idx,
)
from shared_gene_rule import load_concert_genes  # noqa: E402
from perturbmap_common import to_dense  # noqa: E402
from spacseq_slides import load_prepared_slide  # noqa: E402

METHODS = ["concert", "biolord", "gears", "scgen", "celcomen", "steamboat", "cellagentchat"]
LABELS = {"concert": "CONCERT", "biolord": "Biolord", "gears": "GEARS", "scgen": "scGEN",
          "celcomen": "CelComen", "steamboat": "Steamboat", "cellagentchat": "CellAgentChat"}
FLOOR = -10.0  # log(1e-6) is -13.8; anything under -10 is the epsilon floor, not a fold change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, default=rp.SPACSEQ_PREDICTIONS)
    ap.add_argument("--prepared_dir", type=Path, default=rp.SPACSEQ_PREPARED)
    ap.add_argument("--out", type=Path, default=rp.WORKDIR / "spacseq_results" / "spatial_method_diagnosis_unseen.csv")
    ap.add_argument("--n_hvgs", type=int, default=2000)
    args = ap.parse_args()

    bdir = args.prepared_dir / "benchmark"
    panel = load_eval_panel(bdir)
    eu, sa = sec.import_eval_utils()
    concert_genes = set(np.asarray(load_concert_genes()).astype(str).tolist())

    rows = []
    per_slide_pred: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    cache: dict[str, object] = {}
    for unit in panel["eval_units"]:
        gene, slide = unit["gene"], unit["target_slide"]
        if slide not in cache:
            cache.clear()
            cache[slide] = load_prepared_slide(args.prepared_dir, slide)
        adata = cache[slide]
        names = np.asarray(adata.var_names).astype(str)
        hvg = sec.select_hvgs(adata, args.n_hvgs, sa)
        in_concert = np.fromiter((g in concert_genes for g in names), dtype=bool, count=len(names))
        keep = np.asarray(hvg, dtype=bool) & in_concert & (names != gene)

        src_idx = load_source_cells(bdir, slide)
        ko_idx = ko_holdout_indices(bdir, gene, slide)
        ref_idx = unseen_reference_idx(adata, args.prepared_dir, slide)
        o_full = eu.pseudobulk_lfc(to_dense(adata[ko_idx].X), to_dense(adata[ref_idx].X))
        o = np.asarray(o_full, dtype=np.float64)[keep]
        rmse_zero = float(np.sqrt(np.mean(o ** 2)))
        _ = src_idx

        for m in METHODS:
            f = args.tree / m / "unseen" / "unseen_pert" / f"{gene}__slide_{slide}" / "model_lfc.json"
            if not f.is_file():
                continue
            v = json.loads(f.read_text())["L2"].get(gene)
            if not v:
                continue
            # model_lfc.json holds the whole gene axis; the metrics run on the scored genes only.
            p = np.asarray(v, dtype=np.float64)
            if len(p) == len(names):
                p = p[keep]
            if len(p) != len(o):
                continue
            per_slide_pred.setdefault((m, slide), {})[gene] = p
            denom = np.sqrt(np.mean(o ** 2))
            rows.append({
                "method": LABELS[m], "gene": gene, "slide": slide, "n_genes": int(keep.sum()),
                "rmse": float(np.sqrt(np.mean((p - o) ** 2))),
                "rmse_zero": rmse_zero,
                "mag_ratio": float(np.sqrt(np.mean(p ** 2)) / denom) if denom else float("nan"),
                "pearson": float(np.corrcoef(p, o)[0, 1]) if p.std() > 0 else float("nan"),
                "spearman": float(sa.spearman(p, o)) if hasattr(sa, "spearman") else float("nan"),
                "floor_frac": float(np.mean(p < FLOOR)),
                "frac_exact_zero": float(np.mean(np.abs(p) < 1e-9)),
            })
        print(f"[diag] {gene}@{slide} done", flush=True)

    within = {}
    for (m, slide), preds in per_slide_pred.items():
        if len(preds) < 2:
            continue
        vals = []
        for a, b in itertools.combinations(sorted(preds), 2):
            pa, pb = preds[a], preds[b]
            n = min(len(pa), len(pb))
            if pa[:n].std() > 0 and pb[:n].std() > 0:
                vals.append(float(np.corrcoef(pa[:n], pb[:n])[0, 1]))
        if vals:
            within.setdefault(LABELS[m], []).append(float(np.mean(vals)))

    import csv as _csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")

    print("\n=== per method, mean over units")
    hdr = f"{'method':<14}{'rmse':>8}{'rmse_zero':>11}{'|m|/|o|':>9}{'pearson':>9}{'floor%':>8}{'within_slide_r':>16}"
    print(hdr)
    for label in LABELS.values():
        sub = [r for r in rows if r["method"] == label]
        if not sub:
            continue
        g = lambda k: float(np.nanmean([r[k] for r in sub]))  # noqa: E731
        wr = float(np.mean(within.get(label, [float("nan")])))
        print(f"{label:<14}{g('rmse'):>8.2f}{g('rmse_zero'):>11.2f}{g('mag_ratio'):>9.2f}"
              f"{g('pearson'):>9.3f}{100 * g('floor_frac'):>8.1f}{wr:>16.3f}")


if __name__ == "__main__":
    main()
