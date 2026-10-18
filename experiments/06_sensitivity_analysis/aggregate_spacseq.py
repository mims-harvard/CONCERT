#!/usr/bin/env python
"""Aggregate the SPAC-seq SI sensitivity arms into one tidy CSV.

SPAC-seq counterpart of aggregate_perturbmap.py. The layouts differ: SPAC arms are five-slide models
evaluated with evaluate_concert_spacseq.py, which writes
    <root>/seen_<tag>/patch/slide_<sid>/lfc_agg.json
and the sweep output is split across two batch roots. Emits one row per arm x slide x perturbation.

Seen regime only: the regime is fixed at train time (--holdout_file, run_spacseq_arm.sh:79),
so an unseen sweep would be 65 fresh trainings rather than a re-scoring.

Design: experiments/06_sensitivity_analysis/README.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark_evaluation"))
import release_paths as rp  # noqa: E402

# Arms are spread over two batch roots. First root holding a complete
# five-slide arm wins, so a half-written duplicate never shadows a finished one.
ROOTS = [rp.SPACSEQ_SENSITIVITY / "batch1", rp.SPACSEQ_SENSITIVITY / "batch2"]
SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]

# tag -> (factor, value at that arm), anchored at the REPORTED spac config
# (pre-clonegate base model: inducing 16, band [0.15,0.90], coord scale 20).
ARMS: dict[str, tuple[str, float]] = {
    "base42": ("base", 0.0),
    "seed1": ("seed", 1), "seed2": ("seed", 2), "seed3": ("seed", 3), "seed4": ("seed", 4),
    # footprint band reported as ls_max in loc units (= normalized x coord_scale 20)
    "lsb1": ("fpert_ls_max_loc", 6.0),
    "lsb3": ("fpert_ls_max_loc", 54.0),
    "ind8": ("fpert_svgp_inducing", 8), "ind64": ("fpert_svgp_inducing", 64),
    "cs10": ("fpert_coord_scale", 10.0), "cs40": ("fpert_coord_scale", 40.0),
    "mor0": ("moran_lambda", 0.0), "mor2": ("moran_lambda", 2.0),
    "lfc05": ("lfc_lambda", 0.5), "lfc2": ("lfc_lambda", 2.0),
    "dis2": ("disentangle_radius", 2.0), "dis8": ("disentangle_radius", 8.0),
    "nfd4": ("niche_film_dim", 4), "nfd16": ("niche_film_dim", 16),
    "nfk10": ("niche_film_k", 10), "nfk50": ("niche_film_k", 50),
    "brf64": ("basal_rff_features", 64), "brf256": ("basal_rff_features", 256),
    "brb005": ("basal_rff_bandwidth", 0.05), "brb030": ("basal_rff_bandwidth", 0.30),
}
BASE_VALUE = {
    "seed": 42, "fpert_ls_max_loc": 18.0, "fpert_svgp_inducing": 16,
    "fpert_coord_scale": 20.0, "moran_lambda": 1.0, "lfc_lambda": 1.0,
    "disentangle_radius": 4.0, "niche_film_dim": 8, "niche_film_k": 25,
    "basal_rff_features": 128, "basal_rff_bandwidth": 0.15,
}
SEED_TAGS = {"base42", "seed1", "seed2", "seed3", "seed4"}
REP_SEEDS = [42, 1, 2]
BASE_TAG_FOR_SEED = {42: "base42", 1: "seed1", 2: "seed2"}
METRICS = ["spearman_mean", "lfc_dist_mean", "sign_agreement_mean",
           "topk_overlap_mean", "abs_spearman_mean", "spearman_sd", "lfc_dist_sd",
           "spearman_boot_std_mean", "n_sig_mean"]


def arm_dir(tag: str, regime: str) -> Path | None:
    """Root holding a complete five-slide evaluation for this arm, else None."""
    for root in ROOTS:
        d = root / f"{regime}_{tag}" / "patch"
        if all((d / f"slide_{s}" / "lfc_agg.json").is_file() for s in SLIDES):
            return d
    return None


def rows_for(tag: str, factor: str, value: float, regime: str):
    d = arm_dir(tag, regime)
    if d is None:
        return
    for sid in SLIDES:
        agg = json.loads((d / f"slide_{sid}" / "lfc_agg.json").read_text())
        for level, perts in agg.items():
            for pert, m in perts.items():
                row = {"tag": tag, "factor": factor, "value": value, "regime": regime,
                       "root": str(d), "slide": sid, "level": level, "pert": pert,
                       "is_seed_arm": int(tag in SEED_TAGS)}
                row.update({k: m.get(k) for k in METRICS})
                yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="seen")
    ap.add_argument("--out", type=Path,
                    default=rp.WORKDIR / "sensitivity" / "sens_spac_arms.csv")
    args = ap.parse_args()

    rows, missing = [], []
    for tag, (factor, value) in ARMS.items():
        for seed in ([None] if tag in SEED_TAGS else REP_SEEDS):
            full = tag if seed in (None, 42) else f"{tag}_s{seed}"
            got = list(rows_for(full, factor, value, args.regime))
            if not got:
                missing.append(full)
                continue
            for r in got:
                r["seed"] = int(value) if tag in SEED_TAGS else seed
                r["tag"] = full
            rows.extend(got)

    # the base config anchors every factor panel, at the same seeds as the factor arms
    base = [r for r in rows if r["tag"] in BASE_TAG_FOR_SEED.values()]
    for factor, bval in BASE_VALUE.items():
        if factor == "seed":
            rows.extend({**r, "factor": "seed", "value": 42, "seed": 42}
                        for r in base if r["tag"] == "base42")
            continue
        seed_of = {v: k for k, v in BASE_TAG_FOR_SEED.items()}
        for r in base:
            rows.append({**r, "factor": factor, "value": bval, "seed": seed_of[r["tag"]]})
    rows = [r for r in rows if r["factor"] != "base"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    if missing:
        print(f"[WARN] no complete results for arms: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
