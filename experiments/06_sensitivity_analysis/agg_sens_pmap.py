#!/usr/bin/env python
"""Aggregate the SI sensitivity arms into one tidy CSV.

Walks concert_sens_<regime>_<tag>/{seen,unseen}_pert/slide_*/lfc_agg.json and emits one row per
arm x regime x slide x perturbation, carrying the factor each arm moves and its value so the
figure can group panels without re-parsing tags.

Design: docs/superpowers/specs/2026-08-17-concert-sensitivity-analysis-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark_evaluation"))
import release_paths as rp  # noqa: E402

RES = rp.PERTURBMAP_CONCERT / "sensitivity"

# tag -> (factor, value at that arm). The base arm belongs to every factor panel, so it is
# expanded into one row per factor by BASE_IN_ALL below.
ARMS: dict[str, tuple[str, float]] = {
    "base42": ("base", 0.0),
    "seed1": ("seed", 1), "seed2": ("seed", 2), "seed3": ("seed", 3), "seed4": ("seed", 4),
    # footprint band reported as ls_max in loc units (= normalized x coord_scale 20)
    "lsb1": ("fpert_ls_max_loc", 6.0),
    "lsb2": ("fpert_ls_max_loc", 18.0),
    "lsb3": ("fpert_ls_max_loc", 54.0),
    "ind16": ("fpert_svgp_inducing", 16), "ind32": ("fpert_svgp_inducing", 32),
    "ind128": ("fpert_svgp_inducing", 128),
    "cs10": ("fpert_coord_scale", 10.0), "cs40": ("fpert_coord_scale", 40.0),
    "cs70": ("fpert_coord_scale", 70.0),
    "mor0": ("moran_lambda", 0.0), "mor05": ("moran_lambda", 0.5), "mor2": ("moran_lambda", 2.0),
    "lfc05": ("lfc_lambda", 0.5), "lfc2": ("lfc_lambda", 2.0),
    "dis2": ("disentangle_radius", 2.0), "dis8": ("disentangle_radius", 8.0),
    "nfd4": ("niche_film_dim", 4), "nfd16": ("niche_film_dim", 16),
    "nfk10": ("niche_film_k", 10), "nfk50": ("niche_film_k", 50),
    "brf64": ("basal_rff_features", 64), "brf256": ("basal_rff_features", 256),
    "brb005": ("basal_rff_bandwidth", 0.05), "brb030": ("basal_rff_bandwidth", 0.30),
}
# value the base config sits at, per factor -- the base arm is replicated into each panel there.
BASE_VALUE = {
    "seed": 42, "fpert_ls_max_loc": 360.0, "fpert_svgp_inducing": 64,
    "fpert_coord_scale": 20.0, "moran_lambda": 1.0, "lfc_lambda": 1.0,
    "disentangle_radius": 4.0, "niche_film_dim": 8, "niche_film_k": 25,
    "basal_rff_features": 128, "basal_rff_bandwidth": 0.15,
}
SEED_TAGS = {"base42", "seed1", "seed2", "seed3", "seed4"}
# Every factor arm is replicated at these seeds (tag suffix _s<N>); the base point in each
# factor panel is the same three seeds, so arm and base carry comparable error bars.
REP_SEEDS = [42, 1, 2]
BASE_TAG_FOR_SEED = {42: "base42", 1: "seed1", 2: "seed2"}
# lfc_dist is the LFC RMSE reported in the benchmark figure (perturbmap_harness_rescore.py:2)
METRICS = ["spearman_mean", "lfc_dist_mean", "e_dist_mean", "sign_agreement_mean",
           "topk_overlap_mean", "spearman_sd", "lfc_dist_sd", "e_dist_sd",
           "spearman_boot_std_mean", "n_scored_genes"]


def rows_for(tag: str, factor: str, value: float, regime: str):
    d = RES / f"concert_sens_{regime}_{tag}"
    if not d.is_dir():
        return
    for scope in ("seen_pert", "unseen_pert"):
        for slide_dir in sorted((d / scope).glob("slide_*")):
            f = slide_dir / "lfc_agg.json"
            if not f.is_file():
                continue
            agg = json.loads(f.read_text())
            for level, perts in agg.items():
                for pert, m in perts.items():
                    row = {"tag": tag, "factor": factor, "value": value, "regime": regime,
                           "scope": scope, "slide": slide_dir.name.split("_")[-1],
                           "level": level, "pert": pert,
                           "is_seed_arm": int(tag in SEED_TAGS)}
                    row.update({k: m.get(k) for k in METRICS})
                    yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="seen")
    ap.add_argument("--out", type=Path,
                    default=rp.WORKDIR / "sensitivity" / "sens_pmap_arms.csv")
    args = ap.parse_args()

    rows = []
    missing = []
    for tag, (factor, value) in ARMS.items():
        for seed in ([None] if tag in SEED_TAGS else REP_SEEDS):
            # seed 42 is the original un-suffixed arm; replicates carry _s<N>
            full = tag if seed in (None, 42) else f"{tag}_s{seed}"
            got = list(rows_for(full, factor, value, args.regime))
            if not got:
                missing.append(full)
                continue
            for r in got:
                r["seed"] = int(value) if tag in SEED_TAGS else seed
            rows.extend(got)
    # the base config anchors every factor panel, at the same seeds as the factor arms
    base = [r for r in rows if r["tag"] in BASE_TAG_FOR_SEED.values()]
    for factor, bval in BASE_VALUE.items():
        if factor == "seed":
            # the seed panel already holds seeds 1-4 as their own arms; add seed 42 from base
            rows.extend({**r, "factor": "seed", "value": 42, "seed": 42}
                        for r in base if r["tag"] == "base42")
            continue
        for r in base:
            seed = {v: k for k, v in BASE_TAG_FOR_SEED.items()}[r["tag"]]
            rows.append({**r, "factor": factor, "value": bval, "seed": seed})
    rows = [r for r in rows if r["factor"] != "base"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    if missing:
        print(f"[WARN] no results for arms: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
