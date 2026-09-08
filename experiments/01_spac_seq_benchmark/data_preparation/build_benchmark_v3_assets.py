#!/usr/bin/env python3
"""Build benchmark v3 eval panel, holdout files, and train manifests (SPAC + PM)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_v3_lib import (
    DEFAULT_TOP_N_SPAC,
    PM_PREPARED,
    PM_SLIDES,
    SPAC_DONOR_SLIDE,
    SPAC_OTHER_SLIDES,
    SPAC_PREPARED,
    build_pm_eval_panel,
    build_spac_eval_panel,
    write_benchmark_v3_assets,
)
from spacseq_multi_common import DEFAULT_SLIDES

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build benchmark v3 assets for SPAC-seq and/or Perturb-map.")
    p.add_argument("--dataset", choices=["spacseq", "perturbmap", "both"], default="both")
    p.add_argument("--spac_prepared", type=Path, default=SPAC_PREPARED)
    p.add_argument("--pm_prepared", type=Path, default=PM_PREPARED)
    p.add_argument("--top_n_spac", type=int, default=DEFAULT_TOP_N_SPAC)
    p.add_argument("--min_ko", type=int, default=15,
                   help="Default min KO for donor/target when dataset overrides not set")
    p.add_argument("--min_ko_donor", type=int, default=None)
    p.add_argument("--min_ko_target", type=int, default=None,
                   help="If set, overrides both SPAC and PM target thresholds")
    p.add_argument("--min_ko_target_spac", type=int, default=100,
                   help="SPAC target-slide min KO (default 100)")
    p.add_argument("--min_ko_target_pm", type=int, default=15,
                   help="PM target-slide min KO (default 15)")
    p.add_argument("--max_train_cells_spac", type=int, default=50000)
    p.add_argument("--max_train_cells_pm", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    min_ko_donor = args.min_ko_donor if args.min_ko_donor is not None else args.min_ko
    if args.min_ko_target is not None:
        spac_min_ko_target = pm_min_ko_target = args.min_ko_target
    else:
        spac_min_ko_target = args.min_ko_target_spac
        pm_min_ko_target = args.min_ko_target_pm

    if args.dataset in ("spacseq", "both"):
        log.info("=== SPAC-seq benchmark v3 ===")
        panel = build_spac_eval_panel(
            args.spac_prepared,
            donor_slide=SPAC_DONOR_SLIDE,
            other_slides=SPAC_OTHER_SLIDES,
            top_n=args.top_n_spac,
            min_ko_donor=min_ko_donor,
            min_ko_target=spac_min_ko_target,
        )
        out = args.spac_prepared / "benchmark_v3"
        write_benchmark_v3_assets(
            args.spac_prepared,
            out,
            dataset="spacseq",
            slides=list(DEFAULT_SLIDES),
            eval_panel=panel,
            max_train_cells=args.max_train_cells_spac,
            seed=args.seed,
        )
        log.info("SPAC: %d eval units (of %d eligible) -> %s", panel["n_selected"], panel["n_eligible"], out)

    if args.dataset in ("perturbmap", "both"):
        log.info("=== Perturb-map benchmark v3 ===")
        panel = build_pm_eval_panel(
            args.pm_prepared,
            slides=PM_SLIDES,
            min_ko_target=pm_min_ko_target,
            min_ko_donor=min_ko_donor,
        )
        out = args.pm_prepared / "benchmark_v3"
        write_benchmark_v3_assets(
            args.pm_prepared,
            out,
            dataset="perturbmap",
            slides=PM_SLIDES,
            eval_panel=panel,
            max_train_cells=args.max_train_cells_pm,
            seed=args.seed,
        )
        log.info("PM: %d eval units -> %s", panel["n_selected"], out)


if __name__ == "__main__":
    main()
