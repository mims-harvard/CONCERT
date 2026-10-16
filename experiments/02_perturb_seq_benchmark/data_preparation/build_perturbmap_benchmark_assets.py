#!/usr/bin/env python3
"""Build the Perturb-map benchmark evaluation panel, holdout files and training manifests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark_evaluation"))

from benchmark_lib import (  # noqa: E402
    PM_PREPARED,
    PM_SLIDES,
    build_pm_eval_panel,
    write_benchmark_assets,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build the Perturb-map benchmark assets.")
    p.add_argument("--pm_prepared", type=Path, default=PM_PREPARED)
    p.add_argument("--min_ko_donor", type=int, default=15)
    p.add_argument("--min_ko_target", type=int, default=15, help="Target-slide minimum KO cells")
    p.add_argument("--max_train_cells", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    panel = build_pm_eval_panel(
        args.pm_prepared,
        slides=PM_SLIDES,
        min_ko_target=args.min_ko_target,
        min_ko_donor=args.min_ko_donor,
    )
    out = args.pm_prepared / "benchmark"
    write_benchmark_assets(
        args.pm_prepared,
        out,
        dataset="perturbmap",
        slides=PM_SLIDES,
        eval_panel=panel,
        max_train_cells=args.max_train_cells,
        seed=args.seed,
    )
    log.info("Perturb-map: %d eval units -> %s", panel["n_selected"], out)


if __name__ == "__main__":
    main()
