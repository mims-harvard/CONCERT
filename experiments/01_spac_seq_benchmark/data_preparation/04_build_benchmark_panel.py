#!/usr/bin/env python3
"""Build the SPAC-seq benchmark assets -> multi_1-5/prepared/benchmark/.

When --expected_panel_json is given, the built panel must reproduce it (gene, target slide,
n_donor, n_target, n_total).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark_evaluation"))

from benchmark_holdouts import SPAC_DONOR_SLIDE, SPAC_OTHER_SLIDES, SPAC_PREPARED  # noqa: E402
from spacseq_cross_slide_panel import (  # noqa: E402
    MAX_PER_TARGET_SLIDE,
    MAX_TOTAL,
    MIN_KO_DONOR,
    MIN_KO_TARGET,
    build_spac_eval_panel,
    write_benchmark_assets,
)
from spacseq_slides import DEFAULT_SLIDES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared_dir", type=Path, default=SPAC_PREPARED)
    p.add_argument("--out_subdir", default="benchmark")
    p.add_argument("--min_ko_donor", type=int, default=MIN_KO_DONOR)
    p.add_argument("--min_ko_target", type=int, default=MIN_KO_TARGET)
    p.add_argument("--max_per_target_slide", type=int, default=MAX_PER_TARGET_SLIDE,
                   help="Max knockouts per target slide (0 = no cap)")
    p.add_argument("--max_total", type=int, default=MAX_TOTAL, help="Max knockouts overall (0 = no cap)")
    p.add_argument("--expected_panel_json", type=Path, default=None,
                   help="panel_cutoff_scan list of {gene,n_donor,target,n_target,n_total} to assert against")
    p.add_argument("--expected_ko_counts_json", type=Path, default=None,
                   help="panel_cutoff_scan/ko_counts.json; assert the panel equals its rows with "
                        "n_donor>=min_ko_donor and n_target>=min_ko_target (no-cap rule only)")
    p.add_argument("--expected_n", type=int, default=None, help="Assert this many selected units")
    p.add_argument("--max_train_cells", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--panel_only", action="store_true", help="Build + assert the panel, write nothing")
    return p.parse_args()


def assert_panel_vs_ko_counts(panel: dict, ko_counts_path: Path) -> None:
    assert panel["max_per_target_slide"] == 0 and panel["max_total"] == 0, "ko_counts assert is for the no-cap rule"
    rows = json.loads(ko_counts_path.read_text())["rows"]
    want = sorted((r["gene"], str(r["target"]), int(r["n_donor"]), int(r["n_target"]), int(r["n_total"]))
                  for r in rows if r["n_donor"] >= panel["min_ko_donor"] and r["n_target"] >= panel["min_ko_target"])
    got = sorted((u["gene"], u["target_slide"], u["n_ko_donor"], u["n_ko_target"], u["n_ko_total"])
                 for u in panel["eval_units"])
    if got != want:
        raise AssertionError(f"panel mismatch vs {ko_counts_path}: missing={sorted(set(want) - set(got))} "
                             f"extra={sorted(set(got) - set(want))}")
    log.info("PANEL ASSERT PASSED: %d units == ko_counts rows with n_donor>=%d and n_target>=%d (%s)",
             len(got), panel["min_ko_donor"], panel["min_ko_target"], ko_counts_path)


def assert_panel(panel: dict, expected_path: Path | None, expected_n: int | None) -> None:
    got = [(u["gene"], u["target_slide"], u["n_ko_donor"], u["n_ko_target"], u["n_ko_total"])
           for u in panel["eval_units"]]
    if expected_n is not None:
        assert len(got) == expected_n, f"panel size {len(got)} != expected {expected_n}"
    if expected_path is not None:
        exp = json.loads(expected_path.read_text())
        want = [(r["gene"], str(r["target"]), int(r["n_donor"]), int(r["n_target"]), int(r["n_total"])) for r in exp]
        if sorted(got) != sorted(want):
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            raise AssertionError(f"panel mismatch vs {expected_path}: missing={missing} extra={extra}")
        log.info("PANEL ASSERT PASSED: %d units match %s (set equality on gene/target/counts)", len(got), expected_path)
        if [g[0] for g in got] != [w[0] for w in want]:
            log.info("note: order differs from expected file (built order = per-rank, expected order kept as given)")


def main():
    args = parse_args()
    panel = build_spac_eval_panel(
        args.prepared_dir,
        donor_slide=SPAC_DONOR_SLIDE,
        other_slides=SPAC_OTHER_SLIDES,
        min_ko_donor=args.min_ko_donor,
        min_ko_target=args.min_ko_target,
        max_per_target_slide=args.max_per_target_slide,
        max_total=args.max_total,
    )
    log.info("panel: eligible=%d (%s) selected=%d (%s)", panel["n_eligible"],
             panel["n_eligible_by_target_slide"], panel["n_selected"], panel["n_selected_by_target_slide"])
    assert_panel(panel, args.expected_panel_json, args.expected_n)
    if args.expected_ko_counts_json is not None:
        assert_panel_vs_ko_counts(panel, args.expected_ko_counts_json)
    if args.panel_only:
        return
    out = args.prepared_dir / args.out_subdir
    write_benchmark_assets(
        args.prepared_dir, out, slides=list(DEFAULT_SLIDES), eval_panel=panel,
        max_train_cells=args.max_train_cells, seed=args.seed,
    )
    log.info("DONE -> %s", out)


if __name__ == "__main__":
    main()
