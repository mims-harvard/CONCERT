"""Where the released data, checkpoints and working outputs live.

Two folders hold every input: `data/` and `checkpoints/`. Place both in the
repository root; reruns write to `runs/` there. Each root can be moved with an environment
variable (`CONCERT_DATA_DIR`, `CONCERT_CHECKPOINT_ROOT`, `CONCERT_WORKDIR`). Every path below is
derived from these three roots; `data/README.md` describes each file.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


DATA = _env("CONCERT_DATA_DIR", REPO / "data")
CHECKPOINTS = _env("CONCERT_CHECKPOINT_ROOT", REPO / "checkpoints")
WORKDIR = _env("CONCERT_WORKDIR", REPO / "runs")

# SPAC-seq: the five measured sections and the matched cell sets the scoring reads
SPACSEQ_RAW = DATA / "spacseq" / "raw"                    # raw_output_1-{1..5}.h5ad
SPACSEQ_PREPARED = DATA / "spacseq" / "prepared"          # slide_1-*.h5ad, raw_counts/, holdout_patch/

# Perturb-map: the four sections in the model format, the prepared cell sets and the niche patches
PERTURBMAP_RAW = DATA / "perturbmap" / "raw"              # perturbmap_s{0..3}_spacfmt.h5ad
PERTURBMAP_PREPARED = DATA / "perturbmap" / "prepared"    # slide_{0..3}.h5ad, holdout_patch/, ...
PERTURBMAP_PATCHES = DATA / "perturbmap" / "patches"      # slide{0..3}.npz

# Benchmark assets: holdout panels, shared source cells, holdout coordinates, shipped results
ASSETS = DATA / "benchmark_assets"
SPACSEQ_BENCHMARK = ASSETS / "spacseq_benchmark"          # eval_panel.json, holdout_*/, source_cells/
PERTURBMAP_BENCHMARK = ASSETS / "perturbmap_benchmark"    # eval_panel.json, holdout_*/
HOLDOUT_COORDS = ASSETS / "holdout_coords"                # holdout_{spac,pmap}_{seen,unseen}.json
STATE_PREDICTIONS = ASSETS / "state_predictions"          # STATE predictions, scored by the rescore
RESULTS = ASSETS / "results"
SPACSEQ_PREDICTIONS = RESULTS / "spacseq_predictions"     # every method's per-unit SPAC-seq results
SPACSEQ_RESULTS = RESULTS / "spacseq_results"             # the reported SPAC-seq tables and figures
SPACSEQ_REFERENCE_HVGS = RESULTS / "spacseq_reference_hvgs"  # per-slide HVG lists of the reference run
SPACSEQ_SENSITIVITY = RESULTS / "spacseq_sensitivity"     # SI sensitivity sweep, one dir per arm
PERTURBMAP_PREDICTIONS = RESULTS / "perturbmap_predictions"  # competitor per-section results
PERTURBMAP_CONCERT = RESULTS / "perturbmap_concert"       # CONCERT per-section results
PERTURBMAP_RESULTS = RESULTS / "perturbmap_results"       # the reported Perturb-map tables and figures
HARNESS_TABLES = RESULTS / "harness_tables"               # record sets and Perturb-map combined tests

# Case studies
DSS = DATA / "dss_colitis"
STROKE = DATA / "stroke"
PERTURBFISH = DATA / "perturbfish"
DECONVOLUTION = DATA / "deconvolution"


def show(path) -> str:
    """`path` written against the root it sits under ($CONCERT_DATA_DIR/..., <repo>/...), for printing."""
    roots = [("$CONCERT_NB_WORKDIR", os.environ.get("CONCERT_NB_WORKDIR")),
             ("$CONCERT_DATA_DIR", DATA), ("$CONCERT_CHECKPOINT_ROOT", CHECKPOINTS),
             ("$CONCERT_WORKDIR", WORKDIR), ("<repo>", REPO)]
    p = str(path)
    for cand in (p, os.path.realpath(p)):
        for name, root in roots:
            if not root:
                continue
            for r in (str(root), os.path.realpath(str(root))):
                if cand == r or cand.startswith(r.rstrip("/") + "/"):
                    return name + cand[len(r.rstrip("/")):]
    return p
