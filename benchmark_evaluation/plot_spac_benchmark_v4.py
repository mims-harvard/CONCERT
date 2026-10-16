#!/usr/bin/env python3
"""Benchmark v4 figures: run plot_spac_benchmark_bars.py / plot_spac_concert_vs_competitor_dots.py
with v4 roots, without editing either script.

  --regime unseen : panel = benchmark_v4/eval_panel.json, competitor roots = multi_1-5_ho_v4
  --regime seen   : competitor roots stay multi_1-5_ho_v3 (seen unchanged); CONCERT via --concert-v3
All other flags are passed through (e.g. --outdir, --concert-v3, --top-n).
Method colors are drawn at 70% opacity.

Usage: plot_spac_benchmark_v4.py {bars|dots} --regime {seen|unseen} --concert-v3 DIR --outdir DIR [...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_spac_benchmark_bars as pb  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

# Method colors drawn at 70% opacity (author request, 2026-09-13). Applied to the shared COLORS dict
# before the dots module builds its color table from it, so bars, dots and legends all match.
BAR_ALPHA = 0.7
for _k, _c in list(pb.COLORS.items()):
    pb.COLORS[_k] = to_rgba(_c, BAR_ALPHA)

import plot_spac_concert_vs_competitor_dots as pd_  # noqa: E402

# One per-knockout error for every method (author decision, 2026-09-14): s.e. from 200 gene-bootstrap
# resamples (score_lfc_agreement), read from the rescored CSVs. load_lfc_dist_from_csv
# (plot_spac_benchmark_bars.py:423-436) has no entry for the v4 CONCERT keys, so CONCERT rows were
# skipped and CONCERT kept the JSON-path error of summarize() (:123-135: s.e.m. across niche records,
# or the eval-native n_boot=10 bootstrap). Map the v4 keys onto keys the loader knows.
import csv as _csv  # noqa: E402
import tempfile as _tempfile  # noqa: E402

_V4_CONCERT_KEYS = {"concert_nichefix_seen": "concert_niche_seen", "concert_v4_unseen": "concert_niche_unseen"}
_orig_load_csv = pb.load_lfc_dist_from_csv


def _load_csv_v4(csv_path, genes):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return _orig_load_csv(csv_path, genes)
    rows = list(_csv.DictReader(csv_path.open()))
    if not any(r["method"] in _V4_CONCERT_KEYS for r in rows):
        return _orig_load_csv(csv_path, genes)
    for r in rows:
        r["method"] = _V4_CONCERT_KEYS.get(r["method"], r["method"])
    with _tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return _orig_load_csv(Path(fh.name), genes)


pb.load_lfc_dist_from_csv = _load_csv_v4
_orig_plot_grid = pb.plot_grid


def _plot_grid_checked(genes, per_gene, overall, metric_key, *a, **k):
    # Report every plotted per-knockout bar that carries no error (would draw as a bare bar).
    zero = [(g, lab) for g in genes for lab, e in per_gene[g].items()
            if isinstance(e, dict) and pb._stat_valid(e, metric_key) and not (e[metric_key][1] > 0)]
    print(f"[errbar-check] {metric_key}: per-knockout bars without error = {len(zero)} {zero[:10]}")
    return _orig_plot_grid(genes, per_gene, overall, metric_key, *a, **k)


pb.plot_grid = _plot_grid_checked

EVAL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval")
V4 = EVAL / "multi_1-5_ho_v4"
PANEL_V4 = EVAL / "multi_1-5/prepared/benchmark_v4/eval_panel.json"


def main():
    which = sys.argv[1]
    rest = sys.argv[2:]
    regime = rest[rest.index("--regime") + 1] if "--regime" in rest else "seen"
    pb.SPAC_PANEL = PANEL_V4
    if regime == "unseen":
        pb.SPAC_V3 = V4
        pb.METHOD_SOURCES[:] = [(label, V4, sub) for label, _, sub in pb.METHOD_SOURCES]
        pd_.SPAC_V3 = V4
    sys.argv = [f"plot_spac_benchmark_{which}"] + rest
    {"bars": pb.main, "dots": pd_.main}[which]()


if __name__ == "__main__":
    main()
