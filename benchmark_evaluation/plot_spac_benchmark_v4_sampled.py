#!/usr/bin/env python3
"""Sampled-output run of plot_spac_benchmark_v4.py (same plotting code and style, sampled roots).

Per-unit JSON roots for CONCERT and competitors (seen and unseen) point at multi_1-5_ho_v4_sampled;
the rescored CSVs come from --outdir as before.
Usage: plot_spac_benchmark_v4_genefix.py {bars|dots} --regime {seen|unseen} --concert-v3 DIR --outdir DIR [...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_spac_benchmark_v4 as p4  # noqa: E402

GF = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5_ho_v4_sampled")
p4.V4 = GF
p4.pb.SPAC_V3 = GF
p4.pb.METHOD_SOURCES[:] = [(label, GF, sub) for label, _, sub in p4.pb.METHOD_SOURCES]
p4.pd_.SPAC_V3 = GF

if __name__ == "__main__":
    p4.main()
