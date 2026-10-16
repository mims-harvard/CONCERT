#!/usr/bin/env python3
"""Sampled-output run of rescore_spac_lfc_distance_v4.py (unchanged code, sampled roots: competitors --sampled_output True, CONCERT clonegate poisson_depth).

All per-unit results come from multi_1-5_ho_v4_sampled/<method>/{seen,unseen}, written by genefix_wrap.py.
Their hvg_info.json already holds the shared gene set (hvg_rank <= 2000 AND CONCERT model gene), and
build_gene_mask removes the KO gene, so every method is rescored on the same genes.
Seen competitor roots move from multi_1-5_ho_v3/<method> to multi_1-5_ho_v4_sampled/<method>.

Usage: rescore_spac_lfc_distance_v4_genefix.py --regime {seen,unseen,all} --outdir DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_spac_lfc_distance_v4 as rs  # noqa: E402

GF = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5_ho_v4_sampled")
rs.SPAC_V4 = GF
rs.CONCERT_V3 = GF / "concert_clonegate"
rs.METHODS = {m: (GF / m, label) for m, (_, label) in rs.METHODS.items()}
rs.METHODS_UNSEEN = {m: (GF / m, label) for m, (_, label) in rs.METHODS.items()}

if __name__ == "__main__":
    if "--concert-v3" not in sys.argv:
        sys.argv += ["--concert-v3", str(rs.CONCERT_V3)]
    rs.main()
