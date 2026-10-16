#!/usr/bin/env python
"""Build CONCERT_Supplementary_tables.xlsx from the current benchmark statistics.

The SI promises this file and it did not exist. Two sheets, one per metric, each carrying raw and
BH-adjusted p-values for all three benchmark arms.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import pandas as pd
from pathlib import Path
SRC = Path(f"{_rp.WORKDIR}/paper/Outputs/spac_seq_lsfix")
OUT = Path(f"{_rp.WORKDIR}/paper/manuscript/CONCERT_Supplementary_tables.xlsx")
COLS = {"method":"Method",
        "spacseq_seen_p":"SPAC-seq seen: p", "spacseq_seen_p_adj":"SPAC-seq seen: p (BH)",
        "spacseq_unseen_p":"SPAC-seq unseen: p", "spacseq_unseen_p_adj":"SPAC-seq unseen: p (BH)",
        "perturbmap_seen_p":"Perturb-map seen: p", "perturbmap_seen_p_adj":"Perturb-map seen: p (BH)"}
with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
    for metric, fn, sheet in (("LFC rank correlation", "concert_vs_competitor_lfc_rho_stats_combined.csv", "Table 1 - LFC rho"),
                              ("LFC RMSE",             "concert_vs_competitor_lfc_rmse_stats_combined.csv", "Table 2 - LFC RMSE")):
        df = pd.read_csv(SRC/fn).rename(columns=COLS)
        df.to_excel(xl, sheet_name=sheet, index=False)
        print(f"  {sheet}: {len(df)} methods, {df.shape[1]-1} statistic columns  [{metric}]")
print("wrote", OUT)
