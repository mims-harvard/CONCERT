#!/usr/bin/env python
"""Build CONCERT_Supplementary_tables.xlsx from the current benchmark statistics.

The SI promises this file and it did not exist. Two sheets, one per metric, each carrying raw and
BH-adjusted p-values for all three benchmark arms.
"""
import pandas as pd
from pathlib import Path
SRC = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/spac_seq_lsfix")
OUT = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/manuscript/CONCERT_Supplementary_tables.xlsx")
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
