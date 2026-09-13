#!/usr/bin/env python3
"""Build Task2 train AnnData: drop masked patches + keep only eval5 KOs + ctrl.

Keeps: unassigned/control + {Ccn1, Iqgap1, Phgr1, Cttn, H2-Ea}.
Drops: role==1 patch cells, other map KOs, multiple, rare non-map KOs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

EVAL5 = ("Ccn1", "Iqgap1", "Phgr1", "Cttn", "H2-Ea")
CTRL = {
    "unassigned", "Unassigned", "no_guide", "No_guide", "unknown",
    "control", "Control", "CTRL", "ctrl", "NT", "NonTargeting",
    "Non_targeting", "non_targeting",
}

RAW = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_2-2.h5ad"
)
MASK = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/spot_plots/"
    "patch_removal_2-2_top5_cellindex.npz"
)
OUT = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/data_mask/"
    "raw_output_2-2_masked_v2_eval5.h5ad"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--mask", type=Path, default=MASK)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    z = np.load(args.mask, allow_pickle=True)
    drop_mask = set(np.asarray(z["obs_name"]).astype(str)[np.asarray(z["role"]) == 1])

    a = ad.read_h5ad(args.raw)
    names = np.asarray(a.obs_names).astype(str)
    pert = a.obs["perturbation"].astype(str).values
    keep_pert = np.isin(pert, list(EVAL5) + list(CTRL))
    keep_mask = ~np.isin(names, list(drop_mask))
    keep = keep_pert & keep_mask

    b = a[keep].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    b.write_h5ad(args.out)

    vc = b.obs["perturbation"].astype(str).value_counts()
    print(f"raw={a.n_obs} drop_mask={len(drop_mask)} keep_pert+kept={int(keep.sum())}")
    print(f"wrote n={b.n_obs} -> {args.out}")
    print("pert counts:\n", vc.to_string())
    still = drop_mask & set(np.asarray(b.obs_names).astype(str))
    print(f"leakage remaining: {len(still)}")
    assert len(still) == 0
    assert set(vc.index) <= set(EVAL5) | CTRL


if __name__ == "__main__":
    main()
