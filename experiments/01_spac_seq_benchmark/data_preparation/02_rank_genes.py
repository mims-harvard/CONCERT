#!/usr/bin/env python3
"""
Compute HVG ranking metrics for SPAC-seq h5ad files and write them to ``.var``.

Uses ``seurat_v3`` on **raw counts** (no log1p).  Raw counts in ``adata.X`` are
preserved.

Continuous columns added to ``adata.var`` (prefix ``hvg_``):
  - hvg_means
  - hvg_dispersions          (or variances if scanpy falls back)
  - hvg_dispersions_norm     -- primary score; higher = more variable
  - hvg_rank                 -- 1 = top HVG by ``hvg_dispersions_norm``

Use ``hvg_rank <= N`` (or sort by ``hvg_dispersions_norm`` descending) to pick
any number of HVGs later without rerunning this step.

Usage
-----
python 02_rank_genes.py --h5ad /path/to/raw_output_1-1.h5ad
python 02_rank_genes.py --h5ad /path/to/raw_output_1-1.h5ad --out_h5ad /path/to/out.h5ad
"""

from __future__ import annotations

import argparse
import gc
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "means": ("means",),
    "dispersions": ("dispersions", "variances"),
    "dispersions_norm": ("dispersions_norm", "variances_norm"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute and store HVG metrics in adata.var")
    p.add_argument("--h5ad", required=True, type=Path, help="Input h5ad (raw counts in X)")
    p.add_argument(
        "--out_h5ad",
        default=None,
        type=Path,
        help="Output path (default: overwrite --h5ad in place)",
    )
    p.add_argument(
        "--n_top_genes",
        type=int,
        default=2000,
        help="n_top_genes passed to scanpy (sets boolean flag; all gene metrics kept)",
    )
    return p.parse_args()


def _pick_series(var_df: pd.DataFrame, aliases: tuple[str, ...], label: str) -> pd.Series:
    for name in aliases:
        if name in var_df.columns:
            return var_df[name]
    raise RuntimeError(
        f"scanpy did not produce {label!r}; tried {aliases}; "
        f"got columns: {list(var_df.columns)}"
    )


def compute_hvg_metrics(adata, *, n_top_genes: int) -> pd.DataFrame:
    """Return per-gene HVG metrics from a copy; does not modify ``adata.X``."""
    tmp = adata.copy()
    if sp.issparse(tmp.X) and not sp.isspmatrix_csr(tmp.X):
        tmp.X = tmp.X.tocsr()
    if not tmp.var_names.is_unique:
        tmp.var_names_make_unique()

    # seurat_v3 expects raw counts -- do NOT log1p (non-integers trigger fallback columns).
    sc.pp.highly_variable_genes(
        tmp,
        n_top_genes=min(n_top_genes, tmp.n_vars),
        flavor="seurat_v3",
    )

    metrics = pd.DataFrame(index=tmp.var_names)
    for out_key, aliases in METRIC_ALIASES.items():
        metrics[f"hvg_{out_key}"] = _pick_series(tmp.var, aliases, out_key).astype(np.float32)

    score = metrics["hvg_dispersions_norm"].astype(float)
    metrics["hvg_rank"] = (-score).rank(method="first").astype(np.int32)
    del tmp
    gc.collect()
    return metrics.reindex(adata.var_names)


def main() -> None:
    args = parse_args()
    in_path = args.h5ad.resolve()
    out_path = (args.out_h5ad or args.h5ad).resolve()

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    sc.settings.verbosity = 2
    print(f"\n[1] Loading {in_path}")
    adata = sc.read_h5ad(in_path)
    print(f"    Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    print("\n[2] Computing HVG metrics (seurat_v3 on raw counts, copy only)")
    metrics = compute_hvg_metrics(adata, n_top_genes=args.n_top_genes)

    for col in metrics.columns:
        adata.var[col] = metrics[col].values

    n_valid = int(np.isfinite(metrics["hvg_dispersions_norm"]).sum())
    top = metrics.nsmallest(5, "hvg_rank")[["hvg_dispersions_norm", "hvg_rank"]]
    print(f"    Genes with metrics: {n_valid:,}/{adata.n_vars:,}")
    print(f"    Top-5 by hvg_rank:\n{top.to_string()}")

    print(f"\n[3] Writing {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path == in_path:
        fd, tmp_name = tempfile.mkstemp(
            suffix=".h5ad", dir=out_path.parent, prefix=f".{out_path.stem}_"
        )
        os.close(fd)
        try:
            adata.write_h5ad(tmp_name)
            os.replace(tmp_name, out_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise
    else:
        adata.write_h5ad(out_path)
    print("    Done.")


if __name__ == "__main__":
    main()
