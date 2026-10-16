#!/usr/bin/env python3
"""Rescore STATE delivery predictions under OUR LFC rule.

STATE H5ADs (delivery 2026-07-18):
  X[i, :] = predicted expression for source cell source_idx[i]
  columns = score-panel genes (SPAC ~2000; PM ~1056)

Our rule:
  m_lfc = log(mean(pred)) - log(mean(source[source_idx]))
  o_lfc = KO vs unassigned (SPAC niche) or KO vs control (PM)
  score panel and HVG, drop KO gene; Spearman rho + RMSE.
"""
from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
from pathlib import Path

import release_paths as rp  # noqa: E402

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfc_scoring import (  # noqa: E402
    build_gene_mask,
    load_hvg_indices,
    observed_lfc_pm_niche,
    observed_lfc_seen_patch_niche,
    pseudobulk_lfc,
    score_lfc_agreement,
    to_dense,
)

PREP_SPAC = rp.SPACSEQ_PREPARED
# Per-slide HVG lists of the reference CONCERT run; the kNN and PCA features use these columns.
SPAC_HVG_ROOT = rp.SPACSEQ_REFERENCE_HVGS
PM_PREP = rp.PERTURBMAP_PREPARED
PM_CONCERT = rp.PERTURBMAP_CONCERT / "within_niche_alternative"
# CONCERT per-record tables that fix the record set of each reference strategy
OUT = rp.HARNESS_TABLES / "record_sets"
MASTER_SPAC = OUT / "spac_seen_lfc_dist_per_record.csv"
MASTER_SPAC_UNSEEN = OUT / "spac_unseen_lfc_dist_per_record.csv"
MASTER_PM = OUT / "pm_seen_lfc_dist_per_record.csv"

STATE_SPAC = rp.STATE_PREDICTIONS / "spacseq"
STATE_PM = rp.STATE_PREDICTIONS / "perturbmap_seen"

MIN_G = 5
NICHE_RE = re.compile(r"^niche(\d+)_(.+)\.h5ad$")


def concert_units(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    c = df[df["method"].astype(str).str.contains("concert", case=False)].copy()
    c["niche_key"] = c["niche_key"].astype(str)
    c["slide"] = c["slide"].astype(str)
    c["gene"] = c["gene"].astype(str)
    return c[["gene", "slide", "niche_key"]].drop_duplicates()


def hvg_for_spac_slide(slide: str) -> np.ndarray | None:
    return load_hvg_indices(SPAC_HVG_ROOT / "seen" / "patch" / f"slide_{slide}")


def align_score(
    pred: ad.AnnData,
    adata: ad.AnnData,
    o_lfc: np.ndarray,
    keep_full: np.ndarray,
) -> dict | None:
    """Align STATE panel genes to slide, use source_idx, score LFC agreement."""
    slide_genes = np.asarray(adata.var_names).astype(str)
    pred_genes = np.asarray(pred.var_names).astype(str)
    name_to_i = {g: i for i, g in enumerate(slide_genes)}
    cols, pred_cols = [], []
    for j, g in enumerate(pred_genes):
        if g in name_to_i:
            cols.append(name_to_i[g])
            pred_cols.append(j)
    if len(cols) < 10:
        return None
    cols = np.asarray(cols, dtype=int)
    pred_cols = np.asarray(pred_cols, dtype=int)

    src_idx = pred.obs["source_idx"].astype(int).to_numpy()
    ok = (src_idx >= 0) & (src_idx < adata.n_obs)
    if int(ok.sum()) < MIN_G:
        return None
    src_idx = src_idx[ok]
    pred_X = to_dense(pred.X)[ok][:, pred_cols].astype(np.float64)
    src_X = to_dense(adata[src_idx].X[:, cols]).astype(np.float64)
    o = np.asarray(o_lfc, dtype=np.float64)[cols]
    keep = np.asarray(keep_full, dtype=bool)[cols]
    if int(keep.sum()) < 10:
        return None
    m = pseudobulk_lfc(pred_X, src_X)
    sc = score_lfc_agreement(m[keep], o[keep])
    sc["n_scored_genes"] = int(keep.sum())
    return sc


def record(gene, slide, niche_key, sc) -> dict:
    return {
        "method": "state",
        "method_label": "STATE",
        "gene": str(gene),
        "slide": str(slide),
        "niche_key": str(niche_key),
        "n_scored_genes": int(sc["n_scored_genes"]),
        "lfc_spearman": float(sc["spearman"]),
        "lfc_dist": float(sc["lfc_dist"]),
        # Same gene-bootstrap term every other method carries, so the per-gene bar
        # panels can show one comparable error instead of leaving these methods bare.
        "lfc_spearman_boot": float(sc.get("spearman_boot_std", float("nan"))),
        "lfc_dist_boot": float(sc.get("lfc_dist_boot_std", float("nan"))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "method_label", "gene", "slide", "niche_key",
        "n_scored_genes", "lfc_spearman", "lfc_dist",
        "lfc_spearman_boot", "lfc_dist_boot",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} n={len(rows)}", flush=True)


def run_spac_seen() -> list[dict]:
    master = concert_units(MASTER_SPAC)
    rows: list[dict] = []
    n_miss = 0
    for slide, gdf in master.groupby("slide"):
        slide = str(slide)
        print(f"[STATE SPAC seen] slide {slide} units={len(gdf)}", flush=True)
        adata = ad.read_h5ad(PREP_SPAC / f"slide_{slide}.h5ad")
        genes = np.asarray(adata.var_names).astype(str)
        hvg = hvg_for_spac_slide(slide)
        for r in gdf.itertuples(index=False):
            gene = str(r.gene)
            niche = int(float(r.niche_key))
            p = STATE_SPAC / "task_A_seen" / f"slide_{slide}" / f"niche{niche}_{gene}.h5ad"
            if not p.is_file():
                n_miss += 1
                continue
            o = observed_lfc_seen_patch_niche(
                adata, gene, niche, PREP_SPAC, slide, MIN_G
            )
            if o is None:
                n_miss += 1
                continue
            keep = build_gene_mask(genes, gene, hvg)
            pred = ad.read_h5ad(p)
            sc = align_score(pred, adata, o, keep)
            del pred
            if sc is None:
                n_miss += 1
                continue
            rows.append(record(gene, slide, niche, sc))
        del adata
        gc.collect()
        print(f"  scored={len(rows)} miss_so_far={n_miss}", flush=True)
    print(f"[STATE SPAC seen] done n={len(rows)} miss={n_miss}", flush=True)
    return rows


def run_spac_unseen() -> list[dict]:
    """Score task_B_holdout niche files for CONCERT unseen genes (gene-level BM)."""
    master = concert_units(MASTER_SPAC_UNSEEN)
    genes_needed = set(master["gene"].astype(str))
    files: list[tuple[str, int, str, Path]] = []
    for p in sorted((STATE_SPAC / "task_B_holdout").rglob("*.h5ad")):
        m = NICHE_RE.match(p.name)
        if not m:
            continue
        niche, gene = int(m.group(1)), m.group(2)
        if gene not in genes_needed:
            continue
        slide = p.parent.name.replace("slide_", "")
        files.append((slide, niche, gene, p))
    print(
        f"[STATE SPAC unseen] genes={sorted(genes_needed)} "
        f"task_B files={len(files)}",
        flush=True,
    )

    rows: list[dict] = []
    n_miss = 0
    by_slide: dict[str, list] = {}
    for item in files:
        by_slide.setdefault(item[0], []).append(item)

    for slide, items in sorted(by_slide.items()):
        print(f"[STATE SPAC unseen] slide {slide} files={len(items)}", flush=True)
        adata = ad.read_h5ad(PREP_SPAC / f"slide_{slide}.h5ad")
        genes = np.asarray(adata.var_names).astype(str)
        hvg = hvg_for_spac_slide(slide)
        for _, niche, gene, p in items:
            o = observed_lfc_seen_patch_niche(
                adata, gene, niche, PREP_SPAC, slide, MIN_G
            )
            if o is None:
                n_miss += 1
                continue
            keep = build_gene_mask(genes, gene, hvg)
            pred = ad.read_h5ad(p)
            sc = align_score(pred, adata, o, keep)
            del pred
            if sc is None:
                n_miss += 1
                continue
            rows.append(record(gene, slide, niche, sc))
        del adata
        gc.collect()
        print(f"  scored={len(rows)} miss_so_far={n_miss}", flush=True)
    print(f"[STATE SPAC unseen] done n={len(rows)} miss={n_miss}", flush=True)
    return rows


def run_pm() -> list[dict]:
    master = concert_units(MASTER_PM)
    rows: list[dict] = []
    n_miss = 0
    for slide, gdf in master.groupby("slide"):
        slide = str(int(slide)) if str(slide).isdigit() else str(slide)
        print(f"[STATE PM] slide {slide} units={len(gdf)}", flush=True)
        adata = ad.read_h5ad(PM_PREP / f"slide_{slide}.h5ad")
        genes = np.asarray(adata.var_names).astype(str)
        hvg = load_hvg_indices(PM_CONCERT / "seen_pert" / f"slide_{slide}")
        for r in gdf.itertuples(index=False):
            gene = str(r.gene)
            nk = str(r.niche_key)
            b, p = nk.split(":")
            path = STATE_PM / f"slide_{slide}" / f"patch_{p}_bin_{b}_{gene}.h5ad"
            if not path.is_file():
                n_miss += 1
                continue
            o = observed_lfc_pm_niche(adata, gene, int(b), MIN_G)
            if o is None:
                n_miss += 1
                continue
            keep = build_gene_mask(genes, gene, hvg)
            pred = ad.read_h5ad(path)
            sc = align_score(pred, adata, o, keep)
            del pred
            if sc is None:
                n_miss += 1
                continue
            rows.append(record(gene, slide, nk, sc))
        del adata
        gc.collect()
        print(f"  scored={len(rows)} miss_so_far={n_miss}", flush=True)
    print(f"[STATE PM] done n={len(rows)} miss={n_miss}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="spac,spac_unseen,pm")
    ap.add_argument("--outdir", type=Path, default=OUT)
    args = ap.parse_args()
    dsets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    if "spac" in dsets:
        write_csv(args.outdir / "spac_seen_state_our_rule.csv", run_spac_seen())
    if "spac_unseen" in dsets:
        write_csv(args.outdir / "spac_unseen_state_our_rule.csv", run_spac_unseen())
    if "pm" in dsets:
        write_csv(args.outdir / "pm_seen_state_our_rule.csv", run_pm())

    frames = []
    for ds, fname in (
        ("spac_seen", "spac_seen_state_our_rule.csv"),
        ("spac_unseen", "spac_unseen_state_our_rule.csv"),
        ("pm_seen", "pm_seen_state_our_rule.csv"),
    ):
        p = args.outdir / fname
        if p.is_file():
            frames.append(pd.read_csv(p).assign(dataset=ds))
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        macro = (
            all_df.groupby(["dataset", "method_label"], as_index=False)
            .agg(
                n=("lfc_spearman", "count"),
                lfc_rho=("lfc_spearman", "mean"),
                lfc_rmse=("lfc_dist", "mean"),
            )
        )
        outp = args.outdir / "state_our_rule_macro.csv"
        macro.to_csv(outp, index=False)
        print(macro.to_string(index=False), flush=True)
        print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
