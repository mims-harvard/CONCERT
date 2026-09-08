#!/usr/bin/env python3
"""Post-hoc LFC-vector distance (RMSE) for Perturb-map seen from saved model_lfc.json."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from seen_residual_scoring import (
    align_mlfc_vector,
    build_gene_mask,
    load_concert_gene_names_from_checkpoint,
    load_hvg_indices,
    observed_lfc_on_gene_axis,
    observed_lfc_pm_niche,
    score_lfc_agreement,
)
import anndata as ad

PREP = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/prepared_panel1053ko")
PM_V3 = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/results_panel1053ko_v3")
PM_NICHE = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/results_panel1053ko_niche")
OUT = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/spac_seq")

SLIDES = ["0", "1", "2", "3"]
GENES = ["Jak2", "Ifngr2", "Tgfbr2"]
METHODS = {
    "scgen": (PM_V3 / "scgen", "scGEN"),
    "biolord": (PM_V3 / "biolord", "Biolord"),
    "gears": (PM_V3 / "gears", "GEARS"),
    "steamboat": (PM_V3 / "steamboat", "Steamboat"),
    "celcomen": (PM_V3 / "celcomen", "CelComen"),
    "cellagentchat": (PM_V3 / "cellagentchat", "CellAgentChat"),
    "concert_v3_seen": (PM_V3 / "concert_ms_seen", "CONCERT (v3 ckpt)"),
    "concert_niche_seen": (PM_NICHE / "concert_ms_seen", "CONCERT (niche ckpt)"),
    # The model reported in the paper.
    "concert": (PM_NICHE / "concert_ms_seen_bandI", "CONCERT"),
}


@dataclasses.dataclass
class Record:
    method: str
    method_label: str
    gene: str
    slide: str
    niche_key: str
    n_scored_genes: int
    lfc_spearman: float
    lfc_dist: float
    lfc_spearman_boot: float = float("nan")
    lfc_dist_boot: float = float("nan")


def rescore_method_slide(
    prepared_dir: Path,
    result_dir: Path,
    slide_id: str,
    method: str,
    method_label: str,
    min_g_cells: int,
) -> list[Record]:
    mlfc_path = result_dir / "seen_pert" / f"slide_{slide_id}" / "model_lfc.json"
    if not mlfc_path.is_file():
        return []

    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")
    hvg_idx = load_hvg_indices(result_dir / "seen_pert" / f"slide_{slide_id}")
    mlfc = json.loads(mlfc_path.read_text()).get("L2", {})
    rows: list[Record] = []

    if method.startswith("concert"):
        vec_len = 0
        for niche_vecs in mlfc.values():
            for vec in niche_vecs.values():
                if vec:
                    vec_len = len(vec)
                    break
            if vec_len:
                break
        if not vec_len:
            return []
        concert_genes = load_concert_gene_names_from_checkpoint(result_dir, vec_len)
        if concert_genes is None:
            return []
        for gene, niche_vecs in mlfc.items():
            keep = build_gene_mask(concert_genes, gene, hvg_idx)
            for key, vec in niche_vecs.items():
                if not vec:
                    continue
                niche_bin = int(str(key).split(":")[0])
                pred = np.asarray(vec, dtype=np.float64)
                obs = observed_lfc_on_gene_axis(adata, gene, niche_bin, concert_genes, min_g_cells)
                if obs is None or not np.isfinite(pred).any():
                    continue
                sc = score_lfc_agreement(pred[keep], obs[keep])
                rows.append(Record(
                    method=method, method_label=method_label, gene=gene, slide=slide_id,
                    niche_key=str(key), n_scored_genes=int(keep.sum()),
                    lfc_spearman=float(sc["spearman"]),
                    lfc_dist=float(sc["lfc_dist"]),
                    lfc_spearman_boot=float(sc.get("spearman_boot_std", float("nan"))),
                    lfc_dist_boot=float(sc.get("lfc_dist_boot_std", float("nan"))),
                ))
        return rows

    gene_names = np.asarray(adata.var_names)
    for gene, niche_vecs in mlfc.items():
        keep = build_gene_mask(gene_names, gene, hvg_idx)
        for key, vec in niche_vecs.items():
            if not vec:
                continue
            niche_bin = int(str(key).split(":")[0])
            pred = align_mlfc_vector(vec, gene_names, hvg_idx)
            obs = observed_lfc_pm_niche(adata, gene, niche_bin, min_g_cells)
            if obs is None or not np.isfinite(pred).any():
                continue
            sc = score_lfc_agreement(pred[keep], obs[keep])
            rows.append(Record(
                method=method, method_label=method_label, gene=gene, slide=slide_id,
                niche_key=str(key), n_scored_genes=int(keep.sum()),
                lfc_spearman=float(sc["spearman"]),
                lfc_dist=float(sc["lfc_dist"]),
                lfc_spearman_boot=float(sc.get("spearman_boot_std", float("nan"))),
                lfc_dist_boot=float(sc.get("lfc_dist_boot_std", float("nan"))),
            ))
    return rows


def aggregate(records: list[Record]) -> list[dict]:
    from collections import defaultdict
    buckets: dict[tuple, list[Record]] = defaultdict(list)
    for r in records:
        buckets[(r.method, r.method_label, r.gene)].append(r)
    out = []
    for (method, label, gene), recs in sorted(buckets.items()):
        sp = [r.lfc_spearman for r in recs if np.isfinite(r.lfc_spearman)]
        ld = [r.lfc_dist for r in recs if np.isfinite(r.lfc_dist)]
        # Error term = s.e.m. ACROSS the niche-patch records of this perturbation.
        # Every method is scored on the same records, so this is like-for-like. The
        # per-record gene bootstrap is kept separately as a within-record term.
        out.append({
            "method": method,
            "method_label": label,
            "gene": gene,
            "n_records": len(recs),
            "lfc_spearman": float(np.mean(sp)) if sp else np.nan,
            "lfc_dist": float(np.mean(ld)) if ld else np.nan,
            "lfc_spearman_sem": float(np.std(sp, ddof=1) / np.sqrt(len(sp))) if len(sp) > 1 else np.nan,
            "lfc_dist_sem": float(np.std(ld, ddof=1) / np.sqrt(len(ld))) if len(ld) > 1 else np.nan,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=PREP)
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--min-g-cells", type=int, default=5)
    args = ap.parse_args()

    all_records: list[Record] = []
    for method, (root, label) in METHODS.items():
        if not root.is_dir():
            continue
        for slide in SLIDES:
            all_records.extend(rescore_method_slide(
                args.prepared_dir, root, slide, method, label, args.min_g_cells,
            ))

    per_record = [dataclasses.asdict(r) for r in all_records]
    summary = aggregate(all_records)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rec_path = outdir / "pm_seen_lfc_dist_per_record.csv"
    sum_path = outdir / "pm_seen_lfc_dist_per_pert.csv"
    with rec_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_record[0].keys()) if per_record else [])
        if per_record:
            w.writeheader()
            w.writerows(per_record)
    with sum_path.open("w", newline="") as fh:
        fields = ["method", "method_label", "gene", "n_records", "lfc_spearman", "lfc_dist",
                  "lfc_spearman_sem", "lfc_dist_sem"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) and np.isfinite(r[k]) else r[k]) for k in fields})

    print(f"Wrote {rec_path}\nWrote {sum_path}\n")
    for gene in GENES:
        sub = [r for r in summary if r["gene"] == gene and r["n_records"] > 0]
        sub.sort(key=lambda r: r["lfc_spearman"], reverse=True)
        print(f"### {gene}")
        print("| Method | N | LFC ρ | LFC RMSE |")
        print("| --- | ---: | ---: | ---: |")
        for r in sub:
            print(f"| {r['method_label']} | {r['n_records']} | {r['lfc_spearman']:.3f} | {r['lfc_dist']:.3f} |")
        print()


if __name__ == "__main__":
    main()
