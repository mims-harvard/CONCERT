#!/usr/bin/env python3
"""Post-hoc LFC-vector RMSE for SPAC-seq seen: competitors + CONCERT niche eval CSV."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import anndata as ad
import numpy as np

from benchmark_v3_eval_lib import ko_holdout_indices
from seen_residual_scoring import (
    align_mlfc_vector,
    build_gene_mask,
    load_hvg_indices,
    observed_lfc_on_gene_axis,
    observed_lfc_seen_patch_niche,
    observed_lfc_unseen_v3,
    score_lfc_agreement,
)

PREP = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5/prepared")
BENCHMARK_V3 = PREP / "benchmark_v3"
SPAC_PANEL = BENCHMARK_V3 / "eval_panel.json"
SPAC_V3 = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5_ho_v3")
CONCERT_V3 = SPAC_V3 / "concert"
CONCERT_CSV = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/eval/detailed_per_slide_niche_pert.csv"
)
OUT = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/spac_seq"
)

SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]
METHODS = {
    "scgen": (SPAC_V3 / "scgen", "scGEN"),
    "biolord": (SPAC_V3 / "biolord", "Biolord"),
    "gears": (SPAC_V3 / "gears", "GEARS"),
    "steamboat": (SPAC_V3 / "steamboat", "Steamboat"),
    "celcomen": (SPAC_V3 / "celcomen", "CelComen"),
    "cellagentchat": (SPAC_V3 / "cellagentchat", "CellAgentChat"),
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
    # Gene-bootstrap SD of each statistic within this single evaluation. Computed for
    # EVERY method by score_lfc_agreement, so per-gene bar panels can show one
    # comparable error term instead of a spread that only some methods possess.
    lfc_spearman_boot: float = float("nan")
    lfc_dist_boot: float = float("nan")


def load_eval_units() -> list[tuple[str, str]]:
    data = json.loads(SPAC_PANEL.read_text())
    return [
        (str(u["gene"]), str(u["target_slide"]))
        for u in data.get("eval_units", [])
        if u.get("target_slide")
    ]


def load_concert_v3_records(v3_root: Path, task: str, method_key: str, eval_units=None) -> list[Record]:
    rows: list[Record] = []
    if task == "unseen" and eval_units:
        for gene, slide in eval_units:
            path = v3_root / "unseen" / "unseen_pert" / f"{gene}__slide_{slide}" / "lfc_scores.json"
            if not path.is_file():
                continue
            recs = json.loads(path.read_text()).get("L2", {}).get(gene, [])
            for rec in recs:
                ld = float(rec.get("lfc_dist", float("nan")))
                if not np.isfinite(ld):
                    continue
                rows.append(Record(
                    method=method_key, method_label="CONCERT", gene=gene, slide=slide,
                    niche_key=str(int(rec.get("niche", -1))),
                    n_scored_genes=int(rec.get("n_scored_genes", 0)),
                    lfc_spearman=float(rec.get("spearman", 0.0)),
                    lfc_dist=ld,
                    # Eval-native gene bootstrap (n_boot=10). Unseen has no niche, so the niche-gated
                    # attach_concert_gene_bootstrap cannot recompute it and routing CONCERT through the
                    # competitor path corrupts its point estimate -- so read the eval's own boot here. For
                    # unseen these are comparable in width to competitors' gene bootstrap.
                    lfc_spearman_boot=float(rec.get("spearman_boot_std", float("nan"))),
                    lfc_dist_boot=float(rec.get("lfc_dist_boot_std", float("nan"))),
                ))
        return rows
    for slide in SLIDES:
        path = v3_root / "seen" / "patch" / f"slide_{slide}" / "lfc_scores.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text()).get("L2", {})
        for gene, recs in data.items():
            for rec in recs:
                ld = float(rec.get("lfc_dist", float("nan")))
                if not np.isfinite(ld):
                    continue
                rows.append(Record(
                    method=method_key, method_label="CONCERT", gene=gene, slide=slide,
                    niche_key=str(int(rec.get("niche", -1))),
                    n_scored_genes=int(rec.get("n_scored_genes", 0)),
                    lfc_spearman=float(rec.get("spearman", 0.0)),
                    lfc_dist=ld,
                ))
    return rows


def load_concert_records(csv_path: Path, task: str, method_key: str) -> list[Record]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    sub = df[(df["arch"] == "niche") & (df["task"] == task)]
    rows: list[Record] = []
    for _, r in sub.iterrows():
        rows.append(Record(
            method=method_key,
            method_label="CONCERT",
            gene=str(r["gene"]),
            slide=str(r["slide"]),
            niche_key=str(int(r["niche"])),
            n_scored_genes=int(r.get("n_ko", 0) or 0),
            lfc_spearman=float(r["lfc_rho"]),
            lfc_dist=float(r["lfc_dist"]),
        ))
    return rows


def unseen_eval_dir(root: Path, gene: str, target_slide: str) -> Path:
    unit = f"{gene}__slide_{target_slide}"
    boot = root / "unseen" / "unseen_pert" / unit
    if (boot / "model_lfc.json").is_file():
        return boot
    return root / "unseen_pert" / unit


def rescore_method_slide(
    prepared_dir: Path,
    result_dir: Path,
    slide_id: str,
    method: str,
    method_label: str,
    min_g_cells: int,
) -> list[Record]:
    mlfc_path = result_dir / "seen" / "patch" / f"slide_{slide_id}" / "model_lfc.json"
    if not mlfc_path.is_file():
        return []

    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")
    hvg_idx = load_hvg_indices(result_dir / "seen" / "patch" / f"slide_{slide_id}")
    mlfc = json.loads(mlfc_path.read_text()).get("L2", {})
    gene_names = np.asarray(adata.var_names)
    rows: list[Record] = []

    for gene, niche_vecs in mlfc.items():
        keep = build_gene_mask(gene_names, gene, hvg_idx)
        for key, vec in niche_vecs.items():
            if not vec:
                continue
            niche_bin = int(str(key).split(":")[0])
            pred = align_mlfc_vector(vec, gene_names, hvg_idx)
            obs = _observed_lfc_seen_gt(
                adata, prepared_dir, gene, slide_id, niche_bin, min_g_cells,
            )
            if obs is None or not np.isfinite(pred).any():
                continue
            sc = score_lfc_agreement(pred[keep], obs[keep])
            rows.append(Record(
                method=method, method_label=method_label, gene=gene, slide=slide_id,
                niche_key=str(niche_bin), n_scored_genes=int(keep.sum()),
                lfc_spearman=float(sc["spearman"]),
                lfc_dist=float(sc["lfc_dist"]),
                lfc_spearman_boot=float(sc.get("spearman_boot_std", float("nan"))),
                lfc_dist_boot=float(sc.get("lfc_dist_boot_std", float("nan"))),
            ))
    return rows


def _observed_lfc_seen_gt(
    adata: ad.AnnData,
    prepared_dir: Path,
    gene: str,
    slide_id: str,
    niche_bin: int,
    min_g_cells: int,
) -> np.ndarray | None:
    gene_names = np.asarray(adata.var_names)
    return observed_lfc_seen_patch_niche(
        adata, gene, niche_bin, prepared_dir, slide_id, min_g_cells,
        target_gene_names=gene_names,
    )


def _observed_lfc_unseen_gt(
    adata: ad.AnnData,
    benchmark_v3_dir: Path,
    gene: str,
    target_slide: str,
    niche_bin: int,
    min_g_cells: int,
) -> np.ndarray | None:
    if niche_bin < 0:
        ko_idx = ko_holdout_indices(benchmark_v3_dir, gene, target_slide)
        return observed_lfc_unseen_v3(adata, ko_idx, min_g_cells)
    return observed_lfc_on_gene_axis(adata, gene, niche_bin, np.asarray(adata.var_names), min_g_cells)


def rescore_method_unseen(
    prepared_dir: Path,
    result_dir: Path,
    gene: str,
    target_slide: str,
    method: str,
    method_label: str,
    min_g_cells: int,
    benchmark_v3_dir: Path,
) -> list[Record]:
    edir = unseen_eval_dir(result_dir, gene, target_slide)
    mlfc_path = edir / "model_lfc.json"
    if not mlfc_path.is_file():
        return []

    adata = ad.read_h5ad(prepared_dir / f"slide_{target_slide}.h5ad")
    hvg_idx = load_hvg_indices(edir)
    mlfc = json.loads(mlfc_path.read_text()).get("L2", {})
    gene_names = np.asarray(adata.var_names)
    rows: list[Record] = []
    niche_vecs = mlfc.get(gene, {})
    if not niche_vecs:
        return rows

    keep = build_gene_mask(gene_names, gene, hvg_idx)
    # unseen model_lfc may be a single vector (no per-niche dict)
    if isinstance(niche_vecs, list):
        pred = align_mlfc_vector(niche_vecs, gene_names, hvg_idx)
        # match CONCERT niche from holdout CSV via lfc_scores niche if present
        lfc_path = edir / "lfc_scores.json"
        niche_bin = None
        if lfc_path.is_file():
            recs = json.loads(lfc_path.read_text()).get("L2", {}).get(gene, [])
            if recs:
                niche_bin = int(recs[0].get("niche", -1))
        if niche_bin is None:
            niche_bin = -1
        obs = _observed_lfc_unseen_gt(
            adata, benchmark_v3_dir, gene, target_slide, niche_bin, min_g_cells,
        )
        if obs is not None and np.isfinite(pred).any():
            sc = score_lfc_agreement(pred[keep], obs[keep])
            rows.append(Record(
                method=method, method_label=method_label, gene=gene, slide=target_slide,
                niche_key=str(niche_bin if niche_bin is not None else -1),
                n_scored_genes=int(keep.sum()),
                lfc_spearman=float(sc["spearman"]),
                lfc_dist=float(sc["lfc_dist"]),
                lfc_spearman_boot=float(sc.get("spearman_boot_std", float("nan"))),
                lfc_dist_boot=float(sc.get("lfc_dist_boot_std", float("nan"))),
            ))
        return rows

    for key, vec in niche_vecs.items():
        if not vec:
            continue
        niche_bin = int(str(key).split(":")[0])
        pred = align_mlfc_vector(vec, gene_names, hvg_idx)
        obs = _observed_lfc_unseen_gt(
            adata, benchmark_v3_dir, gene, target_slide, niche_bin, min_g_cells,
        )
        if obs is None or not np.isfinite(pred).any():
            continue
        sc = score_lfc_agreement(pred[keep], obs[keep])
        rows.append(Record(
            method=method, method_label=method_label, gene=gene, slide=target_slide,
            niche_key=str(niche_bin), n_scored_genes=int(keep.sum()),
            lfc_spearman=float(sc["spearman"]),
            lfc_dist=float(sc["lfc_dist"]),
            lfc_spearman_boot=float(sc.get("spearman_boot_std", float("nan"))),
            lfc_dist_boot=float(sc.get("lfc_dist_boot_std", float("nan"))),
        ))
    return rows


def rescore_competitor_at_cell(
    prepared_dir: Path,
    result_dir: Path,
    gene: str,
    slide_id: str,
    niche_bin: int,
    method: str,
    method_label: str,
    min_g_cells: int,
    unseen: bool = False,
    benchmark_v3_dir: Path | None = None,
) -> Record | None:
    """Rescore one competitor at a CONCERT-eval (gene, slide, niche) cell."""
    if unseen:
        edir = unseen_eval_dir(result_dir, gene, slide_id)
    else:
        edir = result_dir / "seen" / "patch" / f"slide_{slide_id}"
    mlfc_path = edir / "model_lfc.json"
    lfc_path = edir / "lfc_scores.json"
    if not mlfc_path.is_file():
        return None

    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")
    hvg_idx = load_hvg_indices(edir)
    gene_names = np.asarray(adata.var_names)
    keep = build_gene_mask(gene_names, gene, hvg_idx)
    mlfc = json.loads(mlfc_path.read_text()).get("L2", {})
    niche_vecs = mlfc.get(gene)
    if not niche_vecs:
        return None

    pred = None
    if isinstance(niche_vecs, dict):
        vec = niche_vecs.get(str(niche_bin)) or niche_vecs.get(str(int(niche_bin)))
        if not vec:
            for v in niche_vecs.values():
                if v:
                    vec = v
                    break
        if vec:
            pred = align_mlfc_vector(vec, gene_names, hvg_idx)
    elif isinstance(niche_vecs, list):
        pred = align_mlfc_vector(niche_vecs, gene_names, hvg_idx)

    if unseen or niche_bin < 0:
        if benchmark_v3_dir is None:
            return None
        obs = _observed_lfc_unseen_gt(
            adata, benchmark_v3_dir, gene, slide_id, niche_bin, min_g_cells,
        )
    else:
        obs = _observed_lfc_seen_gt(
            adata, prepared_dir, gene, slide_id, niche_bin, min_g_cells,
        )
    if pred is not None and obs is not None and np.isfinite(pred).any():
        sc = score_lfc_agreement(pred[keep], obs[keep])
        if np.isfinite(sc["lfc_dist"]):
            return Record(
                method=method, method_label=method_label, gene=gene, slide=slide_id,
                niche_key=str(niche_bin), n_scored_genes=int(keep.sum()),
                lfc_spearman=float(sc["spearman"]), lfc_dist=float(sc["lfc_dist"]),
                lfc_spearman_boot=float(sc.get("spearman_boot_std", float("nan"))),
                lfc_dist_boot=float(sc.get("lfc_dist_boot_std", float("nan"))),
            )
    return None


def rescore_seen_competitors_at_master(
    prepared_dir: Path, result_dir: Path, method: str, method_label: str,
    master: list[Record], min_g_cells: int,
) -> list[Record]:
    rows: list[Record] = []
    for m in master:
        r = rescore_competitor_at_cell(
            prepared_dir, result_dir, m.gene, m.slide, int(m.niche_key),
            method, method_label, min_g_cells, unseen=False,
        )
        if r is not None:
            rows.append(r)
    return rows


def rescore_unseen_competitors_at_master(
    prepared_dir: Path, result_dir: Path, method: str, method_label: str,
    master: list[Record], min_g_cells: int, benchmark_v3_dir: Path,
) -> list[Record]:
    rows: list[Record] = []
    for m in master:
        r = rescore_competitor_at_cell(
            prepared_dir, result_dir, m.gene, m.slide, int(m.niche_key),
            method, method_label, min_g_cells, unseen=True,
            benchmark_v3_dir=benchmark_v3_dir,
        )
        if r is not None:
            rows.append(r)
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
        sb = [r.lfc_spearman_boot for r in recs if np.isfinite(r.lfc_spearman_boot)]
        db = [r.lfc_dist_boot for r in recs if np.isfinite(r.lfc_dist_boot)]
        out.append({
            "method": method,
            "method_label": label,
            "gene": gene,
            "n_records": len(recs),
            "lfc_spearman": float(np.mean(sp)) if sp else np.nan,
            "lfc_dist": float(np.mean(ld)) if ld else np.nan,
            "lfc_spearman_boot": float(np.mean(sb)) if sb else np.nan,
            "lfc_dist_boot": float(np.mean(db)) if db else np.nan,
        })
    return out


def _write_outputs(outdir: Path, all_records: list[Record], prefix: str, concert_method: str) -> None:
    per_record = [dataclasses.asdict(r) for r in all_records]
    summary = aggregate(all_records)
    outdir.mkdir(parents=True, exist_ok=True)
    rec_path = outdir / f"{prefix}_lfc_dist_per_record.csv"
    sum_path = outdir / f"{prefix}_lfc_dist_per_pert.csv"
    fields = list(per_record[0].keys()) if per_record else []
    with rec_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if per_record:
            w.writeheader()
            w.writerows(per_record)
    sum_fields = ["method", "method_label", "gene", "n_records", "lfc_spearman", "lfc_dist",
                  "lfc_spearman_boot", "lfc_dist_boot"]
    with sum_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sum_fields)
        w.writeheader()
        for r in summary:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) and np.isfinite(r[k]) else r[k])
                        for k in sum_fields})
    print(f"Wrote {rec_path} ({len(per_record)} records)")
    print(f"Wrote {sum_path}")
    macro = [r for r in summary if r["method"] == concert_method]
    if macro:
        print(f"CONCERT niche ({prefix}): {len(macro)} genes, "
              f"mean LFC ρ={np.nanmean([r['lfc_spearman'] for r in macro]):.3f}, "
              f"mean LFC RMSE={np.nanmean([r['lfc_dist'] for r in macro]):.3f}")


def attach_concert_gene_bootstrap(
    prepared_dir: Path, concert_root: Path, master: list[Record], min_g_cells: int,
    unseen: bool, benchmark_v3_dir: Path | None = None, rho_tol: float = 1e-3,
) -> None:
    """Give CONCERT the SAME gene bootstrap every competitor gets.

    CONCERT's records are read straight from its precomputed lfc_scores.json, so they
    never pass through score_lfc_agreement and would otherwise carry no bootstrap at all
    (the per-gene bar panels then showed an error for all 11 competitors and none for
    CONCERT). Its eval JSON does ship a `spearman_boot_std`, but that is produced by a
    different procedure and lands 3-5x tighter than the gene bootstrap -- copying it would
    make CONCERT look far more precise than every competitor for reasons of bootstrap
    definition, not accuracy. So it is recomputed here from CONCERT's own predictions,
    through the identical code path the competitors use.

    ONLY the bootstrap fields are written back. The stored lfc_spearman / lfc_dist are the
    master values every competitor is rescored against and must not move. If the recomputed
    rho disagrees with the stored one the vectors or gene mask differ, which would make the
    bootstrap describe a different quantity -- those records keep NaN and are counted.
    """
    n_ok = n_skip = n_mismatch = 0
    for m in master:
        r = rescore_competitor_at_cell(
            prepared_dir, concert_root, m.gene, m.slide, int(m.niche_key),
            m.method, m.method_label, min_g_cells, unseen=unseen,
            benchmark_v3_dir=benchmark_v3_dir,
        )
        if r is None:
            n_skip += 1
            continue
        if not (np.isfinite(r.lfc_spearman) and np.isfinite(m.lfc_spearman)):
            n_mismatch += 1
            continue
        if abs(float(r.lfc_spearman) - float(m.lfc_spearman)) > rho_tol:
            n_mismatch += 1
            continue
        m.lfc_spearman_boot = r.lfc_spearman_boot
        m.lfc_dist_boot = r.lfc_dist_boot
        n_ok += 1
    print(f"[concert-boot] attached={n_ok} no-recompute={n_skip} rho-mismatch={n_mismatch}")


def run_seen(args) -> None:
    master = load_concert_v3_records(args.concert_v3, "seen", "concert_v3_seen")
    attach_concert_gene_bootstrap(
        args.prepared_dir, args.concert_v3, master, args.min_g_cells, unseen=False,
    )
    all_records: list[Record] = list(master)
    for method, (root, label) in METHODS.items():
        if not root.is_dir():
            continue
        all_records.extend(rescore_seen_competitors_at_master(
            args.prepared_dir, root, method, label, master, args.min_g_cells,
        ))
    _write_outputs(args.outdir, all_records, "spac_seen", "concert_v3_seen")


def run_unseen(args) -> None:
    # CONCERT master carries its eval-native gene bootstrap (spearman_boot_std / lfc_dist_boot_std,
    # n_boot=10) read straight from lfc_scores.json (see load_concert_v3_records unseen branch). The
    # niche-gated attach_concert_gene_bootstrap CANNOT be used for unseen (niche_key=-1: the rho
    # recompute never matches and every unit is rejected), and routing CONCERT through the competitor
    # recompute path corrupts its point estimate (ρ 0.383 -> 0.011, a model_lfc format mismatch). For
    # unseen the eval-native boot is comparable in width to the competitor gene bootstrap.
    master = load_concert_v3_records(args.concert_v3, "unseen", "concert_v3_unseen", load_eval_units())
    all_records: list[Record] = list(master)
    for method, (root, label) in METHODS.items():
        if not root.is_dir():
            continue
        all_records.extend(rescore_unseen_competitors_at_master(
            args.prepared_dir, root, method, label, master, args.min_g_cells,
            args.benchmark_v3_dir,
        ))
    _write_outputs(args.outdir, all_records, "spac_unseen", "concert_v3_unseen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=PREP)
    ap.add_argument("--benchmark-v3-dir", type=Path, default=BENCHMARK_V3)
    ap.add_argument("--concert-v3", type=Path, default=CONCERT_V3)
    ap.add_argument("--concert-csv", type=Path, default=CONCERT_CSV)
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--min-g-cells", type=int, default=5)
    ap.add_argument("--regime", choices=("seen", "unseen", "all"), default="all")
    args = ap.parse_args()

    if args.regime in ("seen", "all"):
        print("=== seen ===")
        run_seen(args)
    if args.regime in ("unseen", "all"):
        print("=== unseen ===")
        run_unseen(args)


if __name__ == "__main__":
    main()
