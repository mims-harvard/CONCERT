"""Benchmark v3 evaluation: target-slide masked GT (no donor pooling)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

import spacseq_eval_common as sec
import spacseq_multi_common as smc
from perturbmap_common import CTRL_CANON, lib_sizes_from, load_prepared_raw_resources, to_dense
from spacseq_eval_common import CONTROL_LABELS, UNASSIGNED_LABELS

log = logging.getLogger(__name__)


def load_eval_panel(benchmark_v3_dir: Path) -> dict:
    return json.loads((benchmark_v3_dir / "eval_panel.json").read_text())


def load_holdout_unassigned(benchmark_v3_dir: Path, slide_id: str) -> np.ndarray:
    path = benchmark_v3_dir / "holdout_unassigned" / f"slide_{slide_id}.json"
    data = json.loads(path.read_text())
    return np.asarray(data["union_holdout_idx"], dtype=int)


def load_holdout_ko(benchmark_v3_dir: Path, gene: str, target_slide: str) -> dict:
    path = benchmark_v3_dir / "holdout_ko" / f"{gene}__target_slide_{target_slide}.json"
    return json.loads(path.read_text())


def ko_holdout_indices(benchmark_v3_dir: Path, gene: str, target_slide: str) -> np.ndarray:
    data = load_holdout_ko(benchmark_v3_dir, gene, target_slide)
    return np.asarray(data["ko_holdout_idx"], dtype=int)


def unseen_v3_spac(
    adata_target,
    gene: str,
    global_src_idx: np.ndarray,
    ko_idx: np.ndarray,
    pred_g: np.ndarray | None,
    hvg_mask: np.ndarray,
    args,
    *,
    unpredictable: bool = False,
    scoring_method: str | None = None,
):
    """One unseen record: pred on holdout unassigned @ T vs masked G-KO @ T (same slide)."""
    eu, sa = sec.import_eval_utils()
    if scoring_method:
        pred_g = smc.normalize_pred_cache(scoring_method, {gene: pred_g}).get(gene)
    gene_names = np.asarray(adata_target.var_names)
    pert = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    una_idx = np.where(np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"})))[0]
    keep = hvg_mask & (gene_names != gene)
    keep_n = int(keep.sum())
    n_src = int(len(global_src_idx))
    n_ko = int(len(ko_idx))

    if unpredictable or pred_g is None or len(pred_g) != n_src:
        return sec.worst_niche_score(-1, n_src, n_ko, keep_n, 12.0)

    if n_ko < args.min_g_cells or len(una_idx) < args.min_g_cells:
        return sec.worst_niche_score(-1, n_src, n_ko, keep_n, 12.0)

    src_expr = to_dense(adata_target[global_src_idx].X)
    y_ko = to_dense(adata_target[ko_idx].X)
    y_una = to_dense(adata_target[una_idx].X)
    o_lfc = eu.pseudobulk_lfc(y_ko, y_una)
    m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pred_g, src_expr), nan=0.0, posinf=0.0, neginf=0.0)
    sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
    raw_mm, _ = smc.load_prepared_raw_resources(args.prepared_dir, args.slide_id)
    sc_["e_dist"] = sa.aligned_energy_distance(
        pred_g, np.asarray(raw_mm[ko_idx]),
        smc.lib_sizes_from_adata(adata_target, global_src_idx), keep,
    )
    sc_.update(niche=-1, n_src=n_src, n_real_g=n_ko, n_scored_genes=keep_n)
    n_boot, frac, boot_seed = smc.boot_settings(args)
    if n_boot > 0:
        smc.attach_bootstrap_stats(sc_, smc.bootstrap_source_metrics(
            pred_g, src_expr, o_lfc, np.asarray(raw_mm[ko_idx]),
            smc.lib_sizes_from_adata(adata_target, global_src_idx), keep, eu, sa,
            n_boot=n_boot, frac=frac, seed=boot_seed + hash(gene) % 997,
        ))
    return sc_


def unseen_v3_pm(
    adata_target,
    gene: str,
    global_src_idx: np.ndarray,
    ko_idx: np.ndarray,
    pred_g: np.ndarray | None,
    hvg_mask: np.ndarray,
    args,
    *,
    unpredictable: bool = False,
    scoring_method: str | None = None,
):
    """PM unseen v3: masked G-KO @ T vs KP control @ T."""
    eu, sa = sec.import_eval_utils()
    if scoring_method:
        pred_g = smc.normalize_pred_cache(scoring_method, {gene: pred_g}).get(gene)
    gene_names = np.asarray(adata_target.var_names)
    pert = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    ctrl_idx = np.where(pert == CTRL_CANON)[0]
    keep = hvg_mask & (gene_names != gene)
    keep_n = int(keep.sum())
    n_src = int(len(global_src_idx))
    n_ko = int(len(ko_idx))

    if unpredictable or pred_g is None or len(pred_g) != n_src:
        return sec.worst_niche_score(-1, n_src, n_ko, keep_n, 12.0)
    if n_ko < args.min_g_cells or len(ctrl_idx) < args.min_g_cells:
        return sec.worst_niche_score(-1, n_src, n_ko, keep_n, 12.0)

    src_expr = to_dense(adata_target[global_src_idx].X)
    y_ko = to_dense(adata_target[ko_idx].X)
    y_ctrl = to_dense(adata_target[ctrl_idx].X)
    o_lfc = eu.pseudobulk_lfc(y_ko, y_ctrl)
    m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pred_g, src_expr), nan=0.0, posinf=0.0, neginf=0.0)
    sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
    raw_mm, _ = load_prepared_raw_resources(args.prepared_dir, args.slide_id)
    sc_["e_dist"] = sa.aligned_energy_distance(
        pred_g, np.asarray(raw_mm[ko_idx]),
        lib_sizes_from(adata_target, global_src_idx), keep,
    )
    sc_.update(niche=-1, n_src=n_src, n_real_g=n_ko, n_scored_genes=keep_n)
    n_boot, frac, boot_seed = smc.boot_settings(args)
    if n_boot > 0:
        sec.attach_bootstrap_stats(sc_, smc.bootstrap_source_metrics(
            pred_g, src_expr, o_lfc, np.asarray(raw_mm[ko_idx]),
            lib_sizes_from(adata_target, global_src_idx), keep, eu, sa,
            n_boot=n_boot, frac=frac, seed=boot_seed + hash(gene) % 997,
        ))
    return sc_


def save_unseen_unit_results(
    outdir: Path,
    unit: dict,
    score: dict,
    mlfc: list,
    hvg_info: dict,
    extra: dict,
) -> None:
    gene = unit["gene"]
    target = unit["target_slide"]
    unit_dir = outdir / "unseen_pert" / f"{gene}__slide_{target}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "L2": {gene: [score]},
    }
    (unit_dir / "lfc_scores.json").write_text(json.dumps(payload, indent=2))
    (unit_dir / "model_lfc.json").write_text(json.dumps({"L2": {gene: mlfc}}, indent=2))
    (unit_dir / "hvg_info.json").write_text(json.dumps(hvg_info, indent=2))
    meta = {
        "version": "v3",
        "regime": "unseen",
        "eval_unit": unit,
        **extra,
    }
    (unit_dir / "eval_meta.json").write_text(json.dumps(meta, indent=2))


def aggregate_unseen_v3(outdir: Path, panel: dict) -> None:
    """Write lfc_agg.json summarizing all unseen eval units."""
    scores_by_gene: dict[str, list] = {}
    for unit in panel["eval_units"]:
        gene = unit["gene"]
        target = unit["target_slide"]
        path = outdir / "unseen_pert" / f"{gene}__slide_{target}" / "lfc_scores.json"
        if not path.exists():
            log.warning("Missing %s — skip aggregate", path)
            continue
        rec = json.loads(path.read_text())["L2"][gene]
        scores_by_gene.setdefault(gene, []).extend(rec)

    agg = {"L2": {}}
    for g, recs in scores_by_gene.items():
        agg["L2"][g] = smc.aggregate_scores({"L2": {g: recs}}, "L2", g,
                                             {"sign_agreement": 0.5, "topk_overlap": 0.0})

    summary_dir = outdir / "unseen_pert" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "lfc_agg.json").write_text(json.dumps(agg, indent=2))
    log.info("Wrote unseen aggregate -> %s", summary_dir / "lfc_agg.json")
