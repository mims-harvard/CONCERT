"""SPAC-seq benchmark evaluation, unseen arm: target-slide masked KO vs an unassigned reference
that excludes the held-out prediction-source pool.

Three rules:
  1. Observed LFC reference. The reference is the unassigned cells on the target slide OUTSIDE the
     patch holdout: unassigned cells with patch role 1 or 2 are dropped, the same rule the seen arm
     applies in lfc_scoring.observed_lfc_seen_patch_niche. The prediction source (role 1
     unassigned, benchmark/holdout_unassigned) is therefore never part of the reference.
  2. model_lfc.json. unseen_spac returns the model LFC vector it scored (computed from the
     method-normalised prediction, NaN/inf set to 0), and save_unseen_unit_results writes that
     exact vector, so any rescore uses the scored predictions.
  3. Source cells. Every method (CONCERT and competitors) is scored on the SAME source cells:
     benchmark/source_cells/slide_<T>.json (make_source_cells.py) = seed-42 subsample of 5000
     cells from the patch-holdout unassigned pool, drawn among the cells CONCERT's loader keeps, so
     all 5000 map to CONCERT model rows. load_source_cells is the only source loader.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

import spacseq_common as sec
import spacseq_slides as smc
from perturbmap_common import to_dense
from spacseq_common import UNASSIGNED_LABELS

log = logging.getLogger(__name__)

REFERENCE_RULE = "unassigned_on_target_slide_excluding_patch_role_1_2"


def load_eval_panel(bench_dir: Path) -> dict:
    return json.loads((bench_dir / "eval_panel.json").read_text())


def load_holdout_unassigned(bench_dir: Path, slide_id: str) -> np.ndarray:
    data = json.loads((bench_dir / "holdout_unassigned" / f"slide_{slide_id}.json").read_text())
    return np.asarray(data["union_holdout_idx"], dtype=int)


SOURCE_CELLS_SUBDIR = "source_cells"


def load_source_cells(bench_dir: Path, slide_id: str) -> np.ndarray:
    """Shared unseen prediction-source cells (identical for every method)."""
    path = bench_dir / SOURCE_CELLS_SUBDIR / f"slide_{slide_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing shared source-cell file {path} (run make_source_cells.py)")
    return np.asarray(json.loads(path.read_text())["source_idx"], dtype=int)


def ko_holdout_indices(bench_dir: Path, gene: str, target_slide: str) -> np.ndarray:
    data = json.loads((bench_dir / "holdout_ko" / f"{gene}__target_slide_{target_slide}.json").read_text())
    return np.asarray(data["ko_holdout_idx"], dtype=int)


def filter_units(panel: dict, units: list[str] | None) -> list[dict]:
    """units: optional list of 'Gene' or 'Gene@slide' strings."""
    if not units:
        return list(panel["eval_units"])
    want = set(units)
    return [u for u in panel["eval_units"]
            if u["gene"] in want or f"{u['gene']}@{u['target_slide']}" in want]


def unseen_reference_idx(adata_target, prepared_dir: Path, slide_id: str) -> np.ndarray:
    """Unassigned cells on the target slide minus unassigned cells in patch roles 1/2."""
    from spacseq_slides import load_patch_roles

    pert = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    una_mask = np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"}))
    role = load_patch_roles(prepared_dir, slide_id, adata_target)
    if role is None:
        raise FileNotFoundError(f"the reference needs the patch holdout roles for slide {slide_id}")
    keep = una_mask & ~np.isin(role, (1, 2))
    return np.where(keep)[0].astype(int)


def unseen_spac(
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
    """One unseen record. Returns (score_dict, scored_model_lfc_list)."""
    eu, sa = sec.import_eval_utils()
    if scoring_method and pred_g is not None:
        pred_g = smc.normalize_pred_cache(scoring_method, {gene: pred_g}).get(gene)
    gene_names = np.asarray(adata_target.var_names)
    ref_idx = unseen_reference_idx(adata_target, args.prepared_dir, args.slide_id)
    src_set = set(np.asarray(global_src_idx, dtype=int).tolist())
    assert not (src_set & set(ref_idx.tolist())), "reference overlaps the prediction source"
    keep = hvg_mask & (gene_names != gene)
    keep_n = int(keep.sum())
    n_src = int(len(global_src_idx))
    n_ko = int(len(ko_idx))

    if unpredictable or pred_g is None or len(pred_g) != n_src:
        return sec.worst_niche_score(-1, n_src, n_ko, keep_n, 12.0), []
    if n_ko < args.min_g_cells or len(ref_idx) < args.min_g_cells:
        return sec.worst_niche_score(-1, n_src, n_ko, keep_n, 12.0), []

    src_expr = to_dense(adata_target[global_src_idx].X)
    y_ko = to_dense(adata_target[ko_idx].X)
    y_ref = to_dense(adata_target[ref_idx].X)
    o_lfc = eu.pseudobulk_lfc(y_ko, y_ref)
    m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pred_g, src_expr), nan=0.0, posinf=0.0, neginf=0.0)
    sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
    raw_mm, _ = smc.load_prepared_raw_resources(args.prepared_dir, args.slide_id)
    sc_["e_dist"] = sa.aligned_energy_distance(
        pred_g, np.asarray(raw_mm[ko_idx]),
        smc.lib_sizes_from_adata(adata_target, global_src_idx), keep,
    )
    sc_.update(niche=-1, n_src=n_src, n_real_g=n_ko, n_scored_genes=keep_n, n_reference=int(len(ref_idx)))
    n_boot, frac, boot_seed = smc.boot_settings(args)
    if n_boot > 0:
        smc.attach_bootstrap_stats(sc_, smc.bootstrap_source_metrics(
            pred_g, src_expr, o_lfc, np.asarray(raw_mm[ko_idx]),
            smc.lib_sizes_from_adata(adata_target, global_src_idx), keep, eu, sa,
            n_boot=n_boot, frac=frac, seed=boot_seed + hash(gene) % 997,
        ))
    return sc_, m_lfc.tolist()


def save_unseen_unit_results(outdir: Path, unit: dict, score: dict, mlfc: list, hvg_info: dict, extra: dict) -> None:
    gene, target = unit["gene"], unit["target_slide"]
    unit_dir = outdir / "unseen_pert" / f"{gene}__slide_{target}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "lfc_scores.json").write_text(json.dumps({"L2": {gene: [score]}}, indent=2))
    (unit_dir / "model_lfc.json").write_text(json.dumps({"L2": {gene: mlfc}}, indent=2))
    (unit_dir / "hvg_info.json").write_text(json.dumps(hvg_info, indent=2))
    meta = {
        "regime": "unseen",
        "reference_rule": REFERENCE_RULE,
        "model_lfc_rule": "scored_vector_from_normalised_prediction",
        "eval_unit": unit,
        **extra,
    }
    (unit_dir / "eval_meta.json").write_text(json.dumps(meta, indent=2))


def aggregate_unseen(outdir: Path, panel: dict) -> None:
    scores_by_gene: dict[str, list] = {}
    for unit in panel["eval_units"]:
        gene, target = unit["gene"], unit["target_slide"]
        path = outdir / "unseen_pert" / f"{gene}__slide_{target}" / "lfc_scores.json"
        if not path.exists():
            log.warning("Missing %s, skipped in aggregate", path)
            continue
        scores_by_gene.setdefault(gene, []).extend(json.loads(path.read_text())["L2"][gene])
    agg = {"L2": {}}
    for g, recs in scores_by_gene.items():
        agg["L2"][g] = smc.aggregate_scores({"L2": {g: recs}}, "L2", g, {"sign_agreement": 0.5, "topk_overlap": 0.0})
    summary_dir = outdir / "unseen_pert" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "lfc_agg.json").write_text(json.dumps(agg, indent=2))
    log.info("Wrote unseen aggregate (%d units) -> %s", len(scores_by_gene), summary_dir / "lfc_agg.json")
