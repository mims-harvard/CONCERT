"""Benchmark v4 assets (SPAC-seq): v4 perturbation-selection rule for the cross-slide (unseen) arm.

v4 rule (SPAC-seq):
  * donor slide 1-1 carries >= min_ko_donor KO cells (v4: 20)
  * target slide = non-donor slide with the most KO cells of that gene (same rule as v3),
    and it carries >= min_ko_target KO cells (v4: 100)
  * ranking key = total KO cells across the five slides (descending), gene name as tie-break
  * cap: at most `max_per_target_slide` knockouts per target slide (0 = no cap) and at most
    `max_total` knockouts overall (0 = no cap), both applied along the ranking key.

The seen arm (unassigned patch holdout + train_manifest_seen) is produced by the unchanged v3
code path, so its content is identical to benchmark_v3. Helpers are imported from
benchmark_v3_lib (read-only); v3 files are never modified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from benchmark_v3_lib import (
    SPAC_DONOR_SLIDE,
    SPAC_OTHER_SLIDES,
    _ko_counts_per_slide,
    _ko_indices,
    _pick_target_slide_second_abundant,
    _unassigned_holdout_idx,
    build_train_manifest_v3,
    write_ko_holdout,
    write_unassigned_holdout,
)
from train_manifest_lib import save_manifest

log = logging.getLogger(__name__)

V4_MIN_KO_DONOR = 20
V4_MIN_KO_TARGET = 100
V4_MAX_PER_TARGET_SLIDE = 0  # author decision: no cap (all eligible knockouts)
V4_MAX_TOTAL = 0


def build_spac_eval_panel_v4(
    prepared_dir: Path,
    *,
    donor_slide: str = SPAC_DONOR_SLIDE,
    other_slides: list[str] | None = None,
    min_ko_donor: int = V4_MIN_KO_DONOR,
    min_ko_target: int = V4_MIN_KO_TARGET,
    max_per_target_slide: int = V4_MAX_PER_TARGET_SLIDE,
    max_total: int = V4_MAX_TOTAL,
) -> dict:
    other_slides = list(other_slides or SPAC_OTHER_SLIDES)
    all_slides = [donor_slide] + other_slides
    counts = _ko_counts_per_slide(prepared_dir, all_slides)

    eligible: list[dict] = []
    for gene, n_donor in counts.get(donor_slide, {}).items():
        if n_donor < min_ko_donor:
            continue
        on_other = {s: counts[s].get(gene, 0) for s in other_slides if counts[s].get(gene, 0) > 0}
        if not on_other:
            continue
        target = _pick_target_slide_second_abundant(on_other)
        n_target = on_other[target]
        if n_target < min_ko_target:
            continue
        donor_slides = [donor_slide] + [s for s in on_other if s != target]
        n_by_slide = {donor_slide: int(n_donor), **{s: int(on_other[s]) for s in on_other}}
        eligible.append({
            "gene": gene,
            "donor_slides": donor_slides,
            "target_slide": target,
            "target_rule": "second_abundant_non_donor",
            "n_ko_donor": int(n_donor),
            "n_ko_target": int(n_target),
            "n_ko_total": int(sum(n_by_slide.values())),
            "n_ko_by_slide": n_by_slide,
        })

    eligible.sort(key=lambda x: (-x["n_ko_total"], x["gene"]))
    per_slide_kept: dict[str, int] = {}
    selected: list[dict] = []
    for unit in eligible:
        t = unit["target_slide"]
        if max_per_target_slide > 0 and per_slide_kept.get(t, 0) >= max_per_target_slide:
            continue
        if max_total > 0 and len(selected) >= max_total:
            break
        per_slide_kept[t] = per_slide_kept.get(t, 0) + 1
        selected.append(unit)

    n_eligible_by_slide: dict[str, int] = {}
    for u in eligible:
        n_eligible_by_slide[u["target_slide"]] = n_eligible_by_slide.get(u["target_slide"], 0) + 1

    return {
        "version": "v4",
        "dataset": "spacseq",
        "donor_slide_required": donor_slide,
        "other_slides": other_slides,
        "target_rule": "second_abundant_non_donor",
        "min_ko_donor": int(min_ko_donor),
        "min_ko_target": int(min_ko_target),
        "rank_key": "n_ko_total_desc_then_gene",
        "max_per_target_slide": int(max_per_target_slide),
        "max_total": int(max_total),
        "n_eligible": len(eligible),
        "n_eligible_by_target_slide": dict(sorted(n_eligible_by_slide.items())),
        "n_selected": len(selected),
        "n_selected_by_target_slide": dict(sorted(per_slide_kept.items())),
        "eval_units": selected,
    }


def write_benchmark_v4_assets(
    prepared_dir: Path,
    out_root: Path,
    *,
    slides: list[str],
    eval_panel: dict,
    max_train_cells: int = 50000,
    seed: int = 42,
) -> None:
    """Write benchmark_v4/ (SPAC-seq). Mirrors write_benchmark_v3_assets with v4 labels."""
    out_root.mkdir(parents=True, exist_ok=True)
    ho_una = out_root / "holdout_unassigned"
    ho_ko = out_root / "holdout_ko"
    if ho_ko.is_dir():
        # a panel change must never leave KO files of genes that are no longer in the panel
        for f in ho_ko.glob("*.json"):
            f.unlink()

    for sid in slides:
        idx = _unassigned_holdout_idx(prepared_dir, sid)
        write_unassigned_holdout(ho_una / f"slide_{sid}.json", slide_id=sid, idx=idx, dataset="spacseq")

    for unit in eval_panel["eval_units"]:
        gene, target = unit["gene"], unit["target_slide"]
        idx = _ko_indices(prepared_dir, target, gene)
        ko_path = ho_ko / f"{gene}__target_slide_{target}.json"
        una_path = ho_una / f"slide_{target}.json"
        write_ko_holdout(
            ko_path, gene=gene, slide_id=target, idx=idx,
            meta={
                "donor_slides": unit["donor_slides"],
                "n_ko_by_slide": unit["n_ko_by_slide"],
                "eval_source": str(una_path.relative_to(out_root)),
                "eval_gt": "same_slide_ko_vs_unassigned_outside_patch_holdout",
            },
        )
        unit["holdout_ko_file"] = str(ko_path.relative_to(out_root))
        unit["holdout_unassigned_file"] = str(una_path.relative_to(out_root))
        unit["n_ko_holdout"] = int(idx.size)

    (out_root / "eval_panel.json").write_text(json.dumps(eval_panel, indent=2))

    union_doc = {"version": "v4", "regime": "unseen_union", "per_target_slide": {}}
    for sid in slides:
        una = json.loads((ho_una / f"slide_{sid}.json").read_text())
        ko_idx: list[int] = []
        genes: list[str] = []
        for unit in eval_panel["eval_units"]:
            if unit["target_slide"] == sid:
                genes.append(unit["gene"])
                ko_idx.extend(json.loads((ho_ko / f"{unit['gene']}__target_slide_{sid}.json").read_text())["ko_holdout_idx"])
        all_exc = sorted(set(una["union_holdout_idx"]) | set(ko_idx))
        union_doc["per_target_slide"][sid] = {
            "ko_genes_masked": genes,
            "n_ko_holdout": len(set(ko_idx)),
            "n_unassigned_holdout": una["n_holdout"],
            "n_union_exclude": len(all_exc),
        }
    (out_root / "holdout_train_unseen_union.json").write_text(json.dumps(union_doc, indent=2))

    # Seen manifest: unchanged v3 code path (content identical to benchmark_v3).
    seen_manifest = build_train_manifest_v3(
        prepared_dir, slides, dataset="spacseq", eval_panel=eval_panel,
        regime="seen", max_train_cells=max_train_cells, seed=seed,
    )
    unseen_manifest = build_train_manifest_v3(
        prepared_dir, slides, dataset="spacseq", eval_panel=eval_panel,
        regime="unseen_union", max_train_cells=max_train_cells, seed=seed,
    )
    unseen_manifest["version"] = "v4"
    save_manifest(seen_manifest, out_root / "train_manifest_seen.json")
    save_manifest(unseen_manifest, out_root / "train_manifest_unseen_union.json")
    (out_root / "README.md").write_text(_readme_text_v4(eval_panel))
    log.info("Wrote benchmark_v4 assets -> %s", out_root)


def _readme_text_v4(panel: dict) -> str:
    cap_slide = panel["max_per_target_slide"] or "no cap"
    cap_total = panel["max_total"] or "no cap"
    lines = [
        "# Benchmark v4 assets (spacseq)",
        "",
        "Protocol: `spac_seq_eval/BENCHMARK_V4_PROTOCOL.md`.",
        "",
        f"Selection: donor slide {panel['donor_slide_required']} >= {panel['min_ko_donor']} KO cells, "
        f"target slide (non-donor slide with most KO cells) >= {panel['min_ko_target']} KO cells, "
        f"ranked by total KO cells across the five slides; per-target-slide cap: {cap_slide}; "
        f"total cap: {cap_total}.",
        f"Eligible {panel['n_eligible']} -> selected {panel['n_selected']} "
        f"(by target slide: {panel['n_selected_by_target_slide']}).",
        "",
        "Seen-arm files (`holdout_unassigned/`, `train_manifest_seen.json`) are identical to benchmark_v3.",
        "",
        "| File | Purpose |",
        "|------|---------|",
        "| `eval_panel.json` | Knockouts to predict: donor slides, target slide, cell counts |",
        "| `holdout_unassigned/slide_*.json` | Unassigned prediction-source cells (patch role 1), excluded from training |",
        "| `holdout_ko/{gene}__target_slide_{T}.json` | Masked KO cells on the target slide (training exclusion + observed reference) |",
        "| `holdout_train_unseen_union.json` | Summary of all unseen-training exclusions |",
        "| `train_manifest_seen.json` | Seen training cells (identical to v3) |",
        "| `train_manifest_unseen_union.json` | Unseen training: union of all KO + unassigned holdouts |",
        "",
        "## Eval units",
        "",
    ]
    for u in panel["eval_units"]:
        lines.append(
            f"- {u['gene']}: target slide {u['target_slide']} ({u['n_ko_target']} KO cells held out), "
            f"total {u['n_ko_total']}, donors {u['donor_slides']}"
        )
    lines.append("")
    return "\n".join(lines)
