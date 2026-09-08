"""Benchmark v3 assets: eval panel, KO holdouts, seen/unseen train manifests."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np

from spacseq_eval_common import CONTROL_LABELS, UNASSIGNED_LABELS
from spacseq_multi_common import DEFAULT_SLIDES, load_patch_roles
from train_manifest_lib import PM_PREPARED, SPAC_PREPARED, save_manifest

log = logging.getLogger(__name__)

NON_KO = CONTROL_LABELS | UNASSIGNED_LABELS | {"control", "unassigned", "excluded"}

SPAC_DONOR_SLIDE = "1-1"
SPAC_OTHER_SLIDES = ["1-2", "1-3", "1-4", "1-5"]
PM_SLIDES = ["0", "1", "2", "3"]

DEFAULT_MIN_KO_CELLS = 15
DEFAULT_TOP_N_SPAC = 10


def _is_ko(label: str) -> bool:
    return label not in NON_KO


def _ko_counts_per_slide(prepared_dir: Path, slides: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for sid in slides:
        adata = ad.read_h5ad(prepared_dir / f"slide_{sid}.h5ad", backed="r")
        pc = adata.obs["perturbation_canon"].astype(str).to_numpy()
        counts: dict[str, int] = {}
        for g in np.unique(pc):
            if _is_ko(g):
                counts[g] = int((pc == g).sum())
        out[sid] = counts
    return out


def _ko_indices(prepared_dir: Path, slide_id: str, gene: str) -> np.ndarray:
    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad", backed="r")
    pc = adata.obs["perturbation_canon"].astype(str).to_numpy()
    return np.where(pc == gene)[0].astype(int)


def _unassigned_holdout_idx(prepared_dir: Path, slide_id: str) -> np.ndarray:
    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad", backed="r")
    role = load_patch_roles(prepared_dir, slide_id, adata)
    if role is None:
        raise FileNotFoundError(f"Missing patch_holdout for slide {slide_id}")
    return np.where(role == 1)[0].astype(int)


def _pm_unassigned_holdout_idx(prepared_dir: Path, slide_id: str) -> np.ndarray:
    path = prepared_dir / "holdout_patch" / f"slide_{slide_id}.json"
    data = json.loads(path.read_text())
    return np.asarray(data["union_holdout_idx"], dtype=int)


def _pick_target_slide_min(counts_on_other: dict[str, int], tie_break: str = "min_slide_id") -> str:
    """PM target = slide with fewest KO cells (among slides where gene appears)."""
    min_n = min(counts_on_other.values())
    cands = [s for s, n in counts_on_other.items() if n == min_n]
    if tie_break == "min_slide_id":
        return sorted(cands)[0]
    return cands[0]


def _pick_target_slide_second_abundant(
    counts_on_other: dict[str, int],
    tie_break: str = "min_slide_id",
) -> str:
    """SPAC target = non-donor slide with **most** KO cells (2nd abundant after slide 1-1)."""
    max_n = max(counts_on_other.values())
    cands = [s for s, n in counts_on_other.items() if n == max_n]
    if tie_break == "min_slide_id":
        return sorted(cands)[0]
    return cands[0]


def build_spac_eval_panel(
    prepared_dir: Path,
    *,
    donor_slide: str = SPAC_DONOR_SLIDE,
    other_slides: list[str] | None = None,
    top_n: int = DEFAULT_TOP_N_SPAC,
    min_ko_donor: int = DEFAULT_MIN_KO_CELLS,
    min_ko_target: int = DEFAULT_MIN_KO_CELLS,
) -> dict:
    other_slides = other_slides or SPAC_OTHER_SLIDES
    all_slides = [donor_slide] + list(other_slides)
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
        eligible.append({
            "gene": gene,
            "donor_slides": donor_slides,
            "target_slide": target,
            "target_rule": "second_abundant_non_donor",
            "n_ko_donor": int(n_donor),
            "n_ko_target": int(n_target),
            "n_ko_by_slide": {donor_slide: int(n_donor), **{s: int(on_other[s]) for s in on_other}},
        })

    eligible.sort(key=lambda x: (-x["n_ko_donor"], x["gene"]))
    selected = eligible[:top_n]
    return {
        "version": "v3",
        "dataset": "spacseq",
        "donor_slide_required": donor_slide,
        "other_slides": other_slides,
        "target_rule": "second_abundant_non_donor",
        "min_ko_donor": min_ko_donor,
        "min_ko_target": min_ko_target,
        "top_n": top_n,
        "n_eligible": len(eligible),
        "n_selected": len(selected),
        "eval_units": selected,
    }


def build_pm_eval_panel(
    prepared_dir: Path,
    *,
    slides: list[str] | None = None,
    min_ko_target: int = DEFAULT_MIN_KO_CELLS,
    min_ko_donor: int = DEFAULT_MIN_KO_CELLS,
) -> dict:
    slides = slides or PM_SLIDES
    counts = _ko_counts_per_slide(prepared_dir, slides)

    genes = set()
    for sid in slides:
        genes.update(counts[sid].keys())

    selected: list[dict] = []
    for gene in sorted(genes):
        present = {s: counts[s][gene] for s in slides if counts[s].get(gene, 0) > 0}
        if len(present) < 2:
            continue
        target = _pick_target_slide_min(present)
        n_target = present[target]
        if n_target < min_ko_target:
            continue
        donor_slides = [s for s in present if s != target]
        if any(present[s] < min_ko_donor for s in donor_slides):
            continue
        selected.append({
            "gene": gene,
            "donor_slides": donor_slides,
            "target_slide": target,
            "n_ko_target": int(n_target),
            "n_ko_by_slide": {s: int(present[s]) for s in present},
            "n_ko_donor_total": int(sum(present[s] for s in donor_slides)),
        })

    return {
        "version": "v3",
        "dataset": "perturbmap",
        "slides": slides,
        "min_ko_donor": min_ko_donor,
        "min_ko_target": min_ko_target,
        "target_rule": "min_ko_count_slide_among_slides_with_gene",
        "n_selected": len(selected),
        "eval_units": selected,
    }


def write_ko_holdout(path: Path, *, gene: str, slide_id: str, idx: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gene": gene,
        "slide_id": slide_id,
        "ko_holdout_idx": idx.astype(int).tolist(),
        "n_ko_holdout": int(idx.size),
        "role": "masked_gt_excluded_from_unseen_train_used_for_eval",
        **meta,
    }
    path.write_text(json.dumps(payload, indent=2))


def write_unassigned_holdout(path: Path, *, slide_id: str, idx: np.ndarray, dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slide_id": slide_id,
        "union_holdout_idx": idx.astype(int).tolist(),
        "n_holdout": int(idx.size),
        "source": "patch_role1_unassigned" if dataset == "spacseq" else "holdout_patch_kp_role1",
        "role": "prediction_source_excluded_from_train",
    }
    path.write_text(json.dumps(payload, indent=2))


def build_train_manifest_v3(
    prepared_dir: Path,
    slides: list[str],
    *,
    dataset: str,
    eval_panel: dict,
    regime: str,
    max_train_cells: int = 0,
    seed: int = 42,
) -> dict:
    """Build seen or unseen-union training manifest."""
    if regime not in ("seen", "unseen_union"):
        raise ValueError(regime)

    # Map target_slide -> genes to mask on that slide (unseen only)
    ko_mask_by_slide: dict[str, list[str]] = {s: [] for s in slides}
    if regime == "unseen_union":
        for unit in eval_panel["eval_units"]:
            ko_mask_by_slide[unit["target_slide"]].append(unit["gene"])

    unassigned_fn = _unassigned_holdout_idx if dataset == "spacseq" else _pm_unassigned_holdout_idx

    per_slide: dict[str, dict] = {}
    entries: list[tuple[str, int, str]] = []

    for sid in slides:
        adata = ad.read_h5ad(prepared_dir / f"slide_{sid}.h5ad", backed="r")
        n_obs = adata.n_obs
        obs_names = np.asarray(adata.obs_names)
        pc = adata.obs["perturbation_canon"].astype(str).to_numpy()

        unassigned_ho = set(unassigned_fn(prepared_dir, sid).tolist())
        ko_ho: set[int] = set()
        for gene in ko_mask_by_slide.get(sid, []):
            ko_ho.update(_ko_indices(prepared_dir, sid, gene).tolist())

        exclude = unassigned_ho | ko_ho
        train_idx = np.array([i for i in range(n_obs) if i not in exclude], dtype=int)

        for i in train_idx:
            entries.append((sid, int(i), str(obs_names[i])))

        pert = pc[train_idx]
        per_slide[sid] = {
            "n_obs": int(n_obs),
            "unassigned_holdout_idx": sorted(unassigned_ho),
            "n_unassigned_holdout": len(unassigned_ho),
            "ko_holdout_idx": sorted(ko_ho),
            "n_ko_holdout": len(ko_ho),
            "union_train_exclude_idx": sorted(exclude),
            "n_union_exclude": len(exclude),
            "train_idx_before_subsample": train_idx.tolist(),
            "n_train_before_subsample": int(len(train_idx)),
            "ko_genes_masked": list(ko_mask_by_slide.get(sid, [])),
            "n_unassigned_train": int(np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"})).sum()),
            "n_control_train": int(np.isin(pert, list(CONTROL_LABELS | {"control"})).sum()),
        }
        log.info(
            "slide %s [%s]: exclude una=%d ko=%d train=%d",
            sid, regime, len(unassigned_ho), len(ko_ho), len(train_idx),
        )

    selected = entries
    if max_train_cells > 0 and len(entries) > max_train_cells:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(entries), max_train_cells, replace=False)
        selected = [entries[int(i)] for i in np.sort(pick)]

    selected_set = {(s, i) for s, i, _ in selected}
    for sid in slides:
        before = per_slide[sid]["train_idx_before_subsample"]
        train_idx = [i for i in before if (sid, i) in selected_set]
        train_names = [name for s, i, name in selected if s == sid]
        per_slide[sid]["train_idx"] = train_idx
        per_slide[sid]["train_obs_names"] = train_names
        per_slide[sid]["n_train"] = len(train_idx)

    holdout_rule = {
        "seen": "unassigned/KP holdout patches only; all KO kept",
        "unseen_union": "unassigned/KP holdout + masked KO on target slides (eval panel)",
    }[regime]

    return {
        "version": "v3",
        "regime": regime,
        "dataset": dataset,
        "holdout_rule": holdout_rule,
        "eval_panel_ref": "eval_panel.json",
        "slides": slides,
        "max_train_cells": int(max_train_cells),
        "seed": int(seed),
        "n_train_total": int(sum(per_slide[s]["n_train"] for s in slides)),
        "per_slide": per_slide,
    }


def write_benchmark_v3_assets(
    prepared_dir: Path,
    out_root: Path,
    *,
    dataset: str,
    slides: list[str],
    eval_panel: dict,
    max_train_cells: int = 0,
    seed: int = 42,
) -> None:
    """Write full benchmark_v3/ tree under out_root (typically prepared_dir/benchmark_v3)."""
    out_root.mkdir(parents=True, exist_ok=True)

    unassigned_fn = _unassigned_holdout_idx if dataset == "spacseq" else _pm_unassigned_holdout_idx
    ho_una = out_root / "holdout_unassigned"
    ho_ko = out_root / "holdout_ko"

    for sid in slides:
        idx = unassigned_fn(prepared_dir, sid)
        write_unassigned_holdout(
            ho_una / f"slide_{sid}.json",
            slide_id=sid, idx=idx, dataset=dataset,
        )

    for unit in eval_panel["eval_units"]:
        gene = unit["gene"]
        target = unit["target_slide"]
        idx = _ko_indices(prepared_dir, target, gene)
        ko_path = ho_ko / f"{gene}__target_slide_{target}.json"
        una_path = ho_una / f"slide_{target}.json"
        write_ko_holdout(
            ko_path,
            gene=gene,
            slide_id=target,
            idx=idx,
            meta={
                "donor_slides": unit["donor_slides"],
                "n_ko_by_slide": unit["n_ko_by_slide"],
                "eval_source": str(una_path.relative_to(out_root)),
                "eval_gt": "same_slide_ko_vs_control",
            },
        )
        unit["holdout_ko_file"] = str(ko_path.relative_to(out_root))
        unit["holdout_unassigned_file"] = str(una_path.relative_to(out_root))
        unit["n_ko_holdout"] = int(idx.size)

    (out_root / "eval_panel.json").write_text(json.dumps(eval_panel, indent=2))

    # Union of all training exclusions for unseen (reference file)
    union_exclude: dict[str, dict] = {}
    for unit in eval_panel["eval_units"]:
        t = unit["target_slide"]
        union_exclude.setdefault(t, {"ko_genes": [], "unassigned_holdout": None})
        union_exclude[t]["ko_genes"].append(unit["gene"])

    union_doc = {"version": "v3", "regime": "unseen_union", "per_target_slide": {}}
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

    seen_manifest = build_train_manifest_v3(
        prepared_dir, slides, dataset=dataset, eval_panel=eval_panel,
        regime="seen", max_train_cells=max_train_cells, seed=seed,
    )
    unseen_manifest = build_train_manifest_v3(
        prepared_dir, slides, dataset=dataset, eval_panel=eval_panel,
        regime="unseen_union", max_train_cells=max_train_cells, seed=seed,
    )
    save_manifest(seen_manifest, out_root / "train_manifest_seen.json")
    save_manifest(unseen_manifest, out_root / "train_manifest_unseen_union.json")

    readme = _readme_text(dataset, out_root, eval_panel)
    (out_root / "README.md").write_text(readme)
    log.info("Wrote benchmark_v3 assets -> %s", out_root)


def _readme_text(dataset: str, out_root: Path, panel: dict) -> str:
    units = panel["eval_units"]
    lines = [
        f"# Benchmark v3 assets ({dataset})",
        "",
        "See `spac_seq_eval/BENCHMARK_V3_SEEN_UNSEEN_PROTOCOL.md` for the full protocol.",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "|------|---------|",
        "| `eval_panel.json` | Perturbations to predict: donor slide(s), target slide, cell counts |",
        "| `holdout_unassigned/slide_*.json` | Unperturbed prediction-source cells excluded from train |",
        "| `holdout_ko/{gene}__target_slide_{T}.json` | Masked KO GT on target slide (train exclude + eval reference) |",
        "| `holdout_train_unseen_union.json` | Summary of all unseen-train exclusions |",
        "| `train_manifest_seen.json` | Seen training cells (all methods) |",
        "| `train_manifest_unseen_union.json` | One unseen train: union of all KO + unassigned holdouts |",
        "",
        "## Eval units",
        "",
    ]
    for u in units:
        lines.append(
            f"- **{u['gene']}**: learn on {u['donor_slides']} → mask+predict on slide **{u['target_slide']}** "
            f"({u['n_ko_by_slide'].get(u['target_slide'], u.get('n_ko_target', '?'))} KO cells held out)"
        )
    lines.append("")
    return "\n".join(lines)
