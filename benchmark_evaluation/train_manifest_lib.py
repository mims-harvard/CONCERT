"""Shared training cell manifests for fair multi-method benchmarks.

Rules (SPAC-seq):
  * Exclude holdout **unassigned spatial patches** (patch_holdout role==1) from training.
  * Keep all NTC (control), KO, and non-holdout unassigned cells.
  * Optional pooled subsample with one fixed seed -> identical cells for every method.

Rules (perturb-map):
  * Exclude holdout **KP-tumor source patches** (holdout_patch union / role==1).
  * Keep control (KP), KO, unassigned (normal context), context cells.
  * Same optional pooled subsample.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import release_paths as rp  # noqa: E402

import anndata as ad
import numpy as np

from spacseq_eval_common import CONTROL_LABELS, UNASSIGNED_LABELS
from spacseq_multi_common import DEFAULT_SLIDES, load_holdout_mask, load_patch_roles

log = logging.getLogger(__name__)

SPAC_PREPARED = rp.SPACSEQ_PREPARED
PM_PREPARED = rp.PERTURBMAP_PREPARED


def _spacseq_holdout_patch_idx(prepared_dir: Path, slide_id: str) -> np.ndarray:
    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad", backed="r")
    role = load_patch_roles(prepared_dir, slide_id, adata)
    if role is None:
        raise FileNotFoundError(
            f"Missing patch_holdout npz for slide {slide_id} "
            f"(expected {prepared_dir.parent}/patch_holdout/{slide_id}.npz)"
        )
    return np.where(role == 1)[0].astype(int)


def _pm_holdout_idx(prepared_dir: Path, slide_id: str, holdout_dir: str = "holdout_patch") -> np.ndarray:
    mask = load_holdout_mask(prepared_dir, slide_id, holdout_dir)
    if mask is None:
        raise FileNotFoundError(f"No {holdout_dir}/slide_{slide_id}.json under {prepared_dir}")
    return np.asarray(mask["union_holdout_idx"], dtype=int)


def build_manifest(
    prepared_dir: Path,
    slides: list[str],
    *,
    dataset: str,
    holdout_dir: str | None = None,
    max_train_cells: int = 0,
    seed: int = 42,
) -> dict:
    """Build a train manifest. ``max_train_cells=0`` keeps all post-holdout cells."""
    per_slide: dict[str, dict] = {}
    entries: list[tuple[str, int, str]] = []  # (slide, idx, obs_name)

    for sid in slides:
        path = prepared_dir / f"slide_{sid}.h5ad"
        adata = ad.read_h5ad(path, backed="r")
        n_obs = adata.n_obs
        if dataset == "spacseq":
            holdout = _spacseq_holdout_patch_idx(prepared_dir, sid)
        elif dataset == "perturbmap":
            holdout = _pm_holdout_idx(prepared_dir, sid, holdout_dir or "holdout_patch")
        else:
            raise ValueError(dataset)

        holdout_set = set(int(i) for i in holdout)
        train_idx = np.array([i for i in range(n_obs) if i not in holdout_set], dtype=int)
        obs_names = np.asarray(adata.obs_names)
        for i in train_idx:
            entries.append((sid, int(i), str(obs_names[i])))
        pert = adata.obs["perturbation_canon"].astype(str).to_numpy()[train_idx]
        n_una = int(np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"})).sum())
        n_ctrl = int(np.isin(pert, list(CONTROL_LABELS | {"control"})).sum())
        per_slide[sid] = {
            "n_obs": int(n_obs),
            "holdout_idx": holdout.astype(int).tolist(),
            "n_holdout": int(len(holdout)),
            "train_idx_before_subsample": train_idx.tolist(),
            "n_train_before_subsample": int(len(train_idx)),
            "n_unassigned_train": n_una,
            "n_control_train": n_ctrl,
        }
        log.info(
            "slide %s: n_obs=%d holdout=%d train=%d (una=%d ctrl=%d)",
            sid, n_obs, len(holdout), len(train_idx), n_una, n_ctrl,
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

    manifest = {
        "dataset": dataset,
        "control_mode": "ntc" if dataset == "spacseq" else "kp_control",
        "holdout_rule": (
            "patch_holdout role==1 unassigned"
            if dataset == "spacseq"
            else "holdout_patch KP source patches (role==1)"
        ),
        "slides": slides,
        "max_train_cells": int(max_train_cells),
        "seed": int(seed),
        "n_train_total": int(sum(per_slide[s]["n_train"] for s in slides)),
        "per_slide": per_slide,
    }
    return manifest


def save_manifest(manifest: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote train manifest (%d cells) -> %s", manifest["n_train_total"], path)
    return path


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_slide_train_adata(prepared_dir: Path, slide_id: str, manifest: dict) -> ad.AnnData:
    from spacseq_multi_common import _attach_counts_layer

    idx = np.asarray(manifest["per_slide"][slide_id]["train_idx"], dtype=int)
    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")
    _attach_counts_layer(adata, prepared_dir, slide_id)
    return adata[idx].copy()


def load_pooled_train_adatas(
    prepared_dir: Path,
    slides: list[str],
    manifest: dict,
) -> list[ad.AnnData]:
    return [load_slide_train_adata(prepared_dir, sid, manifest) for sid in slides]


def concat_pooled(adatas: list[ad.AnnData]) -> ad.AnnData:
    if len(adatas) == 1:
        return adatas[0]
    return ad.concat(adatas, join="outer", index_unique="-")


def resolve_training_adatas(prepared_dir, slides, args):
    """Load training AnnData list from a train manifest or the holdout exclusion files."""
    from spacseq_multi_common import load_prepared_slide

    manifest_path = getattr(args, "train_manifest", None)
    slides = list(slides)
    if manifest_path is not None:
        manifest = load_manifest(Path(manifest_path))
        log.info(
            "Using train manifest %s: %d cells (%s)",
            manifest_path,
            manifest["n_train_total"],
            manifest.get("regime", manifest.get("version", "?")),
        )
        return load_pooled_train_adatas(prepared_dir, slides, manifest), manifest
    adatas = [
        load_prepared_slide(
            prepared_dir, sid,
            exclude_holdout=getattr(args, "exclude_holdout", False),
            holdout_dir=getattr(args, "holdout_dir", "holdout"),
        )
        for sid in slides
    ]
    return adatas, None


def apply_control_mode_labels(adata: ad.AnnData, control_mode: str = "ntc") -> ad.AnnData:
    """Attach ``condition`` column for methods that need explicit control keys.

    SPAC-seq ``ntc``: NTC -> control, unassigned -> unassigned, KO -> gene.
    PM ``kp_control``: control (KP) -> control, unassigned -> unassigned, KO -> gene.
    """
    adata = adata.copy()
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    cond = np.empty(len(canon), dtype=object)
    ctrl_set = CONTROL_LABELS | {"control"}
    una_set = UNASSIGNED_LABELS | {"unassigned"}
    for i, p in enumerate(canon):
        if p in ctrl_set:
            cond[i] = "control"
        elif p in una_set:
            cond[i] = "unassigned"
        else:
            cond[i] = p
    adata.obs["condition"] = cond.astype(str)
    return adata
