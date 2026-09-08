"""Multi-slide SPAC-seq data prep and cross-slide counterfactual evaluation helpers."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr

from method_input_lib import normalize_pred_cache
from sklearn.preprocessing import MinMaxScaler

EXCLUDED_LABELS = frozenset(["multiple", "Multiple"])

from spacseq_eval_common import (
    CONTROL_LABELS,
    DEFAULT_BOOT_FRAC,
    DEFAULT_N_BOOT,
    DEFAULT_N_PATCH_BOOT,
    UNASSIGNED_LABELS,
    aggregate_scores,
    attach_bootstrap_stats,
    attach_patch_boot_stats,
    boot_settings,
    bootstrap_source_metrics,
    import_eval_utils,
    jsonify,
    patch_boot_settings,
    pick_random_holdout_patch,
    random_local_patch_rows,
    save_results,
    score_with_patch_boot,
    to_dense,
    worst_niche_score,
)


def add_bootstrap_args(p, default_n: int = 0):
    """CLI flags for within-niche source-patch bootstrap (0 = off)."""
    p.add_argument(
        "--n_boot", type=int, default=default_n,
        help="Bootstrap draws per niche by resampling source patch cells (0=off)",
    )
    p.add_argument("--boot_frac", type=float, default=DEFAULT_BOOT_FRAC,
                   help="Fraction of source cells per bootstrap draw")
    p.add_argument("--boot_seed", type=int, default=1234)


def add_random_patch_args(p):
    """Random non-niche holdout patch + patch-level resampling (non-spatial methods)."""
    p.add_argument(
        "--random_patch", action="store_true",
        help="Seen only: score on random holdout spatial patches outside eval niche "
             "(10 resamples; unseen ignores this flag)",
    )
    p.add_argument(
        "--n_patch_boot", type=int, default=DEFAULT_N_PATCH_BOOT,
        help="Resample random patch this many times for variance (uses mean±patch SD)",
    )
    p.add_argument("--patch_boot_seed", type=int, default=1234)


def method_results_subdir(method: str, random_patch: bool = False) -> str:
    """Result folder name: biolord vs biolord_sp (random patch variant)."""
    return f"{method}_sp" if random_patch else method

log = logging.getLogger(__name__)

DATA_DIR = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/"
    "spaVAE/datasets/newdata/spac_seq"
)
DEFAULT_SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]
HVG_MAX_RANK = 10_000
MIN_CELLS_PER_PERT = 25
LOC_RANGE = 20.0


def canonicalize_pert(label: str) -> str:
    if label in UNASSIGNED_LABELS:
        return "unassigned"
    if label in CONTROL_LABELS:
        return "control"
    return label


def slide_path(slide_id: str) -> Path:
    return DATA_DIR / f"raw_output_{slide_id}.h5ad"


def read_perturbations(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        ds = f["obs"]["perturbation"]
        if isinstance(ds, h5py.Dataset):
            arr = ds[:]
        else:
            arr = ds["categories"][:][ds["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr], dtype=object)


def combined_pert_counts(slides: list[str] | None = None) -> Counter:
    slides = slides or DEFAULT_SLIDES
    counts: Counter = Counter()
    for sid in slides:
        pert = read_perturbations(slide_path(sid))
        pert = pert[~np.isin(pert, list(EXCLUDED_LABELS))]
        canon = np.array([canonicalize_pert(p) for p in pert], dtype=object)
        counts.update(canon.tolist())
    return counts


def build_shared_hvg_mask(slides: list[str] | None = None, hvg_max_rank: int = HVG_MAX_RANK) -> np.ndarray:
    slides = slides or DEFAULT_SLIDES
    ranks = []
    gene_names_ref = None
    for sid in slides:
        with h5py.File(slide_path(sid), "r") as f:
            if gene_names_ref is None:
                gene_names_ref = np.array([
                    g.decode() if isinstance(g, bytes) else str(g)
                    for g in f["var"]["_index"][:]
                ])
            ranks.append(np.array(f["var"]["hvg_rank"][:], dtype=np.float32))
    rank_matrix = np.stack(ranks, axis=0)
    avg_rank = rank_matrix.mean(axis=0)
    sorted_idx = np.argsort(avg_rank)
    n_select = min(hvg_max_rank, len(avg_rank))
    mask = np.zeros(len(avg_rank), dtype=bool)
    mask[sorted_idx[:n_select]] = True
    log.info("Shared HVG mask: %d genes (avg-rank top-%d)", mask.sum(), hvg_max_rank)
    return mask, gene_names_ref


def load_and_filter_slide(
    slide_id: str,
    shared_hvg_mask: np.ndarray,
    valid_perts: set[str],
) -> ad.AnnData:
    path = slide_path(slide_id)
    log.info("Loading slide %s from %s", slide_id, path)
    adata = ad.read_h5ad(path)
    adata.var_names_make_unique()

    hvg_idx = np.where(shared_hvg_mask)[0]
    adata = adata[:, hvg_idx].copy()

    pert_raw = adata.obs["perturbation"].astype(str).values
    pert_canon = np.array([canonicalize_pert(p) for p in pert_raw], dtype=object)
    coords = adata.obsm["spatial"].astype(np.float64)

    keep = ~np.isin(pert_canon, list(EXCLUDED_LABELS))
    keep &= np.array(
        [p in valid_perts or p in ("unassigned", "control") for p in pert_canon],
        dtype=bool,
    )
    adata = adata[keep].copy()
    pert_canon = pert_canon[keep]
    coords = coords[keep]

    n_una = int((pert_canon == "unassigned").sum())
    n_ctrl = int((pert_canon == "control").sum())

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    row_sum = np.asarray(adata.X.sum(axis=1)).ravel()
    nz = row_sum > 0
    if (~nz).any():
        adata = adata[nz].copy()
        pert_canon = pert_canon[nz]
        coords = coords[nz]

    coords_scaled = MinMaxScaler().fit_transform(coords) * LOC_RANGE
    eu, _ = import_eval_utils()
    bin_id, _ = eu.build_niche_grid(coords_scaled, 6, LOC_RANGE)

    adata.obs["perturbation_canon"] = pert_canon
    adata.obs["sample_id"] = slide_id
    adata.obs["bin_id"] = bin_id.astype(int)
    adata.obsm["spatial"] = coords_scaled.astype(np.float32)
    log.info(
        "Slide %s filtered: %s, %d unique perts, unassigned=%d control=%d",
        slide_id, adata.shape, len(set(pert_canon)), n_una, n_ctrl,
    )
    return adata


def prepare_all_slides(
    outdir: Path,
    slides: list[str] | None = None,
    hvg_max_rank: int = HVG_MAX_RANK,
) -> Path:
    slides = slides or DEFAULT_SLIDES
    outdir.mkdir(parents=True, exist_ok=True)
    counts = combined_pert_counts(slides)
    valid_perts = {
        p for p, n in counts.items()
        if n >= MIN_CELLS_PER_PERT and p not in UNASSIGNED_LABELS and p not in CONTROL_LABELS
    }
    log.info("Combined valid KO perts: %d", len(valid_perts))

    shared_hvg_mask, gene_names = build_shared_hvg_mask(slides, hvg_max_rank)
    meta = {
        "slides": slides,
        "hvg_max_rank": hvg_max_rank,
        "min_cells_per_pert": MIN_CELLS_PER_PERT,
        "n_genes": int(shared_hvg_mask.sum()),
        "gene_names": gene_names[shared_hvg_mask].tolist(),
    }
    (outdir / "prepared_meta.json").write_text(json.dumps(meta, indent=2))

    for sid in slides:
        adata = load_and_filter_slide(sid, shared_hvg_mask, valid_perts)
        # Raw slide metadata (esp. uns/log1p base=null) breaks anndata 0.10 reads.
        adata.uns = {}
        dest = outdir / f"slide_{sid}.h5ad"
        adata.write_h5ad(dest)
        log.info("Wrote %s", dest)
    return outdir


def _strip_null_uns_h5ad(path: Path) -> None:
    """Remove null-encoded uns entries that anndata 0.10 cannot read."""
    with h5py.File(path, "a") as f:
        uns = f.get("uns")
        if uns is None:
            return
        if "log1p" in uns:
            del uns["log1p"]


def holdout_mask_path(prepared_dir: Path, slide_id: str, holdout_dir: str = "holdout") -> Path:
    return prepared_dir / holdout_dir / f"slide_{slide_id}.json"


def load_holdout_mask(prepared_dir: Path, slide_id: str, holdout_dir: str = "holdout") -> dict | None:
    path = holdout_mask_path(prepared_dir, slide_id, holdout_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _attach_counts_layer(adata: ad.AnnData, prepared_dir: Path, slide_id: str) -> None:
    """Attach raw counts from prepared/raw_counts cache when h5ad lacks layers['counts']."""
    if "counts" in adata.layers:
        return
    rc_path = prepared_dir / "raw_counts" / f"slide_{slide_id}" / "filtered_raw_counts.npy"
    if not rc_path.exists():
        log.warning("Slide %s: no raw counts cache at %s", slide_id, rc_path)
        return
    rc = np.load(rc_path, mmap_mode="r")
    if rc.shape[0] != adata.n_obs or rc.shape[1] != adata.n_vars:
        log.warning(
            "Slide %s: raw counts shape %s != adata %s -- skip counts layer",
            slide_id, rc.shape, adata.shape,
        )
        return
    adata.layers["counts"] = sp.csr_matrix(np.asarray(rc, dtype=np.float32))
    log.info("Slide %s: attached counts layer from %s", slide_id, rc_path)


def load_prepared_slide(
    prepared_dir: Path, slide_id: str, exclude_holdout: bool = False,
    holdout_dir: str = "holdout",
) -> ad.AnnData:
    path = prepared_dir / f"slide_{slide_id}.h5ad"
    try:
        adata = ad.read_h5ad(path)
    except Exception as exc:
        if "encoding_type='null'" not in str(exc) and "IORegistryError" not in type(exc).__name__:
            raise
        log.warning("Repairing null uns in %s then re-reading", path)
        _strip_null_uns_h5ad(path)
        adata = ad.read_h5ad(path)
    _attach_counts_layer(adata, prepared_dir, slide_id)
    if exclude_holdout:
        mask = load_holdout_mask(prepared_dir, slide_id, holdout_dir)
        if mask is None:
            log.warning("exclude_holdout=True but no mask (%s) for slide %s -- using all cells",
                        holdout_dir, slide_id)
        else:
            held = np.asarray(mask["union_holdout_idx"], dtype=int)
            keep = np.ones(adata.n_obs, dtype=bool)
            keep[held] = False
            log.info("Slide %s: excluding %d held-out cells from training (%s)",
                     slide_id, len(held), holdout_dir)
            adata = adata[keep].copy()
    return adata


def subsample_src_indices(
    src_idx: np.ndarray, max_cells: int, seed: int = 42,
) -> np.ndarray:
    src_idx = np.asarray(src_idx, dtype=int)
    if max_cells <= 0 or len(src_idx) <= max_cells:
        return src_idx
    rng = np.random.default_rng(seed)
    return rng.choice(src_idx, max_cells, replace=False)


def load_donor_slides(
    prepared_dir: Path,
    target_slide: str,
    eval_genes: list[str] | None = None,
    slides: list[str] | None = None,
    max_una_per_slide: int = 8000,
    max_cells_per_slide: int = 25000,
    seed: int = 42,
) -> ad.AnnData:
    """Concat donor slides with subsampling so eval fits in memory."""
    slides = slides or DEFAULT_SLIDES
    eval_genes = set(eval_genes or [])
    rng = np.random.default_rng(seed)
    parts: list[ad.AnnData] = []
    for sid in slides:
        if sid == target_slide:
            continue
        adata = load_prepared_slide(prepared_dir, sid)
        pert = adata.obs["perturbation_canon"].astype(str).to_numpy()
        ko_keep = np.isin(pert, list(eval_genes))
        ctrl_keep = pert == "control"
        una_idx = np.where(pert == "unassigned")[0]
        if len(una_idx) > max_una_per_slide:
            una_idx = rng.choice(una_idx, max_una_per_slide, replace=False)
        keep = ko_keep | ctrl_keep
        keep[una_idx] = True
        pick = np.where(keep)[0]
        if len(pick) > max_cells_per_slide:
            must = np.where(ko_keep | ctrl_keep)[0]
            rest = np.setdiff1d(pick, must)
            n_bg = max(max_cells_per_slide - len(must), 0)
            bg = rng.choice(rest, min(n_bg, len(rest)), replace=False) if n_bg and len(rest) else np.array([], dtype=int)
            pick = np.sort(np.union1d(must, bg))
        parts.append(adata[pick].copy())
    if not parts:
        raise ValueError(f"No donor slides for target {target_slide}")
    return ad.concat(parts, join="outer")


def concat_prepared_slides(prepared_dir: Path, slides: list[str] | None = None) -> ad.AnnData:
    slides = slides or DEFAULT_SLIDES
    parts = [load_prepared_slide(prepared_dir, sid) for sid in slides]
    return ad.concat(parts, label="batch", keys=slides, index_unique="-")


def slide_ko_sets(prepared_dir: Path, slides: list[str] | None = None) -> dict[str, set[str]]:
    slides = slides or DEFAULT_SLIDES
    out: dict[str, set[str]] = {}
    for sid in slides:
        adata = load_prepared_slide(prepared_dir, sid)
        pert = adata.obs["perturbation_canon"].astype(str).to_numpy()
        ko = set(pert) - {"control", "unassigned"}
        out[sid] = ko
    return out


def build_cross_slide_manifest(
    prepared_dir: Path,
    out_path: Path | None = None,
    slides: list[str] | None = None,
    k: int = 10,
    seed: int = 42,
    min_donor_ko_cells: int = MIN_CELLS_PER_PERT,
) -> dict:
    slides = slides or DEFAULT_SLIDES
    ko_sets = slide_ko_sets(prepared_dir, slides)
    rng = np.random.default_rng(seed)
    manifest = {
        "slides": slides,
        "k_default": k,
        "seed": seed,
        "min_donor_ko_cells": min_donor_ko_cells,
        "selection_rule": "zero_ko_on_target_and_seen_on_other_slide",
        "per_slide": {},
    }

    for sid in slides:
        local = ko_sets[sid]
        others = set().union(*[ko_sets[o] for o in slides if o != sid])
        candidates = sorted(others - local)
        donor_counts = {}
        ad_donor_parts = []
        for other in slides:
            if other == sid:
                continue
            ad_other = load_prepared_slide(prepared_dir, other)
            ad_donor_parts.append(ad_other)
        ad_donor = ad.concat(ad_donor_parts, join="outer") if ad_donor_parts else None
        pert_all = (
            ad_donor.obs["perturbation_canon"].astype(str).to_numpy()
            if ad_donor is not None
            else np.array([], dtype=object)
        )
        for g in candidates:
            donor_counts[g] = int((pert_all == g).sum())

        ranked = sorted(
            candidates,
            key=lambda g: (-donor_counts.get(g, 0), g),
        )
        pick = ranked[: min(k, len(ranked))]
        if len(ranked) > len(pick):
            rest = [g for g in ranked if g not in pick]
            rng.shuffle(rest)
            # already top by count; keep deterministic top-k

        manifest["per_slide"][sid] = {
            "eval_genes": pick,
            "n_available": len(candidates),
            "donor_ko_counts": {g: donor_counts[g] for g in pick},
        }
        log.info(
            "Slide %s: %d cross-slide candidates, evaluating k=%d",
            sid,
            len(candidates),
            len(pick),
        )

    out_path = out_path or (prepared_dir / "cross_slide_eval_manifest.json")
    out_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest -> %s", out_path)
    return manifest


def target_eval_context(adata_target: ad.AnnData, niche_grid: int = 6, una_floor: int = 50):
    pert_canon = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata_target.obs["bin_id"].to_numpy().astype(int)
    una_mask = np.isin(pert_canon, list(UNASSIGNED_LABELS | {"unassigned"}))
    n_bins = niche_grid ** 2
    una_per_bin = np.bincount(bin_id[una_mask], minlength=n_bins)
    evaluable = una_per_bin >= una_floor
    return pert_canon, bin_id, una_mask, evaluable


def donor_reference_lfc(
    adata_donor: ad.AnnData,
    gene: str,
    hvg_mask: np.ndarray,
    min_cells: int = 20,
) -> np.ndarray | None:
    eu, _ = import_eval_utils()
    pert = adata_donor.obs["perturbation_canon"].astype(str).to_numpy()
    una_mask = np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"}))
    ko_idx = np.where(pert == gene)[0]
    una_idx = np.where(una_mask)[0]
    if len(ko_idx) < min_cells or len(una_idx) < min_cells:
        return None
    y_ko = to_dense(adata_donor[ko_idx].X)
    y_una = to_dense(adata_donor[una_idx].X)
    return eu.pseudobulk_lfc(y_ko, y_una)


def lib_sizes_from_adata(adata: ad.AnnData, idx: np.ndarray) -> np.ndarray:
    if "n_counts" in adata.obs.columns:
        return adata.obs["n_counts"].to_numpy()[idx].astype(np.float64)
    x = adata[idx].X
    if sp.issparse(x):
        return np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    return np.asarray(x, dtype=np.float64).sum(axis=1)


def pseudo_raw_counts(adata: ad.AnnData, idx: np.ndarray) -> np.ndarray:
    _, sa = import_eval_utils()
    x = to_dense(adata[idx].X)
    lib = lib_sizes_from_adata(adata, idx)
    return sa.log_cp10k_to_counts(x, lib)


def niche_lfc_cross_slide_loop(
    eval_genes: list[str],
    adata_target: ad.AnnData,
    adata_donor: ad.AnnData,
    pred_g_cache: dict[str, np.ndarray | None],
    global_src_idx: np.ndarray,
    hvg_mask: np.ndarray,
    args,
    unpredictable_genes: set[str] | None = None,
    bad_edist_default: float = 12.0,
    scoring_method: str | None = None,
    pred_basal=None,
):
    """Score cross-slide transfer: model on target unassigned vs donor-reference LFC.

    If ``pred_basal`` (predicted basal on the source cells, aligned with global_src_idx)
    is given, predicted LFC = DELTA pred_KO - pred_basal (cancels basal reconstruction
    error) instead of pred_KO - observed source."""
    eu, sa = import_eval_utils()
    unpredictable_genes = set(unpredictable_genes or [])
    if scoring_method:
        pred_g_cache = normalize_pred_cache(scoring_method, pred_g_cache)
        if pred_basal is not None:
            pred_basal = normalize_pred_cache(scoring_method, {"_b": pred_basal})["_b"]
    gene_names = np.asarray(adata_target.var_names)
    pert_canon, bin_id, una_mask, evaluable = target_eval_context(
        adata_target, args.niche_grid, args.una_floor,
    )
    eval_bins = np.where(evaluable)[0]
    scores = {"L2": {g: [] for g in eval_genes}}
    agg = {"L2": {}}
    mlfc = {"L2": {g: {} for g in eval_genes}}

    global_src_idx = np.asarray(global_src_idx, dtype=int)
    if args.max_src_cells > 0 and len(global_src_idx) > args.max_src_cells:
        rng = np.random.default_rng(42)
        global_src_idx = rng.choice(global_src_idx, args.max_src_cells, replace=False)
    src_expr = to_dense(adata_target[global_src_idx].X)
    n_src = len(global_src_idx)
    log.info("[L2] target source cells: %d", n_src)

    donor_ko_idx = {}
    for g in eval_genes:
        pert_d = adata_donor.obs["perturbation_canon"].astype(str).to_numpy()
        donor_ko_idx[g] = np.where(pert_d == g)[0]

    worst_metrics = {
        "spearman": 0.0,
        "abs_spearman": 0.0,
        "sign_agreement": 0.5,
        "topk_overlap": 0.0,
        "e_dist": bad_edist_default,
    }

    for g in eval_genes:
        o_lfc_ref = donor_reference_lfc(adata_donor, g, hvg_mask, args.min_g_cells)
        if o_lfc_ref is None:
            log.warning("[L2/%s] insufficient donor reference -- worst-case", g)
            for b in eval_bins:
                una_b = np.where(una_mask & (bin_id == b))[0]
                if len(una_b) < args.min_g_cells:
                    continue
                keep = hvg_mask & (gene_names != g)
                scores["L2"][g].append(
                    worst_niche_score(int(b), n_src, 0, int(keep.sum()), bad_edist_default)
                )
            continue

        pred_g = pred_g_cache.get(g)
        if g in unpredictable_genes or pred_g is None or len(pred_g) != n_src:
            for b in eval_bins:
                una_b = np.where(una_mask & (bin_id == b))[0]
                if len(una_b) < args.min_g_cells:
                    continue
                keep = hvg_mask & (gene_names != g)
                scores["L2"][g].append(
                    worst_niche_score(int(b), n_src, 0, int(keep.sum()), bad_edist_default)
                )
            continue

        donor_ko = donor_ko_idx[g]
        obs_raw_donor = pseudo_raw_counts(adata_donor, donor_ko)
        y_ko_log = to_dense(adata_donor[donor_ko].X)

        for b in eval_bins:
            una_b = np.where(una_mask & (bin_id == b))[0]
            if len(una_b) < args.min_g_cells:
                continue
            src_rows = np.where(bin_id[global_src_idx] == b)[0]
            if len(src_rows) < args.min_g_cells:
                continue
            keep = hvg_mask & (gene_names != g)
            n_boot, frac, boot_seed = boot_settings(args)

            pred_b = pred_g[src_rows]
            ref_rows = pred_basal[src_rows] if pred_basal is not None else src_expr[src_rows]
            m_lfc = eu.pseudobulk_lfc(pred_b, ref_rows)
            m_lfc = np.nan_to_num(m_lfc, nan=0.0, posinf=0.0, neginf=0.0)
            sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc_ref[keep])
            abs_rho = float(
                spearmanr(pred_b[:, keep].mean(axis=0), y_ko_log[:, keep].mean(axis=0)).correlation
            )
            sc_["abs_spearman"] = 0.0 if np.isnan(abs_rho) else abs_rho
            sc_["e_dist"] = sa.aligned_energy_distance(
                pred_b,
                obs_raw_donor,
                lib_sizes_from_adata(adata_target, global_src_idx[src_rows]),
                keep,
            )
            sc_.update(
                niche=int(b),
                n_src=int(len(src_rows)),
                n_real_g=int(len(donor_ko)),
                n_scored_genes=int(keep.sum()),
            )
            if n_boot > 0:
                attach_bootstrap_stats(sc_, bootstrap_source_metrics(
                    pred_b, ref_rows, o_lfc_ref, obs_raw_donor,
                    lib_sizes_from_adata(adata_target, global_src_idx[src_rows]),
                    keep, eu, sa,
                    n_boot=n_boot, frac=frac,
                    seed=boot_seed + int(b) * 1009 + hash(g) % 997,
                ))
            scores["L2"][g].append(sc_)
            mlfc["L2"][g][int(b)] = m_lfc.tolist()

        scored_edists = [s["e_dist"] for s in scores["L2"][g] if np.isfinite(s.get("e_dist", np.nan))]
        if scored_edists:
            worst_metrics["e_dist"] = float(np.max(scored_edists))

    for g in eval_genes:
        agg["L2"][g] = aggregate_scores(scores, "L2", g, worst_metrics)
        if agg["L2"][g]["n_niches"]:
            log.info(
                "[L2] %-12s lfc_rho=%.3f abs_rho=%.3f e_dist=%.3f (n=%d)",
                g,
                agg["L2"][g].get("spearman_mean", float("nan")),
                agg["L2"][g].get("abs_spearman_mean", float("nan")),
                agg["L2"][g].get("e_dist_mean", float("nan")),
                agg["L2"][g]["n_niches"],
            )

    return scores, agg, mlfc


def all_manifest_genes(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    genes = set()
    for info in manifest["per_slide"].values():
        genes.update(info["eval_genes"])
    return sorted(genes)


def load_manifest_genes(manifest_path: Path, slide_id: str) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    return list(manifest["per_slide"][slide_id]["eval_genes"])


def save_cross_slide_results(out_dir: Path, slide_id: str, eval_genes, scores, agg, mlfc, hvg_info, extra=None):
    out_dir = out_dir / f"slide_{slide_id}"
    meta = {"slide_id": slide_id, "eval_genes": eval_genes}
    if extra:
        meta.update(extra)
    save_results(out_dir, scores, agg, mlfc, hvg_info)
    (out_dir / "eval_meta.json").write_text(json.dumps(jsonify(meta), indent=2))


def a1_source_indices(prepared_dir, slide_id, pert_array, max_src_cells, src_mode="subsample", seed=42):
    """Analysis 1 source cells: patch holdout unassigned (v2), JSON holdout, or subsample."""
    if src_mode == "patch":
        return a1_patch_source_indices(prepared_dir, slide_id, max_src_cells, seed=seed)
    if src_mode == "holdout":
        mask = load_holdout_mask(prepared_dir, slide_id)
        if mask is None:
            raise FileNotFoundError(f"No holdout mask for slide {slide_id}")
        idx = np.asarray(mask["a1_holdout_idx"], dtype=int)
        return subsample_src_indices(idx, max_src_cells, seed=seed)
    return subsample_src_indices(
        np.where(np.asarray(pert_array) == "unassigned")[0], max_src_cells, seed=seed,
    )


def a1_patch_source_indices(prepared_dir, slide_id, max_src_cells, seed=42) -> np.ndarray:
    """Analysis 1 source: held-out unassigned spatial patches (role==1)."""
    adata = load_prepared_slide(prepared_dir, slide_id)
    role = load_patch_roles(prepared_dir, slide_id, adata)
    if role is None:
        raise FileNotFoundError(f"No patch_holdout for slide {slide_id}")
    pert = adata.obs["perturbation_canon"].astype(str).to_numpy()
    una = np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"}))
    idx = np.where((role == 1) & una)[0]
    return subsample_src_indices(idx, max_src_cells, seed=seed)


def niche_lfc_seen_holdout_loop(
    panel_niche,
    adata_target,
    src_idx,
    pred_g_cache,
    hvg_mask,
    raw_mm,
    lib_sizes,
    args,
    unpredictable_genes=None,
    bad_edist_default=12.0,
    scoring_method: str | None = None,
    without_noise_out: dict | None = None,
    pred_basal=None,
    ko_gt_rows=None,
):
    """Analysis 2 holdout scoring: each seen KO scored only in its most-abundant niche.

    If ``pred_basal`` (predicted basal counterfactual on the same source cells, aligned
    with src_idx) is given, the predicted LFC uses the DELTA pred_KO - pred_basal (cancels
    basal reconstruction error, isolates μ_action + f_pert) instead of pred_KO - observed
    source. This is how counterfactual models are meant to be scored.

    Source = held-out unassigned cells (passed as src_idx). For gene g with niche b*,
    use the held-out source cells located in b*; observed = real KO g vs unassigned in b*
    (with-noise = μ_action + f_pert).

    If ``without_noise_out`` (a dict) is passed, ALSO compute the without-noise flavour
    per niche where NTC (control) cells exist: observed = KO vs NTC, predicted = with-noise
    m_lfc + empirical offset log(mean_una_src) - log(mean_NTC) (mirrors the CF eval,
    eval_spatial_counterfactual.py:557). Results land in without_noise_out['L2'][g].
    """
    eu, sa = import_eval_utils()
    gene_names = np.asarray(adata_target.var_names)
    pert_canon = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata_target.obs["bin_id"].to_numpy().astype(int)
    una_mask = np.isin(pert_canon, list(UNASSIGNED_LABELS | {"unassigned"}))
    ntc_mask = np.isin(pert_canon, list(CONTROL_LABELS | {"control"}))
    compute_wo = without_noise_out is not None
    if compute_wo:
        without_noise_out.setdefault("L2", {})
    src_idx = np.asarray(src_idx, dtype=int)
    src_expr = to_dense(adata_target[src_idx].X)
    src_bin = bin_id[src_idx]
    panel = list(panel_niche.keys())
    unpredictable_genes = set(unpredictable_genes or [])
    if scoring_method:
        pred_g_cache = normalize_pred_cache(scoring_method, pred_g_cache)
        if pred_basal is not None:
            pred_basal = normalize_pred_cache(scoring_method, {"_b": pred_basal})["_b"]
    # DELTA reference: predicted basal on the source cells (else observed source).
    ref_expr = pred_basal if pred_basal is not None else src_expr

    scores = {"L2": {g: [] for g in panel}}
    agg = {"L2": {}}
    mlfc = {"L2": {g: {} for g in panel}}
    worst_metrics = {
        "spearman": 0.0, "abs_spearman": 0.0, "sign_agreement": 0.5,
        "topk_overlap": 0.0, "e_dist": bad_edist_default,
    }

    for g in panel:
        b = int(panel_niche[g])
        keep = hvg_mask & (gene_names != g)
        src_rows = np.where(src_bin == b)[0]
        ko_idx = np.where((pert_canon == g) & (bin_id == b))[0]
        if ko_gt_rows is not None:
            ko_idx = ko_idx[np.isin(ko_idx, ko_gt_rows)]
        una_b = np.where(una_mask & (bin_id == b))[0]
        pred_g = pred_g_cache.get(g)
        if (
            g in unpredictable_genes or pred_g is None
            or len(src_rows) < args.min_g_cells
            or len(ko_idx) < args.min_g_cells
            or len(una_b) < args.min_g_cells
        ):
            scores["L2"][g].append(
                worst_niche_score(b, len(src_rows), len(ko_idx), int(keep.sum()), bad_edist_default)
            )
            continue
        pred_b = pred_g[src_rows]
        y_ko_b = to_dense(adata_target[ko_idx].X)
        y_una_b = to_dense(adata_target[una_b].X)
        o_lfc = eu.pseudobulk_lfc(y_ko_b, y_una_b)
        random_patch = getattr(args, "random_patch", False)
        n_patch_boot, patch_seed = patch_boot_settings(args)
        n_boot, frac, boot_seed = boot_settings(args)

        if random_patch and n_patch_boot > 0:
            prepared_dir = getattr(args, "prepared_dir", None)
            slide_id = getattr(args, "slide_id", None)
            patch_list = (
                holdout_patch_groups_spacseq(
                    adata_target, prepared_dir, slide_id, src_idx, b, args.min_g_cells,
                )
                if prepared_dir is not None and slide_id is not None
                else []
            )
            if not patch_list:
                scores["L2"][g].append(
                    worst_niche_score(b, len(src_rows), len(ko_idx), int(keep.sum()), bad_edist_default)
                )
                continue
            patch_stats = score_with_patch_boot(
                pred_g, src_expr, o_lfc, np.asarray(raw_mm[ko_idx]),
                lib_sizes[src_idx], keep, eu, sa,
                pick_rows=lambda rng, _pl=patch_list: pick_random_holdout_patch(_pl, rng),
                n_patch_boot=n_patch_boot,
                seed=patch_seed + int(b) * 1009 + hash(g) % 997,
            )
            if patch_stats is None:
                scores["L2"][g].append(
                    worst_niche_score(b, len(src_rows), len(ko_idx), int(keep.sum()), bad_edist_default)
                )
                continue
            rows0 = pick_random_holdout_patch(
                patch_list, np.random.default_rng(patch_seed + int(b)),
            )
            pred_b = pred_g[rows0]
            m_lfc = np.nan_to_num(
                eu.pseudobulk_lfc(pred_b, ref_expr[rows0]), nan=0.0, posinf=0.0, neginf=0.0,
            )
            sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
            abs_rho = float(spearmanr(pred_b[:, keep].mean(axis=0), y_ko_b[:, keep].mean(axis=0)).correlation)
            sc_["abs_spearman"] = 0.0 if np.isnan(abs_rho) else abs_rho
            attach_patch_boot_stats(sc_, patch_stats)
            sc_.update(
                niche=b, n_src=int(len(rows0)), n_real_g=int(len(ko_idx)),
                n_scored_genes=int(keep.sum()), random_patch=True,
            )
        else:
            m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pred_b, ref_expr[src_rows]), nan=0.0, posinf=0.0, neginf=0.0)
            sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
            abs_rho = float(spearmanr(pred_b[:, keep].mean(axis=0), y_ko_b[:, keep].mean(axis=0)).correlation)
            sc_["abs_spearman"] = 0.0 if np.isnan(abs_rho) else abs_rho
            sc_["e_dist"] = sa.aligned_energy_distance(
                pred_b, np.asarray(raw_mm[ko_idx]), lib_sizes[src_idx[src_rows]], keep,
            )
            sc_.update(niche=b, n_src=int(len(src_rows)), n_real_g=int(len(ko_idx)), n_scored_genes=int(keep.sum()))
            if n_boot > 0:
                attach_bootstrap_stats(sc_, bootstrap_source_metrics(
                    pred_b, ref_expr[src_rows], o_lfc, np.asarray(raw_mm[ko_idx]),
                    lib_sizes[src_idx[src_rows]], keep, eu, sa,
                    n_boot=n_boot, frac=frac, seed=boot_seed + int(b) * 1009 + hash(g) % 997,
                ))
        scores["L2"][g].append(sc_)
        mlfc["L2"][g][b] = m_lfc.tolist()

        # Without-noise (KO vs NTC) via empirical re-reference -- only where NTC exists.
        if compute_wo:
            ntc_b = np.where(ntc_mask & (bin_id == b))[0]
            if len(ntc_b) >= args.min_g_cells:
                y_ntc_b = to_dense(adata_target[ntc_b].X)
                emp_offset = eu.pseudobulk_lfc(y_una_b, y_ntc_b)
                m_without = np.nan_to_num(m_lfc + emp_offset, nan=0.0, posinf=0.0, neginf=0.0)
                o_without = eu.pseudobulk_lfc(y_ko_b, y_ntc_b)
                sc_wo = eu.score_lfc_agreement(m_without[keep], o_without[keep])
                sc_wo.update(niche=b, n_real_g=int(len(ko_idx)), n_ntc=int(len(ntc_b)),
                             n_scored_genes=int(keep.sum()))
                without_noise_out["L2"].setdefault(g, []).append(sc_wo)

    for g in panel:
        agg["L2"][g] = aggregate_scores(scores, "L2", g, worst_metrics)
        if agg["L2"][g]["n_niches"]:
            log.info(
                "[L2/holdout] %-12s niche=%d lfc_rho=%.3f abs_rho=%.3f e_dist=%.3f",
                g, int(panel_niche[g]),
                agg["L2"][g].get("spearman_mean", float("nan")),
                agg["L2"][g].get("abs_spearman_mean", float("nan")),
                agg["L2"][g].get("e_dist_mean", float("nan")),
            )
    return scores, agg, mlfc


def patch_holdout_path(prepared_dir: Path, slide_id: str) -> Path:
    """Per-niche contiguous spatial patch split (built by patch_holdout_index.py)."""
    return prepared_dir.parent / "patch_holdout" / f"{slide_id}.npz"


def load_patch_roles(prepared_dir: Path, slide_id: str, adata_target: ad.AnnData) -> np.ndarray | None:
    """Return per-cell role aligned to adata_target order (-1 excluded, 0 seen, 1 holdout patch)."""
    path = patch_holdout_path(prepared_dir, slide_id)
    if not path.exists():
        log.warning("No patch holdout file %s", path)
        return None
    d = np.load(path, allow_pickle=True)
    obs = np.asarray(adata_target.obs_names)
    npz_obs = d["obs_name"].astype(str)
    role = np.asarray(d["role"], dtype=int)
    if np.array_equal(obs, npz_obs):
        return role
    pos = {n: i for i, n in enumerate(npz_obs)}
    if not all(n in pos for n in obs):
        raise ValueError(f"patch holdout {path} obs_names do not cover slide {slide_id}")
    order = np.array([pos[n] for n in obs], dtype=int)
    return role[order]


def _align_patch_npz_field(d, field: str, adata_target: ad.AnnData) -> np.ndarray:
    obs = np.asarray(adata_target.obs_names)
    npz_obs = d["obs_name"].astype(str)
    arr = np.asarray(d[field])
    if np.array_equal(obs, npz_obs):
        return arr
    pos = {n: i for i, n in enumerate(npz_obs)}
    order = np.array([pos[n] for n in obs], dtype=int)
    return arr[order]


def holdout_patch_groups_spacseq(
    adata_target: ad.AnnData,
    prepared_dir: Path,
    slide_id: str,
    global_src_idx: np.ndarray,
    exclude_bin: int,
    min_cells: int,
) -> list[np.ndarray]:
    """Contiguous holdout sub-patches (role==1) outside ``exclude_bin``.

    Uses ``patch_holdout/{slide}.npz`` subpatch labels; returns local indices
    into ``global_src_idx`` (one array per spatial holdout patch).
    """
    path = patch_holdout_path(prepared_dir, slide_id)
    if not path.exists():
        log.warning("No patch holdout file for slide %s -- cannot build _sp patches", slide_id)
        return []
    d = np.load(path, allow_pickle=True)
    role = _align_patch_npz_field(d, "role", adata_target)
    subp = _align_patch_npz_field(d, "subpatch", adata_target)
    bin_id = adata_target.obs["bin_id"].to_numpy().astype(int)
    pert = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    una = np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"}))
    global_src_idx = np.asarray(global_src_idx, dtype=int)
    g2l = {int(g): i for i, g in enumerate(global_src_idx)}
    patches: list[np.ndarray] = []
    for b in np.unique(bin_id):
        if int(b) == int(exclude_bin):
            continue
        in_b = bin_id == b
        for sp in np.unique(subp[in_b & (role == 1) & una]):
            if int(sp) < 0:
                continue
            gidx = np.where(in_b & (subp == sp) & (role == 1) & una)[0]
            gidx = gidx[np.isin(gidx, list(g2l))]
            if len(gidx) < int(min_cells):
                continue
            patches.append(np.array([g2l[int(g)] for g in gidx], dtype=int))
    return patches


def patch_dominant_perts(
    adata_target: ad.AnnData,
    role: np.ndarray,
    min_g_cells: int,
    src_role: int = 1,
    baseline_role: int = 0,
) -> dict[int, str]:
    """Per niche: the most-abundant real KO gene, if the niche is evaluable.

    src_role selects which patch is used as the prediction source: 1 = held-out
    patch (default), 2 = seen patch. baseline_role selects the observed-unassigned
    reference (0 = context unassigned, disjoint from both patches). A niche is kept
    only if it has >= min_g_cells source-patch cells, >= min_g_cells baseline
    unassigned cells, and its dominant KO gene has >= min_g_cells real cells.
    Otherwise the patch/niche is skipped (as before).
    """
    pert = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata_target.obs["bin_id"].to_numpy().astype(int)
    una_mask = np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"}))
    ko_mask = ~np.isin(pert, ["unassigned", "control"])
    niche_dom: dict[int, str] = {}
    for b in np.unique(bin_id):
        in_b = bin_id == b
        n_src = int(((role == src_role) & una_mask & in_b).sum())
        if n_src < min_g_cells:
            continue
        if int(((role == baseline_role) & una_mask & in_b).sum()) < min_g_cells:
            continue
        ko_b = in_b & ko_mask
        if not ko_b.any():
            continue
        genes, counts = np.unique(pert[ko_b], return_counts=True)
        gi = int(np.argmax(counts))
        if int(counts[gi]) < min_g_cells:
            continue
        niche_dom[int(b)] = str(genes[gi])
    return niche_dom


def patch_source_indices(
    adata_target: ad.AnnData,
    role: np.ndarray,
    niche_dom: dict[int, str],
    max_src_cells_per_niche: int,
    seed: int = 42,
    src_role: int = 1,
) -> np.ndarray:
    """Union of source-patch unassigned cells across evaluable niches (per-niche capped)."""
    pert = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata_target.obs["bin_id"].to_numpy().astype(int)
    una_mask = np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"}))
    rng = np.random.default_rng(seed)
    parts = []
    for b in niche_dom:
        idx_b = np.where((role == src_role) & una_mask & (bin_id == b))[0]
        if max_src_cells_per_niche > 0 and len(idx_b) > max_src_cells_per_niche:
            idx_b = rng.choice(idx_b, max_src_cells_per_niche, replace=False)
        parts.append(idx_b)
    if not parts:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(parts))


def niche_lfc_patch_holdout_loop(
    niche_dom,
    adata_target,
    src_idx,
    pred_g_cache,
    hvg_mask,
    raw_mm,
    lib_sizes,
    args,
    unpredictable_genes=None,
    bad_edist_default=12.0,
    exclude_from_baseline_idx=None,
    scoring_method: str | None = None,
    pred_basal=None,
):
    """Patch-holdout scoring: each niche scored on its dominant KO only.

    Source = source-patch unassigned cells located in that niche (subset of
    src_idx). Observed = real dominant-KO cells vs unassigned cells in the niche.
    If exclude_from_baseline_idx is given, those cells (the full source patch) are
    removed from the observed unassigned reference so source and reference are
    disjoint. Results are accumulated per gene (a gene dominant in several niches
    gets several niche entries).
    """
    eu, sa = import_eval_utils()
    gene_names = np.asarray(adata_target.var_names)
    pert_canon = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata_target.obs["bin_id"].to_numpy().astype(int)
    una_mask = np.isin(pert_canon, list(UNASSIGNED_LABELS | {"unassigned"}))
    src_idx = np.asarray(src_idx, dtype=int)
    src_expr = to_dense(adata_target[src_idx].X)
    src_bin = bin_id[src_idx]
    panel = sorted(set(niche_dom.values()))
    unpredictable_genes = set(unpredictable_genes or [])
    if scoring_method:
        pred_g_cache = normalize_pred_cache(scoring_method, pred_g_cache)
        if pred_basal is not None:
            pred_basal = normalize_pred_cache(scoring_method, {"_b": pred_basal})["_b"]
    ref_expr = pred_basal if pred_basal is not None else src_expr
    exclude_baseline = (
        set(np.asarray(exclude_from_baseline_idx, dtype=int).tolist())
        if exclude_from_baseline_idx is not None else set()
    )

    scores = {"L2": {g: [] for g in panel}}
    agg = {"L2": {}}
    mlfc = {"L2": {g: {} for g in panel}}
    worst_metrics = {
        "spearman": 0.0, "abs_spearman": 0.0, "sign_agreement": 0.5,
        "topk_overlap": 0.0, "e_dist": bad_edist_default,
    }

    for b, g in sorted(niche_dom.items()):
        keep = hvg_mask & (gene_names != g)
        src_rows = np.where(src_bin == b)[0]
        ko_idx = np.where((pert_canon == g) & (bin_id == b))[0]
        una_b = np.where(una_mask & (bin_id == b))[0]
        if exclude_baseline:
            una_b = np.array([i for i in una_b if i not in exclude_baseline], dtype=int)
        pred_g = pred_g_cache.get(g)
        if (
            g in unpredictable_genes or pred_g is None
            or len(src_rows) < args.min_g_cells
            or len(ko_idx) < args.min_g_cells
            or len(una_b) < args.min_g_cells
        ):
            sc_ = worst_niche_score(int(b), len(src_rows), len(ko_idx), int(keep.sum()), bad_edist_default)
            scores["L2"][g].append(sc_)
            continue
        pred_b = pred_g[src_rows]
        y_ko_b = to_dense(adata_target[ko_idx].X)
        y_una_b = to_dense(adata_target[una_b].X)
        m_lfc = np.nan_to_num(
            eu.pseudobulk_lfc(pred_b, ref_expr[src_rows]), nan=0.0, posinf=0.0, neginf=0.0,
        )
        o_lfc = eu.pseudobulk_lfc(y_ko_b, y_una_b)
        sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
        abs_rho = float(spearmanr(pred_b[:, keep].mean(axis=0), y_ko_b[:, keep].mean(axis=0)).correlation)
        sc_["abs_spearman"] = 0.0 if np.isnan(abs_rho) else abs_rho
        sc_["e_dist"] = sa.aligned_energy_distance(
            pred_b, np.asarray(raw_mm[ko_idx]), lib_sizes[src_idx[src_rows]], keep,
        )
        sc_.update(
            niche=int(b), n_src=int(len(src_rows)), n_real_g=int(len(ko_idx)),
            n_scored_genes=int(keep.sum()),
        )
        n_boot, frac, boot_seed = boot_settings(args)
        if n_boot > 0:
            attach_bootstrap_stats(sc_, bootstrap_source_metrics(
                pred_b, ref_expr[src_rows], o_lfc, np.asarray(raw_mm[ko_idx]),
                lib_sizes[src_idx[src_rows]], keep, eu, sa,
                n_boot=n_boot, frac=frac, seed=boot_seed + int(b) * 1009 + hash(g) % 997,
            ))
        scores["L2"][g].append(sc_)
        mlfc["L2"][g][int(b)] = m_lfc.tolist()

    for g in panel:
        agg["L2"][g] = aggregate_scores(scores, "L2", g, worst_metrics)
        if agg["L2"][g]["n_niches"]:
            log.info(
                "[L2/patch] %-12s n_niches=%d lfc_rho=%.3f abs_rho=%.3f e_dist=%.3f",
                g, agg["L2"][g]["n_niches"],
                agg["L2"][g].get("spearman_mean", float("nan")),
                agg["L2"][g].get("abs_spearman_mean", float("nan")),
                agg["L2"][g].get("e_dist_mean", float("nan")),
            )
    return scores, agg, mlfc


def select_seen_panel_for_slide(
    adata_target: ad.AnnData,
    niche_grid: int = 6,
    ntc_floor: int = 30,
    min_panel_cells: int = 200,
    min_g_cells: int = 20,
    max_panel: int = 30,
    min_niche_frac: float = 0.0,
) -> tuple[list[str], np.ndarray]:
    """Top seen KO perturbations on a single slide (same rule as one-slide panel)."""
    eu, _ = import_eval_utils()
    pert = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata_target.obs["bin_id"].to_numpy().astype(int)
    una_mask = np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"}))
    n_bins = niche_grid ** 2
    una_per_bin = np.bincount(bin_id[una_mask], minlength=n_bins)
    evaluable = una_per_bin >= ntc_floor
    n_eval = int(evaluable.sum())
    panel = eu.select_perturbation_panel(
        pert,
        bin_id,
        n_eval,
        min_cells=min_panel_cells,
        min_cells_per_niche=min_g_cells,
        min_niche_frac=min_niche_frac,
        max_panel=max_panel,
    )
    log.info("Slide seen panel (%d): %s", len(panel), panel)
    return panel, evaluable


def prepared_raw_counts_dir(prepared_dir: Path, slide_id: str) -> Path:
    return prepared_dir / "raw_counts" / f"slide_{slide_id}"


CONCERT_PYTHON = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env/bin/python"


def ensure_prepared_raw_counts(prepared_dir: Path, slide_id: str) -> Path:
    """Build aligned raw-count cache for a prepared slide (for e-dist scoring)."""
    import subprocess

    outdir = prepared_raw_counts_dir(prepared_dir, slide_id)
    counts = outdir / "filtered_raw_counts.npy"
    if counts.exists():
        return outdir
    outdir.mkdir(parents=True, exist_ok=True)
    prepared_h5ad = prepared_dir / f"slide_{slide_id}.h5ad"
    raw_h5ad = slide_path(slide_id)
    build_script = (
        "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/biolord/biolord/exp_spacseq/"
        "build_filtered_raw_counts.py"
    )
    log.info("Building raw counts for slide %s -> %s", slide_id, outdir)
    subprocess.run(
        [
            CONCERT_PYTHON,
            "-u",
            build_script,
            "--filtered_h5ad",
            str(prepared_h5ad),
            "--raw_h5ad",
            str(raw_h5ad),
            "--outdir",
            str(outdir),
        ],
        check=True,
    )
    return outdir


def load_prepared_raw_resources(prepared_dir: Path, slide_id: str):
    _, sa = import_eval_utils()
    rc_dir = ensure_prepared_raw_counts(prepared_dir, slide_id)
    raw_mm = sa.open_raw_counts(raw_counts_dir=rc_dir)
    lib_sizes = sa.load_lib_sizes(raw_counts_dir=rc_dir)
    return raw_mm, lib_sizes


def save_multi_seen_results(
    out_dir: Path,
    slide_id: str,
    panel: list[str],
    scores,
    agg,
    mlfc,
    hvg_info,
    method: str,
    extra: dict | None = None,
    subdir: str = "seen",
    eval_type: str = "multi_seen_top_panel",
):
    dest = out_dir / subdir / f"slide_{slide_id}"
    meta = {
        "slide_id": slide_id,
        "eval_type": eval_type,
        "method": method,
        "panel": panel,
    }
    if extra:
        meta.update(extra)
    save_results(dest, scores, agg, mlfc, hvg_info)
    (dest / "eval_meta.json").write_text(json.dumps(jsonify(meta), indent=2))

