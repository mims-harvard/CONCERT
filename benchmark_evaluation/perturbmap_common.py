"""Perturb-map (4-slide) benchmark: data prep + counterfactual evaluation.

Adapts the SPAC-seq multi-slide protocol to the perturb-map dataset. Key
differences vs ``spacseq_multi_common``:

* Source cells for the in-silico KO are the **control (KP-tumor)** patch cells
  (``role == 1``), not unassigned cells.
* The LFC baseline / reference is **control (KP-tumor)**: predicted LFC is
  ``pseudobulk_lfc(pred_on_source, source_expr)`` and observed LFC is
  ``pseudobulk_lfc(real_KO_tumor, control_KP)`` -- i.e. the clean KO-vs-control
  tumor effect (control is abundant here, unlike SPAC-seq NTC).
* Niches (``bin_id``) and the contiguous KP-tumor source patches are taken
  directly from ``perturb_map/patches/slide{j}.npz`` (4x4 grid). Per prediction
  niche there are up to N_REP REPLICATE KP patches (distinct ``patch_id``); each
  is scored as an independent record. Holdout source patches (role 1) are
  excluded from training via ``holdout_patch/``. Normal tissue is NOT sampled.
* Scoring uses all 1053 targeted genes (no HVG subsample).

Roles in the patch npz: ``0 = context``, ``1 = KP-tumor source patch``.
Each source cell carries ``patch_id`` (replicate id; -1 for context) and
``target_bin`` / ``target_ko`` = the niche + KO it is meant to predict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

import spacseq_eval_common as sec
from method_input_lib import normalize_pred_cache
from spacseq_eval_common import to_dense

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PM_H5 = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/perturb_map_data_all.h5"
)
# Whole-transcriptome build (GSE193460) row-aligned to PM_H5; contains the KO
# target genes plus HVG/KO flags in .var (built by pm_build_fullgene.py).
PM_FULL_H5AD = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/perturb_map_data_all_fullgene.h5ad"
)
PM_PATCH_DIR = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/patches"
)
DEFAULT_PREPARED = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/prepared_panel1053ko"
)

DEFAULT_SLIDES = ["0", "1", "2", "3"]
LOC_RANGE = 20.0
KO_GENES = ("Tgfbr2", "Ifngr2", "Jak2")
CTRL_CANON = "control"          # KP-tumor (unperturbed tumor genotype)
UNA_CANON = "unassigned"        # None (normal-dominated background)
NON_KO = frozenset({CTRL_CANON, UNA_CANON, "excluded"})
MIN_CELLS_PER_PERT = 15         # min combined KO cells to keep a perturbation


# --------------------------------------------------------------------------- #
# Data preparation: h5 + patch npz  ->  prepared/slide_{j}.h5ad (+ raw cache)
# --------------------------------------------------------------------------- #
def _load_patch_npz(slide: str) -> dict:
    d = np.load(PM_PATCH_DIR / f"slide{slide}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def prepare_all_slides(outdir: Path, slides: list[str] | None = None) -> Path:
    import h5py
    import scanpy as sc
    slides = slides or DEFAULT_SLIDES
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(PM_H5, "r") as f:
        gene_names = np.array([g.decode() if isinstance(g, bytes) else str(g)
                               for g in f["gene"][:]])
        Xall = f["X"][:]  # (6363, 1053) raw counts

    for sid in slides:
        d = _load_patch_npz(sid)
        ci = d["cell_index"].astype(int)
        Xraw = np.asarray(Xall[ci], dtype=np.float32)
        n = Xraw.shape[0]

        obs = {
            "perturbation_canon": d["perturbation_canon"].astype(str),
            "perturbation": d["perturbation"].astype(str),
            "tissue": d["tissue"].astype(str),
            "bin_id": d["bin_id"].astype(int),
            "role": d["role"].astype(int),
            "patch_id": (d["patch_id"].astype(int) if "patch_id" in d
                         else np.full(d["role"].shape, -1, dtype=int)),
            "target_bin": d["target_bin"].astype(int),
            "target_ko": d["target_ko"].astype(str),
            "sample_id": np.array([sid] * n),
            "n_counts": Xraw.sum(axis=1).astype(np.float64),
        }
        obs_names = d["obs_name"].astype(str)

        coords = np.stack([d["pos_x"].astype(np.float64),
                           d["pos_y"].astype(np.float64)], axis=1)
        cmin = coords.min(axis=0)
        cspan = np.clip(coords.max(axis=0) - cmin, 1e-9, None)
        coords_scaled = ((coords - cmin) / cspan * LOC_RANGE).astype(np.float32)

        adata = ad.AnnData(
            X=sp.csr_matrix(Xraw),
            obs={k: v for k, v in obs.items()},
        )
        adata.obs_names = obs_names
        adata.var_names = gene_names
        adata.obsm["spatial"] = coords_scaled
        # X -> log-CP10K (model expression scale), keep raw in a layer.
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.uns = {}
        dest = outdir / f"slide_{sid}.h5ad"
        adata.write_h5ad(dest)
        log.info("slide %s -> %s  (%d cells)", sid, dest, n)

        # Raw-count cache (memmap-compatible) for energy distance.
        rc_dir = outdir / "raw_counts" / f"slide_{sid}"
        rc_dir.mkdir(parents=True, exist_ok=True)
        np.save(rc_dir / "filtered_raw_counts.npy", Xraw)
        np.save(rc_dir / "filtered_lib_sizes.npy", Xraw.sum(axis=1).astype(np.float64))
        (rc_dir / "raw_counts_meta.json").write_text(json.dumps(
            {"shape": list(Xraw.shape), "dtype": "float32"}, indent=2))

    (outdir / "prepared_meta.json").write_text(json.dumps({
        "slides": slides,
        "n_genes": int(len(gene_names)),
        "gene_names": gene_names.tolist(),
        "loc_range": LOC_RANGE,
        "ctrl_canon": CTRL_CANON,
        "una_canon": UNA_CANON,
        "ko_genes": list(KO_GENES),
    }, indent=2))
    log.info("prepared_meta.json written; %d genes", len(gene_names))
    return outdir


def prepare_all_slides_fullgene(outdir: Path, slides: list[str] | None = None) -> Path:
    """Like prepare_all_slides but from the whole-transcriptome GEO build.

    Uses the ``use_for_analysis`` (HVG + KO) gene subset so the KO target genes
    are present in the panel (required for graph/LR methods), while scoring still
    happens on the HVG genes only (KO genes carry a ``ko_gene`` flag in .var).
    Cells/labels/niches come from the same patch npz files via the global
    ``cell_index`` (verified row-aligned to the full-gene AnnData).
    """
    import scanpy as sc
    slides = slides or DEFAULT_SLIDES
    outdir.mkdir(parents=True, exist_ok=True)

    full = ad.read_h5ad(PM_FULL_H5AD)
    use = full.var["use_for_analysis"].to_numpy().astype(bool)
    gene_names = np.asarray(full.var_names)[use]
    ko_sub = full.var["ko_gene"].to_numpy().astype(bool)[use]
    hvg_sub = full.var["highly_variable"].to_numpy().astype(bool)[use]
    counts = full.layers["counts"] if "counts" in full.layers else full.X
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    counts = counts[:, use]

    for sid in slides:
        d = _load_patch_npz(sid)
        ci = d["cell_index"].astype(int)
        Xraw = np.asarray(counts[ci].todense(), dtype=np.float32)
        n = Xraw.shape[0]

        obs = {
            "perturbation_canon": d["perturbation_canon"].astype(str),
            "perturbation": d["perturbation"].astype(str),
            "tissue": d["tissue"].astype(str),
            "bin_id": d["bin_id"].astype(int),
            "role": d["role"].astype(int),
            "patch_id": (d["patch_id"].astype(int) if "patch_id" in d
                         else np.full(d["role"].shape, -1, dtype=int)),
            "target_bin": d["target_bin"].astype(int),
            "target_ko": d["target_ko"].astype(str),
            "sample_id": np.array([sid] * n),
            "n_counts": Xraw.sum(axis=1).astype(np.float64),
        }
        obs_names = d["obs_name"].astype(str)
        coords = np.stack([d["pos_x"].astype(np.float64),
                           d["pos_y"].astype(np.float64)], axis=1)
        cmin = coords.min(axis=0)
        cspan = np.clip(coords.max(axis=0) - cmin, 1e-9, None)
        coords_scaled = ((coords - cmin) / cspan * LOC_RANGE).astype(np.float32)

        adata = ad.AnnData(X=sp.csr_matrix(Xraw), obs={k: v for k, v in obs.items()})
        adata.obs_names = obs_names
        adata.var_names = gene_names
        adata.var["ko_gene"] = ko_sub
        adata.var["highly_variable"] = hvg_sub
        adata.obsm["spatial"] = coords_scaled
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.uns = {}
        dest = outdir / f"slide_{sid}.h5ad"
        adata.write_h5ad(dest)
        log.info("slide %s -> %s  (%d cells, %d genes)", sid, dest, n, len(gene_names))

        rc_dir = outdir / "raw_counts" / f"slide_{sid}"
        rc_dir.mkdir(parents=True, exist_ok=True)
        np.save(rc_dir / "filtered_raw_counts.npy", Xraw)
        np.save(rc_dir / "filtered_lib_sizes.npy", Xraw.sum(axis=1).astype(np.float64))
        (rc_dir / "raw_counts_meta.json").write_text(json.dumps(
            {"shape": list(Xraw.shape), "dtype": "float32"}, indent=2))

    (outdir / "prepared_meta.json").write_text(json.dumps({
        "slides": slides,
        "source": str(PM_FULL_H5AD),
        "n_genes": int(len(gene_names)),
        "gene_names": gene_names.tolist(),
        "ko_genes": list(KO_GENES),
        "n_hvg_score_genes": int((hvg_sub & ~ko_sub).sum()),
        "loc_range": LOC_RANGE,
        "ctrl_canon": CTRL_CANON,
        "una_canon": UNA_CANON,
    }, indent=2))
    log.info("prepared_meta.json written; %d genes (%d HVG score genes + %d KO)",
             len(gene_names), int((hvg_sub & ~ko_sub).sum()), int(ko_sub.sum()))
    return outdir


def prepare_all_slides_panel1053ko(outdir: Path, slides: list[str] | None = None) -> Path:
    """GEO whole-transcriptome counts with the curated 1053-gene panel + KO targets.

    Same cell/niche alignment as ``prepare_all_slides_fullgene``, but the gene
    panel is the original curated 1053 genes (``in_panel_1053``) plus
    Tgfbr2/Ifngr2/Jak2 so CelComen can knock them out while LFC scoring stays
    on the 1053 panel genes (KO genes flagged via ``ko_gene`` in .var).
    """
    import scanpy as sc
    slides = slides or DEFAULT_SLIDES
    outdir.mkdir(parents=True, exist_ok=True)

    full = ad.read_h5ad(PM_FULL_H5AD)
    in1053 = full.var["in_panel_1053"].to_numpy().astype(bool)
    is_ko = full.var["ko_gene"].to_numpy().astype(bool)
    use = in1053 | is_ko
    gene_names = np.asarray(full.var_names)[use]
    ko_sub = is_ko[use]
    score_sub = in1053[use] & ~ko_sub
    counts = full.layers["counts"] if "counts" in full.layers else full.X
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    counts = counts[:, use]

    for sid in slides:
        d = _load_patch_npz(sid)
        ci = d["cell_index"].astype(int)
        Xraw = np.asarray(counts[ci].todense(), dtype=np.float32)
        n = Xraw.shape[0]

        obs = {
            "perturbation_canon": d["perturbation_canon"].astype(str),
            "perturbation": d["perturbation"].astype(str),
            "tissue": d["tissue"].astype(str),
            "bin_id": d["bin_id"].astype(int),
            "role": d["role"].astype(int),
            "patch_id": (d["patch_id"].astype(int) if "patch_id" in d
                         else np.full(d["role"].shape, -1, dtype=int)),
            "target_bin": d["target_bin"].astype(int),
            "target_ko": d["target_ko"].astype(str),
            "sample_id": np.array([sid] * n),
            "n_counts": Xraw.sum(axis=1).astype(np.float64),
        }
        obs_names = d["obs_name"].astype(str)
        coords = np.stack([d["pos_x"].astype(np.float64),
                           d["pos_y"].astype(np.float64)], axis=1)
        cmin = coords.min(axis=0)
        cspan = np.clip(coords.max(axis=0) - cmin, 1e-9, None)
        coords_scaled = ((coords - cmin) / cspan * LOC_RANGE).astype(np.float32)

        adata = ad.AnnData(X=sp.csr_matrix(Xraw), obs={k: v for k, v in obs.items()})
        adata.obs_names = obs_names
        adata.var_names = gene_names
        adata.var["ko_gene"] = ko_sub
        adata.var["in_panel_1053"] = score_sub | ko_sub  # panel membership
        adata.var["score_gene"] = score_sub             # genes used for LFC/e-dist
        adata.obsm["spatial"] = coords_scaled
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.uns = {}
        dest = outdir / f"slide_{sid}.h5ad"
        adata.write_h5ad(dest)
        log.info("slide %s -> %s  (%d cells, %d genes)", sid, dest, n, len(gene_names))

        rc_dir = outdir / "raw_counts" / f"slide_{sid}"
        rc_dir.mkdir(parents=True, exist_ok=True)
        np.save(rc_dir / "filtered_raw_counts.npy", Xraw)
        np.save(rc_dir / "filtered_lib_sizes.npy", Xraw.sum(axis=1).astype(np.float64))
        (rc_dir / "raw_counts_meta.json").write_text(json.dumps(
            {"shape": list(Xraw.shape), "dtype": "float32"}, indent=2))

    (outdir / "prepared_meta.json").write_text(json.dumps({
        "slides": slides,
        "source": str(PM_FULL_H5AD),
        "panel": "curated_1053_plus_ko",
        "n_genes": int(len(gene_names)),
        "n_score_genes": int(score_sub.sum()),
        "gene_names": gene_names.tolist(),
        "ko_genes": list(KO_GENES),
        "loc_range": LOC_RANGE,
        "ctrl_canon": CTRL_CANON,
        "una_canon": UNA_CANON,
    }, indent=2))
    log.info("prepared_meta.json written; %d genes (%d score + %d KO)",
             len(gene_names), int(score_sub.sum()), int(ko_sub.sum()))
    return outdir


def load_prepared_slide(prepared_dir: Path, slide_id: str) -> ad.AnnData:
    return ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")


def load_prepared_raw_resources(prepared_dir: Path, slide_id: str):
    rc = prepared_dir / "raw_counts" / f"slide_{slide_id}"
    meta = json.loads((rc / "raw_counts_meta.json").read_text())
    raw_mm = np.load(rc / "filtered_raw_counts.npy", mmap_mode="r")
    lib_sizes = np.load(rc / "filtered_lib_sizes.npy")
    assert list(raw_mm.shape) == meta["shape"]
    return raw_mm, lib_sizes


def concat_prepared_slides(prepared_dir: Path, slides: list[str] | None = None) -> ad.AnnData:
    slides = slides or DEFAULT_SLIDES
    parts = []
    for sid in slides:
        a = load_prepared_slide(prepared_dir, sid)
        parts.append(a)
    return ad.concat(parts, join="inner", index_unique=None)


def load_donor_concat(prepared_dir: Path, target_slide: str,
                      slides: list[str] | None = None) -> ad.AnnData:
    """Concat all slides except the target (for cross-slide KO substitution)."""
    slides = slides or DEFAULT_SLIDES
    donors = [s for s in slides if s != target_slide]
    return ad.concat([load_prepared_slide(prepared_dir, s) for s in donors],
                     join="inner", index_unique=None)


# --------------------------------------------------------------------------- #
# Present/absent KO sets + cross-slide manifest (unseen pert)
# --------------------------------------------------------------------------- #
def slide_ko_counts(prepared_dir: Path, slides: list[str] | None = None) -> dict[str, dict[str, int]]:
    """{slide: {ko_gene: n_real_KO_tumor_cells}} (only KO genes)."""
    slides = slides or DEFAULT_SLIDES
    out = {}
    for sid in slides:
        a = load_prepared_slide(prepared_dir, sid)
        canon = a.obs["perturbation_canon"].astype(str).to_numpy()
        counts = {}
        for g in KO_GENES:
            n = int((canon == g).sum())
            if n > 0:
                counts[g] = n
        out[sid] = counts
    return out


def build_cross_slide_manifest(prepared_dir: Path, slides: list[str] | None = None,
                               min_donor_ko_cells: int = MIN_CELLS_PER_PERT) -> dict:
    slides = slides or DEFAULT_SLIDES
    counts = slide_ko_counts(prepared_dir, slides)
    per_slide = {}
    for sid in slides:
        present = set(counts[sid])
        donor_counts = {}
        for g in KO_GENES:
            if g in present:
                continue
            tot = sum(counts[o].get(g, 0) for o in slides if o != sid)
            if tot >= min_donor_ko_cells:
                donor_counts[g] = tot
        eval_genes = sorted(donor_counts, key=lambda g: -donor_counts[g])
        per_slide[sid] = {"eval_genes": eval_genes, "donor_ko_counts": donor_counts,
                          "present": sorted(present)}
    manifest = {
        "slides": slides,
        "min_donor_ko_cells": min_donor_ko_cells,
        "selection_rule": "KO present on >=1 donor slide and absent on target",
        "slide_ko_counts": counts,
        "per_slide": per_slide,
    }
    (prepared_dir / "cross_slide_eval_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("cross-slide manifest: %s", {s: per_slide[s]["eval_genes"] for s in slides})
    return manifest


def load_manifest_genes(prepared_dir: Path, slide_id: str) -> list[str]:
    m = json.loads((prepared_dir / "cross_slide_eval_manifest.json").read_text())
    return m["per_slide"][slide_id]["eval_genes"]


# --------------------------------------------------------------------------- #
# Source / reference helpers
# --------------------------------------------------------------------------- #
def source_indices(adata: ad.AnnData, src_role: int = 1) -> np.ndarray:
    """Global indices of the source patch cells (1 = KP-tumor)."""
    role = adata.obs["role"].to_numpy().astype(int)
    return np.where(role == src_role)[0]


def seen_niche_groups(adata: ad.AnnData, src_role: int = 1) -> dict:
    """{(target_bin, target_ko, patch_id): local-rows within source_indices}.

    Each replicate KP patch (distinct ``patch_id``) becomes its own group so it
    is scored as an independent record (more samples for niche-level error bars).
    Falls back to a single group per (bin, KO) when ``patch_id`` is absent.
    """
    role = adata.obs["role"].to_numpy().astype(int)
    tb = adata.obs["target_bin"].to_numpy().astype(int)
    tk = adata.obs["target_ko"].to_numpy().astype(str)
    if "patch_id" in adata.obs:
        pid = adata.obs["patch_id"].to_numpy().astype(int)
    else:
        pid = np.full(role.shape, -1, dtype=int)
    gsi = np.where(role == src_role)[0]
    groups: dict[tuple[int, str, int], list[int]] = {}
    for local, gi in enumerate(gsi):
        gene = str(tk[gi])
        if gene not in KO_GENES:
            continue
        key = (int(tb[gi]), gene, int(pid[gi]))
        groups.setdefault(key, []).append(local)
    return {k: np.asarray(v, dtype=int) for k, v in groups.items()}


def _ref_ko_idx(adata, canon, bin_id, gene, b, min_g_cells):
    """Real KO-tumor cell indices: niche-local if >= min_g_cells else slide-level."""
    local = np.where((canon == gene) & (bin_id == b))[0]
    if len(local) >= min_g_cells:
        return local, "niche"
    return np.where(canon == gene)[0], "slide"


def _ref_ctrl_idx(adata, canon, bin_id, b, min_g_cells):
    """Control (KP-tumor) baseline indices: niche-local if enough else slide-level."""
    local = np.where((canon == CTRL_CANON) & (bin_id == b))[0]
    if len(local) >= min_g_cells:
        return local, "niche"
    return np.where(canon == CTRL_CANON)[0], "slide"


# --------------------------------------------------------------------------- #
# Evaluation loops
# --------------------------------------------------------------------------- #
def niche_lfc_seen_loop(adata_target, raw_mm, lib_sizes, global_src_idx,
                        groups, pred_g_cache, hvg_mask, args,
                        unpredictable_genes=None, scoring_method: str | None = None,
                        pred_basal=None, ko_gt_rows=None):
    """Seen pert: per (niche, dominant KO) record.

    predicted LFC = pseudobulk_lfc(pred_on_source, reference)
    observed  LFC = pseudobulk_lfc(real_KO_tumor, control_KP)

    If ``pred_basal`` (predicted basal on the source cells, aligned with
    global_src_idx) is given, reference = pred_basal -> DELTA pred_KO - pred_basal
    (cancels basal reconstruction error) instead of pred_KO - observed source.
    """
    eu, sa = sec.import_eval_utils()
    unpredictable_genes = set(unpredictable_genes or [])
    if scoring_method:
        pred_g_cache = normalize_pred_cache(scoring_method, pred_g_cache)
        if pred_basal is not None:
            pred_basal = normalize_pred_cache(scoring_method, {"_b": pred_basal})["_b"]
    gene_names = np.asarray(adata_target.var_names)
    canon = adata_target.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata_target.obs["bin_id"].to_numpy().astype(int)
    src_expr = to_dense(adata_target[global_src_idx].X)
    ref_expr = pred_basal if pred_basal is not None else src_expr
    src_bin_local = bin_id[global_src_idx]
    random_patch = getattr(args, "random_patch", False)
    n_patch_boot, patch_seed = sec.patch_boot_settings(args)

    panel = sorted({g for (_, g, _) in groups})
    scores = {"L2": {g: [] for g in panel}}
    agg = {"L2": {}}
    mlfc = {"L2": {g: {} for g in panel}}

    pending_worst = []   # evaluable niches the method could not predict
    finite_edists = []
    for (b, g, rep), rows in sorted(groups.items()):
        ko_idx, _ko_scope = _ref_ko_idx(adata_target, canon, bin_id, g, b, args.min_g_cells)
        if ko_gt_rows is not None:
            ko_idx = ko_idx[np.isin(ko_idx, ko_gt_rows)]
        if len(ko_idx) < args.min_g_cells:
            continue
        ctrl_idx, _ = _ref_ctrl_idx(adata_target, canon, bin_id, b, args.min_g_cells)
        if len(ctrl_idx) < args.min_g_cells:
            continue
        keep_n = int((hvg_mask & (gene_names != g)).sum())

        pred_g = pred_g_cache.get(g)
        if g in unpredictable_genes or pred_g is None:
            pending_worst.append((int(b), g, len(rows), len(ko_idx), keep_n))
            continue

        o_lfc = eu.pseudobulk_lfc(to_dense(adata_target[ko_idx].X),
                                  to_dense(adata_target[ctrl_idx].X))
        keep = hvg_mask & (gene_names != g)
        n_boot, frac, boot_seed = sec.boot_settings(args)

        if random_patch and n_patch_boot > 0:
            patch_list = sec.holdout_patch_groups_pm(groups, b, args.min_g_cells)
            if not patch_list:
                pending_worst.append((int(b), g, len(rows), len(ko_idx), keep_n))
                continue
            patch_stats = sec.score_with_patch_boot(
                pred_g, ref_expr, o_lfc, np.asarray(raw_mm[ko_idx]),
                lib_sizes[global_src_idx], keep, eu, sa,
                pick_rows=lambda rng, _pl=patch_list: sec.pick_random_holdout_patch(_pl, rng),
                n_patch_boot=n_patch_boot,
                seed=patch_seed + int(b) * 1009 + int(rep) * 17 + hash(g) % 997,
            )
            if patch_stats is None:
                pending_worst.append((int(b), g, len(rows), len(ko_idx), keep_n))
                continue
            rows0 = sec.pick_random_holdout_patch(
                patch_list, np.random.default_rng(patch_seed + int(b) + int(rep)),
            )
            pred_b = pred_g[rows0]
            m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pred_b, ref_expr[rows0]),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
            sec.attach_patch_boot_stats(sc_, patch_stats)
        else:
            pred_b = pred_g[rows]
            m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pred_b, ref_expr[rows]),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
            sc_["e_dist"] = sa.aligned_energy_distance(
                pred_b, np.asarray(raw_mm[ko_idx]),
                lib_sizes[global_src_idx[rows]], keep,
            )
            if n_boot > 0:
                sec.attach_bootstrap_stats(sc_, sec.bootstrap_source_metrics(
                    pred_b, ref_expr[rows], o_lfc, np.asarray(raw_mm[ko_idx]),
                    lib_sizes[global_src_idx[rows]], keep, eu, sa,
                    n_boot=n_boot, frac=frac,
                    seed=boot_seed + int(b) * 1009 + int(rep) * 17 + hash(g) % 997,
                ))
        sc_.update(niche=int(b), patch_id=int(rep), n_src=int(len(rows)),
                   n_real_g=int(len(ko_idx)), n_scored_genes=int(keep.sum()),
                   random_patch=bool(random_patch and n_patch_boot > 0))
        if not (random_patch and n_patch_boot > 0):
            if np.isfinite(sc_.get("e_dist", np.nan)):
                finite_edists.append(sc_["e_dist"])
        else:
            if np.isfinite(sc_.get("e_dist", np.nan)):
                finite_edists.append(sc_["e_dist"])
        scores["L2"][g].append(sc_)
        mlfc["L2"][g][f"{int(b)}:{int(rep)}"] = m_lfc.tolist()

    # Penalise evaluable niches a method could not predict (worst-case), so
    # unpredictable methods score poorly instead of disappearing (SPAC-seq rule).
    bad_edist = float(max(finite_edists)) if finite_edists else 12.0
    for (b, g, n_src, n_real, keep_n) in pending_worst:
        scores["L2"][g].append(sec.worst_niche_score(b, n_src, n_real, keep_n, bad_edist))

    for g in panel:
        agg["L2"][g] = sec.aggregate_scores(scores, "L2", g,
                                            {"sign_agreement": 0.5, "topk_overlap": 0.0})
    return scores, agg, mlfc


def niche_lfc_unseen_loop(adata_target, prepared_dir, slide_id, global_src_idx,
                          eval_genes, pred_g_cache, hvg_mask, args,
                          slides=None, unpredictable_genes=None,
                          scoring_method: str | None = None):
    """Unseen pert: one record per absent KO gene, donor-pooled reference.

    predicted LFC = pseudobulk_lfc(pred_on_source, source_expr)   (target KP source)
    observed  LFC = pseudobulk_lfc(donor real KO, donor control)  (pooled donors)
    e-dist = pred (target source) vs donor real-KO raw counts.
    """
    eu, sa = sec.import_eval_utils()
    slides = slides or DEFAULT_SLIDES
    unpredictable_genes = set(unpredictable_genes or [])
    if scoring_method:
        pred_g_cache = normalize_pred_cache(scoring_method, pred_g_cache)
    gene_names = np.asarray(adata_target.var_names)
    src_expr = to_dense(adata_target[global_src_idx].X)
    ref_expr = src_expr  # prepared-dir loop: reference is always observed source

    scores = {"L2": {g: [] for g in eval_genes}}
    agg = {"L2": {}}
    mlfc = {"L2": {g: {} for g in eval_genes}}

    # Donor pools (log-CP10K for LFC; raw counts for e-dist).
    donor_ko_log: dict[str, list] = {g: [] for g in eval_genes}
    donor_ko_raw: dict[str, list] = {g: [] for g in eval_genes}
    donor_ctrl_log: list = []
    for sid in slides:
        if sid == slide_id:
            continue
        a = load_prepared_slide(prepared_dir, sid)
        canon = a.obs["perturbation_canon"].astype(str).to_numpy()
        raw_mm, _ = load_prepared_raw_resources(prepared_dir, sid)
        ci = np.where(canon == CTRL_CANON)[0]
        if len(ci):
            donor_ctrl_log.append(to_dense(a[ci].X))
        for g in eval_genes:
            gi = np.where(canon == g)[0]
            if len(gi):
                donor_ko_log[g].append(to_dense(a[gi].X))
                donor_ko_raw[g].append(np.asarray(raw_mm[gi]))
    donor_ctrl = np.vstack(donor_ctrl_log) if donor_ctrl_log else None

    pending_worst = []
    finite_edists = []
    for g in eval_genes:
        if not donor_ko_log[g] or donor_ctrl is None:
            continue  # no donor reference -> not evaluable
        ko_raw = np.vstack(donor_ko_raw[g])
        keep = hvg_mask & (gene_names != g)
        keep_n = int(keep.sum())
        pred_g = pred_g_cache.get(g)
        if g in unpredictable_genes or pred_g is None:
            pending_worst.append((g, int(len(global_src_idx)), int(ko_raw.shape[0]), keep_n))
            continue
        ko_log = np.vstack(donor_ko_log[g])
        o_lfc = eu.pseudobulk_lfc(ko_log, donor_ctrl)
        n_boot, frac, boot_seed = sec.boot_settings(args)
        m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pred_g, ref_expr),
                              nan=0.0, posinf=0.0, neginf=0.0)
        sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
        sc_["e_dist"] = sa.aligned_energy_distance(
            pred_g, ko_raw, lib_sizes_from(adata_target, global_src_idx), keep,
        )
        sc_.update(niche=-1, n_src=int(len(global_src_idx)), n_real_g=int(ko_raw.shape[0]),
                   n_scored_genes=keep_n)
        if n_boot > 0:
            sec.attach_bootstrap_stats(sc_, sec.bootstrap_source_metrics(
                pred_g, ref_expr, o_lfc, ko_raw,
                lib_sizes_from(adata_target, global_src_idx), keep, eu, sa,
                n_boot=n_boot, frac=frac, seed=boot_seed + hash(g) % 997,
            ))
        if np.isfinite(sc_.get("e_dist", np.nan)):
            finite_edists.append(sc_["e_dist"])
        scores["L2"][g].append(sc_)
        mlfc["L2"][g][-1] = m_lfc.tolist()

    bad_edist = float(max(finite_edists)) if finite_edists else 12.0
    for (g, n_src, n_real, keep_n) in pending_worst:
        scores["L2"][g].append(sec.worst_niche_score(-1, n_src, n_real, keep_n, bad_edist))

    for g in eval_genes:
        agg["L2"][g] = sec.aggregate_scores(scores, "L2", g,
                                            {"sign_agreement": 0.5, "topk_overlap": 0.0})
    return scores, agg, mlfc


def niche_lfc_unseen_score(adata_target, global_src_idx, eval_genes, pred_g_cache,
                           hvg_mask, args, donor_ko_log, donor_ko_raw, donor_ctrl,
                           unpredictable_genes=None, scoring_method: str | None = None,
                           pred_basal=None):
    """Same scoring as niche_lfc_unseen_loop but donor pools are supplied explicitly
    (model-space donors), instead of loaded from prepared_dir. Used by the CONCERT
    adapter, which scores in MODEL-ROW space (like the seen adapter). donor_ko_log /
    donor_ko_raw are {gene: [arrays]}; donor_ctrl is a stacked (n_ctrl, G) array.

    If ``pred_basal`` (predicted basal on the source cells, aligned with global_src_idx)
    is given, reference = pred_basal -> DELTA pred_KO - pred_basal (cancels basal
    reconstruction error) instead of pred_KO - observed source."""
    eu, sa = sec.import_eval_utils()
    unpredictable_genes = set(unpredictable_genes or [])
    if scoring_method:
        pred_g_cache = normalize_pred_cache(scoring_method, pred_g_cache)
        if pred_basal is not None:
            pred_basal = normalize_pred_cache(scoring_method, {"_b": pred_basal})["_b"]
    gene_names = np.asarray(adata_target.var_names)
    src_expr = to_dense(adata_target[global_src_idx].X)
    ref_expr = pred_basal if pred_basal is not None else src_expr
    scores = {"L2": {g: [] for g in eval_genes}}
    agg = {"L2": {}}
    mlfc = {"L2": {g: {} for g in eval_genes}}
    pending_worst = []
    finite_edists = []
    for g in eval_genes:
        if not donor_ko_log.get(g) or donor_ctrl is None:
            continue  # no donor reference -> not evaluable
        ko_raw = np.vstack(donor_ko_raw[g])
        keep = hvg_mask & (gene_names != g)
        keep_n = int(keep.sum())
        pred_g = pred_g_cache.get(g)
        if g in unpredictable_genes or pred_g is None:
            pending_worst.append((g, int(len(global_src_idx)), int(ko_raw.shape[0]), keep_n))
            continue
        ko_log = np.vstack(donor_ko_log[g])
        o_lfc = eu.pseudobulk_lfc(ko_log, donor_ctrl)
        n_boot, frac, boot_seed = sec.boot_settings(args)
        m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pred_g, ref_expr),
                              nan=0.0, posinf=0.0, neginf=0.0)
        sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])
        sc_["e_dist"] = sa.aligned_energy_distance(
            pred_g, ko_raw, lib_sizes_from(adata_target, global_src_idx), keep,
        )
        sc_.update(niche=-1, n_src=int(len(global_src_idx)), n_real_g=int(ko_raw.shape[0]),
                   n_scored_genes=keep_n)
        if n_boot > 0:
            sec.attach_bootstrap_stats(sc_, sec.bootstrap_source_metrics(
                pred_g, ref_expr, o_lfc, ko_raw,
                lib_sizes_from(adata_target, global_src_idx), keep, eu, sa,
                n_boot=n_boot, frac=frac, seed=boot_seed + hash(g) % 997,
            ))
        if np.isfinite(sc_.get("e_dist", np.nan)):
            finite_edists.append(sc_["e_dist"])
        scores["L2"][g].append(sc_)
        mlfc["L2"][g][-1] = m_lfc.tolist()

    bad_edist = float(max(finite_edists)) if finite_edists else 12.0
    for (g, n_src, n_real, keep_n) in pending_worst:
        scores["L2"][g].append(sec.worst_niche_score(-1, n_src, n_real, keep_n, bad_edist))
    for g in eval_genes:
        agg["L2"][g] = sec.aggregate_scores(scores, "L2", g,
                                            {"sign_agreement": 0.5, "topk_overlap": 0.0})
    return scores, agg, mlfc


def lib_sizes_from(adata, idx) -> np.ndarray:
    if "n_counts" in adata.obs:
        return adata.obs["n_counts"].to_numpy()[idx].astype(np.float64)
    return np.asarray(to_dense(adata[idx].X).sum(axis=1), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Result writer
# --------------------------------------------------------------------------- #
def save_results(outdir: Path, slide_id: str, setting: str, panel, scores, agg, mlfc,
                 method: str, hvg_info=None, extra=None):
    out_dir = outdir / setting / f"slide_{slide_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sec.save_results(out_dir, scores, agg, mlfc, hvg_info or {})
    meta = {"method": method, "setting": setting, "slide": slide_id, "panel": list(panel)}
    if extra:
        meta.update(extra)
    (out_dir / "eval_meta.json").write_text(json.dumps(sec.jsonify(meta), indent=2))
    log.info("[%s] %s slide %s -> %s", method, setting, slide_id, out_dir)
