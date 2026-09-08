"""Seen v4 residual spatial scoring (inference-free rescoring from saved LFC vectors)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import anndata as ad
import h5py
import numpy as np
from scipy.stats import spearmanr

EPS = 1e-6
PM_DATA = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map")
CONCERT_SPACFMT = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets")


@dataclass
class ResidualScore:
    gene: str
    slide: str
    method: str
    n_niches: int
    n_genes_scored: int
    raw_lfc_rho: float
    niche_residual_rho: float
    cross_niche_pattern_rho: float
    global_pool_raw_rho: float
    global_pool_residual_rho: float = 0.0
    delta_residual_over_global_pool: float = field(init=False)

    def __post_init__(self) -> None:
        self.delta_residual_over_global_pool = (
            self.niche_residual_rho - self.global_pool_residual_rho
        )


def to_dense(x) -> np.ndarray:
    if hasattr(x, "todense"):
        return np.asarray(x.todense())
    return np.asarray(x)


def pseudobulk_lfc(pert: np.ndarray, ntc: np.ndarray, eps: float = EPS) -> np.ndarray:
    mp = np.asarray(pert, dtype=np.float64).mean(axis=0)
    mn = np.asarray(ntc, dtype=np.float64).mean(axis=0)
    return np.log(mp + eps) - np.log(mn + eps)


def score_lfc_agreement(model_lfc: np.ndarray, obs_lfc: np.ndarray,
                        n_boot: int = 200, seed: int = 0) -> dict:
    """Agreement between predicted and observed LFC, with a gene-bootstrap uncertainty.

    The bootstrap resamples the scored GENES with replacement and re-computes both
    statistics, giving the sampling uncertainty of this one evaluation. It is computed
    here, at the single chokepoint every method's rescoring passes through, so CONCERT,
    the competitors, the naive reference strategies and STATE all get the same quantity
    by construction — previously only CONCERT and the competitors carried a bootstrap
    term (from their own eval JSONs), so per-gene bar panels could not show a comparable
    error for the others. Deterministic: fixed seed, so reruns reproduce byte-for-byte.
    """
    nan_result = {"spearman": float("nan"), "lfc_dist": float("nan"),
                  "spearman_boot_std": float("nan"), "lfc_dist_boot_std": float("nan")}
    m = np.asarray(model_lfc, dtype=np.float64)
    o = np.asarray(obs_lfc, dtype=np.float64)
    if m.size == 0 or o.size == 0:
        return dict(nan_result)
    mask = np.isfinite(m) & np.isfinite(o)
    if mask.sum() == 0:
        return dict(nan_result)
    m, o = m[mask], o[mask]
    rho = spearmanr(m, o).correlation
    lfc_dist = float(np.sqrt(np.mean((m - o) ** 2)))

    rho_boot = float("nan")
    dist_boot = float("nan")
    n = m.size
    if n >= 3 and n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        mb, ob = m[idx], o[idx]
        # Spearman on each resample = Pearson on the within-resample ranks.
        rm = np.argsort(np.argsort(mb, axis=1), axis=1).astype(np.float64)
        ro = np.argsort(np.argsort(ob, axis=1), axis=1).astype(np.float64)
        rm -= rm.mean(axis=1, keepdims=True)
        ro -= ro.mean(axis=1, keepdims=True)
        denom = np.sqrt((rm ** 2).sum(axis=1) * (ro ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            rhos = (rm * ro).sum(axis=1) / denom
        rhos = rhos[np.isfinite(rhos)]
        if rhos.size > 1:
            rho_boot = float(np.std(rhos, ddof=1))
        dists = np.sqrt(np.mean((mb - ob) ** 2, axis=1))
        dists = dists[np.isfinite(dists)]
        if dists.size > 1:
            dist_boot = float(np.std(dists, ddof=1))

    return {
        "spearman": float(rho) if rho == rho else float("nan"),
        "lfc_dist": lfc_dist,
        "spearman_boot_std": rho_boot,
        "lfc_dist_boot_std": dist_boot,
    }


def residualize_across_niches(niche_lfc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """niche_lfc: (n_niche, n_gene). Returns (global_per_gene, residual)."""
    arr = np.asarray(niche_lfc, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.array([]), np.array([])
    global_lfc = arr.mean(axis=0)
    residual = arr - global_lfc
    return global_lfc, residual


def score_residual_flat(pred_resid: np.ndarray, obs_resid: np.ndarray) -> float:
    p = np.asarray(pred_resid, dtype=np.float64).ravel()
    o = np.asarray(obs_resid, dtype=np.float64).ravel()
    mask = np.isfinite(p) & np.isfinite(o)
    p, o = p[mask], o[mask]
    if p.size < 3 or p.std() < 1e-12 or o.std() < 1e-12:
        return 0.0
    rho = spearmanr(p, o).correlation
    return float(rho) if rho == rho else 0.0


def build_gene_mask(gene_names: np.ndarray, ko_gene: str, hvg_indices: np.ndarray | None = None) -> np.ndarray:
    keep = np.ones(len(gene_names), dtype=bool)
    if ko_gene in gene_names:
        keep[list(gene_names).index(ko_gene)] = False
    if hvg_indices is not None and len(hvg_indices):
        hvg_mask = np.zeros(len(gene_names), dtype=bool)
        idx = np.asarray(hvg_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(gene_names))]
        if idx.size == 0:
            return keep
        hvg_mask[idx] = True
        keep &= hvg_mask
    return keep


def load_concert_hvg_names(slide_id: str, hvg_max_rank: int = 2000) -> np.ndarray:
    spacfmt = CONCERT_SPACFMT / f"perturbmap_s{slide_id}_spacfmt.h5ad"
    with h5py.File(str(spacfmt), "r") as vf:
        names = vf["var"]["_index"][:].astype(str)
        hvg = vf["var"]["highly_variable"][:].astype(bool)
        rank = vf["var"]["hvg_rank"][:]
    hvg_names = names[hvg]
    hvg_ranks = rank[hvg]
    order = np.argsort(hvg_ranks)
    hvg_names = hvg_names[order]
    if hvg_max_rank > 0:
        hvg_names = hvg_names[:hvg_max_rank]
    return np.asarray(hvg_names)


def load_concert_panel_mask(slide_id: str, concert_gene_names: np.ndarray) -> np.ndarray:
    spacfmt = CONCERT_SPACFMT / f"perturbmap_s{slide_id}_spacfmt.h5ad"
    with h5py.File(str(spacfmt), "r") as vf:
        all_names = vf["var"]["_index"][:].astype(str)
        in_panel = vf["var"]["in_panel_1053"][:].astype(bool)
    panel_set = set(all_names[in_panel])
    return np.array([g in panel_set for g in concert_gene_names], dtype=bool)


def observed_lfc_on_gene_axis(
    adata: ad.AnnData,
    gene: str,
    niche_bin: int,
    target_gene_names: np.ndarray,
    min_g_cells: int,
) -> np.ndarray | None:
    """Observed niche LFC mapped onto an arbitrary gene-name axis."""
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata.obs["bin_id"].to_numpy().astype(int)
    prep_names = np.asarray(adata.var_names).astype(str)
    ko = np.where((canon == gene) & (bin_id == niche_bin))[0]
    if len(ko) < min_g_cells:
        ko = np.where(canon == gene)[0]
    ctrl = np.where((canon == "control") & (bin_id == niche_bin))[0]
    if len(ko) < min_g_cells or len(ctrl) < min_g_cells:
        return None
    obs_full = pseudobulk_lfc(to_dense(adata[ko].X), to_dense(adata[ctrl].X))
    pmap = {g: i for i, g in enumerate(prep_names)}
    out = np.full(len(target_gene_names), np.nan, dtype=np.float64)
    for i, g in enumerate(target_gene_names):
        pi = pmap.get(str(g))
        if pi is not None:
            out[i] = obs_full[pi]
    return out


def observed_lfc_seen_patch_niche(
    adata: ad.AnnData,
    gene: str,
    niche_bin: int,
    prepared_dir: Path,
    slide_id: str,
    min_g_cells: int,
    *,
    exclude_patch_holdout: bool = True,
    target_gene_names: np.ndarray | None = None,
) -> np.ndarray | None:
    """SPAC-seq seen patch-holdout GT: dominant KO vs unassigned in niche."""
    from spacseq_eval_common import UNASSIGNED_LABELS
    from spacseq_multi_common import load_patch_roles

    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata.obs["bin_id"].to_numpy().astype(int)
    prep_names = np.asarray(adata.var_names).astype(str)
    una_mask = np.isin(canon, list(UNASSIGNED_LABELS | {"unassigned"}))

    ko_idx = np.where((canon == gene) & (bin_id == niche_bin))[0]
    una_b = np.where(una_mask & (bin_id == niche_bin))[0]

    if exclude_patch_holdout:
        role = load_patch_roles(prepared_dir, slide_id, adata)
        if role is not None:
            exclude = set(np.where(np.isin(role, (1, 2)) & una_mask)[0].tolist())
            if exclude:
                una_b = np.array([i for i in una_b if i not in exclude], dtype=int)

    if len(ko_idx) < min_g_cells or len(una_b) < min_g_cells:
        return None

    obs_full = pseudobulk_lfc(to_dense(adata[ko_idx].X), to_dense(adata[una_b].X))
    if target_gene_names is None:
        return obs_full
    out = np.full(len(target_gene_names), np.nan, dtype=np.float64)
    pmap = {g: i for i, g in enumerate(prep_names)}
    for i, g in enumerate(target_gene_names):
        pi = pmap.get(str(g))
        if pi is not None:
            out[i] = obs_full[pi]
    return out


def observed_lfc_unseen_v3(
    adata: ad.AnnData,
    ko_idx: np.ndarray,
    min_g_cells: int,
) -> np.ndarray | None:
    """V3 unseen GT: masked G-KO @ target slide vs unassigned @ target slide."""
    from spacseq_eval_common import UNASSIGNED_LABELS

    pert = adata.obs["perturbation_canon"].astype(str).to_numpy()
    una_idx = np.where(np.isin(pert, list(UNASSIGNED_LABELS | {"unassigned"})))[0]
    ko_idx = np.asarray(ko_idx, dtype=int)
    if len(ko_idx) < min_g_cells or len(una_idx) < min_g_cells:
        return None
    return pseudobulk_lfc(to_dense(adata[ko_idx].X), to_dense(adata[una_idx].X))


def load_concert_gene_names_from_checkpoint(result_dir: Path, vec_len: int) -> np.ndarray | None:
    """Load model-row gene names (2000 HVG or full 2260 niche panel)."""
    candidates = []
    for meta in (result_dir / "seen_pert").glob("slide_*/eval_meta.json"):
        candidates.append(meta)
    # Common checkpoint dirs for pmap niche / v3 / v4
    for ck in (
        Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_pmap_niche_seen"),
        Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_pmap_niche_unseen"),
        Path("/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT/outputs/ms_pmap_v3_seen"),
        Path("/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT/outputs/ms_pmap_v4_seen"),
    ):
        gpath = ck / "hvg_gene_names.txt"
        if gpath.is_file():
            candidates.append(gpath)
    for path in candidates:
        if path.name == "eval_meta.json":
            # infer from sibling eval - try niche ckpt first
            gpath = Path(
                "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_pmap_niche_seen"
            ) / "hvg_gene_names.txt"
            if not gpath.is_file():
                continue
        else:
            gpath = path
        names = np.asarray([ln.strip() for ln in gpath.read_text().splitlines() if ln.strip()])
        if len(names) == vec_len:
            return names
    if vec_len == 2000:
        return load_concert_hvg_names("0", hvg_max_rank=2000)
    return None


def rescore_concert_pm_slide(
    prepared_dir: Path,
    result_dir: Path,
    slide_id: str,
    method: str,
    min_g_cells: int = 20,
) -> list[ResidualScore]:
    mlfc_path = result_dir / "seen_pert" / f"slide_{slide_id}" / "model_lfc.json"
    if not mlfc_path.is_file():
        return []
    hvg_idx = load_hvg_indices(result_dir / "seen_pert" / f"slide_{slide_id}")
    mlfc = json.loads(mlfc_path.read_text()).get("L2", {})
    # infer concert gene axis length from first non-empty vector
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
    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")
    out: list[ResidualScore] = []
    for gene, niche_vecs in mlfc.items():
        keep = build_gene_mask(concert_genes, gene, hvg_idx)
        if keep.sum() < 10:
            continue
        pred_by_key: dict[str, np.ndarray] = {}
        obs_by_key: dict[str, np.ndarray] = {}
        for key, vec in niche_vecs.items():
            if not vec:
                continue
            niche_bin = int(str(key).split(":")[0])
            pred = np.asarray(vec, dtype=np.float64)
            if pred.size != len(concert_genes):
                continue
            obs = observed_lfc_on_gene_axis(adata, gene, niche_bin, concert_genes, min_g_cells)
            if obs is None or not np.isfinite(obs).any():
                continue
            pred_by_key[key] = pred
            obs_by_key[key] = obs
        sc = score_gene_from_niche_lfc(gene, slide_id, method, pred_by_key, obs_by_key, keep)
        if sc is not None:
            out.append(sc)
    return out


def align_mlfc_vector(
    vec: list | np.ndarray,
    gene_names: np.ndarray,
    hvg_indices: np.ndarray | None,
) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    if v.size == len(gene_names):
        return v
    if hvg_indices is not None and v.size == len(hvg_indices):
        out = np.full(len(gene_names), np.nan, dtype=np.float64)
        idx = np.asarray(hvg_indices, dtype=int)
        valid = (idx >= 0) & (idx < len(gene_names))
        out[idx[valid]] = v[valid]
        return out
    return np.full(len(gene_names), np.nan, dtype=np.float64)


def observed_lfc_pm_niche(
    adata: ad.AnnData,
    gene: str,
    niche_bin: int,
    min_g_cells: int,
) -> np.ndarray | None:
    """Observed LFC for KO gene vs KP control in niche (PM seen eval).

    Control falls back to slide-wide when the niche has < min_g_cells,
    matching CONCERT / competitor ``_ref_ctrl_idx`` behavior.
    """
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata.obs["bin_id"].to_numpy().astype(int)
    ko = np.where((canon == gene) & (bin_id == niche_bin))[0]
    if len(ko) < min_g_cells:
        ko = np.where(canon == gene)[0]
    ctrl = np.where((canon == "control") & (bin_id == niche_bin))[0]
    if len(ctrl) < min_g_cells:
        ctrl = np.where(canon == "control")[0]
    if len(ko) < min_g_cells or len(ctrl) < min_g_cells:
        return None
    return pseudobulk_lfc(to_dense(adata[ko].X), to_dense(adata[ctrl].X))


def observed_global_lfc_pm(adata: ad.AnnData, gene: str, min_g_cells: int) -> np.ndarray | None:
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    ko = np.where(canon == gene)[0]
    ctrl = np.where(canon == "control")[0]
    if len(ko) < min_g_cells or len(ctrl) < min_g_cells:
        return None
    return pseudobulk_lfc(to_dense(adata[ko].X), to_dense(adata[ctrl].X))


def score_gene_from_niche_lfc(
    gene: str,
    slide: str,
    method: str,
    pred_by_key: dict[str, np.ndarray],
    obs_by_key: dict[str, np.ndarray],
    keep: np.ndarray,
) -> ResidualScore | None:
    keys = sorted(set(pred_by_key) & set(obs_by_key))
    if len(keys) < 2:
        return None

    pred_mat = np.stack([pred_by_key[k][keep] for k in keys])
    obs_mat = np.stack([obs_by_key[k][keep] for k in keys])

    raw_scores = [score_lfc_agreement(pred_by_key[k][keep], obs_by_key[k][keep])["spearman"] for k in keys]
    raw_lfc_rho = float(np.nanmean(raw_scores))

    obs_global, obs_resid = residualize_across_niches(obs_mat)
    pred_global, pred_resid = residualize_across_niches(pred_mat)

    niche_residual_rho = score_residual_flat(pred_resid, obs_resid)
    cross_niche_pattern_rho = niche_residual_rho  # same flattening for (niche × gene)

    # Global pooled baseline: predict slide-global observed LFC in every niche
    pool_pred = np.tile(obs_global, (len(keys), 1))
    pool_raw_scores = [score_lfc_agreement(pool_pred[i], obs_mat[i])["spearman"] for i in range(len(keys))]
    global_pool_raw_rho = float(np.nanmean(pool_raw_scores))
    _, pool_resid = residualize_across_niches(pool_pred)
    global_pool_residual_rho = score_residual_flat(pool_resid, obs_resid)

    return ResidualScore(
        gene=gene,
        slide=slide,
        method=method,
        n_niches=len(keys),
        n_genes_scored=int(keep.sum()),
        raw_lfc_rho=raw_lfc_rho,
        niche_residual_rho=niche_residual_rho,
        cross_niche_pattern_rho=cross_niche_pattern_rho,
        global_pool_raw_rho=global_pool_raw_rho,
        global_pool_residual_rho=global_pool_residual_rho,
    )


def load_hvg_indices(result_dir: Path) -> np.ndarray | None:
    path = result_dir / "hvg_info.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    idx = data.get("hvg_gene_indices")
    return np.asarray(idx, dtype=int) if idx else None


def rescore_pm_slide_method(
    prepared_dir: Path,
    result_dir: Path,
    slide_id: str,
    method: str,
    min_g_cells: int = 20,
) -> list[ResidualScore]:
    """Rescore one method/slide from saved model_lfc.json (no inference)."""
    if method.startswith("concert"):
        return rescore_concert_pm_slide(prepared_dir, result_dir, slide_id, method, min_g_cells)

    mlfc_path = result_dir / "seen_pert" / f"slide_{slide_id}" / "model_lfc.json"
    if not mlfc_path.is_file():
        return []

    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")
    gene_names = np.asarray(adata.var_names)
    hvg_idx = load_hvg_indices(result_dir / "seen_pert" / f"slide_{slide_id}")
    mlfc = json.loads(mlfc_path.read_text()).get("L2", {})

    out: list[ResidualScore] = []
    for gene, niche_vecs in mlfc.items():
        keep = build_gene_mask(gene_names, gene, hvg_idx)
        if keep.sum() < 10:
            continue

        pred_by_key: dict[str, np.ndarray] = {}
        obs_by_key: dict[str, np.ndarray] = {}
        for key in niche_vecs:
            if not niche_vecs[key]:
                continue
            parts = str(key).split(":")
            niche_bin = int(parts[0])
            vec = align_mlfc_vector(niche_vecs[key], gene_names, hvg_idx)
            if not np.isfinite(vec).any():
                continue
            obs = observed_lfc_pm_niche(adata, gene, niche_bin, min_g_cells)
            if obs is None:
                continue
            pred_by_key[key] = vec
            obs_by_key[key] = obs

        sc = score_gene_from_niche_lfc(gene, slide_id, method, pred_by_key, obs_by_key, keep)
        if sc is not None:
            out.append(sc)
    return out


def global_pooled_baseline_pm_slide(
    prepared_dir: Path,
    slide_id: str,
    genes: Iterable[str],
    min_g_cells: int = 20,
) -> list[ResidualScore]:
    """Baseline: predict slide-global observed LFC in every niche (residual ≈ 0)."""
    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")
    gene_names = np.asarray(adata.var_names)
    out: list[ResidualScore] = []

    for gene in genes:
        keep = build_gene_mask(gene_names, gene, None)
        global_lfc = observed_global_lfc_pm(adata, gene, min_g_cells)
        if global_lfc is None:
            continue

        obs_by_key: dict[str, np.ndarray] = {}
        pred_by_key: dict[str, np.ndarray] = {}
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        bin_id = adata.obs["bin_id"].to_numpy().astype(int)
        for niche_bin in np.unique(bin_id):
            obs = observed_lfc_pm_niche(adata, gene, int(niche_bin), min_g_cells)
            if obs is None:
                continue
            key = f"{int(niche_bin)}:pool"
            obs_by_key[key] = obs
            pred_by_key[key] = global_lfc.copy()

        sc = score_gene_from_niche_lfc(
            gene, slide_id, "global_pooled", pred_by_key, obs_by_key, keep,
        )
        if sc is not None:
            out.append(sc)
    return out


def zero_residual_baseline_pm_slide(
    prepared_dir: Path,
    slide_id: str,
    genes: Iterable[str],
    min_g_cells: int = 20,
) -> list[ResidualScore]:
    """Baseline: predict zero residual (non-spatial null)."""
    adata = ad.read_h5ad(prepared_dir / f"slide_{slide_id}.h5ad")
    gene_names = np.asarray(adata.var_names)
    out: list[ResidualScore] = []

    for gene in genes:
        keep = build_gene_mask(gene_names, gene, None)
        global_lfc = observed_global_lfc_pm(adata, gene, min_g_cells)
        if global_lfc is None:
            continue

        obs_by_key: dict[str, np.ndarray] = {}
        pred_by_key: dict[str, np.ndarray] = {}
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        bin_id = adata.obs["bin_id"].to_numpy().astype(int)
        for niche_bin in np.unique(bin_id):
            obs = observed_lfc_pm_niche(adata, gene, int(niche_bin), min_g_cells)
            if obs is None:
                continue
            key = f"{int(niche_bin)}:zero"
            obs_by_key[key] = obs
            pred_by_key[key] = global_lfc.copy()  # constant → zero residual

        sc = score_gene_from_niche_lfc(
            gene, slide_id, "zero_residual", pred_by_key, obs_by_key, keep,
        )
        if sc is not None:
            out.append(sc)
    return out
