"""Shared helpers for SPAC-seq counterfactual baseline evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from method_input_lib import normalize_pred_cache

log = logging.getLogger(__name__)

BIOLORD_TRAIN_OUTPUT = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/"
    "biolord/biolord/exp_spacseq/outputs"
)
EVAL_UTILS_DIR = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/"
    "spaVAE/CONCERT-main/src/CONCERT-spac"
)

UNASSIGNED_LABELS = frozenset([
    "unassigned", "Unassigned", "No_guide", "no_guide", "unknown",
])
CONTROL_LABELS = frozenset([
    "Non_targeting", "non_targeting", "NonTargeting",
    "control", "Control", "CTRL", "ctrl", "NT",
])


def import_eval_utils():
    import sys

    sys.path.insert(0, str(EVAL_UTILS_DIR))
    import eval_utils as eu
    import spac_edist_align as sa

    return eu, sa


def to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Per-niche bootstrap: resample the *source (holdout prediction) patch* cells
# --------------------------------------------------------------------------- #
DEFAULT_N_BOOT = 50
DEFAULT_BOOT_FRAC = 0.5
DEFAULT_N_PATCH_BOOT = 10
NON_SPATIAL_METHODS = frozenset({"biolord", "gears", "scgen"})


def patch_boot_settings(args) -> tuple[int, int]:
    n_patch = int(getattr(args, "n_patch_boot", 0))
    seed = int(getattr(args, "patch_boot_seed", getattr(args, "boot_seed", getattr(args, "seed", 0) or 0)))
    return n_patch, seed


def random_local_patch_rows(
    src_bin: np.ndarray,
    exclude_bin: int,
    n_want: int,
    min_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick holdout-source rows from a niche *other than* ``exclude_bin``.

    Returns local indices into the source pool (``src_bin``). Falls back to any
    niche if too few cells exist outside the eval niche.
    """
    src_bin = np.asarray(src_bin, dtype=int)
    n_want = max(int(n_want), int(min_cells))
    pool = np.where(src_bin != int(exclude_bin))[0]
    if len(pool) < min_cells:
        pool = np.arange(len(src_bin))
    n_take = min(n_want, len(pool))
    if n_take <= 0:
        return np.array([], dtype=int)
    return rng.choice(pool, n_take, replace=False)


def pick_random_holdout_patch(patches: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Return one full holdout patch (local row indices) chosen uniformly."""
    if not patches:
        return np.array([], dtype=int)
    return np.asarray(patches[int(rng.integers(len(patches)))], dtype=int)


def holdout_patch_groups_pm(
    groups: dict,
    exclude_bin: int,
    min_cells: int,
) -> list[np.ndarray]:
    """KP holdout replicate patches with ``target_bin != exclude_bin``."""
    out = []
    for (tb, _g, _rep), rows in groups.items():
        if int(tb) == int(exclude_bin):
            continue
        rows = np.asarray(rows, dtype=int)
        if len(rows) >= int(min_cells):
            out.append(rows)
    return out


def score_with_patch_boot(
    pred_g: np.ndarray,
    src_expr: np.ndarray,
    obs_lfc: np.ndarray,
    obs_ko_raw: np.ndarray,
    lib_sizes: np.ndarray,
    keep: np.ndarray,
    eu,
    sa,
    pick_rows,
    n_patch_boot: int,
    seed: int,
) -> dict | None:
    """Score one record by resampling random patches ``n_patch_boot`` times.

    ``pick_rows(rng)`` returns local source indices for one patch draw.
    Returns mean metrics plus std across draws (stored as ``*_boot_std``).
    """
    if n_patch_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    sp_l, ed_l, ld_l = [], [], []
    keep = np.asarray(keep, dtype=bool)
    o_keep = np.asarray(obs_lfc, dtype=np.float64)[keep]
    obs_ko_raw = np.asarray(obs_ko_raw)
    for _ in range(int(n_patch_boot)):
        rows = np.asarray(pick_rows(rng), dtype=int)
        if rows.size < 2:
            continue
        pred_b = pred_g[rows]
        m_lfc = np.nan_to_num(
            eu.pseudobulk_lfc(pred_b, src_expr[rows]), nan=0.0, posinf=0.0, neginf=0.0,
        )
        sc_ = eu.score_lfc_agreement(m_lfc[keep], o_keep)
        ed = float(sa.aligned_energy_distance(
            pred_b, obs_ko_raw, lib_sizes[rows], keep,
        ))
        sp_l.append(float(sc_["spearman"]))
        ed_l.append(ed)
        ld_l.append(float(sc_["lfc_dist"]))
    if not sp_l:
        return None
    return {
        "spearman": float(np.mean(sp_l)),
        "lfc_dist": float(np.mean(ld_l)),
        "e_dist": float(np.mean(ed_l)),
        "spearman_boot_std": float(np.std(sp_l, ddof=1)) if len(sp_l) > 1 else 0.0,
        "lfc_dist_boot_std": float(np.std(ld_l, ddof=1)) if len(ld_l) > 1 else 0.0,
        "e_dist_boot_std": float(np.std(ed_l, ddof=1)) if len(ed_l) > 1 else 0.0,
        "n_boot": int(len(sp_l)),
    }


def attach_patch_boot_stats(sc_: dict, patch_stats: dict | None) -> dict:
    if not patch_stats:
        return sc_
    sc_.update(
        spearman=patch_stats["spearman"],
        lfc_dist=patch_stats.get("lfc_dist", float("nan")),
        e_dist=patch_stats["e_dist"],
        spearman_boot_std=patch_stats["spearman_boot_std"],
        lfc_dist_boot_std=patch_stats.get("lfc_dist_boot_std", 0.0),
        e_dist_boot_std=patch_stats["e_dist_boot_std"],
        n_boot=patch_stats["n_boot"],
    )
    return sc_


def bootstrap_source_metrics(
    pred_rows: np.ndarray,
    src_rows: np.ndarray,
    obs_lfc: np.ndarray,
    obs_ko_raw: np.ndarray,
    lib_rows: np.ndarray,
    keep: np.ndarray,
    eu,
    sa,
    n_boot: int = DEFAULT_N_BOOT,
    frac: float = DEFAULT_BOOT_FRAC,
    seed: int = 0,
) -> dict | None:
    """Within-niche variance by resampling only the source (holdout) patch cells.

    The source patch = held-out *unperturbed* cells the model predicts the KO on
    (perturb-map: role==1 KP-tumor; SPAC-seq: held-out unassigned source). The
    observed KO/control reference is ground truth and stays fixed, so we resample
    ``frac`` of the source cells (without replacement) ``n_boot`` times and
    recompute both metrics each draw:

    * ``spearman`` of predicted vs observed per-gene pseudobulk LFC
    * ``lfc_dist``  RMSE between predicted and observed LFC vectors
    * ``e_dist``   aligned energy distance (predicted subset vs fixed observed KO)

    Parameters mirror the point-estimate call sites: ``pred_rows``/``src_rows`` are
    the (n_src, n_genes) predicted and source-expression matrices *restricted to
    this record's source cells*; ``obs_lfc`` is the already-computed observed LFC
    (fixed); ``obs_ko_raw`` the observed KO raw counts; ``lib_rows`` the per-source
    library sizes; ``keep`` the scored-gene mask (KO gene removed).

    Returns ``{"spearman_boot": [...], "lfc_dist_boot": [...], "e_dist_boot": [...]}``
    (length == number of successful draws) or ``None`` when the source patch is too small (< 2 cells).
    """
    pred_rows = np.asarray(pred_rows, dtype=np.float64)
    src_rows = np.asarray(src_rows, dtype=np.float64)
    n = int(pred_rows.shape[0])
    if n < 2 or n_boot <= 0:
        return None
    k = max(2, int(round(frac * n)))
    if k >= n:
        k = n - 1
    keep = np.asarray(keep, dtype=bool)
    o_keep = np.asarray(obs_lfc, dtype=np.float64)[keep]
    lib_rows = np.asarray(lib_rows, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sp_l, ed_l, ld_l = [], [], []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, size=k, replace=False)
        pr = pred_rows[idx]
        m_lfc = np.nan_to_num(eu.pseudobulk_lfc(pr, src_rows[idx]),
                              nan=0.0, posinf=0.0, neginf=0.0)
        sc_ = eu.score_lfc_agreement(m_lfc[keep], o_keep)
        rho = float(sc_["spearman"])
        ld = float(sc_["lfc_dist"])
        ed = float(sa.aligned_energy_distance(pr, obs_ko_raw, lib_rows[idx], keep))
        sp_l.append(rho)
        ld_l.append(ld)
        ed_l.append(ed)
    return {"spearman_boot": sp_l, "lfc_dist_boot": ld_l, "e_dist_boot": ed_l}


def boot_settings(args) -> tuple[int, float, int]:
    """Read bootstrap knobs off argparse namespace with safe defaults."""
    n_boot = int(getattr(args, "n_boot", 0))
    frac = float(getattr(args, "boot_frac", DEFAULT_BOOT_FRAC))
    seed = int(getattr(args, "boot_seed", getattr(args, "seed", 0) or 0))
    return n_boot, frac, seed


def attach_bootstrap_stats(sc_: dict, boot: dict | None) -> dict:
    """Add per-record bootstrap summary (mean/std of resampled metrics) to ``sc_``."""
    if not boot:
        sc_.update(spearman_boot_std=0.0, lfc_dist_boot_std=0.0, e_dist_boot_std=0.0, n_boot=0)
        return sc_
    sp = boot["spearman_boot"]
    ld = boot.get("lfc_dist_boot", [])
    ed = boot["e_dist_boot"]
    sc_.update(
        spearman_boot_std=float(np.std(sp, ddof=1)) if len(sp) > 1 else 0.0,
        lfc_dist_boot_std=float(np.std(ld, ddof=1)) if len(ld) > 1 else 0.0,
        e_dist_boot_std=float(np.std(ed, ddof=1)) if len(ed) > 1 else 0.0,
        n_boot=int(len(sp)),
    )
    return sc_


def jsonify(d):
    if isinstance(d, dict):
        return {k: jsonify(v) for k, v in d.items()}
    if isinstance(d, (np.floating, np.integer)):
        return float(d)
    return d


def select_hvgs(adata_all: ad.AnnData, n_top: int, sa) -> np.ndarray:
    mask = sa.hvg_mask_from_rank(adata_all, n_top=n_top)
    log.info(
        "HVG scoring mask: %d / %d genes (hvg_rank <= %d)",
        mask.sum(),
        len(mask),
        n_top,
    )
    return mask


def build_per_pert_summary(agg):
    summary = {}
    for lv, genes in agg.items():
        for g, m in genes.items():
            if g not in summary:
                summary[g] = {}
            summary[g][lv] = dict(m)
    return summary


def print_per_pert_table(per_pert, out_dir: Path):
    lines = []
    header = (
        f"{'gene':<14} {'level':<6} "
        f"{'lfc_rho':>8} {'abs_rho':>8} {'e_dist':>8} {'sign':>6} {'topk':>6} {'niches':>6}"
    )
    sep = "-" * len(header)
    lines += [sep, header, sep]
    for g in sorted(per_pert):
        for lv in sorted(per_pert[g]):
            m = per_pert[g][lv]
            lines.append(
                f"{g:<14} {lv:<6} "
                f"{m.get('spearman_mean', float('nan')):>8.3f} "
                f"{m.get('abs_spearman_mean', float('nan')):>8.3f} "
                f"{m.get('e_dist_mean', float('nan')):>8.3f} "
                f"{m.get('sign_agreement_mean', float('nan')):>6.3f} "
                f"{m.get('topk_overlap_mean', float('nan')):>6.3f} "
                f"{m.get('n_niches', 0):>6d}"
            )
    lines.append(sep)
    table_str = "\n".join(lines)
    log.info("\n%s", table_str)
    (out_dir / "per_pert_table.txt").write_text(table_str + "\n")


def save_results(out_dir: Path, scores, agg, mlfc, hvg_info, convergence=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("lfc_scores.json").write_text(json.dumps(jsonify(scores), indent=2))
    out_dir.joinpath("lfc_agg.json").write_text(json.dumps(jsonify(agg), indent=2))
    out_dir.joinpath("model_lfc.json").write_text(json.dumps(jsonify(mlfc), indent=2))
    out_dir.joinpath("hvg_info.json").write_text(json.dumps(hvg_info, indent=2))
    per_pert = build_per_pert_summary(agg)
    out_dir.joinpath("per_pert_summary.json").write_text(json.dumps(jsonify(per_pert), indent=2))
    if convergence is not None:
        out_dir.joinpath("convergence.json").write_text(json.dumps(jsonify(convergence), indent=2))
    log.info("Results → %s", out_dir)
    print_per_pert_table(per_pert, out_dir)


def aggregate_scores(scores, lv, g, worst_metric_values=None):
    if not scores[lv][g]:
        out = dict(
            spearman_mean=0.0,
            spearman_sd=0.0,
            abs_spearman_mean=0.0,
            abs_spearman_sd=0.0,
            sign_agreement_mean=0.5,
            sign_agreement_sd=0.0,
            topk_overlap_mean=0.0,
            topk_overlap_sd=0.0,
            e_dist_mean=float("nan"),
            e_dist_sd=float("nan"),
            n_niches=0,
            n_scored_genes=0,
        )
        if worst_metric_values:
            for base_key, worst_val in worst_metric_values.items():
                k = base_key + "_mean"
                if k in out and np.isnan(out[k]):
                    out[k] = worst_val
        return out
    metric_keys = [
        k for k in scores[lv][g][0]
        if k not in ("niche", "n_src", "n_real_g", "n_scored_genes")
    ]
    agg = {
        k + "_mean": float(np.nanmean([s.get(k, np.nan) for s in scores[lv][g]]))
        for k in metric_keys
    }
    agg.update({
        k + "_sd": float(np.nanstd([s.get(k, np.nan) for s in scores[lv][g]]))
        for k in metric_keys
    })
    agg["n_niches"] = len(scores[lv][g])
    agg["n_scored_genes"] = scores[lv][g][0]["n_scored_genes"]
    worst_defaults = {"sign_agreement": 0.5, "topk_overlap": 0.0}
    if worst_metric_values:
        worst_defaults.update(worst_metric_values)
    for base_key, worst_val in worst_defaults.items():
        k = base_key + "_mean"
        if k in agg and np.isnan(agg[k]):
            agg[k] = worst_val
    return agg


def worst_niche_score(
    niche: int,
    n_src: int,
    n_real_g: int,
    n_scored_genes: int,
    bad_edist: float,
) -> dict:
    """Worst-case per-niche metrics when a method cannot predict a perturbation."""
    return dict(
        spearman=0.0,
        abs_spearman=0.0,
        sign_agreement=0.5,
        topk_overlap=0.0,
        e_dist=bad_edist,
        lfc_dist=float("nan"),
        niche=int(niche),
        n_src=int(n_src),
        n_real_g=int(n_real_g),
        n_scored_genes=int(n_scored_genes),
    )


def _fill_worst_case_scores(
    scores,
    panel,
    lv,
    eval_bins,
    una_mask,
    bin_id,
    pert_canon,
    hvg_mask,
    gene_names,
    unpredictable_genes,
    n_src,
    args,
    bad_edist_default,
):
    """Insert worst-case niche scores for unpredicted genes / missing niches."""
    unpredictable_genes = unpredictable_genes or set()
    scored_edists = [
        s["e_dist"]
        for g in panel
        for s in scores[lv][g]
        if np.isfinite(s.get("e_dist", np.nan))
    ]
    bad_edist = float(np.max(scored_edists)) if scored_edists else float(bad_edist_default)
    scored_niches = {g: {s["niche"] for s in scores[lv][g]} for g in panel}
    worst_metrics = {
        "spearman": 0.0,
        "abs_spearman": 0.0,
        "sign_agreement": 0.5,
        "topk_overlap": 0.0,
        "e_dist": bad_edist,
    }

    for b in eval_bins:
        una_idx = np.where(una_mask & (bin_id == b))[0]
        if len(una_idx) < args.min_g_cells:
            continue
        for g in panel:
            real_g_idx = np.where((pert_canon == g) & (bin_id == b))[0]
            if len(real_g_idx) < args.min_g_cells:
                continue
            if int(b) in scored_niches[g]:
                continue
            n_scored = int((hvg_mask & (gene_names != g)).sum())
            reason = "unpredictable" if g in unpredictable_genes else "missing prediction"
            log.warning(
                "[%s] %s niche %d: %s — inserting worst-case metrics",
                lv,
                g,
                b,
                reason,
            )
            scores[lv][g].append(
                worst_niche_score(int(b), n_src, len(real_g_idx), n_scored, bad_edist)
            )
    return worst_metrics


def niche_lfc_loop(
    panel,
    adata_all,
    bin_id,
    evaluable,
    pert_canon,
    source_masks,
    pred_g_cache,
    hvg_mask,
    args,
    raw_mm,
    lib_sizes,
    unpredictable_genes=None,
    bad_edist_default=12.0,
    scoring_method: str | None = None,
):
    """Score per-niche LFC agreement using precomputed pred_g arrays.

    Competing methods with limited perturbation support (e.g. CellAgentChat LR-only)
    should pass those genes in ``unpredictable_genes``. Failed / missing predictions
    (``pred_g_cache[g] is None``) receive worst-case per-niche scores so they count
    against the method in bar-plot aggregates.
    """
    eu, sa = import_eval_utils()

    gene_names = np.asarray(adata_all.var_names)
    eval_bins = np.where(evaluable)[0]
    una_mask = np.isin(pert_canon, list(UNASSIGNED_LABELS | {"unassigned"}))
    unpredictable_genes = set(unpredictable_genes or [])
    if scoring_method:
        pred_g_cache = normalize_pred_cache(scoring_method, pred_g_cache)

    scores = {lv: {g: [] for g in panel} for lv in source_masks}
    agg = {lv: {} for lv in source_masks}
    mlfc = {lv: {g: {} for g in panel} for lv in source_masks}

    for lv, src_mask in source_masks.items():
        global_src_idx = np.where(src_mask)[0]
        if len(global_src_idx) == 0:
            log.warning("[%s] empty source pool", lv)
            continue
        if args.max_src_cells > 0 and len(global_src_idx) > args.max_src_cells:
            rng = np.random.default_rng(42)
            global_src_idx = rng.choice(global_src_idx, args.max_src_cells, replace=False)
        src_expr = to_dense(adata_all[global_src_idx].X)
        n_src = len(global_src_idx)
        log.info("[%s] using %d global source cells", lv, n_src)

        for b in eval_bins:
            una_idx = np.where(una_mask & (bin_id == b))[0]
            if len(una_idx) < args.min_g_cells:
                continue
            y_una_b = to_dense(adata_all[una_idx].X)

            for g in panel:
                if g in unpredictable_genes:
                    continue
                pred_g = pred_g_cache.get(g)
                if pred_g is None:
                    continue
                real_g_idx = np.where((pert_canon == g) & (bin_id == b))[0]
                if len(real_g_idx) < args.min_g_cells:
                    continue
                y_g_b = to_dense(adata_all[real_g_idx].X)

                m_lfc = eu.pseudobulk_lfc(pred_g, src_expr)
                o_lfc = eu.pseudobulk_lfc(y_g_b, y_una_b)
                m_lfc = np.nan_to_num(m_lfc, nan=0.0, posinf=0.0, neginf=0.0)

                keep = hvg_mask & (gene_names != g)
                sc_ = eu.score_lfc_agreement(m_lfc[keep], o_lfc[keep])

                mean_pred_hvg = pred_g[:, keep].mean(axis=0)
                mean_obs_hvg = y_g_b[:, keep].mean(axis=0)
                abs_rho = float(spearmanr(mean_pred_hvg, mean_obs_hvg).correlation)
                sc_["abs_spearman"] = 0.0 if np.isnan(abs_rho) else abs_rho

                sc_["e_dist"] = sa.aligned_energy_distance(
                    pred_g,
                    np.asarray(raw_mm[real_g_idx]),
                    lib_sizes[global_src_idx],
                    keep,
                )
                sc_.update(
                    niche=int(b),
                    n_src=n_src,
                    n_real_g=int(len(real_g_idx)),
                    n_scored_genes=int(keep.sum()),
                )
                scores[lv][g].append(sc_)
                mlfc[lv][g][int(b)] = m_lfc.tolist()

        worst_metrics = _fill_worst_case_scores(
            scores,
            panel,
            lv,
            eval_bins,
            una_mask,
            bin_id,
            pert_canon,
            hvg_mask,
            gene_names,
            unpredictable_genes,
            n_src,
            args,
            bad_edist_default,
        )

        for g in panel:
            agg[lv][g] = aggregate_scores(scores, lv, g, worst_metrics)
            if agg[lv][g]["n_niches"]:
                log.info(
                    "[%s] %-12s lfc_rho=%.3f abs_rho=%.3f e_dist=%.3f sign=%.3f  (n=%d)",
                    lv,
                    g,
                    agg[lv][g].get("spearman_mean", float("nan")),
                    agg[lv][g].get("abs_spearman_mean", float("nan")),
                    agg[lv][g].get("e_dist_mean", float("nan")),
                    agg[lv][g].get("sign_agreement_mean", float("nan")),
                    agg[lv][g]["n_niches"],
                )

    return scores, agg, mlfc
