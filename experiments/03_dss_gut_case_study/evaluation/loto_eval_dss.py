"""LOTO-D9 evaluation for the CONCERT-spac-niche-temporal gut DSS case study.

Rebuts the reviewer critique "the temporal prediction is just interpolation -- the mean
between two observed timepoints." We hold out DSS9 (the acute, NON-MONOTONE injury peak)
entirely from training and test whether CONCERT predicts the day-9 inflammation program
BETTER than linear interpolation of DSS3 and DSS21.

See docs/LOTO_D9_EVAL_DESIGN.md for the full design. Key points implemented here:

  * Leakage-free: day-9 predictions are produced by taking NON-day-9 source cells and
    re-decoding them at t = target_day/denom via counterfactualPrediction(time_t=...). No
    day-9 expression ever touches a day-9 prediction.
  * Perturbation-state-STRATIFIED pseudobulk (control-only / perturbed-only) is the PRIMARY
    metric -- it isolates the INTENSITY dynamics the time×pert term actually models. The
    whole-tissue (composition-conflated) number is reported alongside for honesty.
  * States are decoded in SEPARATE counterfactualPrediction calls: rk_b (the f_pert/time
    gate) is batch-uniform from target_cell_perturbation, so perturbed and control source
    cells must not be mixed in one call.
  * PRIMARY metric = LFC-RMSE on the perturbed-vs-control log-fold-change vector (scale-robust,
    matches the niche benchmark's lfc_dist). WIN: CONCERT LFC-RMSE(day9) < linear-interp.

This module is invoked from run_concert_spac.py's `loto_eval` stage, which reconstructs the
model on the SAME training file list (guaranteeing the architecture matches the checkpoint)
and loads the trained weights before calling run_loto_eval(...).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

# Curated inflammation / DSS-damage marker panel for the learned-time-function curve and the
# pathway response readout. Intersected with the 942-gene MERFISH panel at run time.
INFLAMMATION_MARKERS = [
    "Saa3", "Cxcl10", "Il1b", "Mmp13", "Nos2", "Tnf", "Isg15", "Ifit1",
    "Cxcl1", "Cxcl5", "Il11", "Mmp3", "Mmp10", "Plau", "Igfbp5", "Osm",
    "Trem1", "Socs3", "Duoxa2", "Il18", "Aldh1a3", "Krt13", "Krt14",
]

EPS = 1e-3
CP_TARGET = 1e4  # counts-per-CP_TARGET library normalization for pseudobulks


# ---
# helpers
# ---
def _dense(M) -> np.ndarray:
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _cp_normalize(counts: np.ndarray, target: float = CP_TARGET) -> np.ndarray:
    """Row-normalize a (cells × genes) count matrix to a fixed library size."""
    counts = np.asarray(counts, dtype=np.float64)
    lib = counts.sum(axis=1, keepdims=True)
    lib = np.clip(lib, 1e-8, None)
    return counts / lib * target


def _pseudobulk(counts: np.ndarray) -> np.ndarray:
    """Mean CP-normalized expression over a set of cells -> (genes,) pseudobulk vector."""
    if counts.shape[0] == 0:
        raise ValueError("empty cell set for pseudobulk")
    return _cp_normalize(counts).mean(axis=0)


def _lfc(pb_pert: np.ndarray, pb_ctrl: np.ndarray) -> np.ndarray:
    """Perturbed-vs-control log fold-change vector."""
    return np.log(pb_pert + EPS) - np.log(pb_ctrl + EPS)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return _pearson(ra.astype(float), rb.astype(float))


# ---
# ground truth loading (held-out day-9 files)
# ---
def _load_holdout_pseudobulks(
    files: List[str], gene_names: np.ndarray, pert_col: str = "perturbation",
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Pool held-out (day-9) slice files -> CP-normalized pseudobulks for perturbed & control.

    Returns (pb_pert, pb_ctrl, n_pert, n_ctrl). Genes are reindexed to `gene_names` order
    (the model's output column order) by symbol.
    """
    import anndata as ad

    pert_chunks: List[np.ndarray] = []
    ctrl_chunks: List[np.ndarray] = []
    for pth in files:
        a = ad.read_h5ad(pth)
        genes = np.asarray(a.var.index)
        # reindex to model gene order
        pos = {g: i for i, g in enumerate(genes)}
        if not all(g in pos for g in gene_names):
            missing = [g for g in gene_names if g not in pos][:5]
            raise ValueError(f"{pth}: missing model genes e.g. {missing}")
        col = np.array([pos[g] for g in gene_names], dtype=int)
        X = _dense(a.X)[:, col]
        if "filter_pass" in a.obs:
            fp = a.obs["filter_pass"].to_numpy().astype(bool)
            X = X[fp]; state = a.obs[pert_col].to_numpy()[fp]
        else:
            state = a.obs[pert_col].to_numpy()
        state = np.asarray([str(s) for s in state])
        pert_chunks.append(X[state == "perturbed"])
        ctrl_chunks.append(X[state == "control"])

    Xp = np.concatenate(pert_chunks, axis=0)
    Xc = np.concatenate(ctrl_chunks, axis=0)
    return _pseudobulk(Xp), _pseudobulk(Xc), Xp.shape[0], Xc.shape[0]


# ---
# CONCERT day-t prediction (leakage-free counterfactual re-decode)
# ---
def _predict_state_at_t(
    model, *, pos_batched, sample_indices, cell_atts, Y_expr, subset_idx,
    target_pert_code, target_t, n_samples, batch_size, is_real_KO: bool,
) -> np.ndarray:
    """Decode a single perturbation-state subset of source cells at normalized day `target_t`.

    `target_pert_code` is the INTEGER perturbation code (counterfactualPrediction does
    int(target_cell_perturbation) internally): 1=perturbed, 0=__basal__/control.

    Returns the (n_subset × genes) predicted mean-count matrix. States are decoded in their
    OWN call because rk_b (the f_pert/time gate) is batch-uniform from target_cell_perturbation.
    """
    import torch

    n = int(subset_idx.size)
    if n == 0:
        return np.zeros((0, Y_expr.shape[1]), dtype=np.float32)
    tt = (None if target_t is None
          else np.full(n, float(target_t), dtype=np.float32))
    mu, _ = model.counterfactualPrediction(
        X=np.asarray(pos_batched)[subset_idx],
        sample_index=torch.as_tensor(np.asarray(sample_indices)[subset_idx]),
        cell_atts=np.asarray(cell_atts)[subset_idx].copy(),
        perturb_cell_id=torch.arange(n),
        target_cell_perturbation=int(target_pert_code),
        target_cell_tissue=int(cell_atts[subset_idx[0], 0]),
        target_has_guide=True if is_real_KO else False,
        target_is_real_KO=is_real_KO,
        n_samples=n_samples,
        batch_size=batch_size,
        Y=np.asarray(Y_expr)[subset_idx],
        time_t=tt,
    )
    return np.asarray(mu, dtype=np.float64)


# ---
# main entry point
# ---
def run_loto_eval(
    model, adata, cell_atts, pos_batched, sample_indices, sample_id_all,
    multi_paths, gene_names, pert_map, cfg, outdir,
) -> Dict:
    """Run the LOTO-D9 evaluation and write metrics/figures to `outdir`.

    Source cells = the loaded training cells (Healthy + DSS3 + DSS21; DSS9 excluded from
    training). Ground truth = the held-out DSS9 files (cfg.loto_holdout_files).
    """
    import anndata as _ad

    outdir = Path(outdir)
    target_t = float(cfg.loto_target_day) / float(cfg.loto_day_denom)
    gene_names = np.asarray(gene_names)
    G = gene_names.size

    # per-cell source day_norm (each slide-file = one timepoint) ---
    day_by_file = np.array([
        float(_ad.read_h5ad(p, backed="r").obs[cfg.time_col].iloc[0]) for p in multi_paths
    ], dtype=np.float32)
    src_day = day_by_file[np.asarray(sample_id_all).astype(int)]   # (N,)
    uniq_days = np.unique(np.round(src_day, 4))
    logging.info("[loto] target_t=%.4f (day %.1f/%.1f); source day_norms=%s",
                 target_t, cfg.loto_target_day, cfg.loto_day_denom, uniq_days.tolist())

    # source expression matrices ---
    Y_expr = _dense(adata.X).astype(np.float32)          # normalized (expr encoder input)
    raw_src = _dense(adata.raw.X).astype(np.float64)     # raw counts (for real pseudobulks)
    state = np.asarray(cell_atts)[:, 1].astype(int)      # 0=control/basal, 1=perturbed
    pert_code = int(pert_map.get("perturbed", 1))
    ctrl_code = int(pert_map.get("__basal__", 0))

    n_samples = int(getattr(cfg, "loto_n_samples", 10))
    bs = int(getattr(cfg, "loto_batch_size", 512))

    # --- 1. CONCERT day-9 predictions (leakage-free), per state ---
    pert_src = np.where(state == pert_code)[0]
    ctrl_src = np.where(state != pert_code)[0]
    logging.info("[loto] source cells: %d perturbed, %d control", pert_src.size, ctrl_src.size)

    pred_pert = _predict_state_at_t(
        model, pos_batched=pos_batched, sample_indices=sample_indices, cell_atts=cell_atts,
        Y_expr=Y_expr, subset_idx=pert_src, target_pert_code=pert_code,
        target_t=target_t, n_samples=n_samples, batch_size=bs, is_real_KO=True)
    pred_ctrl = _predict_state_at_t(
        model, pos_batched=pos_batched, sample_indices=sample_indices, cell_atts=cell_atts,
        Y_expr=Y_expr, subset_idx=ctrl_src, target_pert_code=ctrl_code,
        target_t=target_t, n_samples=n_samples, batch_size=bs, is_real_KO=False)

    concert_pb_pert = _pseudobulk(pred_pert)
    concert_pb_ctrl = _pseudobulk(pred_ctrl)
    concert_lfc = _lfc(concert_pb_pert, concert_pb_ctrl)

    # --- 2. Ground-truth day-9 (held out) ---
    y9_pert, y9_ctrl, n9p, n9c = _load_holdout_pseudobulks(cfg.loto_holdout_files, gene_names)
    real_lfc = _lfc(y9_pert, y9_ctrl)
    logging.info("[loto] real DSS9: %d perturbed, %d control cells", n9p, n9c)

    # --- 3. Baselines from observed timepoints (D3, D21) ---
    def _day_state_pb(day_val_norm: float, want_pert: bool) -> Optional[np.ndarray]:
        want = pert_code if want_pert else None
        m = np.isclose(src_day, day_val_norm, atol=1e-3)
        m &= (state == pert_code) if want_pert else (state != pert_code)
        if m.sum() == 0:
            return None
        return _pseudobulk(raw_src[m])

    d3n = 3.0 / float(cfg.loto_day_denom)
    d21n = 21.0 / float(cfg.loto_day_denom)
    pb = {}
    for tag, dn in [("d3", d3n), ("d21", d21n)]:
        pp = _day_state_pb(dn, True); pc = _day_state_pb(dn, False)
        if pp is None or pc is None:
            logging.warning("[loto] missing %s source cells (pert=%s ctrl=%s)", tag, pp is not None, pc is not None)
        pb[tag] = (pp, pc)

    lfc_d3 = _lfc(*pb["d3"]) if pb["d3"][0] is not None else None
    lfc_d21 = _lfc(*pb["d21"]) if pb["d21"][0] is not None else None

    w = (9.0 - 3.0) / (21.0 - 3.0)   # 0.3333
    baselines: Dict[str, np.ndarray] = {}
    if lfc_d3 is not None and lfc_d21 is not None:
        baselines["linear_interp"] = lfc_d3 + w * (lfc_d21 - lfc_d3)
    if lfc_d3 is not None:
        baselines["nearest_d3"] = lfc_d3        # |9-3|=6 < |9-21|=12
    if lfc_d21 is not None:
        baselines["endpoint_d21"] = lfc_d21
    # global-mean baseline: pooled non-D9 pert-vs-ctrl LFC (ignores time)
    pooled_p = raw_src[state == pert_code]; pooled_c = raw_src[state != pert_code]
    if pooled_p.shape[0] and pooled_c.shape[0]:
        baselines["global_pooled"] = _lfc(_pseudobulk(pooled_p), _pseudobulk(pooled_c))

    # --- 4. Metrics: LFC-RMSE (primary) + per-gene expression ---
    methods = {"CONCERT": concert_lfc, **baselines}
    lfc_metrics = {
        name: {"lfc_rmse": _rmse(v, real_lfc),
               "lfc_pearson": _pearson(v, real_lfc),
               "lfc_spearman": _spearman(v, real_lfc)}
        for name, v in methods.items()
    }

    # per-gene stratified expression RMSE/Pearson (CONCERT vs interp) in CP-normalized space
    def _expr_stats(pred_pb_pert, pred_pb_ctrl):
        return {
            "pert_rmse": _rmse(pred_pb_pert, y9_pert),
            "pert_pearson": _pearson(pred_pb_pert, y9_pert),
            "ctrl_rmse": _rmse(pred_pb_ctrl, y9_ctrl),
            "ctrl_pearson": _pearson(pred_pb_ctrl, y9_ctrl),
        }
    expr_metrics = {"CONCERT": _expr_stats(concert_pb_pert, concert_pb_ctrl)}
    # interp expression baseline (interp the pseudobulk vectors directly)
    if pb["d3"][0] is not None and pb["d21"][0] is not None:
        ip = pb["d3"][0] + w * (pb["d21"][0] - pb["d3"][0])
        ic = pb["d3"][1] + w * (pb["d21"][1] - pb["d3"][1])
        expr_metrics["linear_interp"] = _expr_stats(ip, ic)

    # --- 5. Whole-tissue (composition-conflated) honesty metric ---
    # Predict every source cell at its OWN state at t=9, pool all -> compare to real D9 pooled.
    concert_pool = np.concatenate([pred_pert, pred_ctrl], axis=0)
    concert_pool_pb = _pseudobulk(concert_pool)
    # real D9 pooled (weight pert/ctrl by their real cell counts)
    real_pool_pb = (y9_pert * n9p + y9_ctrl * n9c) / max(n9p + n9c, 1)
    whole_tissue = {
        "CONCERT_rmse": _rmse(concert_pool_pb, real_pool_pb),
        "CONCERT_pearson": _pearson(concert_pool_pb, real_pool_pb),
    }
    if pb["d3"][0] is not None and pb["d21"][0] is not None:
        # interp pooled: interp the pooled pseudobulks
        pool_d3 = _pseudobulk(raw_src[np.isclose(src_day, d3n, atol=1e-3)])
        pool_d21 = _pseudobulk(raw_src[np.isclose(src_day, d21n, atol=1e-3)])
        interp_pool = pool_d3 + w * (pool_d21 - pool_d3)
        whole_tissue["linear_interp_rmse"] = _rmse(interp_pool, real_pool_pb)
        whole_tissue["linear_interp_pearson"] = _pearson(interp_pool, real_pool_pb)

    # --- 6. Learned time-function curve (non-linearity evidence) ---
    marker_idx = np.array([i for i, g in enumerate(gene_names) if g in set(INFLAMMATION_MARKERS)], dtype=int)
    markers_present = [gene_names[i] for i in marker_idx]
    ts = np.linspace(0.0, 1.0, 15)
    # decode the perturbed source cells at each t; track mean marker CP-expression.
    # subsample for speed -- the curve only needs the MEAN trajectory, not per-cell.
    TC_CAP = 30000
    if pert_src.size > TC_CAP:
        sub = np.sort(np.random.default_rng(0).choice(pert_src, TC_CAP, replace=False))
    else:
        sub = pert_src
    time_curve = []
    for t in ts:
        mu_t = _predict_state_at_t(
            model, pos_batched=pos_batched, sample_indices=sample_indices, cell_atts=cell_atts,
            Y_expr=Y_expr, subset_idx=sub, target_pert_code=pert_code,
            target_t=float(t), n_samples=max(3, n_samples // 2), batch_size=bs, is_real_KO=True)
        pb_t = _pseudobulk(mu_t)
        time_curve.append(float(pb_t[marker_idx].mean()) if marker_idx.size else float("nan"))
    time_curve = np.array(time_curve)

    # --- 7. Persist ---
    win = None
    if "linear_interp" in lfc_metrics:
        win = bool(lfc_metrics["CONCERT"]["lfc_rmse"] < lfc_metrics["linear_interp"]["lfc_rmse"])

    summary = {
        "target_day": cfg.loto_target_day,
        "target_t_norm": target_t,
        "interp_weight": w,
        "n_source_perturbed": int(pert_src.size),
        "n_source_control": int(ctrl_src.size),
        "n_real_d9_perturbed": int(n9p),
        "n_real_d9_control": int(n9c),
        "lfc_metrics": lfc_metrics,
        "expr_metrics": expr_metrics,
        "whole_tissue": whole_tissue,
        "WIN_concert_beats_interp_lfc": win,
        "time_curve_t": ts.tolist(),
        "time_curve_marker_mean": time_curve.tolist(),
        "markers_used": markers_present,
    }
    (outdir / "loto_d9_metrics.json").write_text(json.dumps(summary, indent=2))

    # per-gene table
    import csv
    with open(outdir / "loto_d9_pseudobulk.csv", "w", newline="") as fh:
        wtr = csv.writer(fh)
        header = ["gene", "real_pert", "real_ctrl", "real_lfc",
                  "concert_pert", "concert_ctrl", "concert_lfc"]
        if "linear_interp" in baselines:
            header += ["interp_lfc"]
        wtr.writerow(header)
        for gi, g in enumerate(gene_names):
            row = [g, y9_pert[gi], y9_ctrl[gi], real_lfc[gi],
                   concert_pb_pert[gi], concert_pb_ctrl[gi], concert_lfc[gi]]
            if "linear_interp" in baselines:
                row += [baselines["linear_interp"][gi]]
            wtr.writerow(row)

    _plot_summary(outdir, ts, time_curve, real_lfc, concert_lfc,
                  baselines.get("linear_interp"), lfc_metrics, target_t)

    # --- console report ---
    logging.info("=" * 68)
    logging.info("LOTO-D9 RESULT (PRIMARY = stratified perturbed-vs-control LFC-RMSE):")
    for name, m in lfc_metrics.items():
        logging.info("  %-16s LFC-RMSE=%.4f  Pearson=%.3f  Spearman=%.3f",
                     name, m["lfc_rmse"], m["lfc_pearson"], m["lfc_spearman"])
    if win is not None:
        logging.info("  >>> CONCERT %s linear interpolation (LFC-RMSE).",
                     "BEATS" if win else "does NOT beat")
    logging.info("  Whole-tissue (composition-conflated, honesty): %s", whole_tissue)
    logging.info("  Wrote loto_d9_metrics.json, loto_d9_pseudobulk.csv, loto_d9_summary.png")
    logging.info("=" * 68)
    return summary


def _plot_summary(outdir, ts, time_curve, real_lfc, concert_lfc, interp_lfc,
                  lfc_metrics, target_t) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        logging.warning("[loto] matplotlib unavailable, skipping figure: %s", e)
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) learned time-function of the inflammation program
    ax = axes[0]
    ax.plot(ts, time_curve, "-o", color="#c0392b", lw=2, ms=5)
    ax.axvline(target_t, ls="--", color="#2c3e50", alpha=0.7, label=f"day 9 (t={target_t:.3f})")
    ax.set_xlabel("normalized day  t = day/21")
    ax.set_ylabel("mean inflammation-marker expr (CP10K)")
    ax.set_title("Learned time-function f_pert(t)\n(non-monotone ⇒ not linear interp)")
    ax.legend(frameon=False)

    # (b) LFC scatter: CONCERT vs interp against real day-9
    ax = axes[1]
    if interp_lfc is not None:
        ax.scatter(real_lfc, interp_lfc, s=8, alpha=0.4, color="#7f8c8d",
                   label=f"linear-interp (RMSE={lfc_metrics.get('linear_interp',{}).get('lfc_rmse',float('nan')):.3f})")
    ax.scatter(real_lfc, concert_lfc, s=8, alpha=0.5, color="#c0392b",
               label=f"CONCERT (RMSE={lfc_metrics['CONCERT']['lfc_rmse']:.3f})")
    lim = [min(real_lfc.min(), concert_lfc.min()), max(real_lfc.max(), concert_lfc.max())]
    ax.plot(lim, lim, "k--", alpha=0.5, lw=1)
    ax.set_xlabel("real DSS9 perturbed-vs-control LFC")
    ax.set_ylabel("predicted LFC")
    ax.set_title("Day-9 inflammation LFC: predicted vs real")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "loto_d9_summary.png", dpi=130)
    plt.close(fig)
