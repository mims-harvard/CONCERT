#!/usr/bin/env python3
"""Data-only signal-detection diagnostics (no GPU, no model).

Computes per-perturbation, per-niche:
  1. Split-half LFC reliability (ceiling for any model)
  2. Cell-state consistency test (same-pert vs cross-state pair Spearman)

Outputs diag_signal_detection.json and prints a summary table.
"""
import json, logging, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
from make_umap import load_data, classify_perts

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")

DATA_FILE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq/raw_output_1-1.h5ad"
OUTDIR    = Path("/n/holylabs/mzitnik_lab/Users/xianglin226/CONCERT/eval_cf_v5a")

def main():
    logging.info("Loading data...")
    (adata, pos_batched, cell_atts, batch_oh, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(DATA_FILE)
    N = adata.n_obs
    logging.info("N=%d", N)

    Y_raw = adata.raw.X if adata.raw is not None else adata.X
    Y_obs = (Y_raw.toarray() if sp.issparse(Y_raw) else np.asarray(Y_raw)).astype(np.float32)
    gene_names = np.asarray(adata.var_names)
    del adata
    import gc; gc.collect()

    cls        = classify_perts(perturb_raw)
    perturb_raw = np.asarray(perturb_raw)
    coords     = pos_batched[:, :2].astype(np.float64)

    bin_id, n_bins = eu.build_niche_grid(coords, grid=6, loc_range=20.0)
    ntc_mask   = cls == "control"
    ntc_per_bin = np.bincount(bin_id[ntc_mask], minlength=n_bins)
    evaluable  = ntc_per_bin >= 30
    n_eval     = int(evaluable.sum())
    eval_bins  = np.where(evaluable)[0]
    logging.info("Evaluable niches: %d/%d", n_eval, n_bins)

    labels_for_panel = np.where(cls == "gene_KO", perturb_raw, "unassigned")
    panel = eu.select_perturbation_panel(labels_for_panel, bin_id, n_eval, min_cells_per_niche=20)
    logging.info("Panel (%d): %s", len(panel), panel)

    hvg_mask = eu.select_hvgs(Y_obs, n_top=2000)
    logging.info("HVGs: %d", int(hvg_mask.sum()))

    # 1. Split-half LFC reliability
    logging.info("Computing split-half LFC reliability...")
    reliability = {}
    for g in panel:
        keep = hvg_mask & (gene_names != g)
        rels = []
        for b in eval_bins:
            g_b  = np.where((perturb_raw == g) & (bin_id == b))[0]
            ntc_b = np.where(ntc_mask & (bin_id == b))[0]
            if len(g_b) < 20 or len(ntc_b) < 20:
                continue
            r, sb = eu.split_half_lfc_reliability(Y_obs[g_b], Y_obs[ntc_b],
                                                   gene_mask=keep, n_seeds=5)
            if np.isfinite(r):
                rels.append((r, sb))
        if rels:
            reliability[g] = dict(n_niches=len(rels),
                                  split_half_mean=float(np.nanmean([x[0] for x in rels])),
                                  spearman_brown_mean=float(np.nanmean([x[1] for x in rels])))

    # 2. Cell-state consistency
    logging.info("Computing cell-state consistency (same-pert vs cross-state pairs)...")
    cell_cons = eu.cell_state_consistency_test(
        Y_obs, perturb_raw, bin_id, panel,
        ntc_label="control", gene_mask=hvg_mask, n_pairs=500, n_seeds=5)

    # Summary table
    print("\n{:<10} {:>8} {:>12} {:>10} {:>10} {:>8}".format(
        "gene", "niches", "SB-ceiling", "same-rho", "cross-rho", "p-med"))
    print("-" * 62)
    for g in panel:
        rv = reliability.get(g, {}); cv = cell_cons.get(g, {})
        print("{:<10} {:>8} {:>12.3f} {:>10.4f} {:>10.4f} {:>8.3g}".format(
            g,
            rv.get("n_niches", "-"),
            rv.get("spearman_brown_mean", float("nan")),
            cv.get("same_mean", float("nan")),
            cv.get("cross_mean", float("nan")),
            cv.get("mw_pvalue_median", float("nan"))))

    out = {"panel": panel, "reliability": reliability, "cell_consistency": cell_cons}
    (OUTDIR / "diag_signal_detection.json").write_text(json.dumps(out, indent=2))
    logging.info("Written diag_signal_detection.json")

if __name__ == "__main__":
    main()
