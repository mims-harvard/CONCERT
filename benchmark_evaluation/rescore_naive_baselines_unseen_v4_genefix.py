#!/usr/bin/env python3
"""Reference strategies (mean target, mean niche, local kNN, GEX kNN) for the SPAC-seq unseen arm, v4.

Scoring rule is the v4 eval rule: units = benchmark_v4/eval_panel.json (niche_key -1); observed LFC =
masked KO on the target slide vs unassigned cells outside patch roles 1/2
(benchmark_v4_eval_lib.unseen_reference_idx_v4); genes = slide HVG rank mask (2000) minus the KO gene;
m_lfc = log mean(pred) - log mean(source); Spearman + RMSE via seen_residual_scoring.score_lfc_agreement.

Information available to each estimator (same information the models are trained on):
  * source cells    = benchmark_v4/source_cells/slide_T.json, the exact 5000 cells every model is
                      scored on (no further subsampling);
  * mean target     = mean expression of this knockout's cells on its DONOR slides only
                      (unit["donor_slides"]); target-slide KO cells are never read;
  * mean niche      = mean expression of the target-slide cells in the v4 unseen training manifest
                      (benchmark_v4/train_manifest_unseen_union.json);
  * local / GEX kNN = donors drawn from the target-slide cells in the v4 unseen training manifest;
                      the GEX PCA is fitted on those cells too.
The training manifest excludes the patch-holdout unassigned cells and the masked KO cells of every
panel knockout, and the script asserts that no donor pool touches a masked KO cell or a source cell.
Estimator helpers (kNN mean, constant-prediction scoring, record format) come unchanged from
rescore_collab_naive_baselines.py.
Output: <outdir>/spac_unseen_naive_baselines_our_rule.csv

GENEFIX copy (2026-09-14) of rescore_naive_baselines_unseen_v4.py. One change only: the scored-gene mask
is keep = hvg_mask & (gene != KO) & (gene in CONCERT's model gene list), the shared rule applied to every
method (genefix_wrap.py). The estimators themselves (donor pools, kNN features = the slide HVG columns, PCA)
are unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_collab_naive_baselines as nb  # noqa: E402
import spacseq_eval_common as sec  # noqa: E402
from benchmark_v4_eval_lib import (  # noqa: E402
    filter_units,
    ko_holdout_indices,
    load_eval_panel,
    load_source_cells_v4,
    unseen_reference_idx_v4,
)
from seen_residual_scoring import pseudobulk_lfc, score_lfc_agreement, to_dense  # noqa: E402

PREP = nb.PREP_SPAC
from genefix_wrap import load_concert_genes  # noqa: E402

CONCERT_GENES = np.asarray(load_concert_genes()).astype(str)
SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]


def poisson_at_observed_depth(pred_logcp10k: np.ndarray, lib_sizes: np.ndarray, seed: int) -> np.ndarray:
    """Poisson draw of a log-CP10K prediction at each source cell's observed library size.

    Same transform the models are scored under (run_benchmark_v4_unseen_eval.
    poisson_at_observed_depth_scoring_space) with the scoring-space conversion skipped, because the
    reference-strategy predictions are means of adata.X and are already in log-CP10K.
    """
    from method_input_lib import LOG_CP10K_CLIP, raw_counts_to_logcp10k

    rel = np.expm1(np.clip(np.asarray(pred_logcp10k, dtype=np.float64), 0.0, LOG_CP10K_CLIP))
    rel = rel / np.maximum(rel.sum(1, keepdims=True), 1e-12)
    lib = np.asarray(lib_sizes, dtype=np.float64).reshape(-1, 1)
    counts = np.random.default_rng(int(seed)).poisson(rel * lib)
    return raw_counts_to_logcp10k(counts)


def unit_seed(base: int, gene: str) -> int:
    return int(base) + (zlib.crc32(gene.encode()) % 997)


def src_lib_sizes(adata, src: np.ndarray) -> np.ndarray:
    assert "n_counts" in adata.obs.columns, "prepared slide lacks obs['n_counts'] (raw library size)"
    return adata.obs["n_counts"].to_numpy()[src].astype(np.float64)


def donor_target_means(units: list[dict]) -> dict[str, np.ndarray]:
    """gene -> mean expression over its KO cells on its donor slides (target slide excluded)."""
    acc: dict[str, list] = {}
    for slide in SLIDES:
        genes_here = [u["gene"] for u in units if slide in u["donor_slides"] and slide != u["target_slide"]]
        if not genes_here:
            continue
        print(f"  [mean target] donor slide {slide}: {len(genes_here)} knockouts", flush=True)
        adata = ad.read_h5ad(PREP / f"slide_{slide}.h5ad")
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        for g in genes_here:
            idx = np.where(canon == g)[0]
            if len(idx) == 0:
                continue
            s, n = nb.row_sum(adata.X, idx)
            prev = acc.get(g)
            acc[g] = [s, n] if prev is None else [prev[0] + s, prev[1] + n]
        del adata
        gc.collect()
    return {g: s / n for g, (s, n) in acc.items() if n >= nb.MIN_G}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_dir", type=Path, default=PREP / "benchmark_v4")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--methods", default="mean_target,mean_niche,local_knn,gex_knn")
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--n_hvgs", type=int, default=2000)
    # Sampled output: draw Poisson counts at each source cell's observed library size from the
    # baseline's predicted profile, the transform every model is scored under with --sampled_output.
    ap.add_argument("--sampled_output", type=lambda v: str(v).lower() == "true", default=False)
    ap.add_argument("--sampled_seed_base", type=int, default=0)
    args = ap.parse_args()
    assert args.benchmark_dir.name != "benchmark_v3"
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    panel = load_eval_panel(args.benchmark_dir)
    units = filter_units(panel, args.units)
    manifest = json.loads((args.benchmark_dir / "train_manifest_unseen_union.json").read_text())
    assert manifest["version"] == "v4" and manifest["regime"] == "unseen_union"
    _, sa = sec.import_eval_utils()

    tmeans = donor_target_means(units) if "mean_target" in methods else {}

    rows: list[dict] = []
    by_slide: dict[str, list[dict]] = {}
    for u in units:
        by_slide.setdefault(u["target_slide"], []).append(u)

    for slide, sunits in sorted(by_slide.items()):
        print(f"[v4 unseen reference strategies] slide {slide} units={len(sunits)}", flush=True)
        adata = ad.read_h5ad(PREP / f"slide_{slide}.h5ad")
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        genes = np.asarray(adata.var_names).astype(str)
        hvg_mask = sec.select_hvgs(adata, args.n_hvgs, sa)
        hvg_cols = np.where(hvg_mask)[0]
        in_concert = np.isin(genes, CONCERT_GENES)
        print(f"  genefix: scored genes = HVG {int(hvg_mask.sum())} AND CONCERT {int(in_concert.sum())} "
              f"-> {int((hvg_mask & in_concert).sum())} (before removing the KO gene)", flush=True)
        ref_idx = unseen_reference_idx_v4(adata, PREP, slide)
        y_ref = to_dense(adata[ref_idx].X)
        src = load_source_cells_v4(args.benchmark_dir, slide)
        train_idx = np.asarray(manifest["per_slide"][slide]["train_idx"], dtype=int)

        # masked KO cells of ALL panel knockouts on this slide: must never be in a donor pool
        masked = set()
        for u in panel["eval_units"]:
            if u["target_slide"] == slide:
                masked.update(ko_holdout_indices(args.benchmark_dir, u["gene"], slide).tolist())
        assert not (set(train_idx.tolist()) & masked), "training manifest contains masked KO cells"
        assert not (set(train_idx.tolist()) & set(src.tolist())), "training manifest contains source cells"
        print(f"  source={len(src)} train_pool={len(train_idx)} masked_ko={len(masked)} ref={len(ref_idx)}", flush=True)

        src_X_full = to_dense(adata[src].X).astype(np.float64)
        src_X_hvg = src_X_full[:, hvg_cols]

        niche_mu = None
        if "mean_niche" in methods:
            s, n = nb.row_sum(adata.X, train_idx)
            niche_mu = s / n if n >= nb.MIN_G else None

        pca = None
        if "gex_knn" in methods:
            rng = np.random.default_rng(42)
            fit = train_idx if len(train_idx) <= nb.PCA_FIT_MAX else rng.choice(train_idx, nb.PCA_FIT_MAX, replace=False)
            X_fit = to_dense(adata[fit].X[:, hvg_cols]).astype(np.float64)
            n_comp = int(min(nb.PCA_N, len(fit) - 1, X_fit.shape[1]))
            if n_comp >= 2:
                pca = PCA(n_components=n_comp, random_state=42).fit(X_fit)
            del X_fit

        cand_X = cand_s = cand_X_full = None
        if "local_knn" in methods or "gex_knn" in methods:
            cand_s = nb.subsample(train_idx, nb.MAX_CAND, seed=hash((slide, "train_pool")) % (2**31))
            cand_X = to_dense(adata[cand_s].X[:, hvg_cols]).astype(np.float64)
            if args.sampled_output:
                # Poisson sampling renormalises over the gene axis, so the donor mean has to span
                # every gene; the neighbour search stays in the HVG / PCA space it always used.
                cand_X_full = to_dense(adata[cand_s].X).astype(np.float64)
        # kNN predictions depend only on (source, donor pool), not on the knockout: compute once per slide
        m_local = m_gex = None
        p_local = p_gex = None
        if "local_knn" in methods:
            val = cand_X_full if args.sampled_output else cand_X
            pred = nb.knn_predict_mean(src_X_hvg, val, xy[cand_s], xy[src])
            if args.sampled_output:
                p_local = pred
            else:
                m_local = None if pred is None else pseudobulk_lfc(pred, src_X_hvg)
        if "gex_knn" in methods and pca is not None:
            val = cand_X_full if args.sampled_output else cand_X
            pred = nb.knn_predict_mean(src_X_hvg, val, pca.transform(cand_X), pca.transform(src_X_hvg))
            if args.sampled_output:
                p_gex = pred
            else:
                m_gex = None if pred is None else pseudobulk_lfc(pred, src_X_hvg)
        lib = src_lib_sizes(adata, src) if args.sampled_output else None

        for ui, u in enumerate(sunits):
            gene = u["gene"]
            niche_key = "-1"
            ko_idx = ko_holdout_indices(args.benchmark_dir, gene, slide)
            if len(ko_idx) < nb.MIN_G or len(ref_idx) < nb.MIN_G:
                print(f"  skip {gene}: too few cells", flush=True)
                continue
            o = pseudobulk_lfc(to_dense(adata[ko_idx].X), y_ref)
            keep = hvg_mask & (genes != gene) & in_concert
            keep_hvg = keep[hvg_cols]
            o_hvg = o[hvg_cols]

            seed = unit_seed(args.sampled_seed_base, gene)

            def score_full(pred_full):
                """Score a full-gene prediction matrix against the source cells."""
                if args.sampled_output:
                    pred_full = poisson_at_observed_depth(pred_full, lib, seed)
                m = pseudobulk_lfc(pred_full, src_X_full)
                return score_lfc_agreement(m[keep], o[keep])

            if "mean_target" in methods and gene in tmeans:
                rows.append(nb.record("mean_target", gene, slide, niche_key, keep,
                                      score_full(np.broadcast_to(tmeans[gene], src_X_full.shape))
                                      if args.sampled_output else
                                      nb.score_const_pred(tmeans[gene], src_X_full, o, keep)))
            if "mean_niche" in methods and niche_mu is not None:
                rows.append(nb.record("mean_niche", gene, slide, niche_key, keep,
                                      score_full(np.broadcast_to(niche_mu, src_X_full.shape))
                                      if args.sampled_output else
                                      nb.score_const_pred(niche_mu, src_X_full, o, keep)))
            if p_local is not None:
                rows.append(nb.record("local_knn", gene, slide, niche_key, keep, score_full(p_local)))
            elif m_local is not None:
                rows.append(nb.record("local_knn", gene, slide, niche_key, keep,
                                      score_lfc_agreement(m_local[keep_hvg], o_hvg[keep_hvg])))
            if p_gex is not None:
                rows.append(nb.record("gex_knn", gene, slide, niche_key, keep, score_full(p_gex)))
            elif m_gex is not None:
                rows.append(nb.record("gex_knn", gene, slide, niche_key, keep,
                                      score_lfc_agreement(m_gex[keep_hvg], o_hvg[keep_hvg])))
            print(f"  {ui + 1}/{len(sunits)} {gene} rows={len(rows)}", flush=True)
        del adata, xy, y_ref, pca, cand_X, cand_X_full, src_X_full, src_X_hvg
        gc.collect()

    nb.write_csv(args.outdir / "spac_unseen_naive_baselines_our_rule.csv", rows)


if __name__ == "__main__":
    main()
