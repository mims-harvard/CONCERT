#!/usr/bin/env python3
"""Benchmark v4, Perturb-map within-niche (seen) arm: rescore every method with the v4 reference.

No inference and no retraining. For every record the predicted LFC is built exactly as the
reported rescorers build it, then scored twice against the observed LFC:
  *_v3 : the reported reference (reproduction check)
  *_v4 : control cells of patch role 0 (perturbmap_v4_scoring.pm_reference_v4)

Sources of the predicted LFC (all unchanged):
  model methods (CONCERT checkpoints, scGEN, Biolord, GEARS, Steamboat, CelComen):
      stored seen_pert/slide_*/model_lfc.json vectors, read as in rescore_pm_lfc_distance.py:60-132.
      These vectors are the scored LFCs of the eval (perturbmap_common.py:568), computed after the
      method's scoring-space normalisation (perturbmap_common.py:473-476), so no normalisation is
      applied here.
  reference strategies (mean target, mean niche, local kNN, GEX kNN):
      recomputed on the fly exactly as rescore_collab_naive_baselines.run_pm (lines 586-704);
      they are not stored on disk. Predictions depend on the source cells only, never on the
      reference, and are computed once and scored against both references.
  STATE: delivered prediction h5ad files, scored as rescore_state_our_rule.run_pm/align_score
      (lines 239-273, 78-122); no normalisation, as in the reported STATE rescore.

Output: <outdir>/pm_seen_v4_vs_v3_per_record.csv (long, one row per method x record).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

EVAL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perturbmap_v4_scoring as v4  # noqa: E402
import rescore_collab_naive_baselines as rcnb  # noqa: E402
import rescore_pm_lfc_distance as rpm  # noqa: E402
import rescore_state_our_rule as rsor  # noqa: E402
import seen_residual_scoring as srs  # noqa: E402

MIN_G = 5  # rescore_pm_lfc_distance.py:164 default, rescore_collab_naive_baselines.py:77
MODEL_METHODS = ["concert", "concert_niche_seen", "concert_v3_seen",
                 "scgen", "biolord", "gears", "steamboat", "celcomen"]
DEFAULT_OUT = EVAL / "multi_pm_ho_v4"


def _score_pair(pred_k, obs3_k, obs4_k):
    s3 = srs.score_lfc_agreement(pred_k, obs3_k) if obs3_k is not None else None
    s4 = srs.score_lfc_agreement(pred_k, obs4_k) if obs4_k is not None else None
    nan = float("nan")
    return dict(
        lfc_spearman_v3=s3["spearman"] if s3 else nan, lfc_dist_v3=s3["lfc_dist"] if s3 else nan,
        lfc_spearman_boot_v3=s3["spearman_boot_std"] if s3 else nan,
        lfc_dist_boot_v3=s3["lfc_dist_boot_std"] if s3 else nan,
        lfc_spearman_v4=s4["spearman"] if s4 else nan, lfc_dist_v4=s4["lfc_dist"] if s4 else nan,
        lfc_spearman_boot_v4=s4["spearman_boot_std"] if s4 else nan,
        lfc_dist_boot_v4=s4["lfc_dist_boot_std"] if s4 else nan,
    )


def model_rows(adata, slide: str) -> list[dict]:
    rows = []
    gene_names = np.asarray(adata.var_names)
    for method in MODEL_METHODS:
        root, label = rpm.METHODS[method]
        edir = root / "seen_pert" / f"slide_{slide}"
        mlfc_path = edir / "model_lfc.json"
        if not mlfc_path.is_file():
            continue
        hvg_idx = srs.load_hvg_indices(edir)
        mlfc = json.loads(mlfc_path.read_text()).get("L2", {})
        is_c = method.startswith("concert")
        axis = None
        if is_c:
            vl = next((len(v) for nv in mlfc.values() for v in nv.values() if v), 0)
            axis = srs.load_concert_gene_names_from_checkpoint(root, vl) if vl else None
            if axis is None:
                print(f"[warn] {method} slide {slide}: no gene axis", flush=True)
                continue
        for gene, niche_vecs in mlfc.items():
            keep = srs.build_gene_mask(axis if is_c else gene_names, gene, hvg_idx)
            for key, vec in niche_vecs.items():
                if not vec:
                    continue
                b = int(str(key).split(":")[0])
                if is_c:  # rescore_pm_lfc_distance.py:96-101
                    pred = np.asarray(vec, dtype=np.float64)
                    obs3 = srs.observed_lfc_on_gene_axis(adata, gene, b, axis, MIN_G)
                else:     # rescore_pm_lfc_distance.py:118-123
                    pred = srs.align_mlfc_vector(vec, gene_names, hvg_idx)
                    obs3 = srs.observed_lfc_pm_niche(adata, gene, b, MIN_G)
                if not np.isfinite(pred).any():
                    continue
                full4, info = v4.observed_lfc_pm_v4(adata, gene, key, MIN_G)
                obs4 = None if full4 is None else (v4.to_axis(full4, gene_names, axis) if is_c else full4)
                info.update(v4.reference_overlap_v3(adata, key, MIN_G, slide_fallback=not is_c))
                row = dict(method=method, method_label=label, gene=gene, slide=str(slide),
                           niche_key=str(key), n_scored_genes=int(keep.sum()),
                           scored_v3=obs3 is not None, scored_v4=obs4 is not None)
                row.update(_score_pair(pred[keep], None if obs3 is None else obs3[keep],
                                       None if obs4 is None else obs4[keep]))
                row.update(info)
                rows.append(row)
    return rows


def naive_rows(slides_units: pd.DataFrame, target_acc) -> list[dict]:
    """rescore_collab_naive_baselines.run_pm (586-704), scored against both references."""
    methods = {"mean_target", "mean_niche", "local_knn", "gex_knn"}
    rows = []
    sub_log = {"src_subsampled": 0, "cand_subsampled": 0}
    for slide, gdf in slides_units.groupby("slide"):
        slide = str(int(slide)) if str(slide).isdigit() else str(slide)
        adata = ad.read_h5ad(rcnb.PM_PREP / f"slide_{slide}.h5ad")
        X = rcnb.to_dense(adata.X).astype(np.float64)
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        bin_id = adata.obs["bin_id"].to_numpy().astype(int)
        genes = np.asarray(adata.var_names).astype(str)
        hvg = rcnb.load_hvg_indices(rcnb.PM_CONCERT / "seen_pert" / f"slide_{slide}")
        tumor = adata.obs["tissue"].astype(str).to_numpy() == "tumor" if "tissue" in adata.obs else None
        print(f"[naive] slide {slide} n_obs={adata.n_obs} units={len(gdf)}", flush=True)

        niche_mu = {}
        for nk in gdf["niche_key"].astype(str).unique():
            b = int(nk.split(":")[0])
            if b in niche_mu:
                continue
            idx = np.where(bin_id == b)[0]
            if tumor is not None:
                idx = idx[tumor[idx]]
            if len(idx) >= rcnb.MIN_G:
                niche_mu[b] = X[idx].mean(axis=0)

        excl = []
        for r in gdf.itertuples(index=False):
            b = int(str(r.niche_key).split(":")[0])
            excl.extend(np.where((canon == str(r.gene)) & (bin_id == b))[0].tolist())
        keep_fit = np.ones(len(X), dtype=bool)
        if excl:
            keep_fit[np.unique(excl)] = False
        pool = np.where(keep_fit)[0]
        rng = np.random.default_rng(42)
        if len(pool) > rcnb.PCA_FIT_MAX:
            pool = rng.choice(pool, size=rcnb.PCA_FIT_MAX, replace=False)
        n_comp = int(min(rcnb.PCA_N, len(pool) - 1, X.shape[1]))
        pca = None
        if n_comp >= 2:
            pca = rcnb.PCA(n_components=n_comp, random_state=42)
            pca.fit(X[pool])

        for r in gdf.itertuples(index=False):
            gene, nk = str(r.gene), str(r.niche_key)
            b = int(nk.split(":")[0])
            o3 = rcnb.observed_lfc_pm_niche(adata, gene, b, rcnb.MIN_G)
            if o3 is None:
                continue
            keep = rcnb.build_gene_mask(genes, gene, hvg)
            src = rcnb.pm_source_idx(adata, nk)
            if len(src) < rcnb.MIN_G:
                continue
            o4, info = v4.observed_lfc_pm_v4(adata, gene, nk, rcnb.MIN_G, source_idx=src)
            info.update(v4.reference_overlap_v3(adata, nk, rcnb.MIN_G, slide_fallback=True))
            if len(src) > rcnb.MAX_SRC:
                sub_log["src_subsampled"] += 1
            src_s = rcnb.subsample(src, rcnb.MAX_SRC, seed=hash((slide, nk, gene)) % (2**31))
            src_X = X[src_s]

            def emit(method, m_lfc):
                row = dict(method=method, method_label=rcnb.METHOD_LABELS[method], gene=gene, slide=slide,
                           niche_key=nk, n_scored_genes=int(keep.sum()), scored_v3=True, scored_v4=o4 is not None)
                row.update(_score_pair(m_lfc[keep], o3[keep], None if o4 is None else o4[keep]))
                row.update(info)
                rows.append(row)

            unit_m = (canon == gene) & (bin_id == b)
            if tumor is not None:
                unit_m = unit_m & tumor
            unit_idx = np.where(unit_m)[0]
            u_sum = X[unit_idx].sum(axis=0) if len(unit_idx) else np.zeros(X.shape[1])
            mu = rcnb.unit_leaveout_target_mean(target_acc, gene, u_sum, len(unit_idx))
            if mu is not None:
                emit("mean_target", srs.pseudobulk_lfc(np.broadcast_to(mu, src_X.shape), src_X))
            if b in niche_mu:
                emit("mean_niche", srs.pseudobulk_lfc(np.broadcast_to(niche_mu[b], src_X.shape), src_X))
            cand, _tag = rcnb.knn_donor_pool(len(X), src_s)
            if len(cand) > rcnb.MAX_CAND:
                sub_log["cand_subsampled"] += 1
            cand_s = rcnb.subsample(cand, rcnb.MAX_CAND, seed=hash((slide, gene, "all_cells")) % (2**31))
            pred = rcnb.knn_predict_mean(src_X, X[cand_s], xy[cand_s], xy[src_s])
            if pred is not None:
                emit("local_knn", srs.pseudobulk_lfc(pred, src_X))
            if pca is not None:
                pred = rcnb.knn_predict_mean(src_X, X[cand_s], pca.transform(X[cand_s]), pca.transform(src_X))
                if pred is not None:
                    emit("gex_knn", srs.pseudobulk_lfc(pred, src_X))
        del adata, X
    print(f"[naive] subsampling events (hash-seeded, not reproducible across processes): {sub_log}", flush=True)
    return rows


def state_rows(units: pd.DataFrame) -> list[dict]:
    """rescore_state_our_rule.run_pm (239-273), scored against both references."""
    rows = []
    scale_log = []
    for slide, gdf in units.groupby("slide"):
        slide = str(int(slide)) if str(slide).isdigit() else str(slide)
        adata = ad.read_h5ad(rsor.PM_PREP / f"slide_{slide}.h5ad")
        genes = np.asarray(adata.var_names).astype(str)
        hvg = rsor.load_hvg_indices(rsor.PM_CONCERT / "seen_pert" / f"slide_{slide}")
        for r in gdf.itertuples(index=False):
            gene, nk = str(r.gene), str(r.niche_key)
            b, p = nk.split(":")
            path = rsor.STATE_PM / f"slide_{slide}" / f"patch_{p}_bin_{b}_{gene}.h5ad"
            if not path.is_file():
                print(f"[state] missing {path}", flush=True)
                continue
            o3 = rsor.observed_lfc_pm_niche(adata, gene, int(b), rsor.MIN_G)
            if o3 is None:
                continue
            keep = rsor.build_gene_mask(genes, gene, hvg)
            src = v4.pm_source_idx(adata, nk)
            o4, info = v4.observed_lfc_pm_v4(adata, gene, nk, rsor.MIN_G, source_idx=src)
            info.update(v4.reference_overlap_v3(adata, nk, rsor.MIN_G, slide_fallback=True))
            pred = ad.read_h5ad(path)
            sidx = pred.obs["source_idx"].astype(int).to_numpy()
            # STATE's own source cells must be the record's source cells and outside the v4 reference.
            b4, _ = v4.pm_reference_v4(adata, int(b), rsor.MIN_G)
            assert len(np.intersect1d(sidx, b4)) == 0, "STATE source cells inside v4 reference"
            info["state_src_equals_role1_patch"] = bool(set(sidx.tolist()) == set(src.tolist()))
            s3 = rsor.align_score(pred, adata, o3, keep)
            s4 = rsor.align_score(pred, adata, o4, keep) if o4 is not None else None
            common = [g for g in np.asarray(pred.var_names).astype(str) if g in set(genes)]
            pX = np.asarray(rsor.to_dense(pred[:, common].X), dtype=np.float64)
            sX = np.asarray(rsor.to_dense(adata[sidx][:, common].X), dtype=np.float64)
            scale_log.append(float(pX.mean() / sX.mean()))
            del pred
            if s3 is None:
                continue
            nan = float("nan")
            row = dict(method="state", method_label="STATE", gene=gene, slide=slide, niche_key=nk,
                       n_scored_genes=int(s3["n_scored_genes"]), scored_v3=True, scored_v4=s4 is not None,
                       lfc_spearman_v3=s3["spearman"], lfc_dist_v3=s3["lfc_dist"],
                       lfc_spearman_boot_v3=s3["spearman_boot_std"], lfc_dist_boot_v3=s3["lfc_dist_boot_std"],
                       lfc_spearman_v4=s4["spearman"] if s4 else nan, lfc_dist_v4=s4["lfc_dist"] if s4 else nan,
                       lfc_spearman_boot_v4=s4["spearman_boot_std"] if s4 else nan,
                       lfc_dist_boot_v4=s4["lfc_dist_boot_std"] if s4 else nan)
            row.update(info)
            rows.append(row)
        del adata
    if scale_log:
        print(f"[state] mean(pred)/mean(source expr) over records: median {np.median(scale_log):.4f}, "
              f"range {min(scale_log):.4f}-{max(scale_log):.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print("PYTHONHASHSEED =", os.environ.get("PYTHONHASHSEED"), flush=True)

    rows = []
    for slide in rpm.SLIDES:
        adata = ad.read_h5ad(rpm.PREP / f"slide_{slide}.h5ad")
        role = v4.pm_roles(adata)
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        print(f"[slide {slide}] n_obs={adata.n_obs} role counts={dict(zip(*np.unique(role, return_counts=True)))} "
              f"role1 control={int(((role == 1) & (canon == 'control')).sum())}/{int((role == 1).sum())} "
              f"role2 by label={pd.Series(canon[role == 2]).value_counts().to_dict()}", flush=True)
        rows.extend(model_rows(adata, slide))
        del adata

    units = rcnb.load_master_pm()  # the CONCERT unit set the reported reference strategies used
    units["slide"] = units["slide"].astype(str)
    target_acc, _ = rcnb.precompute_target_means_pm(
        sorted(units["slide"].unique()), set(units["gene"].astype(str)))
    rows.extend(naive_rows(units, target_acc))
    rows.extend(state_rows(units))

    df = pd.DataFrame(rows)
    out = args.outdir / "pm_seen_v4_vs_v3_per_record.csv"
    assert not out.exists() or "multi_pm_ho_v4" in str(out)
    df.to_csv(out, index=False)
    print("wrote", out, len(df), flush=True)
    print(df.groupby("method")[["lfc_spearman_v3", "lfc_spearman_v4", "lfc_dist_v3", "lfc_dist_v4"]]
          .agg(["mean", "count"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
