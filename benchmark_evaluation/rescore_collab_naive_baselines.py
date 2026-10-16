#!/usr/bin/env python3
"""Rescore naive baselines under OUR LFC rule (not ntc20 / not LOO response transfer).

Baselines (user-specified):
  mean_target  -- cell-weighted mean expression of this KO across all slides
                 in the dataset (leave out this unit's real KO cells)
  mean_niche   -- mean expression of all cells in the evaluated niche/bin
  local_knn    -- each source cell <- mean of k spatially nearest donors
  gex_knn      -- each source cell <- mean of k nearest donors in 50-PC GEX

kNN donor pool: all other cells on the slide (no pert / niche / tissue filter).

Our rule:
  m_lfc = log(mean(pred)) - log(mean(source))
  o_lfc = KO vs unassigned (SPAC) or KP-control (PM)
  score HVG and panel, drop KO gene; Spearman + RMSE.
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_v3_eval_lib import ko_holdout_indices  # noqa: E402
from seen_residual_scoring import (  # noqa: E402
    build_gene_mask,
    load_hvg_indices,
    observed_lfc_pm_niche,
    observed_lfc_seen_patch_niche,
    observed_lfc_unseen_v3,
    pseudobulk_lfc,
    score_lfc_agreement,
    to_dense,
)
from spacseq_eval_common import UNASSIGNED_LABELS  # noqa: E402
from spacseq_multi_common import load_patch_roles  # noqa: E402

PREP_SPAC = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5/prepared")
SPAC_V3 = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5_ho_v3")
BENCH_V3 = PREP_SPAC / "benchmark_v3"
PM_PREP = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/prepared_panel1053ko"
)
PM_CONCERT = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/"
    "results_panel1053ko_niche/concert_ms_seen"
)
OUT = Path(
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/spac_seq"
)
MASTER_SPAC = OUT / "spac_seen_lfc_dist_per_record.csv"
MASTER_SPAC_UNSEEN = OUT / "spac_unseen_lfc_dist_per_record.csv"
MASTER_PM = OUT / "pm_seen_lfc_dist_per_record.csv"

STATE_SPAC_ROOTS = [
    Path("/n/home12/bof695/scratch_global/shared/Concert/spac_patch_concert_benchmark/outputs/state"),
]
STATE_PM_ROOTS = [
    Path(
        "/n/home12/bof695/scratch_global/shared/Concert/perturb_map_state_celcomen_benchmark/"
        "outputs/state_protocol_v1/seen_pert"
    ),
]

K_NN = 20
# Match CONCERT PM LFC-distance rescore default (--min-g-cells 5).
# With MIN_G=10 we dropped units CONCERT scored (in-bin control 5-9, source 6-9).
MIN_G = 5
# Sampled output (set by rescore_collab_naive_baselines_seen_genefix.py --sampled_output True): each
# SEEN SPAC prediction is drawn as Poisson counts at the source cells' observed library size, the same
# transform the models and the unseen reference strategies are scored under.
SAMPLED = False
SAMPLED_SEED_BASE = 0
MAX_SRC = 250
MAX_CAND = 2500
PCA_N = 50
PCA_FIT_MAX = 4000

METHOD_LABELS = {
    "mean_target": "Mean target",
    "mean_niche": "Mean niche",
    "local_knn": "Local kNN",
    "gex_knn": "GEX kNN",
    "state": "STATE",
}


def una_mask(canon: np.ndarray) -> np.ndarray:
    return np.isin(canon, list(UNASSIGNED_LABELS | {"unassigned"}))


def load_master_spac() -> pd.DataFrame:
    df = pd.read_csv(MASTER_SPAC)
    c = df[df["method"].astype(str).str.contains("concert", case=False)].copy()
    c["niche_key"] = c["niche_key"].astype(str)
    return c[["gene", "slide", "niche_key"]].drop_duplicates()


def load_master_spac_unseen() -> pd.DataFrame:
    df = pd.read_csv(MASTER_SPAC_UNSEEN)
    c = df[df["method"].astype(str).str.contains("concert", case=False)].copy()
    c["niche_key"] = c["niche_key"].astype(str)
    return c[["gene", "slide", "niche_key"]].drop_duplicates()


def load_master_pm() -> pd.DataFrame:
    df = pd.read_csv(MASTER_PM)
    c = df[df["method"].astype(str).str.contains("concert", case=False)].copy()
    c["niche_key"] = c["niche_key"].astype(str)
    return c[["gene", "slide", "niche_key"]].drop_duplicates()


def hvg_for_spac_slide(slide: str) -> np.ndarray | None:
    return load_hvg_indices(SPAC_V3 / "concert" / "seen" / "patch" / f"slide_{slide}")


def spac_source_idx(adata, role, niche: int) -> np.ndarray:
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    bin_id = adata.obs["bin_id"].to_numpy().astype(int)
    una = una_mask(canon)
    idx = np.where(una & (role == 1) & (bin_id == niche))[0]
    if len(idx) < MIN_G:
        idx = np.where(una & np.isin(role, (1, 2)) & (bin_id == niche))[0]
    return idx


def pm_source_idx(adata, niche_key: str) -> np.ndarray:
    bin_s, patch_s = niche_key.split(":")
    b, p = int(bin_s), int(patch_s)
    role = adata.obs["role"].to_numpy().astype(int)
    patch = adata.obs["patch_id"].to_numpy().astype(int)
    tbin = adata.obs["target_bin"].to_numpy().astype(int)
    return np.where((role == 1) & (patch == p) & (tbin == b))[0]


def record(method, gene, slide, niche_key, keep, sc):
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "gene": gene,
        "slide": str(slide),
        "niche_key": str(niche_key),
        "n_scored_genes": int(keep.sum()),
        "lfc_spearman": float(sc["spearman"]),
        "lfc_dist": float(sc["lfc_dist"]),
        # Same gene-bootstrap term every other method carries, so the per-gene bar
        # panels can show one comparable error instead of leaving these methods bare.
        "lfc_spearman_boot": float(sc.get("spearman_boot_std", float("nan"))),
        "lfc_dist_boot": float(sc.get("lfc_dist_boot_std", float("nan"))),
    }


def row_sum(X, idx: np.ndarray) -> tuple[np.ndarray, int]:
    if len(idx) == 0:
        return None, 0  # type: ignore[return-value]
    sub = X[idx]
    if hasattr(sub, "sum"):
        s = np.asarray(sub.sum(axis=0)).ravel().astype(np.float64)
    else:
        s = np.asarray(sub, dtype=np.float64).sum(axis=0)
    return s, int(len(idx))


def subsample(idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_n, replace=False))


def knn_predict_mean(src_X, cand_X, cand_z, src_z, k=K_NN):
    if len(cand_X) == 0 or len(src_X) == 0:
        return None
    kk = min(k, len(cand_X))
    nn = NearestNeighbors(n_neighbors=kk, algorithm="auto").fit(cand_z)
    _, ind = nn.kneighbors(src_z)
    return cand_X[ind].mean(axis=1)


def knn_donor_pool(n_obs: int, src_idx: np.ndarray):
    """All other cells on the slide -- no pert / niche / tissue filter."""
    src_set = set(np.asarray(src_idx, dtype=int).tolist())
    pool = np.array([i for i in range(n_obs) if i not in src_set], dtype=int)
    if len(pool) < MIN_G:
        pool = np.arange(n_obs, dtype=int)
    return pool, "all_cells"


def score_const_pred(mu: np.ndarray, src_X: np.ndarray, o: np.ndarray, keep: np.ndarray):
    """Broadcast a mean vector as prediction for every source cell."""
    pred = np.broadcast_to(mu, src_X.shape)
    m = pseudobulk_lfc(pred, src_X)
    return score_lfc_agreement(m[keep], o[keep])


def precompute_target_means_spac(slides: list[str], genes_needed: set[str]):
    """gene -> (sum_vec, n) over all KO cells of that gene across slides."""
    acc: dict[str, tuple[np.ndarray, int]] = {}
    gene_names = None
    for slide in slides:
        print(f"  [target-mean] accumulate slide {slide}", flush=True)
        adata = ad.read_h5ad(PREP_SPAC / f"slide_{slide}.h5ad")
        if gene_names is None:
            gene_names = np.asarray(adata.var_names).astype(str)
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        for gene in sorted(genes_needed):
            idx = np.where(canon == gene)[0]
            if len(idx) == 0:
                continue
            s, n = row_sum(adata.X, idx)
            if gene not in acc:
                acc[gene] = (s, n)
            else:
                prev_s, prev_n = acc[gene]
                acc[gene] = (prev_s + s, prev_n + n)
        del adata
        gc.collect()
    return acc, gene_names


def precompute_target_means_pm(slides: list[str], genes_needed: set[str]):
    acc: dict[str, tuple[np.ndarray, int]] = {}
    gene_names = None
    for slide in slides:
        print(f"  [target-mean] accumulate PM slide {slide}", flush=True)
        adata = ad.read_h5ad(PM_PREP / f"slide_{slide}.h5ad")
        if gene_names is None:
            gene_names = np.asarray(adata.var_names).astype(str)
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        tumor = None
        if "tissue" in adata.obs:
            tumor = adata.obs["tissue"].astype(str).to_numpy() == "tumor"
        for gene in sorted(genes_needed):
            m = canon == gene
            if tumor is not None:
                m = m & tumor
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            s, n = row_sum(adata.X, idx)
            if gene not in acc:
                acc[gene] = (s, n)
            else:
                prev_s, prev_n = acc[gene]
                acc[gene] = (prev_s + s, prev_n + n)
        del adata
        gc.collect()
    return acc, gene_names


def unit_leaveout_target_mean(acc, gene, unit_sum, unit_n):
    """Leave-out this unit's KO cells; if too few remain, use full target mean
    so every CONCERT unit still gets a score."""
    if gene not in acc:
        return None
    tot_s, tot_n = acc[gene]
    n = tot_n - unit_n
    if n >= MIN_G:
        return (tot_s - unit_sum) / n
    if tot_n >= MIN_G:
        return tot_s / tot_n
    return None


_STATE_DENIED = False


def try_load_state_spac(slide: str, niche: int, gene: str) -> np.ndarray | None:
    global _STATE_DENIED
    if _STATE_DENIED:
        return None
    for root in STATE_SPAC_ROOTS:
        for task in ("task_A_seen", "task_B_seen"):
            p = root / task / f"slide_{slide}" / f"niche{niche}_{gene}.h5ad"
            try:
                if not p.is_file():
                    continue
                return to_dense(ad.read_h5ad(p).X)
            except PermissionError:
                print(f"[state] permission denied -- skipping STATE: {p}", flush=True)
                _STATE_DENIED = True
                return None
            except (OSError, ValueError) as e:
                print(f"[state] skip {p}: {e}", flush=True)
                return None
    return None


def try_load_state_pm(slide: str, niche_key: str, gene: str) -> np.ndarray | None:
    global _STATE_DENIED
    if _STATE_DENIED:
        return None
    b, p = niche_key.split(":")
    fname = f"patch_{p}_bin_{b}_{gene}.h5ad"
    for root in STATE_PM_ROOTS:
        path = root / f"slide_{slide}" / fname
        try:
            if not path.is_file():
                continue
            return to_dense(ad.read_h5ad(path).X)
        except PermissionError:
            print(f"[state] permission denied -- skipping STATE: {path}", flush=True)
            _STATE_DENIED = True
            return None
        except (OSError, ValueError) as e:
            print(f"[state] skip {path}: {e}", flush=True)
            return None
    return None


def run_spac(methods: set[str], slides: set[str] | None = None) -> list[dict]:
    master_all = load_master_spac()
    # mean_target always pools KO cells across the full dataset, even for --slides runs
    all_slides = sorted(master_all["slide"].astype(str).unique())
    genes_needed = set(master_all["gene"].astype(str))
    master = master_all
    if slides:
        master = master[master["slide"].astype(str).isin(slides)].copy()

    target_acc = None
    if "mean_target" in methods:
        print("[SPAC] precomputing mean-target across ALL slides...", flush=True)
        target_acc, _ = precompute_target_means_spac(all_slides, genes_needed)

    rows: list[dict] = []
    for slide, gdf in master.groupby("slide"):
        print(f"[SPAC] slide {slide} units={len(gdf)}", flush=True)
        adata = ad.read_h5ad(PREP_SPAC / f"slide_{slide}.h5ad")
        role = load_patch_roles(PREP_SPAC, str(slide), adata)
        if role is None:
            print("  no patch roles -- skip", flush=True)
            del adata
            gc.collect()
            continue
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        bin_id = adata.obs["bin_id"].to_numpy().astype(int)
        genes = np.asarray(adata.var_names).astype(str)
        hvg = hvg_for_spac_slide(str(slide))
        if hvg is not None and len(hvg):
            hvg_cols = np.asarray(hvg, dtype=int)
            hvg_cols = hvg_cols[(hvg_cols >= 0) & (hvg_cols < adata.n_vars)]
        else:
            hvg_cols = np.arange(min(2000, adata.n_vars), dtype=int)
        keep_hvg_bool = np.zeros(adata.n_vars, dtype=bool)
        keep_hvg_bool[hvg_cols] = True

        # Niche means (all cells in niche)
        niche_mu: dict[int, np.ndarray] = {}
        if "mean_niche" in methods:
            for niche in sorted(gdf["niche_key"].astype(int).unique()):
                idx = np.where(bin_id == int(niche))[0]
                s, n = row_sum(adata.X, idx)
                if n >= MIN_G:
                    niche_mu[int(niche)] = s / n

        excl = []
        for r in gdf.itertuples(index=False):
            excl.extend(
                np.where((canon == str(r.gene)) & (bin_id == int(r.niche_key)))[0].tolist()
            )
        pca = None
        if "gex_knn" in methods:
            print(
                f"  fitting GEX PCA on HVG (G={len(hvg_cols)}, n_obs={adata.n_obs})...",
                flush=True,
            )
            keep_fit = np.ones(adata.n_obs, dtype=bool)
            if excl:
                keep_fit[np.unique(excl)] = False
            pool = np.where(keep_fit)[0]
            rng = np.random.default_rng(42)
            if len(pool) > PCA_FIT_MAX:
                pool = rng.choice(pool, size=PCA_FIT_MAX, replace=False)
            X_fit = to_dense(adata[pool].X[:, hvg_cols]).astype(np.float32)
            n_comp = int(min(PCA_N, len(pool) - 1, X_fit.shape[1]))
            if n_comp >= 2:
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X_fit.astype(np.float64))
            del X_fit
            print(f"  PCA ready n_comp={n_comp if pca is not None else 0}", flush=True)

        pool_stats = {"lono": 0, "same_gene": 0, "all_cells": 0}
        for ui, r in enumerate(gdf.itertuples(index=False)):
            gene, niche = str(r.gene), int(r.niche_key)
            o = observed_lfc_seen_patch_niche(
                adata, gene, niche, PREP_SPAC, str(slide), MIN_G
            )
            if o is None:
                continue
            keep = build_gene_mask(genes, gene, hvg) & keep_hvg_bool
            src = spac_source_idx(adata, role, niche)
            if len(src) < MIN_G:
                continue
            src_s = subsample(src, MAX_SRC, seed=hash((str(slide), niche, gene)) % (2**31))
            src_X_full = to_dense(adata[src_s].X).astype(np.float64)
            src_X_hvg = src_X_full[:, hvg_cols]
            keep_hvg = keep[hvg_cols]
            o_hvg = o[hvg_cols]

            if SAMPLED:
                from rescore_naive_baselines_unseen_v4_genefix import (
                    poisson_at_observed_depth, src_lib_sizes, unit_seed)
                lib = src_lib_sizes(adata, src_s)
                seed = unit_seed(SAMPLED_SEED_BASE, gene)

                def score_full(pred_full):
                    pred_full = poisson_at_observed_depth(pred_full, lib, seed)
                    m = pseudobulk_lfc(pred_full, src_X_full)
                    return score_lfc_agreement(m[keep], o[keep])

            if "mean_target" in methods and target_acc is not None:
                unit_idx = np.where((canon == gene) & (bin_id == niche))[0]
                u_sum, u_n = row_sum(adata.X, unit_idx)
                if u_sum is None:
                    u_sum, u_n = np.zeros(adata.n_vars), 0
                mu = unit_leaveout_target_mean(target_acc, gene, u_sum, u_n)
                if mu is not None:
                    sc = (score_full(np.broadcast_to(mu, src_X_full.shape)) if SAMPLED
                          else score_const_pred(mu, src_X_full, o, keep))
                    rows.append(record("mean_target", gene, slide, niche, keep, sc))

            if "mean_niche" in methods and niche in niche_mu:
                mu = niche_mu[niche]
                sc = (score_full(np.broadcast_to(mu, src_X_full.shape)) if SAMPLED
                      else score_const_pred(mu, src_X_full, o, keep))
                rows.append(record("mean_niche", gene, slide, niche, keep, sc))

            need_knn = "local_knn" in methods or "gex_knn" in methods
            if need_knn:
                cand, tag = knn_donor_pool(adata.n_obs, src_s)
                pool_stats[tag] = pool_stats.get(tag, 0) + 1
                cand_s = subsample(cand, MAX_CAND, seed=hash((str(slide), gene, tag)) % (2**31))
                cand_X = to_dense(adata[cand_s].X[:, hvg_cols]).astype(np.float64)
                # Poisson sampling renormalises over the gene axis, so the donor mean has to span
                # every gene; the neighbour search stays in the HVG / PCA space.
                val = to_dense(adata[cand_s].X).astype(np.float64) if SAMPLED else cand_X
                if "local_knn" in methods:
                    pred = knn_predict_mean(src_X_hvg, val, xy[cand_s], xy[src_s])
                    if pred is not None:
                        if SAMPLED:
                            sc = score_full(pred)
                        else:
                            m = pseudobulk_lfc(pred, src_X_hvg)
                            sc = score_lfc_agreement(m[keep_hvg], o_hvg[keep_hvg])
                        rows.append(record("local_knn", gene, slide, niche, keep, sc))
                if "gex_knn" in methods and pca is not None:
                    pred = knn_predict_mean(
                        src_X_hvg, val, pca.transform(cand_X), pca.transform(src_X_hvg)
                    )
                    if pred is not None:
                        if SAMPLED:
                            sc = score_full(pred)
                        else:
                            m = pseudobulk_lfc(pred, src_X_hvg)
                            sc = score_lfc_agreement(m[keep_hvg], o_hvg[keep_hvg])
                        rows.append(record("gex_knn", gene, slide, niche, keep, sc))
                del val
                del cand_X

            if "state" in methods:
                pred_mat = try_load_state_spac(str(slide), niche, gene)
                if pred_mat is not None and pred_mat.shape[1] == adata.n_vars:
                    n = min(pred_mat.shape[0], len(src_s))
                    m = pseudobulk_lfc(pred_mat[:n], src_X_full[:n])
                    sc = score_lfc_agreement(m[keep], o[keep])
                    rows.append(record("state", gene, slide, niche, keep, sc))

            del src_X_full, src_X_hvg
            if (ui + 1) % 5 == 0 or ui + 1 == len(gdf):
                print(f"  units {ui + 1}/{len(gdf)} rows={len(rows)}", flush=True)

        print(f"  knn donor tags: {pool_stats}", flush=True)
        del adata, xy, canon, bin_id, pca, niche_mu
        gc.collect()
    return rows


def spac_unseen_source_idx(adata) -> np.ndarray:
    canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
    return np.where(una_mask(canon))[0]


def hvg_for_spac_unseen(gene: str, slide: str) -> np.ndarray | None:
    # Prefer unit-specific HVG under concert unseen; fall back to slide seen HVG
    unit_dir = SPAC_V3 / "concert" / "unseen" / "unseen_pert" / f"{gene}__slide_{slide}"
    h = load_hvg_indices(unit_dir)
    if h is not None and len(h):
        return h
    return hvg_for_spac_slide(slide)


def run_spac_unseen(methods: set[str]) -> list[dict]:
    """SPAC unseen units from CONCERT master (gene x target slide, niche=-1)."""
    master = load_master_spac_unseen()
    genes_needed = set(master["gene"].astype(str))
    prep_slides = ["1-1", "1-2", "1-3", "1-4", "1-5"]
    all_slides = sorted({str(s) for s in master["slide"].unique()} | set(prep_slides))

    target_acc = None
    if "mean_target" in methods:
        print("[SPAC-unseen] precomputing mean-target across ALL slides...", flush=True)
        target_acc, _ = precompute_target_means_spac(all_slides, genes_needed)

    rows: list[dict] = []
    for slide, gdf in master.groupby("slide"):
        slide = str(slide)
        print(f"[SPAC-unseen] slide {slide} units={len(gdf)}", flush=True)
        adata = ad.read_h5ad(PREP_SPAC / f"slide_{slide}.h5ad")
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        genes = np.asarray(adata.var_names).astype(str)

        # Slide-wide mean = "mean niche" when niche_key is -1 (no niche)
        slide_mu = None
        if "mean_niche" in methods:
            s, n = row_sum(adata.X, np.arange(adata.n_obs))
            if n >= MIN_G:
                slide_mu = s / n

        pca = None
        # Fit PCA once on HVG from first unit's mask (approx); refit per unit if HVG differs
        for ui, r in enumerate(gdf.itertuples(index=False)):
            gene = str(r.gene)
            niche_key = str(r.niche_key)
            hvg = hvg_for_spac_unseen(gene, slide)
            if hvg is not None and len(hvg):
                hvg_cols = np.asarray(hvg, dtype=int)
                hvg_cols = hvg_cols[(hvg_cols >= 0) & (hvg_cols < adata.n_vars)]
            else:
                hvg_cols = np.arange(min(2000, adata.n_vars), dtype=int)
            keep_hvg_bool = np.zeros(adata.n_vars, dtype=bool)
            keep_hvg_bool[hvg_cols] = True

            ko_idx = ko_holdout_indices(BENCH_V3, gene, slide)
            if ko_idx is None or len(ko_idx) < MIN_G:
                print(f"  skip {gene}: no holdout KO", flush=True)
                continue
            o = observed_lfc_unseen_v3(adata, np.asarray(ko_idx, dtype=int), MIN_G)
            if o is None:
                continue
            keep = build_gene_mask(genes, gene, hvg) & keep_hvg_bool
            src = spac_unseen_source_idx(adata)
            if len(src) < MIN_G:
                continue
            src_s = subsample(src, MAX_SRC, seed=hash((slide, gene, "unseen")) % (2**31))
            src_X_full = to_dense(adata[src_s].X).astype(np.float64)
            src_X_hvg = src_X_full[:, hvg_cols]
            keep_hvg = keep[hvg_cols]
            o_hvg = o[hvg_cols]

            if "mean_target" in methods and target_acc is not None:
                # leave out this target-slide's KO cells
                unit_idx = np.asarray(ko_idx, dtype=int)
                u_sum, u_n = row_sum(adata.X, unit_idx)
                if u_sum is None:
                    u_sum, u_n = np.zeros(adata.n_vars), 0
                mu = unit_leaveout_target_mean(target_acc, gene, u_sum, u_n)
                if mu is not None:
                    sc = score_const_pred(mu, src_X_full, o, keep)
                    rows.append(record("mean_target", gene, slide, niche_key, keep, sc))

            if "mean_niche" in methods and slide_mu is not None:
                sc = score_const_pred(slide_mu, src_X_full, o, keep)
                rows.append(record("mean_niche", gene, slide, niche_key, keep, sc))

            need_knn = "local_knn" in methods or "gex_knn" in methods
            if need_knn:
                if "gex_knn" in methods and pca is None:
                    print(f"  fitting GEX PCA (G={len(hvg_cols)})...", flush=True)
                    pool = np.arange(adata.n_obs)
                    rng = np.random.default_rng(42)
                    if len(pool) > PCA_FIT_MAX:
                        pool = rng.choice(pool, size=PCA_FIT_MAX, replace=False)
                    X_fit = to_dense(adata[pool].X[:, hvg_cols]).astype(np.float32)
                    n_comp = int(min(PCA_N, len(pool) - 1, X_fit.shape[1]))
                    if n_comp >= 2:
                        pca = PCA(n_components=n_comp, random_state=42)
                        pca.fit(X_fit.astype(np.float64))
                    del X_fit
                cand, _ = knn_donor_pool(adata.n_obs, src_s)
                cand_s = subsample(cand, MAX_CAND, seed=hash((slide, gene, "all")) % (2**31))
                cand_X = to_dense(adata[cand_s].X[:, hvg_cols]).astype(np.float64)
                if "local_knn" in methods:
                    pred = knn_predict_mean(src_X_hvg, cand_X, xy[cand_s], xy[src_s])
                    if pred is not None:
                        m = pseudobulk_lfc(pred, src_X_hvg)
                        sc = score_lfc_agreement(m[keep_hvg], o_hvg[keep_hvg])
                        rows.append(record("local_knn", gene, slide, niche_key, keep, sc))
                if "gex_knn" in methods and pca is not None:
                    pred = knn_predict_mean(
                        src_X_hvg, cand_X, pca.transform(cand_X), pca.transform(src_X_hvg)
                    )
                    if pred is not None:
                        m = pseudobulk_lfc(pred, src_X_hvg)
                        sc = score_lfc_agreement(m[keep_hvg], o_hvg[keep_hvg])
                        rows.append(record("gex_knn", gene, slide, niche_key, keep, sc))
                del cand_X
            del src_X_full, src_X_hvg
            print(f"  {ui + 1}/{len(gdf)} {gene} rows={len(rows)}", flush=True)
        del adata, xy, canon, pca
        gc.collect()
    return rows


def run_pm(methods: set[str]) -> list[dict]:
    master = load_master_pm()
    all_slides = sorted(
        str(int(s)) if str(s).isdigit() else str(s) for s in master["slide"].unique()
    )
    genes_needed = set(master["gene"].astype(str))

    target_acc = None
    if "mean_target" in methods:
        print("[PM] precomputing mean-target across slides...", flush=True)
        target_acc, _ = precompute_target_means_pm(all_slides, genes_needed)

    rows: list[dict] = []
    for slide, gdf in master.groupby("slide"):
        slide = str(int(slide)) if str(slide).isdigit() else str(slide)
        print(f"[PM] slide {slide} units={len(gdf)}", flush=True)
        adata = ad.read_h5ad(PM_PREP / f"slide_{slide}.h5ad")
        X = to_dense(adata.X).astype(np.float64)
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        canon = adata.obs["perturbation_canon"].astype(str).to_numpy()
        bin_id = adata.obs["bin_id"].to_numpy().astype(int)
        genes = np.asarray(adata.var_names).astype(str)
        hvg = load_hvg_indices(PM_CONCERT / "seen_pert" / f"slide_{slide}")
        tumor = None
        if "tissue" in adata.obs:
            tumor = adata.obs["tissue"].astype(str).to_numpy() == "tumor"

        niche_mu: dict[int, np.ndarray] = {}
        if "mean_niche" in methods:
            for nk in gdf["niche_key"].astype(str).unique():
                b = int(str(nk).split(":")[0])
                if b in niche_mu:
                    continue
                idx = np.where(bin_id == b)[0]
                if tumor is not None:
                    idx = idx[tumor[idx]]
                if len(idx) >= MIN_G:
                    niche_mu[b] = X[idx].mean(axis=0)

        excl = []
        for r in gdf.itertuples(index=False):
            b = int(str(r.niche_key).split(":")[0])
            excl.extend(np.where((canon == str(r.gene)) & (bin_id == b))[0].tolist())
        pca = None
        if "gex_knn" in methods:
            keep_fit = np.ones(len(X), dtype=bool)
            if excl:
                keep_fit[np.unique(excl)] = False
            pool = np.where(keep_fit)[0]
            rng = np.random.default_rng(42)
            if len(pool) > PCA_FIT_MAX:
                pool = rng.choice(pool, size=PCA_FIT_MAX, replace=False)
            n_comp = int(min(PCA_N, len(pool) - 1, X.shape[1]))
            if n_comp >= 2:
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X[pool])

        pool_stats = {"lono": 0, "same_gene": 0, "all_cells": 0}
        for r in gdf.itertuples(index=False):
            gene, nk = str(r.gene), str(r.niche_key)
            b = int(nk.split(":")[0])
            o = observed_lfc_pm_niche(adata, gene, b, MIN_G)
            if o is None:
                continue
            keep = build_gene_mask(genes, gene, hvg)
            src = pm_source_idx(adata, nk)
            if len(src) < MIN_G:
                continue
            src_s = subsample(src, MAX_SRC, seed=hash((slide, nk, gene)) % (2**31))
            src_X = X[src_s]

            if "mean_target" in methods and target_acc is not None:
                unit_m = (canon == gene) & (bin_id == b)
                if tumor is not None:
                    unit_m = unit_m & tumor
                unit_idx = np.where(unit_m)[0]
                u_sum = X[unit_idx].sum(axis=0) if len(unit_idx) else np.zeros(X.shape[1])
                u_n = len(unit_idx)
                mu = unit_leaveout_target_mean(target_acc, gene, u_sum, u_n)
                if mu is not None:
                    sc = score_const_pred(mu, src_X, o, keep)
                    rows.append(record("mean_target", gene, slide, nk, keep, sc))

            if "mean_niche" in methods and b in niche_mu:
                sc = score_const_pred(niche_mu[b], src_X, o, keep)
                rows.append(record("mean_niche", gene, slide, nk, keep, sc))

            need_knn = "local_knn" in methods or "gex_knn" in methods
            if need_knn:
                cand, tag = knn_donor_pool(len(X), src_s)
                pool_stats[tag] = pool_stats.get(tag, 0) + 1
                cand_s = subsample(cand, MAX_CAND, seed=hash((slide, gene, tag)) % (2**31))
                if "local_knn" in methods:
                    pred = knn_predict_mean(src_X, X[cand_s], xy[cand_s], xy[src_s])
                    if pred is not None:
                        m = pseudobulk_lfc(pred, src_X)
                        sc = score_lfc_agreement(m[keep], o[keep])
                        rows.append(record("local_knn", gene, slide, nk, keep, sc))
                if "gex_knn" in methods and pca is not None:
                    pred = knn_predict_mean(
                        src_X, X[cand_s], pca.transform(X[cand_s]), pca.transform(src_X)
                    )
                    if pred is not None:
                        m = pseudobulk_lfc(pred, src_X)
                        sc = score_lfc_agreement(m[keep], o[keep])
                        rows.append(record("gex_knn", gene, slide, nk, keep, sc))

            if "state" in methods:
                pred_mat = try_load_state_pm(slide, nk, gene)
                if pred_mat is not None and pred_mat.shape[1] == X.shape[1]:
                    n = min(pred_mat.shape[0], len(src_s))
                    m = pseudobulk_lfc(pred_mat[:n], src_X[:n])
                    sc = score_lfc_agreement(m[keep], o[keep])
                    rows.append(record("state", gene, slide, nk, keep, sc))

        print(f"  knn donor tags: {pool_stats}", flush=True)
        del adata, X, pca
        gc.collect()
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "method_label", "gene", "slide", "niche_key",
        "n_scored_genes", "lfc_spearman", "lfc_dist",
        "lfc_spearman_boot", "lfc_dist_boot",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} n={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--methods",
        default="mean_target,mean_niche,local_knn,gex_knn",
    )
    ap.add_argument("--datasets", default="spac,spac_unseen,pm")
    ap.add_argument("--slides", default="", help="Comma-separated SPAC slides (optional)")
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--append-spac", action="store_true")
    args = ap.parse_args()
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    # aliases from older names
    alias = {"global_mean": "mean_target", "local_mean": "mean_niche"}
    methods = {alias.get(m, m) for m in methods}
    dsets = {d.strip() for d in args.datasets.split(",") if d.strip()}
    slide_set = {s.strip() for s in args.slides.split(",") if s.strip()} or None

    if "pm" in dsets:
        rows = run_pm(methods)
        write_csv(args.outdir / "pm_seen_naive_baselines_our_rule.csv", rows)
    if "spac" in dsets:
        rows = run_spac(methods, slides=slide_set)
        out_spac = args.outdir / "spac_seen_naive_baselines_our_rule.csv"
        if args.append_spac and out_spac.is_file() and rows:
            prev = pd.read_csv(out_spac)
            slides_done = {str(r["slide"]) for r in rows}
            prev = prev[~prev["slide"].astype(str).isin(slides_done)]
            merged = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
            write_csv(out_spac, merged.to_dict("records"))
        else:
            write_csv(out_spac, rows)
    if "spac_unseen" in dsets:
        rows = run_spac_unseen(methods)
        write_csv(args.outdir / "spac_unseen_naive_baselines_our_rule.csv", rows)

    frames = []
    for ds_name, fname in (
        ("pm_seen", "pm_seen_naive_baselines_our_rule.csv"),
        ("spac_seen", "spac_seen_naive_baselines_our_rule.csv"),
        ("spac_unseen", "spac_unseen_naive_baselines_our_rule.csv"),
    ):
        p = args.outdir / fname
        if p.is_file():
            frames.append(pd.read_csv(p).assign(dataset=ds_name))
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # drop obsolete method rows if mixed
        keep_m = {"mean_target", "mean_niche", "local_knn", "gex_knn", "state"}
        all_df = all_df[all_df["method"].isin(keep_m)]
        macro = (
            all_df.groupby(["dataset", "method", "method_label"], as_index=False)
            .agg(
                n=("lfc_spearman", "count"),
                lfc_rho=("lfc_spearman", "mean"),
                lfc_rmse=("lfc_dist", "mean"),
            )
        )
        outp = args.outdir / "naive_baselines_our_rule_macro.csv"
        macro.to_csv(outp, index=False)
        print(macro.to_string(index=False), flush=True)
        print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
