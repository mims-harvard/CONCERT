#!/usr/bin/env python3
"""Mean target reference strategy with a capped number of knockout cells, SPAC-seq v4, both arms, sampled output.

The published Mean target averages every knockout cell of the full prepared slides (median 560 cells
unseen, 973 seen), while the trained models see the same knockout through the 50,000-cell training
manifest (median 18 unseen, 47 seen). This sweep rebuilds the Mean target from a random subset of
N knockout cells and scores it under the unchanged rule of
  unseen: rescore_naive_baselines_unseen_v4_genefix.py (donor slides only, v4 source cells + reference)
  seen:   rescore_collab_naive_baselines.py via the genefix wrapper (all slides, unit cells left out)
with Poisson output at the observed source library size (--sampled_output True in both scripts).

Pools (per knockout unit):
  all       every eligible knockout cell (reproduces the published Mean target; validation row)
  manifest  eligible knockout cells inside train_idx of the matching training manifest
  n<N>      N cells drawn uniformly from the eligible pool (the whole pool when it is smaller)
Each n<N> pool is drawn for several subsampling seeds; the Poisson seed is the published one.
Output: <outdir>/mean_target_ncells_{seen,unseen}.csv   (run with PYTHONHASHSEED=0)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

EVAL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark_evaluation"))
import rescore_collab_naive_baselines_seen_genefix  # noqa: E402,F401  (installs the genefix gene mask on nb)
import rescore_collab_naive_baselines as nb  # noqa: E402
import rescore_naive_baselines_unseen_v4_genefix as uv  # noqa: E402
import spacseq_eval_common as sec  # noqa: E402
from benchmark_v4_eval_lib import (  # noqa: E402
    ko_holdout_indices, load_eval_panel, load_source_cells_v4, unseen_reference_idx_v4)
from seen_residual_scoring import (  # noqa: E402
    observed_lfc_seen_patch_niche, pseudobulk_lfc, score_lfc_agreement, to_dense)
from spacseq_multi_common import load_patch_roles  # noqa: E402

PREP = nb.PREP_SPAC
B4 = PREP / "benchmark_v4"
SLIDES = uv.SLIDES


def pools_for(eligible: dict[str, np.ndarray], train: dict[str, set], ns: list[int], seeds: list[int],
              key: str) -> dict[tuple, dict[str, np.ndarray]]:
    """(config, seed) -> {slide: selected cell indices}."""
    out = {("all", -1): eligible,
           ("manifest", -1): {s: np.asarray([i for i in v if i in train[s]], dtype=int) for s, v in eligible.items()}}
    flat = np.concatenate([np.stack([np.full(len(v), si), v], 1) for si, (s, v) in enumerate(eligible.items())]) \
        if eligible else np.zeros((0, 2), dtype=int)
    names = list(eligible)
    for n in ns:
        for sd in seeds:
            rng = np.random.default_rng([sd, n, abs(hash(key)) % (2**31)])
            pick = flat if len(flat) <= n else flat[rng.choice(len(flat), n, replace=False)]
            out[(f"n{n}", sd)] = {s: np.sort(pick[pick[:, 0] == si, 1]) for si, s in enumerate(names)}
    return out


def accumulate(selections: dict, slide_X: dict) -> dict:
    """selection key -> mean vector (None when fewer than MIN_G cells)."""
    means = {}
    for k, per_slide in selections.items():
        tot, n = None, 0
        for s, idx in per_slide.items():
            if len(idx) == 0:
                continue
            v, c = nb.row_sum(slide_X[s], idx)
            tot = v if tot is None else tot + v
            n += c
        means[k] = (tot / n, n) if n >= nb.MIN_G else (None, n)
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--ns", default="5,10,20,50,100,200,500")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="unseen,seen")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    arms = args.arms.split(",")

    obs, slide_X = {}, {}
    for s in SLIDES:
        a = ad.read_h5ad(PREP / f"slide_{s}.h5ad")
        obs[s] = (a.obs["perturbation_canon"].astype(str).to_numpy(), a.obs["bin_id"].to_numpy().astype(int))
        slide_X[s] = a.X
        del a
        print(f"loaded slide {s}", flush=True)
    man = {r: json.loads((B4 / f"train_manifest_{r}.json").read_text()) for r in ("seen", "unseen_union")}
    train = {r: {s: set(m["per_slide"][s]["train_idx"]) for s in SLIDES} for r, m in man.items()}

    # unseen
    if "unseen" in arms:
        panel = load_eval_panel(B4)
        _, sa = sec.import_eval_utils()
        rows = []
        for T in sorted({u["target_slide"] for u in panel["eval_units"]}):
            a = ad.read_h5ad(PREP / f"slide_{T}.h5ad")
            genes = np.asarray(a.var_names).astype(str)
            hvg_mask = sec.select_hvgs(a, 2000, sa)
            in_concert = np.isin(genes, uv.CONCERT_GENES)
            ref_idx = unseen_reference_idx_v4(a, PREP, T)
            y_ref = to_dense(a[ref_idx].X)
            src = load_source_cells_v4(B4, T)
            src_X = to_dense(a[src].X).astype(np.float64)
            lib = uv.src_lib_sizes(a, src)
            for u in [u for u in panel["eval_units"] if u["target_slide"] == T]:
                g = u["gene"]
                ko_idx = ko_holdout_indices(B4, g, T)
                if len(ko_idx) < nb.MIN_G:
                    continue
                o = pseudobulk_lfc(to_dense(a[ko_idx].X), y_ref)
                keep = hvg_mask & (genes != g) & in_concert
                elig = {s: np.where(obs[s][0] == g)[0] for s in u["donor_slides"] if s != T}
                means = accumulate(pools_for(elig, train["unseen_union"], ns, seeds, f"U{g}{T}"), slide_X)
                seed = uv.unit_seed(0, g)
                for (cfg, sd), (mu, n) in means.items():
                    if mu is None:
                        continue
                    pred = uv.poisson_at_observed_depth(np.broadcast_to(mu, src_X.shape), lib, seed)
                    sc = score_lfc_agreement(pseudobulk_lfc(pred, src_X)[keep], o[keep], n_boot=0)
                    rows.append(dict(arm="unseen", config=cfg, sub_seed=sd, n_cells=n, gene=g, slide=T,
                                     niche_key="-1", lfc_spearman=sc["spearman"], lfc_dist=sc["lfc_dist"]))
                print(f"  unseen {T} {g} done ({len(rows)} rows)", flush=True)
            del a, y_ref, src_X
            gc.collect()
        pd.DataFrame(rows).to_csv(args.outdir / "mean_target_ncells_unseen.csv", index=False)

    # seen
    if "seen" in arms:
        nb.MASTER_SPAC = EVAL / "v4_aggregate_sampled" / "spac_seen_lfc_dist_per_record.csv"
        master = nb.load_master_spac()
        rows = []
        for slide, gdf in master.groupby("slide"):
            slide = str(slide)
            a = ad.read_h5ad(PREP / f"slide_{slide}.h5ad")
            role = load_patch_roles(PREP, slide, a)
            canon, bin_id = obs[slide]
            genes = np.asarray(a.var_names).astype(str)
            hvg = nb.hvg_for_spac_slide(slide)
            hvg_cols = np.asarray(hvg, dtype=int)
            hvg_cols = hvg_cols[(hvg_cols >= 0) & (hvg_cols < a.n_vars)]
            keep_hvg_bool = np.zeros(a.n_vars, dtype=bool)
            keep_hvg_bool[hvg_cols] = True
            for r in gdf.itertuples(index=False):
                g, niche = str(r.gene), int(r.niche_key)
                o = observed_lfc_seen_patch_niche(a, g, niche, PREP, slide, nb.MIN_G)
                if o is None:
                    continue
                keep = nb.build_gene_mask(genes, g, hvg) & keep_hvg_bool
                src = nb.spac_source_idx(a, role, niche)
                if len(src) < nb.MIN_G:
                    continue
                src_s = nb.subsample(src, nb.MAX_SRC, seed=hash((slide, niche, g)) % (2**31))
                src_X = to_dense(a[src_s].X).astype(np.float64)
                lib = uv.src_lib_sizes(a, src_s)
                elig = {}
                for s in SLIDES:
                    m = obs[s][0] == g
                    if s == slide:
                        m &= bin_id != niche
                    elig[s] = np.where(m)[0]
                means = accumulate(pools_for(elig, train["seen"], ns, seeds, f"S{g}{slide}{niche}"), slide_X)
                seed = uv.unit_seed(0, g)
                for (cfg, sd), (mu, n) in means.items():
                    if mu is None:
                        continue
                    pred = uv.poisson_at_observed_depth(np.broadcast_to(mu, src_X.shape), lib, seed)
                    sc = score_lfc_agreement(pseudobulk_lfc(pred, src_X)[keep], o[keep], n_boot=0)
                    rows.append(dict(arm="seen", config=cfg, sub_seed=sd, n_cells=n, gene=g, slide=slide,
                                     niche_key=str(niche), lfc_spearman=sc["spearman"], lfc_dist=sc["lfc_dist"]))
            print(f"  seen slide {slide} done ({len(rows)} rows)", flush=True)
            del a
            gc.collect()
        pd.DataFrame(rows).to_csv(args.outdir / "mean_target_ncells_seen.csv", index=False)


if __name__ == "__main__":
    main()
