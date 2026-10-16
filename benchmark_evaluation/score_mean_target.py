#!/usr/bin/env python3
"""Mean target reference strategy with a fixed knockout-cell budget (default 50), SPAC-seq, both arms, sampled output.

Adopted 2026-09-21 as the Mean target of the benchmark. The earlier Mean target averaged every knockout cell of the
full prepared slides (median 560 unseen, 973 seen), 20-30x more knockout cells than the trained models receive through
the 50,000-cell training manifest (median 18 unseen, 47 seen). Here each unit's Mean target is the mean of N knockout
cells drawn uniformly (one fixed draw) from the same eligible pool as before:
  unseen: knockout cells on the donor slides only (reference_strategies_unseen.donor_target_means)
  seen:   knockout cells on all five slides minus the scored (gene, slide, niche) cells
          (reference_strategies.unit_leaveout_target_mean)
Scoring is unchanged: Poisson output at the observed source library size, shared gene mask, gene bootstrap.
The cell-count sweep behind the choice of N: mean_target_audit/mean_target_ncells_summary.csv.
Output: <outdir>/spac_{unseen,seen}_mean_target_n<N>.csv, in the reference-strategy CSV format
(run with PYTHONHASHSEED=0).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import release_paths as rp  # noqa: E402

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_strategies_seen  # noqa: E402,F401  (installs the shared gene mask on nb)
import reference_strategies as nb  # noqa: E402
import reference_strategies_unseen as uv  # noqa: E402
import spacseq_common as sec  # noqa: E402
from spacseq_cross_slide_scoring import (  # noqa: E402
    ko_holdout_indices, load_eval_panel, load_source_cells, unseen_reference_idx)
from lfc_scoring import (  # noqa: E402
    observed_lfc_seen_patch_niche, pseudobulk_lfc, score_lfc_agreement, to_dense)
from spacseq_slides import load_patch_roles  # noqa: E402

PREP = nb.PREP_SPAC
BENCH = PREP / "benchmark"
SLIDES = uv.SLIDES


def pools_for(eligible: dict[str, np.ndarray], ns: list[int], seeds: list[int],
              key: str) -> dict[tuple, dict[str, np.ndarray]]:
    """(config, seed) -> {slide: selected cell indices}; identical draws to sweep_mean_target_cells.py."""
    out = {}
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
    ap.add_argument("--n_cells", type=int, default=50)
    ap.add_argument("--sub_seed", type=int, default=0)
    ap.add_argument("--arms", default="unseen,seen")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg_key = (f"n{args.n_cells}", args.sub_seed)
    arms = args.arms.split(",")

    obs, slide_X = {}, {}
    for s in SLIDES:
        a = ad.read_h5ad(PREP / f"slide_{s}.h5ad")
        obs[s] = (a.obs["perturbation_canon"].astype(str).to_numpy(), a.obs["bin_id"].to_numpy().astype(int))
        slide_X[s] = a.X
        del a
        print(f"loaded slide {s}", flush=True)

    def capped_mean(elig, key):
        sel = pools_for(elig, [args.n_cells], [args.sub_seed], key)
        return accumulate({cfg_key: sel[cfg_key]}, slide_X)[cfg_key]

    def score(mu, src_X, lib, gene, o, keep):
        pred = uv.poisson_at_observed_depth(np.broadcast_to(mu, src_X.shape), lib, uv.unit_seed(0, gene))
        return score_lfc_agreement(pseudobulk_lfc(pred, src_X)[keep], o[keep])

    if "unseen" in arms:
        panel = load_eval_panel(BENCH)
        _, sa = sec.import_eval_utils()
        rows = []
        for T in sorted({u["target_slide"] for u in panel["eval_units"]}):
            a = ad.read_h5ad(PREP / f"slide_{T}.h5ad")
            genes = np.asarray(a.var_names).astype(str)
            hvg_mask = sec.select_hvgs(a, 2000, sa)
            in_concert = np.isin(genes, uv.CONCERT_GENES)
            ref_idx = unseen_reference_idx(a, PREP, T)
            y_ref = to_dense(a[ref_idx].X)
            src = load_source_cells(BENCH, T)
            src_X = to_dense(a[src].X).astype(np.float64)
            lib = uv.src_lib_sizes(a, src)
            for u in [u for u in panel["eval_units"] if u["target_slide"] == T]:
                g = u["gene"]
                ko_idx = ko_holdout_indices(BENCH, g, T)
                if len(ko_idx) < nb.MIN_G:
                    continue
                o = pseudobulk_lfc(to_dense(a[ko_idx].X), y_ref)
                keep = hvg_mask & (genes != g) & in_concert
                mu, n = capped_mean({s: np.where(obs[s][0] == g)[0] for s in u["donor_slides"] if s != T}, f"U{g}{T}")
                if mu is None:
                    continue
                rows.append(nb.record("mean_target", g, T, "-1", keep, score(mu, src_X, lib, g, o, keep)) | {"n_target_cells": n})
            print(f"  unseen {T} done ({len(rows)} rows)", flush=True)
            del a, y_ref, src_X
            gc.collect()
        pd.DataFrame(rows).to_csv(args.outdir / f"spac_unseen_mean_target_n{args.n_cells}.csv", index=False)

    if "seen" in arms:
        nb.MASTER_SPAC = rp.SPACSEQ_RESULTS / "spac_seen_lfc_dist_per_record.csv"
        master = nb.load_master_spac()
        rows = []
        for slide, gdf in master.groupby("slide"):
            slide = str(slide)
            a = ad.read_h5ad(PREP / f"slide_{slide}.h5ad")
            role = load_patch_roles(PREP, slide, a)
            bin_id = obs[slide][1]
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
                mu, n = capped_mean(elig, f"S{g}{slide}{niche}")
                if mu is None:
                    continue
                rows.append(nb.record("mean_target", g, slide, niche, keep, score(mu, src_X, lib, g, o, keep)) | {"n_target_cells": n})
            print(f"  seen slide {slide} done ({len(rows)} rows)", flush=True)
            del a
            gc.collect()
        pd.DataFrame(rows).to_csv(args.outdir / f"spac_seen_mean_target_n{args.n_cells}.csv", index=False)


if __name__ == "__main__":
    main()
