#!/usr/bin/env python3
"""Systematic transplantation: how much does a knockout's predicted response depend on WHERE.

The shipped case study transplants one patch into two niches, which is an anecdote. This
places each knockout into K niches sampled across the slide and measures the spread of the
predicted responses. That spread is the quantity "perturbation responses depend on the
microenvironment" actually refers to, and it can be compared to the model's own learned
niche gain and to how variable the OBSERVED response of that knockout is across niches.

Predicted response of a patch = mean over its cells of log(perturbed) - log(unperturbed),
both from the model, so the basal state cancels and what remains is the perturbation effect.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import json
import sys
from pathlib import Path

import os

import numpy as np
import h5py
import torch
from scipy.spatial import cKDTree

CODEBASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CODEBASE / "benchmark_evaluation"))
import evaluate_concert_perturbmap as R   # noqa: E402
import case_study as CS       # noqa: E402

OUT = Path(os.environ.get(
    "PMAP_SYSTEMATIC_OUT",
    f"{_rp.WORKDIR}/scoring/figures/pmap_systematic"))
PERTS = ["Tgfbr2", "Ifngr2", "Jak2"]
K_NICHES = 40
PATCH_SIZE = 20
EPS = 1e-6


def log(*a):
    print(*a, flush=True)


def sample_niches(pos, eligible, k, size, seed=0):
    """k spatially compact patches of `size` spots, seeded on eligible (unperturbed) spots."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(pos[eligible])
    seeds = rng.choice(len(eligible), size=min(k, len(eligible)), replace=False)
    patches = []
    for s in seeds:
        _d, nn = tree.query(pos[eligible][s], k=size)
        patches.append(eligible[np.atleast_1d(nn)])
    return patches


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    args = CS.build_args()
    args.batch_size = 2048

    with h5py.File(CS.RAW, "r") as f:
        pos_raw = np.array(f["pos"]).T.astype(float)
        pert_raw = np.array(f["perturbation"]).astype(str)

    ctx = R.build_pm_ctx(args, "0")
    model = ctx["model"]
    rows = CS.locate_slide0_rows(ctx, pos_raw)
    maps, _ = CS.decode_vocabularies(ctx, rows, pert_raw)
    gene_names = np.asarray(ctx["gene_names"]).astype(str)

    # Unperturbed spots only -- transplanting into a spot that already carries a knockout
    # would confound the niche with that knockout's own effect.
    eligible = np.where(np.isin(pert_raw, ["None", "KP"]))[0]
    patches = sample_niches(pos_raw, eligible, K_NICHES, PATCH_SIZE)
    log(f"  {len(patches)} niches of {PATCH_SIZE} spots, seeded on {len(eligible)} unperturbed spots")

    base = CS.predict(ctx, args, rows, [], None, None)          # own-state reconstruction
    results = {"n_niches": len(patches), "patch_size": PATCH_SIZE, "per_perturbation": {}}
    curves = {}

    name2code = {str(k): int(v) for k, v in ctx["name2code"].items()}
    for p in PERTS:
        if p not in name2code:
            continue
        lfcs = []
        for patch in patches:
            pred = CS.predict(ctx, args, rows, patch, maps["tissue"]["tumor"],
                              name2code[p], has_guide=True, is_real_ko=True)
            lfc = (np.log(pred[patch] + EPS) - np.log(base[patch] + EPS)).mean(axis=0)
            lfcs.append(lfc)
        L = np.vstack(lfcs)
        # Spread across niches: per-gene s.d. of the predicted effect, and how similar two
        # randomly chosen niches' responses are.
        per_gene_sd = L.std(axis=0)
        C = np.corrcoef(L)
        iu = np.triu_indices(len(L), k=1)
        results["per_perturbation"][p] = {
            "mean_effect_magnitude": float(np.abs(L).mean()),
            "across_niche_sd": float(per_gene_sd.mean()),
            "across_niche_sd_over_effect": float(per_gene_sd.mean() / max(np.abs(L).mean(), 1e-9)),
            "mean_pairwise_correlation_between_niches": float(C[iu].mean()),
            "min_pairwise_correlation_between_niches": float(C[iu].min()),
        }
        curves[p] = L.astype(np.float32)
        log(f"  {p}: " + json.dumps({k: round(v, 4)
                                     for k, v in results['per_perturbation'][p].items()}))

    np.savez_compressed(OUT / "systematic_transplant.npz",
                        gene_names=gene_names.astype("U32"),
                        patches=np.array([p for p in patches], dtype=object),
                        **{f"lfc_{k}": v for k, v in curves.items()})
    (OUT / "systematic_stats.json").write_text(json.dumps(results, indent=2))
    log("wrote " + str(OUT / "systematic_stats.json"))


if __name__ == "__main__":
    main()
