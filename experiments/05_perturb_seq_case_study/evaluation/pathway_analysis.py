#!/usr/bin/env python3
"""Powered pathway analysis for the Perturb-map case study.

The shipped analysis contrasts ONE transplanted patch against the normal-spot mean, i.e.
n=1 vs n=1, and nothing survives FDR (best q = 0.105 of 45 hallmark sets). This replaces it
with a design that has real replication: each knockout was placed into 40 sampled niches,
so every gene gets a statistic computed ACROSS 40 independent niches.

Per gene: Spearman correlation between its predicted response and the niche's tumour
content. Genes whose response tracks the microenvironment rank high, and the enrichment is
run on that ranking. This also tests the biological question the case study actually asks --
which programmes follow the microenvironment -- rather than comparing two hand-picked spots.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.spatial import cKDTree
from scipy.stats import rankdata

# Reads what transplant_analysis.py wrote and writes its results alongside it.
SYS = Path(os.environ.get(
    "PMAP_SYSTEMATIC_OUT",
    f"{_rp.WORKDIR}/scoring/figures/pmap_systematic"))
RAW = Path(os.environ.get(
    "PMAP_RAW", f"{_rp.DATA}/source_datasets/GSM5808054_data.h5"))
OUT = SYS
PERTS = ["Tgfbr2", "Ifngr2", "Jak2"]
NEIGHBOURHOOD = 25


def log(*a):
    print(*a, flush=True)


def niche_tumour_content(pos, patches, pert_raw, k=NEIGHBOURHOOD):
    """Fraction of each patch's spatial neighbourhood that is tumour (KP or a knockout)."""
    is_tumour = ~np.isin(pert_raw, ["None"])
    tree = cKDTree(pos)
    out = []
    for patch in patches:
        nb = tree.query(pos[patch], k=k + 1)[1][:, 1:].ravel()
        out.append(float(is_tumour[nb].mean()))
    return np.asarray(out)


def spearman_rows(L, x):
    """Spearman correlation of every gene (column of L) with x, across niches."""
    xr = rankdata(x)
    xr = (xr - xr.mean()) / max(xr.std(), 1e-12)
    Lr = np.apply_along_axis(rankdata, 0, L)
    Lr = (Lr - Lr.mean(axis=0)) / np.maximum(Lr.std(axis=0), 1e-12)
    return (Lr * xr[:, None]).mean(axis=0)


def main() -> None:
    import gseapy

    d = np.load(SYS / "systematic_transplant.npz", allow_pickle=True)
    genes = np.char.upper(d["gene_names"].astype(str))
    # Saved as an object array of equal-length patches, which round-trips as dtype=object
    # and cannot index a float array until it is cast back to int.
    patches = [np.asarray(p, dtype=int).ravel() for p in d["patches"]]
    with h5py.File(RAW, "r") as f:
        pos = np.array(f["pos"]).T.astype(float)
        pert_raw = np.array(f["perturbation"]).astype(str)

    tum = niche_tumour_content(pos, patches, pert_raw)
    log(f"  {len(patches)} niches, tumour content {tum.min():.2f}-{tum.max():.2f} "
        f"(mean {tum.mean():.2f}, sd {tum.std():.2f})")

    results = {"n_niches": len(patches),
               "tumour_content": {"min": float(tum.min()), "max": float(tum.max()),
                                  "mean": float(tum.mean()), "sd": float(tum.std())}}
    tables = {}
    for p in PERTS:
        key = f"lfc_{p}"
        if key not in d:
            continue
        L = d[key]
        # AMPLITUDE vs SHAPE. The overall size of the predicted effect is strongly
        # anti-correlated with niche tumour content, and that single global effect shifts
        # every gene's correlation away from zero, which makes GSEA report the same finding
        # as many "significant" pathways. Scale each niche's response to unit norm first, so
        # the ranking asks whether the SHAPE of the response tracks the microenvironment.
        amp = np.abs(L).mean(axis=1)
        amp_rho = float(np.corrcoef(rankdata(amp), rankdata(tum))[0, 1])
        Ln = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-12)
        rho = spearman_rows(Ln, tum)
        rho = rho - rho.mean()          # remove any residual global offset
        log(f"\n  {p}: amplitude-vs-tumour-content rho = {amp_rho:+.3f} "
            f"(the global effect); shape-correlation spread sd = {rho.std():.3f}")
        results.setdefault("amplitude_effect", {})[p] = amp_rho
        df = (pd.DataFrame({"gene": genes, "score": rho})
              .groupby("gene", as_index=False).mean()
              .sort_values("score", ascending=False))
        pre = gseapy.prerank(rnk=df, gene_sets="MSigDB_Hallmark_2020", permutation_num=1000,
                             seed=0, no_plot=True, min_size=10, max_size=1000, verbose=False)
        r = pre.res2d.set_index("Term")[["NES", "FDR q-val"]].astype(float)
        r.columns = ["NES", "q"]
        tables[p] = r
        sig = r[r.q < 0.25].sort_values("NES", ascending=False)
        log(f"\n  === {p} === genes correlated with niche tumour content; "
            f"{int((r.q < 0.05).sum())} sets q<0.05, {int((r.q < 0.25).sum())} q<0.25")
        if len(sig):
            log(sig.head(10).round(3).to_string())
        results[p] = {"n_q05": int((r.q < 0.05).sum()), "n_q25": int((r.q < 0.25).sum()),
                      "top": {t: {"NES": float(r.NES[t]), "q": float(r.q[t])}
                              for t in r.reindex(r.q.sort_values().index).head(8).index}}

    if tables:
        pd.concat(tables, axis=1).to_csv(OUT / "hallmark_niche_correlation.csv")
    (OUT / "pathway_powered_stats.json").write_text(json.dumps(results, indent=2))
    log("\nwrote " + str(OUT / "pathway_powered_stats.json"))


if __name__ == "__main__":
    main()
