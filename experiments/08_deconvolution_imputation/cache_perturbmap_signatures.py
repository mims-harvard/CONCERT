#!/usr/bin/env python3
"""Cache location-matched CONCERT mu for Task-3 peri: KO / KP-tumor / normal.

Writes/updates location_signatures.npz with keys:
  spot_ids, mu_ko, mu_un, mu_normal, y_obs, genes

NEVER uses empirical panel means as signatures.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

# Reuse CONCERT loaders / cf_query from the amortized runner
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deconvolve_perturbmap_jak2 import (  # noqa: E402
    CKPT_DEFAULT,
    DATA,
    KO_GENE,
    LOC_RANGE,
    OUT_DEFAULT,
    cf_query,
    hex_neighbors,
    jak2_contiguous_patch,
    load_concert_slide0,
)


def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def display_coords(xy):
    x, y = xy[:, 0].astype(float), xy[:, 1].astype(float)
    return np.column_stack([y, -x])


def task3_peri_idx(meta):
    """Same deconv-target selection as plot_perturbmap_subspots.py."""
    pert = meta.obs["perturbation"].astype(str).to_numpy()
    tissue = meta.obs["tissue"].astype(str).to_numpy()
    xy = np.asarray(meta.obsm["spatial"], dtype=np.float64)
    rows = meta.obs["array_row"].astype(int).to_numpy()
    cols = meta.obs["array_col"].astype(int).to_numpy()
    is_jak2 = pert == KO_GENE
    is_peri = tissue == "periphery"
    is_normal = tissue == "normal"
    hex_nbrs = hex_neighbors(rows, cols)
    in_patch = jak2_contiguous_patch(is_jak2, is_normal, hex_nbrs) >= 0
    xy_d = display_coords(xy)
    focus = np.where(in_patch)[0]
    fxy = xy_d[focus]
    span = max(float(fxy[:, 0].ptp()), float(fxy[:, 1].ptp()), 1.0)
    pad = 0.35 * span
    in_view = (
        (xy_d[:, 0] >= fxy[:, 0].min() - pad) & (xy_d[:, 0] <= fxy[:, 0].max() + pad) &
        (xy_d[:, 1] >= fxy[:, 1].min() - pad) & (xy_d[:, 1] <= fxy[:, 1].max() + pad)
    )
    core = in_patch & (~is_jak2) & (~is_normal)
    peri_in_view = is_peri & (~is_jak2) & in_view
    return np.where(core | peri_in_view)[0], xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--ckpt_dir", type=Path, default=CKPT_DEFAULT)
    ap.add_argument("--outdir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--force", action="store_true", help="Requery even if mu_normal present")
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "location_signatures.npz"

    meta = ad.read_h5ad(args.data)
    peri_idx, xy = task3_peri_idx(meta)
    tissue = meta.obs["tissue"].astype(str).to_numpy()
    is_peri = tissue == "periphery"
    print(f"Task3 deconv targets n={peri_idx.size}", flush=True)

    need_query = True
    if cache_path.exists() and not args.force:
        z = np.load(cache_path, allow_pickle=True)
        have = set(z.files)
        old_ids = set(z["spot_ids"].astype(int).tolist()) if "spot_ids" in have else set()
        if "mu_normal" in have and set(map(int, peri_idx)).issubset(old_ids):
            print(f"cache already has mu_normal covering all peri -> {cache_path}", flush=True)
            need_query = False
        else:
            print(
                f"cache incomplete (keys={sorted(have)}; "
                f"missing_peri={len(set(map(int, peri_idx)) - old_ids)}) -> refresh",
                flush=True,
            )

    if not need_query:
        return

    print("loading CONCERT ...", flush=True)
    ctx = load_concert_slide0(args.data, args.ckpt_dir, args.ckpt_dir / "hvg_gene_names.txt")
    model = ctx["model"]
    pert_map = ctx["pert_map"]
    tissue_dict = ctx["tissue_dict"]
    ko_code = int(pert_map[KO_GENE])
    unpert_code = int(pert_map.get("unperturbed", pert_map.get("__basal__", 0)))
    t_peri = int(tissue_dict["periphery"])
    t_tumor = int(tissue_dict["tumor"])
    t_normal = int(tissue_dict["normal"])
    assert len(ctx["pos_raw"]) == meta.n_obs

    G = ctx["Y_norm"].shape[1]
    n = len(peri_idx)
    mu_ko = np.zeros((n, G), dtype=np.float32)
    mu_un = np.zeros((n, G), dtype=np.float32)
    mu_normal = np.zeros((n, G), dtype=np.float32)
    y_obs = np.zeros((n, G), dtype=np.float32)

    for j, i in enumerate(peri_idx):
        y_sp = ctx["Y_norm"][i : i + 1]
        tcode_ko = t_peri if is_peri[i] else t_tumor
        mu_ko[j] = cf_query(
            model, ctx, xy[i : i + 1], y_sp, ko_code, tcode_ko, True,
            args.n_samples, args.batch_size,
        )[0]
        mu_un[j] = cf_query(
            model, ctx, xy[i : i + 1], y_sp, unpert_code, t_tumor, False,
            args.n_samples, args.batch_size,
        )[0]
        mu_normal[j] = cf_query(
            model, ctx, xy[i : i + 1], y_sp, unpert_code, t_normal, False,
            args.n_samples, args.batch_size,
        )[0]
        y_obs[j] = y_sp[0]
        if (j + 1) % 5 == 0 or j == 0 or j + 1 == n:
            print(f"  signatures {j+1}/{n}", flush=True)

    np.savez_compressed(
        cache_path,
        spot_ids=peri_idx.astype(int),
        mu_ko=mu_ko,
        mu_un=mu_un,
        mu_normal=mu_normal,
        y_obs=y_obs,
        genes=np.asarray(ctx["gene_names"]).astype(str),
    )
    meta_out = {
        "n_spots": int(n),
        "states": ["Jak2", "unperturbed_tumor", "normal"],
        "note": "All three are CONCERT counterfactuals; no empirical panel means.",
        "path": str(cache_path),
    }
    (outdir / "location_signatures_meta.json").write_text(json.dumps(meta_out, indent=2))
    print(json.dumps(meta_out, indent=2), flush=True)
    print(f"wrote {cache_path}", flush=True)


if __name__ == "__main__":
    main()
