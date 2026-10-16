#!/usr/bin/env python3
"""Build spot-only deconv assets from existing wholeslide pseudo-bulk.

Creates:
  - membership.npz (cell -> spot_id; rebuilt with same knn_partition seed/size)
  - spot_roles.csv (pure / mixture / other)
  - pure_spots.h5ad (spots as observations; for signature training)
  - mixture_spots.h5ad (eval spots)
  - manifest.json

Cell expression is NOT written into training assets beyond spot means already in PB.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spot_aggregate_plots import knn_partition  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_spot_only")

DEFAULT_KO = [
    "Icam1", "Arap1", "Cd276", "Sema4b", "Slc43a3",
    "Bcam", "H2-Q7", "Cxcr4", "Slc2a1", "Utrn",
]
RAW_DEFAULT = Path(
    f"{_rp.DATA}/spacseq/raw/raw_output_2-1.h5ad"
)
PB_DEFAULT = Path(
    f"{_rp.WORKDIR}/scoring/spot_plots/pseudobulk_2-1_wholeslide.h5ad"
)
COMP_DEFAULT = Path(
    f"{_rp.WORKDIR}/scoring/spot_plots/spot_composition_2-1_wholeslide.csv"
)


def classify_spots(comp: pd.DataFrame, ko_list: list[str], pure_frac: float,
                   mix_lo: float, mix_hi: float, una_min: float) -> pd.DataFrame:
    """Assign role per spot using composition GT (benchmark construction only)."""
    F = comp.copy()
    for g in ko_list + ["unassigned"]:
        c = f"frac_{g}"
        if c not in F.columns:
            F[c] = 0.0
        F[c] = F[c].fillna(0.0)

    roles, genes, fko, funa = [], [], [], []
    for i in range(len(F)):
        row = F.iloc[i]
        una = float(row["frac_unassigned"])
        ko_fracs = {g: float(row[f"frac_{g}"]) for g in ko_list}
        present = [g for g, v in ko_fracs.items() if v > 0]
        pure_g = [g for g, v in ko_fracs.items() if v >= pure_frac]
        if len(pure_g) == 1 and una < (1.0 - pure_frac + 1e-9):
            roles.append("pure_KO")
            genes.append(pure_g[0])
            fko.append(ko_fracs[pure_g[0]])
            funa.append(una)
            continue
        if una >= pure_frac and len(present) == 0:
            roles.append("pure_una")
            genes.append("unassigned")
            fko.append(0.0)
            funa.append(una)
            continue
        if len(present) == 1:
            g = present[0]
            fk = ko_fracs[g]
            if mix_lo <= fk <= mix_hi and una >= una_min:
                roles.append("mixture")
                genes.append(g)
                fko.append(fk)
                funa.append(una)
                continue
        roles.append("other")
        genes.append(present[0] if present else "none")
        fko.append(max(ko_fracs.values()) if ko_fracs else 0.0)
        funa.append(una)

    out = pd.DataFrame({
        "spot": F["spot"].astype(int).to_numpy() if "spot" in F.columns else np.arange(len(F)),
        "role": roles,
        "gene": genes,
        "frac_ko": fko,
        "frac_una": funa,
        "n_cells": F["n_cells"].astype(int).to_numpy() if "n_cells" in F.columns else 9,
        "x": F["x"].to_numpy() if "x" in F.columns else np.nan,
        "y": F["y"].to_numpy() if "y" in F.columns else np.nan,
        "dominant": F["dominant"].astype(str).to_numpy() if "dominant" in F.columns else genes,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--pseudobulk", type=Path, default=PB_DEFAULT)
    ap.add_argument("--composition", type=Path, default=COMP_DEFAULT)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--ko_list", type=str, default=",".join(DEFAULT_KO))
    ap.add_argument("--pure_frac", type=float, default=0.8)
    ap.add_argument("--mix_lo", type=float, default=0.15)
    ap.add_argument("--mix_hi", type=float, default=0.85)
    ap.add_argument("--una_min", type=float, default=0.1)
    ap.add_argument("--spot_size_cells", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_membership", action="store_true",
                    help="Skip rebuilding cell->spot map (faster; needed for cell-xy eval)")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    ko_list = [g.strip() for g in args.ko_list.split(",") if g.strip()]

    log.info("Loading composition %s", args.composition)
    comp = pd.read_csv(args.composition)
    roles = classify_spots(comp, ko_list, args.pure_frac, args.mix_lo, args.mix_hi, args.una_min)
    roles = roles.sort_values("spot").reset_index(drop=True)
    roles.to_csv(outdir / "spot_roles.csv", index=False)
    vc = roles["role"].value_counts().to_dict()
    log.info("roles: %s", vc)
    for g in ko_list:
        n_p = int(((roles.role == "pure_KO") & (roles.gene == g)).sum())
        n_m = int(((roles.role == "mixture") & (roles.gene == g)).sum())
        log.info("  %s: pure=%d mixture=%d", g, n_p, n_m)

    log.info("Loading pseudobulk %s", args.pseudobulk)
    pb = ad.read_h5ad(args.pseudobulk)
    assert len(roles) == pb.n_obs, (len(roles), pb.n_obs)

    pure_mask = roles["role"].isin(["pure_KO", "pure_una"]).to_numpy()
    mix_mask = (roles["role"] == "mixture").to_numpy()

    pure = pb[pure_mask].copy()
    pure.obs["perturbation"] = roles.loc[pure_mask, "gene"].astype(str).to_numpy()
    pure.obs["role"] = roles.loc[pure_mask, "role"].astype(str).to_numpy()
    pure.obs["frac_ko"] = roles.loc[pure_mask, "frac_ko"].to_numpy()
    pure.obs["frac_una"] = roles.loc[pure_mask, "frac_una"].to_numpy()
    pure.write_h5ad(outdir / "pure_spots.h5ad")
    log.info("wrote pure_spots.h5ad (%d spots)", pure.n_obs)

    mix = pb[mix_mask].copy()
    mix.obs["perturbation"] = roles.loc[mix_mask, "gene"].astype(str).to_numpy()
    mix.obs["role"] = "mixture"
    mix.obs["frac_ko"] = roles.loc[mix_mask, "frac_ko"].to_numpy()
    mix.obs["frac_una"] = roles.loc[mix_mask, "frac_una"].to_numpy()
    mix.obs["spot_index"] = roles.loc[mix_mask, "spot"].to_numpy()
    mix.write_h5ad(outdir / "mixture_spots.h5ad")
    log.info("wrote mixture_spots.h5ad (%d spots)", mix.n_obs)

    membership = None
    if not args.skip_membership:
        log.info("Loading raw cells for membership rebuild %s", args.raw)
        raw = ad.read_h5ad(args.raw, backed="r")
        coords = np.asarray(raw.obsm["spatial"], dtype=np.float64)
        n_cells = raw.n_obs
        raw.file.close()
        log.info("knn_partition n_cells=%d size=%d seed=%d", n_cells, args.spot_size_cells, args.seed)
        spot_id = knn_partition(coords, args.spot_size_cells, args.seed)
        n_spots = int(spot_id.max()) + 1
        if n_spots != pb.n_obs:
            log.warning("Rebuilt n_spots=%d but PB has %d -- membership may be misaligned",
                        n_spots, pb.n_obs)
        np.savez_compressed(outdir / "membership.npz", spot_id=spot_id.astype(np.int64))
        membership = {"n_cells": int(n_cells), "n_spots_rebuilt": n_spots}
        log.info("wrote membership.npz")

    manifest = {
        "slide": "2-1",
        "pseudobulk": str(args.pseudobulk),
        "composition": str(args.composition),
        "raw": str(args.raw),
        "pure_frac": args.pure_frac,
        "mix_lo": args.mix_lo,
        "mix_hi": args.mix_hi,
        "una_min": args.una_min,
        "spot_size_cells": args.spot_size_cells,
        "seed": args.seed,
        "ko_list": ko_list,
        "n_spots": int(pb.n_obs),
        "n_genes": int(pb.n_vars),
        "role_counts": {str(k): int(v) for k, v in vc.items()},
        "n_pure_spots": int(pure_mask.sum()),
        "n_mixture_spots": int(mix_mask.sum()),
        "pure_per_gene": {
            g: int(((roles.role == "pure_KO") & (roles.gene == g)).sum()) for g in ko_list
        },
        "mixture_per_gene": {
            g: int(((roles.role == "mixture") & (roles.gene == g)).sum()) for g in ko_list
        },
        "n_pure_una": int((roles.role == "pure_una").sum()),
        "membership": membership,
        "note": "Model must not use cell expression; composition GT used only to define pure/mixture for benchmark.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote manifest.json -> %s", outdir)


if __name__ == "__main__":
    main()
