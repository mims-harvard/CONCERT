#!/usr/bin/env python
"""Bridge v3 benchmark manifests -> CONCERT trainer holdout-coords files.

The v3 manifests store *slide-local prepared indices* (union_holdout_idx for the
unassigned prediction-source patch; ko_holdout_idx for the masked target KO).
CONCERT trains on the raw_output_*.h5ad files and keys cells by
(sample_id, array_coords) -- barcodes are dropped in the loader. Prepared and raw
share both obs_names and obsm['array_coords'] (prepared is a filtered subset), so
we translate manifest indices -> array_coords via the prepared slide adata.

Output (one JSON per dataset x regime):
  {
    "dataset": "spac", "regime": "seen"|"unseen",
    "slide_to_sid": {"1-1":0,...},
    "unassigned": {"0":[[x,y],...], ...},   # held out as source (guard: unassigned only)
    "ko":         {"4":[[x,y],...], ...}    # unseen only: masked KO (no unassigned guard)
  }
Consumed by run_concert_spac.py / run_concert_map.py via --holdout_v3_file.
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad


def slide_coords(prep_dir, slide_id, raw_path=None):
    """Per-prep-cell RAW array_coords (int x,y), aligned to prepared row order.

    spac: prepared carries obsm['array_coords'] identical to raw -> use directly.
    pmap: prepared carries only a normalized obsm['spatial'] (min-max of raw
          array_coords to [0,20]); map each prep cell to its raw array_coords by
          KDTree on the scaled raw coords (exact bijection, verified ~2e-7).
    """
    a = ad.read_h5ad(os.path.join(prep_dir, f"slide_{slide_id}.h5ad"), backed="r")
    if "array_coords" in a.obsm:
        return np.rint(np.asarray(a.obsm["array_coords"])).astype(np.int64)
    # pmap path: recover raw array_coords from normalized prep 'spatial'
    from scipy.spatial import cKDTree
    ps = np.asarray(a.obsm["spatial"])
    r = ad.read_h5ad(raw_path, backed="r")
    ra = np.asarray(r.obsm["array_coords"])
    lo, hi = ra.min(0), ra.max(0)
    scaled = (ra - lo) / (hi - lo) * 20.0
    d, idx = cKDTree(scaled).query(ps, k=1)
    if d.max() > 1e-3:
        raise SystemExit(f"[FATAL] pmap {slide_id}: prep->raw coord match dist {d.max():.2e} too large")
    return np.rint(ra[idx]).astype(np.int64)


def build(prep_dir, regime, slide_to_sid, out_path, raw_files=None):
    v3 = os.path.join(prep_dir, "benchmark_v3")
    unassigned = {}
    ko = {}
    stats = []

    # unassigned source patch -- present for BOTH seen and unseen
    hu_dir = os.path.join(v3, "holdout_unassigned")
    for fn in sorted(os.listdir(hu_dir)):
        d = json.load(open(os.path.join(hu_dir, fn)))
        sid_str = d["slide_id"]
        if sid_str not in slide_to_sid:
            continue
        coords = slide_coords(prep_dir, sid_str, (raw_files or {}).get(sid_str))
        idx = np.asarray(d["union_holdout_idx"], dtype=np.int64)
        bad = (idx < 0) | (idx >= len(coords))
        if bad.any():
            raise SystemExit(f"[FATAL] {fn}: {bad.sum()} idx out of range (n={len(coords)})")
        sel = coords[idx]
        # dedup coords within slide
        sel = np.unique(sel, axis=0)
        unassigned[str(slide_to_sid[sid_str])] = sel.tolist()
        stats.append(f"  unassigned {sid_str}(sid{slide_to_sid[sid_str]}): {len(idx)} idx -> {len(sel)} uniq coords")

    # masked KO -- UNSEEN only
    if regime == "unseen":
        ko_dir = os.path.join(v3, "holdout_ko")
        agg = {}  # sid_str -> list of coord arrays
        for fn in sorted(os.listdir(ko_dir)):
            d = json.load(open(os.path.join(ko_dir, fn)))
            sid_str = d["slide_id"]
            if sid_str not in slide_to_sid:
                continue
            coords = slide_coords(prep_dir, sid_str, (raw_files or {}).get(sid_str))
            idx = np.asarray(d["ko_holdout_idx"], dtype=np.int64)
            bad = (idx < 0) | (idx >= len(coords))
            if bad.any():
                raise SystemExit(f"[FATAL] {fn}: {bad.sum()} ko idx out of range (n={len(coords)})")
            agg.setdefault(sid_str, []).append(coords[idx])
        for sid_str, arrs in agg.items():
            sel = np.unique(np.concatenate(arrs, axis=0), axis=0)
            ko[str(slide_to_sid[sid_str])] = sel.tolist()
            stats.append(f"  ko {sid_str}(sid{slide_to_sid[sid_str]}): {sum(len(a) for a in arrs)} idx -> {len(sel)} uniq coords")

    out = {"dataset": os.path.basename(prep_dir), "regime": regime,
           "slide_to_sid": slide_to_sid, "unassigned": unassigned, "ko": ko}
    json.dump(out, open(out_path, "w"))
    print(f"[{regime}] wrote {out_path}")
    print("\n".join(stats))
    return out


def verify_against_raw(out, raw_files, slide_to_sid):
    """Assert held-out coords actually exist in the raw files CONCERT trains on."""
    sid_to_raw = {slide_to_sid[s]: raw_files[s] for s in slide_to_sid}
    for kind in ("unassigned", "ko"):
        for sid_str, coords in out[kind].items():
            raw = sid_to_raw[int(sid_str)]
            a = ad.read_h5ad(raw, backed="r")
            rc = np.rint(np.asarray(a.obsm["array_coords"])).astype(np.int64)
            rset = set(map(tuple, rc.tolist()))
            hit = sum(1 for c in coords if tuple(c) in rset)
            rate = hit / max(1, len(coords))
            flag = "OK" if rate > 0.98 else "*** LOW ***"
            print(f"  verify {kind} sid{sid_str} vs {os.path.basename(raw)}: "
                  f"{hit}/{len(coords)} = {rate:.3f} {flag}")
            if rate <= 0.98:
                raise SystemExit(f"[FATAL] coord match rate {rate:.3f} too low -- bridge unsafe")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["spac", "pmap"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.dataset == "spac":
        prep = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5/prepared"
        SP = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq"
        slide_to_sid = {"1-1": 0, "1-2": 1, "1-3": 2, "1-4": 3, "1-5": 4}
        raw = {s: os.path.join(SP, f"raw_output_{s}.h5ad") for s in slide_to_sid}
    else:
        prep = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/prepared_panel1053ko"
        BASE = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets"
        slide_to_sid = {"0": 0, "1": 1, "2": 2, "3": 3}
        raw = {s: os.path.join(BASE, f"perturbmap_s{s}_spacfmt.h5ad") for s in slide_to_sid}

    for regime in ("seen", "unseen"):
        out = build(prep, regime, slide_to_sid,
                    os.path.join(args.outdir, f"holdout_v3_{args.dataset}_{regime}.json"),
                    raw_files=raw)
        if args.verify:
            verify_against_raw(out, raw, slide_to_sid)
