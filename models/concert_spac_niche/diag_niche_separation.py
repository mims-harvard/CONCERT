#!/usr/bin/env python
"""Go/no-go gate for the niche-FiLM Tgfbr2 fix (Task 2, self-contained).

Question: does an UNSUPERVISED, batch-local coord-kNN aggregate of the content latent gs_mu
(the candidate z_niche descriptor) SEPARATE Tgfbr2 patches whose GT effect points in OPPOSITE
directions? If yes, a niche-keyed sign-flip head can learn when to flip -> proceed. If no, the
descriptor is too weak -> revisit before spending GPU.

Checks:
  - candidate z_niche (batch-local kNN of gs_mu) vs BASELINES gp_p_m (old 2-D coord latent) and raw coords
  - swept over k
  - per-slide + pooled
  - Ifngr2/Jak2 as HOMOGENEITY CONTROLS (should be ~single-sign -> nothing to separate)
Uses the v4 pmap checkpoint + the shared eval loader. CPU-only.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=<CC> \
    concert_env/bin/python diag_niche_separation.py --model_file <ms_pmap_v4_seen/model.pt> --outdir diag_out
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

EVAL = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval"
CC   = "/n/holylabs/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/src/CONCERT-spac-rffbasal"
sys.path.insert(0, CC); sys.path.insert(0, EVAL)
import run_concert_pm_eval as R   # build_pm_ctx, _patch_obs_on_model_rows, SLIDES, parse_args

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def make_args(model_file, outdir):
    """v4 pmap arch defaults (match sub_train_pmap_v4_fix.sh) + ls band + inducing 64."""
    argv = ["--model_file", str(model_file), "--outdir", str(outdir),
            "--ls_prior_min", "3", "--ls_prior_max", "18", "--fpert_svgp_inducing", "64"]
    old = sys.argv
    sys.argv = ["diag"] + argv
    try:
        args = R.parse_args()
    finally:
        sys.argv = old
    return args


def batch_local_zniche(gs, coords, k):
    """Candidate z_niche: distance-weighted kNN mean of gs (surround) ++ self gs. Untrained
    proxy of NicheDescriptor (probe only needs the raw separable signal). (N, 2D)."""
    gs = torch.as_tensor(gs, dtype=torch.float32)
    coords = torch.as_tensor(coords, dtype=torch.float32)
    N = gs.shape[0]
    kk = int(min(k, N - 1))
    d2 = torch.cdist(coords, coords) ** 2
    d2 = d2 + torch.eye(N) * 1e12
    dist, idx = torch.topk(d2, kk, dim=1, largest=False)
    w = 1.0 / (1.0 + dist)                      # (N,kk)
    w = w / w.sum(1, keepdim=True)
    surround = (w.unsqueeze(-1) * gs[idx]).sum(1)   # (N,D)
    z = torch.cat([surround, gs], dim=-1).numpy()
    return z


def probe_auc(Z, y):
    """5-fold logistic AUC predicting binary sign y from features Z. Guards degenerate cases."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3 or not HAVE_SK:
        return None
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    try:
        return float(cross_val_score(LogisticRegression(max_iter=2000), Zs, y,
                                     cv=min(5, min(np.bincount(y))), scoring="roc_auc").mean())
    except Exception:
        return None


def patch_signs(Yobs, canon, patch_id, pert, slide_bg_mask):
    """Per-cell sign label for cells of `pert`: +1/0 by whether their patch's GT direction
    agrees/opposes the pooled `pert` direction. GT direction = patchKO_mean - slide_background_mean.
    Returns (cell_idx array, labels 0/1, n_patches, n_flipped)."""
    m = np.array([str(c).lower() == pert.lower() for c in canon])
    idx = np.where(m)[0]
    if len(idx) < 6:
        return idx, np.array([]), 0, 0
    bg = Yobs[slide_bg_mask].mean(0)                     # background expression (this slide)
    pooled_dir = Yobs[idx].mean(0) - bg                  # pooled pert GT direction
    pooled_dir = pooled_dir / (np.linalg.norm(pooled_dir) + 1e-8)
    pids = patch_id[idx]
    labels = np.zeros(len(idx), dtype=int)
    signs_by_patch = {}
    for p in np.unique(pids):
        sel = pids == p
        if sel.sum() < 3:
            signs_by_patch[p] = 1; labels[sel] = 1; continue
        d = Yobs[idx[sel]].mean(0) - bg
        d = d / (np.linalg.norm(d) + 1e-8)
        s = float(np.dot(d, pooled_dir))
        lab = 1 if s >= 0 else 0
        signs_by_patch[p] = lab
        labels[sel] = lab
    n_patch = len(signs_by_patch)
    n_flip = sum(1 for v in signs_by_patch.values() if v == 0)
    return idx, labels, n_patch, n_flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_file", required=True)
    ap.add_argument("--outdir", default="diag_niche_out")
    ap.add_argument("--ks", type=int, nargs="+", default=[6, 15, 25, 40])
    ap.add_argument("--perts", nargs="+", default=["Tgfbr2", "Ifngr2", "Jak2"])
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    args = make_args(a.model_file, a.outdir)

    report = {"model": str(a.model_file), "ks": a.ks, "perts": a.perts, "slides": {}, "pooled": {}}
    # accumulate pooled (across slides) features/labels per pert
    pool = {p: {"z": {k: [] for k in a.ks}, "gp": [], "coord": [], "y": []} for p in a.perts}

    for sid in R.SLIDES:
        print(f"\n===== slide {sid} =====", flush=True)
        ctx = R.build_pm_ctx(args, sid)
        obs = R._patch_obs_on_model_rows(args, sid, ctx)
        canon = np.asarray(obs["perturbation_canon"])
        patch_id = np.asarray(obs["patch_id"])
        uniq = {str(c) for c in canon}
        print("canon values (sample):", sorted(x for x in uniq if x != "unassigned")[:15], flush=True)

        model = ctx["model"]; GPd = int(model.GP_dim)
        lat = model.batching_latent_samples(
            X=ctx["pos_batched"], sample_index=ctx["sample_indices"],
            cell_atts=ctx["cell_atts"], Y=ctx["Y_expr"], batch_size=512)
        gp = lat[:, :GPd]; gs = lat[:, GPd:]
        coords = np.asarray(ctx["coords"], dtype=np.float32)
        Yobs = np.asarray(ctx["Y_obs"], dtype=np.float32)
        assigned = canon != "unassigned"      # background = other assigned/unassigned cells

        zc = {k: batch_local_zniche(gs, coords, k) for k in a.ks}
        report["slides"][sid] = {}
        for pert in a.perts:
            bg_mask = ~np.array([str(c).lower() == pert.lower() for c in canon])  # everything not this pert
            idx, y, n_patch, n_flip = patch_signs(Yobs, canon, patch_id, pert, bg_mask)
            entry = {"n_cells": int(len(idx)), "n_patch": int(n_patch),
                     "n_flipped_patches": int(n_flip), "auc": {}}
            if len(y) and n_patch >= 2:
                entry["auc"]["gp_p_m"] = probe_auc(gp[idx], y)
                entry["auc"]["coords"] = probe_auc(coords[idx], y)
                for k in a.ks:
                    entry["auc"][f"zniche_k{k}"] = probe_auc(zc[k][idx], y)
                # accumulate pooled
                pool[pert]["gp"].append((gp[idx], y))
                pool[pert]["coord"].append((coords[idx], y))
                for k in a.ks:
                    pool[pert]["z"][k].append((zc[k][idx], y))
            report["slides"][sid][pert] = entry
            print(f"  {pert:8s} cells={len(idx):4d} patches={n_patch:2d} flipped={n_flip:2d} "
                  f"auc(zniche_k{a.ks[-1]})={entry['auc'].get(f'zniche_k{a.ks[-1]}')} "
                  f"auc(gp)={entry['auc'].get('gp_p_m')}", flush=True)

    # pooled across slides
    print("\n===== POOLED across slides =====", flush=True)
    for pert in a.perts:
        def _cat(lst):
            if not lst: return None, None
            X = np.concatenate([x for x, _ in lst], 0); Y = np.concatenate([y for _, y in lst], 0)
            return X, Y
        e = {"auc": {}}
        Xg, Yg = _cat(pool[pert]["gp"])
        if Xg is not None:
            e["n_cells"] = int(len(Yg)); e["n_flipped_frac"] = float((Yg == 0).mean())
            e["auc"]["gp_p_m"] = probe_auc(Xg, Yg)
            Xc, _ = _cat(pool[pert]["coord"]); e["auc"]["coords"] = probe_auc(Xc, Yg)
            for k in a.ks:
                Xk, _ = _cat(pool[pert]["z"][k]); e["auc"][f"zniche_k{k}"] = probe_auc(Xk, Yg)
        report["pooled"][pert] = e
        best = max([v for kk, v in e["auc"].items() if kk.startswith("zniche") and v is not None] or [None],
                   default=None) if e["auc"] else None
        print(f"  {pert:8s} n={e.get('n_cells')} flipped_frac={e.get('n_flipped_frac')} "
              f"best_zniche_auc={best} gp_auc={e['auc'].get('gp_p_m') if e['auc'] else None}", flush=True)

    # verdict
    tg = report["pooled"].get("Tgfbr2", {}).get("auc", {})
    best_z = max([v for k, v in tg.items() if k.startswith("zniche") and v is not None] or [0.0])
    verdict = "PASS" if best_z >= 0.70 else "FAIL"
    report["verdict"] = {"tgfbr2_best_zniche_auc": best_z, "threshold": 0.70, "result": verdict}
    print(f"\n=== VERDICT: Tgfbr2 best z_niche sign-separation AUC = {best_z:.3f} -> {verdict} "
          f"(>=0.70 to proceed) ===", flush=True)
    outp = Path(a.outdir) / "diag_niche_separation.json"
    json.dump(report, open(outp, "w"), indent=2, default=str)
    print("wrote", outp, flush=True)


if __name__ == "__main__":
    main()
