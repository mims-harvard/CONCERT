#!/usr/bin/env python3
"""Rebuild the spatial Perturb-seq case study on the shipped model.

Two analyses, both on slide GSM5808054, using the ORIGINAL patch selections:

  Transplantation -- perturb a patch of unperturbed spots to Jak2 in two different
  microenvironments (another tumour core, and a normal/surface-like region) and compare
  the predicted profiles to each other and to the observed Jak2 spots.

  Niche editing -- take the observed Tgfbr2 patch, edit ONLY the ring of spots around it
  to tumour periphery, and read the predicted response of the core, which is never edited.

The selection files are 1-based (written from R). Row order in GSM5808054_data.h5 is
identical to perturbmap_s0_spacfmt.h5ad (verified: coordinates match element-wise), so the
indices land directly on model rows once the slide-0 block is located.

Also reports whether the core actually moves under niche editing -- in this architecture
the niche descriptor is built from observed expression, so that propagation is a claim to
be measured, not assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import os

import numpy as np
import h5py
import torch

CODEBASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODEBASE / "benchmark_evaluation"))
import run_concert_pm_niche as R  # noqa: E402

# Inputs. RAW is the measured section; SEL/SEL2 hold the spot selections that define the
# patches. CKPT is the trained checkpoint from experiment 02, OUT is where results are
# written; both can be overridden without editing this file.
RAW = Path(os.environ.get(
    "PMAP_RAW", "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/GSM5808054_data.h5"))
SEL = Path(os.environ.get(
    "PMAP_SELECTIONS", "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/pert_cells"))
SEL2 = Path(os.environ.get(
    "PMAP_SELECTIONS2",
    "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/select_cells"))
CKPT = Path(os.environ.get(
    "PMAP_CKPT",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_pmap_niche_seen_bandI"))
OUT = Path(os.environ.get(
    "PMAP_CASE_OUT",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/figures/pmap_case_study"))

SELECTIONS = {
    "dest_core":   SEL2 / "pert_cells_GSM5808054_patchclose_tumor_Jak2.txt",
    "dest_normal": SEL / "patch_trans.txt",
    "tgfbr2_core": SEL / "GSM5808054_patch_tumor_tgfbr2_both.txt",
    "peri_ring":   SEL / "GSM5808054_border_peri.txt",
}


def log(*a):
    print(*a, flush=True)


def build_args():
    # The band, coordinate scale and inducing-point count must match the checkpoint; they are
    # fixed here rather than left to the caller because a mismatch reconstructs a different
    # spatial field rather than failing.
    #
    # batch_size must stay at or above the number of spots in a section. The niche descriptor
    # is a nearest-neighbour aggregate computed WITHIN EACH BATCH, so a smaller batch quietly
    # replaces every spot's neighbourhood with a subsample of the section and every niche
    # quantity downstream is then measured against the wrong neighbours. It is set here because
    # the argument vector below replaces the command line, so passing --batch_size to this
    # script has no effect.
    argv = [
        "--model_file", str(CKPT / "model.pt"),
        "--outdir", str(OUT / "_scratch"),
        "--fpert_ls_min", "0.01", "--fpert_ls_max", "0.60",
        "--fpert_coord_scale", "70.0",
        "--fpert_svgp_inducing", "64",
        "--n_samples", "25",
        "--batch_size", "2048",
    ]
    old = sys.argv
    sys.argv = ["pmap_case_study.py"] + argv
    try:
        return R.parse_args()
    finally:
        sys.argv = old


def locate_slide0_rows(ctx, pos_raw):
    """Model rows belonging to slide 0, in raw-file row order.

    The loader MinMax-scales each slide's coordinates to [0, 20] and tiles the slides
    apart, so slide 0's local coordinates are reproducible from the raw positions.
    """
    from sklearn.preprocessing import MinMaxScaler
    local = MinMaxScaler().fit_transform(pos_raw) * 20.0
    pos_model = np.asarray(ctx["pos_batched"])[:, :2]
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(pos_model).query(local, k=1)
    log(f"  slide-0 row match: max coordinate residual {dist.max():.6f}, "
        f"unique rows {len(np.unique(idx))}/{len(idx)}")
    if dist.max() > 1e-3 or len(np.unique(idx)) != len(idx):
        raise RuntimeError("could not place slide-0 rows on model rows unambiguously")
    return idx.astype(int)


def decode_vocabularies(ctx, rows, pert_raw):
    """Identify which cell_atts column is tissue and which is perturbation, and the codes.

    The perturbation vocabulary comes from the model itself (ctx['name2code']). Tissue is
    derived the way the original notebook derived it -- every knockout plus KP is tumour,
    'periphery' is its own class, 'None' is normal -- and matched against the remaining
    column by cross-tabulation, which is unambiguous because row order is shared.
    """
    atts = np.asarray(ctx["cell_atts"])[rows]
    log(f"  cell_atts shape {atts.shape}, per-column uniques "
        f"{[sorted(np.unique(atts[:, c]).tolist()) for c in range(atts.shape[1])]}")
    log(f"  ctx['name2code'] = {ctx.get('name2code')}")
    for key in ("cls", "pert_name"):
        v = ctx.get(key)
        log(f"  ctx['{key}'] = {v if not hasattr(v, 'shape') else np.unique(np.asarray(v))}")

    tissue_lab = np.where(pert_raw == "periphery", "periphery",
                          np.where(pert_raw == "None", "normal", "tumor"))

    def crosstab(col, labels):
        table = {}
        for lab in np.unique(labels):
            codes, counts = np.unique(atts[labels == lab, col], return_counts=True)
            table[str(lab)] = {int(c): int(n) for c, n in zip(codes, counts)}
        return table

    for col in range(atts.shape[1]):
        log(f"  column {col} vs tissue: {crosstab(col, tissue_lab)}")
        log(f"  column {col} vs perturbation: {crosstab(col, pert_raw)}")

    # Purity score: a column is the tissue axis if each tissue label maps to one code.
    def purity(col, labels):
        tot = ok = 0
        for lab in np.unique(labels):
            codes, counts = np.unique(atts[labels == lab, col], return_counts=True)
            tot += counts.sum()
            ok += counts.max()
        return ok / max(tot, 1)

    scores = {col: (purity(col, tissue_lab), purity(col, pert_raw))
              for col in range(atts.shape[1])}
    log(f"  purity (tissue, perturbation) per column: {scores}")
    tissue_col = max(scores, key=lambda c: scores[c][0] - scores[c][1])
    pert_col = 1 - tissue_col if atts.shape[1] == 2 else max(
        scores, key=lambda c: scores[c][1] - scores[c][0])

    tissue_map = {}
    for lab in np.unique(tissue_lab):
        codes, counts = np.unique(atts[tissue_lab == lab, tissue_col], return_counts=True)
        tissue_map[str(lab)] = int(codes[np.argmax(counts)])
    pert_map = {str(k): int(v) for k, v in (ctx.get("name2code") or {}).items()}
    log(f"  -> tissue column {tissue_col} map {tissue_map}")
    log(f"  -> perturbation column {pert_col} map {pert_map}")
    return {"tissue": tissue_map, "perturbation": pert_map,
            "tissue_col": int(tissue_col), "pert_col": int(pert_col)}, atts


def encoder_space_transform(Y_obs: np.ndarray, Y_expr: np.ndarray):
    """Recover the training normalization: raw counts -> z-scored log expression.

    Y_expr is adata.X after size-factor normalization, log1p and per-gene scaling, so it is
    an affine function of L = log1p(counts / size_factor). Both matrices are available for
    the observed cells, so the per-gene (mean, sd) are recovered exactly by matching moments
    rather than guessed. Returns a callable that maps predicted counts into encoder space,
    and the round-trip error on the observed data as a check that the mapping is right.
    """
    def to_log(counts):
        lib = counts.sum(axis=1, keepdims=True).astype(np.float64)
        sf = lib / np.median(lib[lib > 0]) if np.any(lib > 0) else np.ones_like(lib)
        sf[sf == 0] = 1.0
        return np.log1p(counts / sf)

    L = to_log(np.asarray(Y_obs, dtype=np.float64))
    sd_L, sd_E = L.std(axis=0), Y_expr.std(axis=0)
    scale = np.where(sd_E > 1e-8, sd_L / np.maximum(sd_E, 1e-8), 1.0)
    centre = L.mean(axis=0) - scale * Y_expr.mean(axis=0)

    def to_encoder(counts):
        return ((to_log(np.asarray(counts, dtype=np.float64)) - centre)
                / np.maximum(scale, 1e-8)).astype(np.float32)

    err = float(np.abs(to_encoder(Y_obs) - Y_expr).max())
    return to_encoder, err


def predict(ctx, args, rows, perturb_local_ids, tissue_code, pert_code,
            has_guide=True, is_real_ko=True, Y_override=None):
    model = ctx["model"]
    kwargs = dict(
        X=ctx["pos_batched"][rows],
        sample_index=ctx["sample_indices"][rows],
        cell_atts=np.asarray(ctx["cell_atts"])[rows].copy(),
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        Y=ctx["Y_expr"][rows] if Y_override is None else Y_override,
    )
    if perturb_local_ids is not None:
        # An EMPTY id list is meaningful: it selects the per-cell flag path with nothing
        # edited, i.e. the honest reconstruction baseline in which every cell keeps the
        # perturbation state it actually has. Passing None instead falls back to the legacy
        # whole-batch semantics, which switch every cell's perturbation shift OFF.
        kwargs.update(
            perturb_cell_id=torch.as_tensor(np.asarray(perturb_local_ids, dtype=int)),
            target_has_guide=bool(has_guide),
            target_is_real_KO=bool(is_real_ko),
        )
        if tissue_code is not None:
            kwargs["target_cell_tissue"] = int(tissue_code)
        if pert_code is not None:
            kwargs["target_cell_perturbation"] = int(pert_code)
    means, _ = model.counterfactualPrediction(**kwargs)
    return np.asarray(means)


def nex(mat, rows, gene_idx):
    """Mean normalized expression of one gene over a set of rows (log1p of CP10K)."""
    lib = mat.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return float(np.log1p(mat[rows][:, gene_idx] / lib[rows, 0] * 1e4).mean())


def profile_r2(a, b):
    """R^2 between two mean expression profiles, in log space."""
    x = np.log1p(a / max(a.sum(), 1e-9) * 1e4)
    y = np.log1p(b / max(b.sum(), 1e-9) * 1e4)
    r = np.corrcoef(x, y)[0, 1]
    return float(r ** 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    args = build_args()

    log("== loading raw slide ==")
    with h5py.File(RAW, "r") as f:
        pos_raw = np.array(f["pos"]).T.astype(float)
        pert_raw = np.array(f["perturbation"]).astype(str)
        tissue_raw = np.array(f["tissue"]).astype(str)
        genes_raw = np.array(f["gene"]).astype(str)
    log(f"  raw: {pos_raw.shape[0]} spots, labels {dict(zip(*np.unique(pert_raw, return_counts=True)))}")

    log("== building model context (slide 0) ==")
    ctx = R.build_pm_ctx(args, "0")
    log(f"  ctx keys: {sorted(ctx.keys())}")
    for k in ("pos_batched", "cell_atts", "Y_expr", "Y_obs", "sample_indices"):
        if k in ctx:
            log(f"    {k}: {np.asarray(ctx[k]).shape}")
    gene_names = np.asarray(ctx["gene_names"]).astype(str)
    log(f"  genes in model: {len(gene_names)}")

    rows = locate_slide0_rows(ctx, pos_raw)
    maps, atts = decode_vocabularies(ctx, rows, pert_raw)

    sel = {}
    for name, path in SELECTIONS.items():
        idx0 = np.loadtxt(path, dtype=int) - 1          # R is 1-based
        sel[name] = idx0
        log(f"  {name}: n={len(idx0)} labels={dict(zip(*np.unique(pert_raw[idx0], return_counts=True)))}")

    # Gene indices for the reported markers.
    gidx = {}
    for g in ("Plac8", "Fn1"):
        hit = np.where(gene_names == g)[0]
        gidx[g] = int(hit[0]) if len(hit) else None
        log(f"  gene {g}: {'index ' + str(gidx[g]) if len(hit) else 'NOT IN MODEL GENE SET'}")

    results = {"checkpoint": str(CKPT), "selections": {k: len(v) for k, v in sel.items()},
               "vocabularies": maps, "genes_present": {k: v is not None for k, v in gidx.items()}}

    tissue_col, pert_col = maps["tissue_col"], maps["pert_col"]
    log(f"  using tissue column {tissue_col}, perturbation column {pert_col}")

    log("== baseline (no edit) ==")
    base = predict(ctx, args, rows, [], None, None)
    log(f"  baseline prediction {base.shape} (per-cell path, nothing edited)")
    np.save(OUT / "pred_baseline.npy", base.astype(np.float32))

    obs_jak2 = np.where(pert_raw == "Jak2")[0]
    obs_peri = np.where(pert_raw == "periphery")[0]

    # Analysis 1: transplantation
    if True:
        jak2_code = maps["perturbation"].get("Jak2")
        tumor_code = maps["tissue"].get("tumor")
        log(f"== transplantation (Jak2 code {jak2_code}, tumour tissue code {tumor_code}) ==")
        trans = {}
        for dest in ("dest_core", "dest_normal"):
            pred = predict(ctx, args, rows, sel[dest], tumor_code, jak2_code)
            np.save(OUT / f"pred_transplant_{dest}.npy", pred.astype(np.float32))
            entry = {
                "n_spots": int(len(sel[dest])),
                "r2_to_observed_jak2": profile_r2(pred[sel[dest]].mean(0),
                                                  ctx["Y_obs"][rows][obs_jak2].mean(0)),
                "r2_baseline_to_observed_jak2": profile_r2(base[sel[dest]].mean(0),
                                                           ctx["Y_obs"][rows][obs_jak2].mean(0)),
            }
            if gidx["Plac8"] is not None:
                entry["Plac8_predicted"] = nex(pred, sel[dest], gidx["Plac8"])
                entry["Plac8_baseline"] = nex(base, sel[dest], gidx["Plac8"])
                entry["Plac8_observed_jak2"] = nex(np.asarray(ctx["Y_obs"])[rows],
                                                   obs_jak2, gidx["Plac8"])
            trans[dest] = entry
            log(f"  {dest}: {json.dumps(entry, indent=4)}")
        results["transplantation"] = trans

    # Analysis 2: niche editing
    if True:
        peri_tissue = maps["tissue"].get("periphery")
        # The ring keeps a basal (non-knockout) perturbation identity: only its tissue
        # context is edited, which is what "perturb the surrounding niche" means here.
        keep_pert = maps["perturbation"].get("__basal__", 0)
        log(f"== niche editing (ring -> periphery tissue code {peri_tissue}) ==")
        pred = predict(ctx, args, rows, sel["peri_ring"], peri_tissue, keep_pert,
                       has_guide=False, is_real_ko=False)
        np.save(OUT / "pred_niche_edit.npy", pred.astype(np.float32))
        core = sel["tgfbr2_core"]
        delta = np.abs(pred[core] - base[core]).mean()
        entry = {
            "n_core": int(len(core)), "n_ring": int(len(sel["peri_ring"])),
            "mean_abs_change_in_core": float(delta),
            "core_moved": bool(delta > 1e-8),
            "r2_core_to_observed_periphery_before":
                profile_r2(base[core].mean(0), np.asarray(ctx["Y_obs"])[rows][obs_peri].mean(0)),
            "r2_core_to_observed_periphery_after":
                profile_r2(pred[core].mean(0), np.asarray(ctx["Y_obs"])[rows][obs_peri].mean(0)),
        }
        if gidx["Fn1"] is not None:
            entry["Fn1_core_before"] = nex(base, core, gidx["Fn1"])
            entry["Fn1_core_after"] = nex(pred, core, gidx["Fn1"])
            entry["Fn1_observed_periphery"] = nex(np.asarray(ctx["Y_obs"])[rows],
                                                  obs_peri, gidx["Fn1"])
        results["niche_editing"] = entry
        log(f"  {json.dumps(entry, indent=4)}")
        # The edited ring itself must move -- if it does not, the edit never applied.
        ring_delta = float(np.abs(pred[sel["peri_ring"]] - base[sel["peri_ring"]]).mean())
        results["niche_editing"]["mean_abs_change_in_edited_ring"] = ring_delta
        log(f"  sanity: mean |change| in the EDITED ring = {ring_delta:.6f}")

    # Niche-propagation probe
    # The core's prediction is only worth reporting as "propagated" if it moves by more
    # than decoder sampling noise, so measure that floor, and try a much stronger ring
    # edit to establish whether ANY surrounding edit reaches an unedited core.
    log("== niche propagation probe ==")
    core = sel["tgfbr2_core"]
    ring = sel["peri_ring"]
    # Baseline with NOTHING edited but every cell keeping its own perturbation state --
    # the only correct reference for "did editing the ring move the core".
    base_own = predict(ctx, args, rows, [], None, None)
    base_own2 = predict(ctx, args, rows, [], None, None)
    np.save(OUT / "pred_baseline_owncells.npy", base_own.astype(np.float32))
    base2 = base_own2
    base = base_own
    noise_core = float(np.abs(base_own2[core] - base_own[core]).mean())
    noise_all = float(np.abs(base_own2 - base_own).mean())
    log(f"  sampling-noise floor: core {noise_core:.6f}, all spots {noise_all:.6f}")

    probe = {"noise_floor_core": noise_core, "noise_floor_all": noise_all}
    variants = {
        "ring_to_periphery_tissue": dict(tissue=maps["tissue"]["periphery"],
                                         pert=maps["perturbation"]["__basal__"],
                                         guide=False, ko=False),
        "ring_to_tumour_tissue": dict(tissue=maps["tissue"]["tumor"],
                                      pert=maps["perturbation"]["__basal__"],
                                      guide=False, ko=False),
        "ring_to_Jak2_KO": dict(tissue=maps["tissue"]["tumor"],
                                pert=maps["perturbation"]["Jak2"],
                                guide=True, ko=True),
        "ring_to_Tgfbr2_KO": dict(tissue=maps["tissue"]["tumor"],
                                  pert=maps["perturbation"]["Tgfbr2"],
                                  guide=True, ko=True),
    }
    # Placement control: the SAME edit applied to an equally sized patch far from the core.
    # A spatial field must move the core less from far away; anything global moves it equally.
    centroid = pos_raw[core].mean(axis=0)
    far_order = np.argsort(-((pos_raw - centroid) ** 2).sum(axis=1))
    far_ring = np.array([i for i in far_order if i not in set(core.tolist())][:len(ring)])
    log(f"  distant control patch: n={len(far_ring)}, mean distance to core centroid "
        f"{np.sqrt(((pos_raw[far_ring] - centroid) ** 2).sum(axis=1)).mean():.1f} "
        f"(ring: {np.sqrt(((pos_raw[ring] - centroid) ** 2).sum(axis=1)).mean():.1f})")
    variants["distant_patch_to_Tgfbr2_KO"] = dict(tissue=maps["tissue"]["tumor"],
                                                  pert=maps["perturbation"]["Tgfbr2"],
                                                  guide=True, ko=True, ids=far_ring)
    for name, v in variants.items():
        edit_ids = v.get("ids", ring)
        p = predict(ctx, args, rows, edit_ids, v["tissue"], v["pert"],
                    has_guide=v["guide"], is_real_ko=v["ko"])
        np.save(OUT / f"pred_{name}.npy", p.astype(np.float32))
        d_core = float(np.abs(p[core] - base[core]).mean())
        d_ring = float(np.abs(p[edit_ids] - base[edit_ids]).mean())
        probe[name] = {
            "change_in_edited_ring": d_ring,
            "change_in_unedited_core": d_core,
            "core_over_noise": d_core / max(noise_core, 1e-12),
            "ring_over_noise": d_ring / max(noise_core, 1e-12),
        }
        log(f"  {name}: ring {d_ring:.6f} ({probe[name]['ring_over_noise']:.2f}x noise), "
            f"core {d_core:.6f} ({probe[name]['core_over_noise']:.2f}x noise)")
    results["niche_propagation_probe"] = probe

    # Two-pass niche refresh
    # The single-pass edit cannot propagate: the neighbourhood descriptor is built from
    # OBSERVED expression, while the edit changes labels, and with use_basal_encoder_only
    # the label never reaches the encoder. So run the edit, take the edited cells' PREDICTED
    # expression, put it back into encoder space, and re-predict. Now the core's neighbours
    # genuinely carry a perturbed state and the descriptor sees it.
    log("== two-pass niche refresh ==")
    to_enc, rt_err = encoder_space_transform(np.asarray(ctx["Y_obs"])[rows],
                                             np.asarray(ctx["Y_expr"])[rows])
    log(f"  encoder-space round-trip max error on observed data: {rt_err:.4f} "
        f"(must be ~0 for the mapping to be trusted)")
    results["encoder_space_roundtrip_max_err"] = rt_err

    Y_base = np.asarray(ctx["Y_expr"])[rows]
    twopass = {}
    for name, edit_ids in (("ring", ring), ("distant_patch", far_ring)):
        pass1 = predict(ctx, args, rows, edit_ids, maps["tissue"]["tumor"],
                        maps["perturbation"]["Tgfbr2"], has_guide=True, is_real_ko=True)
        Y2 = Y_base.copy()
        Y2[edit_ids] = to_enc(pass1[edit_ids])
        moved = float(np.abs(Y2[edit_ids] - Y_base[edit_ids]).mean())
        pass2 = predict(ctx, args, rows, [], None, None, Y_override=Y2)
        np.save(OUT / f"pred_twopass_{name}.npy", pass2.astype(np.float32))
        d_core = float(np.abs(pass2[core] - base[core]).mean())
        twopass[name] = {
            "encoder_input_change_in_edited": moved,
            "change_in_unedited_core": d_core,
            "core_over_noise": d_core / max(noise_core, 1e-12),
        }
        log(f"  {name}: edited-cell encoder input moved {moved:.4f}; "
            f"unedited core {d_core:.6f} ({twopass[name]['core_over_noise']:.2f}x noise)")
    results["two_pass_niche_refresh"] = twopass

    # Second baseline draw kept: panels need the noise floor to draw a "no effect" band.
    np.save(OUT / "pred_baseline_draw2.npy", base2.astype(np.float32))
    np.savez_compressed(
        OUT / "case_study_bundle.npz",
        pos_raw=pos_raw,
        pert_raw=pert_raw.astype("U16"),
        gene_names=gene_names.astype("U32"),
        Y_obs=np.asarray(ctx["Y_obs"])[rows].astype(np.float32),
        **{f"sel_{k}": v for k, v in sel.items()},
    )
    log("wrote " + str(OUT / "case_study_bundle.npz"))

    (OUT / "case_study_stats.json").write_text(json.dumps(results, indent=2))
    log("wrote " + str(OUT / "case_study_stats.json"))


if __name__ == "__main__":
    main()
