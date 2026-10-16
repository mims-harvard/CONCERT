"""Descriptive-reconstruction validation for the DSS composition model (spec Sec. 6).

Trained on ALL 4 timepoints. Validates that the membership field pi reproduces the observed
inflammation COMPOSITION over time, is spatially contiguous, and is individualized per mouse --
the capabilities the fixed-state (intensity-only) model lacked. NOT a held-out-peak test.

Metrics written to <outdir>/composition_eval/:
  1. composition_calibration.csv  -- predicted pi-fraction vs observed label-fraction, per day
  2. spatial_moran.csv            -- Moran's I of predicted pi vs observed label, per slice/day
  3. heterogeneity.csv            -- cross-mouse variance of pi at a common projected day
  4. counterfactual_flip.json     -- pi->0 collapses the inflammation program (mechanism/identifiability)
  + composition_summary.png
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import scipy.sparse as sp

INERT_SHIFT_TOL = 0.05


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _moran_np(vals: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    """Simple kNN Moran's I of a scalar field over 2D coords (numpy)."""
    from sklearn.neighbors import NearestNeighbors
    n = len(vals)
    if n < k + 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]                                  # drop self
    z = vals - vals.mean()
    num = np.sum(z[:, None] * z[idx]) / (n * k)
    den = np.sum(z ** 2) / n
    return float(num / (den + 1e-12))


def run_composition_eval(model, adata, cell_atts, pos_batched, sample_indices, sample_id_all,
                         multi_paths, gene_names, pert_map, cfg, outdir) -> Dict:
    import anndata as _ad
    outdir = Path(outdir) / "composition_eval"
    outdir.mkdir(parents=True, exist_ok=True)

    # per-cell source day + niche + observed label
    day_by_file = np.array([
        float(_ad.read_h5ad(p, backed="r").obs[cfg.time_col].iloc[0]) for p in multi_paths],
        dtype=np.float32)
    src_day = day_by_file[np.asarray(sample_id_all).astype(int)]
    state = np.asarray(cell_atts)[:, 1].astype(int)
    pert_code = int(pert_map.get("perturbed", 1))
    y_obs = (state == pert_code).astype(float)
    Y_expr = _dense(adata.X).astype(np.float32)
    coords = np.asarray(pos_batched)[:, :2]
    bs = int(getattr(cfg, "loto_batch_size", 512))
    denom = float(getattr(cfg, "loto_day_denom", 21.0))

    # pi at each cell's OWN observed day (calibration should match observed fractions)
    pi_obs = model.predict_membership(
        X=pos_batched, sample_index=sample_indices, cell_atts=cell_atts,
        Y=Y_expr, time_t=src_day, batch_size=bs)

    # 1. composition calibration per day
    rows = []
    for d in np.unique(np.round(src_day, 4)):
        m = np.isclose(src_day, d, atol=1e-3)
        rows.append({"day_norm": float(d), "day": float(d * denom),
                     "n": int(m.sum()), "obs_frac": float(y_obs[m].mean()),
                     "pred_pi_mean": float(pi_obs[m].mean())})
    import csv
    with open(outdir / "composition_calibration.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    logging.info("[comp] calibration by day (day, obs_frac, pred_pi): %s",
                 [(r["day"], round(r["obs_frac"], 3), round(r["pred_pi_mean"], 3)) for r in rows])

    # 2. spatial Moran of pi vs observed label, per slice
    srows = []
    for s in np.unique(np.asarray(sample_id_all)):
        m = np.asarray(sample_id_all) == s
        if m.sum() < 20:
            continue
        srows.append({"slide": int(s), "day": float(src_day[m][0] * denom),
                      "n": int(m.sum()),
                      "moran_pred_pi": _moran_np(pi_obs[m], coords[m]),
                      "moran_obs_label": _moran_np(y_obs[m], coords[m])})
    with open(outdir / "spatial_moran.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys())); w.writeheader(); w.writerows(srows)

    # 3. per-mouse heterogeneity: project every source cell to the PEAK day, measure cross-slide spread
    t_peak = float(getattr(cfg, "loto_target_day", 9.0)) / denom
    pi_peak = model.predict_membership(
        X=pos_batched, sample_index=sample_indices, cell_atts=cell_atts,
        Y=Y_expr, time_t=np.full(len(src_day), t_peak, np.float32), batch_size=bs)
    hrows = []
    for s in np.unique(np.asarray(sample_id_all)):
        m = np.asarray(sample_id_all) == s
        if m.sum() < 20:
            continue
        hrows.append({"slide": int(s), "proj_day": t_peak * denom,
                      "pred_pi_mean_at_peak": float(pi_peak[m].mean())})
    with open(outdir / "heterogeneity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hrows[0].keys())); w.writeheader(); w.writerows(hrows)
    spread = float(np.std([r["pred_pi_mean_at_peak"] for r in hrows]))
    logging.info("[comp] cross-slide pi spread at peak day = %.4f (interp baseline = 0)", spread)

    # 4. counterfactual-flip / mechanism (spec Sec. 5 identifiability): decode inflamed cells with
    #    the program ON (perturbed) vs OFF (basal); the inflammation program must collapse.
    import torch
    inflamed = np.where(y_obs > 0.5)[0]
    sub = inflamed if inflamed.size <= 20000 else np.sort(
        np.random.default_rng(0).choice(inflamed, 20000, replace=False))
    common = dict(
        X=np.asarray(pos_batched)[sub], sample_index=torch.as_tensor(np.asarray(sample_indices)[sub]),
        cell_atts=np.asarray(cell_atts)[sub].copy(), perturb_cell_id=torch.arange(sub.size),
        target_cell_tissue=int(cell_atts[sub[0], 0]), n_samples=5,
        Y=Y_expr[sub], time_t=src_day[sub])
    mu_on, _ = model.counterfactualPrediction(
        target_cell_perturbation=pert_code, target_has_guide=True, target_is_real_KO=True, **common)
    mu_off, _ = model.counterfactualPrediction(
        target_cell_perturbation=int(pert_map.get("__basal__", 0)),
        target_has_guide=False, target_is_real_KO=False, **common)

    def _cp(counts):
        c = np.asarray(counts, float); c = c / np.clip(c.sum(1, keepdims=True), 1e-8, None) * 1e4
        return c.mean(0)
    collapse = float(np.linalg.norm(np.log(_cp(mu_on) + 1e-3) - np.log(_cp(mu_off) + 1e-3)))
    (outdir / "counterfactual_flip.json").write_text(json.dumps(
        {"program_on_vs_off_lognorm_L2": collapse, "n_cells": int(sub.size),
         "interpretation": "large L2 => pi gates a real inflammation program (mechanism confirmed)"},
        indent=2))
    logging.info("[comp] counterfactual-flip program collapse (log-CP10K L2) = %.4f", collapse)

    _plot(outdir, rows, srows, hrows, spread)
    logging.info("[comp] wrote composition_eval/ outputs to %s", outdir)
    return {"calibration": rows, "heterogeneity_spread": spread, "flip_collapse": collapse}


def run_field_trajectory_test(model, adata, cell_atts, pos_batched, sample_indices,
                              sample_id_all, multi_paths, gene_names, pert_map, cfg, outdir,
                              tier1_id=None, arch_desc=None, infl_score_all=None) -> Dict:
    """THE MAKE-OR-BREAK TEST (v3 continuous inflammation field, spec Sec. 6).

    Re-decode each cell at target days using counterfactualPrediction(time_t=...) and read the
    predicted DSS marker composition. SUCCESS = inflammation MOVES with time under re-decode:
      * day0 (healthy) cells decoded at t=9 RISE above the same cells decoded at t=0
      * day9 cells decoded at t=21 FALL below the same cells decoded at t=9

    counterfactualPrediction slices self.mask_cutoff positionally, so we decode the FULL cell
    set at each target day (constant time_t) and subset the OUTPUT by source day afterward.
    Pert code is passed as INT; the continuous field s is per-cell (not batch-uniform), so a
    single call per target day suffices (unlike the old binary gate).
    """
    import torch
    outdir = Path(outdir) / "field_trajectory"
    outdir.mkdir(parents=True, exist_ok=True)

    denom = float(getattr(cfg, "loto_day_denom", 21.0))
    day_by_file = np.array([
        float(_import_ad().read_h5ad(p, backed="r").obs[cfg.time_col].iloc[0]) for p in multi_paths],
        dtype=np.float32)
    src_day = day_by_file[np.asarray(sample_id_all).astype(int)] * denom      # actual day (0/3/9/21)
    Y_expr = _dense(adata.X).astype(np.float32)
    X_full = np.asarray(pos_batched)
    N = X_full.shape[0]
    pert_code = int(pert_map.get("perturbed", 1))
    bs = int(getattr(cfg, "loto_batch_size", 512))

    if bool(getattr(model, "use_shared_basal", False)):
        if tier1_id is None or arch_desc is None:
            raise ValueError(
                "run_field_trajectory_test: model.use_shared_basal is True but tier1_id/"
                "arch_desc were not provided; the leakage-free re-decode cannot derive the "
                "basal without the per-cell Tier1 code and architectural descriptor. Pass "
                "tier1_idx_all/arch_desc_all from run() (they are row-aligned to adata).")
        tier1_codes = np.asarray(tier1_id)
        arch_codes = np.asarray(arch_desc, dtype=np.float32)
        if tier1_codes.shape[0] != N or arch_codes.shape[0] != N:
            raise ValueError(
                f"run_field_trajectory_test: code arrays misaligned with cell set "
                f"(tier1={tier1_codes.shape[0]}, arch={arch_codes.shape[0]}, N={N}).")
        if arch_codes.ndim != 2 or arch_codes.shape[1] != int(getattr(model, "arch_dim", -1)):
            raise ValueError(
                f"run_field_trajectory_test: arch_desc shape {arch_codes.shape} does not match "
                f"model arch_dim={getattr(model, 'arch_dim', None)}.")
    else:
        tier1_codes = None
        arch_codes = None

    # DSS marker panel -> indices in the model gene axis (gene_names order == counterfactual output)
    DSS_MARKERS = ['Isg15', 'Ifit1', 'Tnf', 'Cxcl1', 'Nos2', 'Il11', 'Il1b', 'Mmp3', 'Mmp10',
                   'Mmp13', 'Plau', 'Igfbp5', 'Col18a1', 'Mmp2', 'S100a8', 'S100a9', 'Osm',
                   'Cxcl10', 'Cxcl5', 'Ly6a']
    gnames = list(map(str, gene_names))
    mk_idx = [gnames.index(g) for g in DSS_MARKERS if g in gnames]
    logging.info("[traj] %d/%d marker genes present in model gene axis", len(mk_idx), len(DSS_MARKERS))

    # shift-sign diagnostic
    # a(t) = softplus(...) >= 0 only scales the MAGNITUDE of `shift` (= Phi*f_pert), which is
    # SIGNED (clamped to +/-f_pert_clamp). So the gate can be passed two ways:
    #   shift > 0  -> basal is healthy and a(t)*shift ADDS inflammation      (intended)
    #   shift < 0  -> basal carries inflammation and the term SUBTRACTS it   (INVERTED:
    #                 a smaller a(9) attenuates a negative shift and markers rise, which
    #                 would pass the trajectory test while "f_pert = inflammation program"
    #                 is false). Accumulate the sign over the marker genes to tell them apart.
    _shift_stats: Dict = {}
    _cur_t = [None]
    _orig_shift_fn = model._pert_log_shift_from_lord

    # Per-cell a(t) capture, keyed by target day. _pert_log_shift_from_lord is called exactly
    # ONCE per batch (before the n_samples loop), and counterfactualPrediction walks batches in
    # order over perturb_cell_id=arange(N), so concatenating these gives an (N,) vector aligned
    # to the same cell order as src_day. This lets us split a(t) by SOURCE day, which the global
    # mean_a_t hides: training only ever supervises the field at each cell's OWN day, so
    # a(day-0 cell, t=9) is an unsupervised off-diagonal extrapolation.
    _a_by_t: Dict = {}

    def _instrumented_shift_fn(*a, **k):
        out = _orig_shift_fn(*a, **k)
        s = getattr(model, "_last_shift", None)
        at = getattr(model, "_last_a", None)
        if s is not None and _cur_t[0] is not None:
            sm = s[:, mk_idx] if mk_idx else s
            acc = _shift_stats.setdefault(_cur_t[0],
                                          {"sum": 0.0, "pos": 0, "n": 0, "a_sum": 0.0, "a_n": 0})
            acc["sum"] += float(sm.sum().item())
            acc["pos"] += int((sm > 0).sum().item())
            acc["n"] += int(sm.numel())
            if at is not None:
                acc["a_sum"] += float(at.sum().item())
                acc["a_n"] += int(at.numel())
                _a_by_t.setdefault(_cur_t[0], []).append(
                    at.detach().float().reshape(-1).cpu().numpy())
        return out

    model._pert_log_shift_from_lord = _instrumented_shift_fn

    def _decode_marker_score(t_day: float) -> np.ndarray:
        """Full-set re-decode at target day -> per-cell mean log1p predicted marker expression."""
        _cur_t[0] = float(t_day)
        tt = np.full(N, float(t_day) / denom, dtype=np.float32)
        # Same cell_atts + Y + arch_desc every call -> identical basal across t (the basal path
        # takes no time input at all); ONLY time_t (-> a(t)) varies,
        # so any change in the marker score is attributable purely to the field s(coords,niche,t).
        # tissue kept per-cell (target_cell_tissue=None); pert code set to the shared inflamed
        # program (INT) so f_pert is the inflammation signature that s scales.
        mu, _ = model.counterfactualPrediction(
            X=X_full, sample_index=torch.as_tensor(np.asarray(sample_indices)),
            cell_atts=np.asarray(cell_atts).copy(), perturb_cell_id=torch.arange(N),
            target_cell_tissue=None, target_cell_perturbation=pert_code,
            target_has_guide=True, target_is_real_KO=True,
            n_samples=3, batch_size=bs, Y=Y_expr, time_t=tt,
            tier1_id=tier1_codes, arch_desc=arch_codes)
        # NOT size-factor-scaled on purpose: the trajectory gate compares the SAME cells
        # across t, and a per-cell time-independent factor cannot change its direction.
        # Scaling here would perturb an already-settled gate result for no gain.
        mu = np.asarray(mu, dtype=np.float32)
        return np.log1p(mu[:, mk_idx]).mean(1) if mk_idx else np.zeros(N, np.float32)

    # Decode the full set at the three target days used by the forward/backward trajectory.
    score_t0  = _decode_marker_score(0.0)
    score_t9  = _decode_marker_score(9.0)
    score_t21 = _decode_marker_score(21.0)
    model._pert_log_shift_from_lord = _orig_shift_fn          # restore

    shift_diag = {}
    logging.info("[shift] === MECHANISM SIGN CHECK (marker genes) ===")
    for t_day in sorted(_shift_stats):
        acc = _shift_stats[t_day]
        mean_shift = acc["sum"] / max(1, acc["n"])
        frac_pos = acc["pos"] / max(1, acc["n"])
        mean_a = acc["a_sum"] / max(1, acc["a_n"])
        shift_diag[f"t{int(t_day)}"] = {"mean_shift": mean_shift, "frac_shift_positive": frac_pos,
                                        "mean_a_t": mean_a, "mean_a_times_shift": mean_a * mean_shift}
        logging.info("[shift] t=%2d  mean_shift=%+.4f  frac>0=%.3f  a(t)=%.4f  a(t)*shift=%+.4f",
                     int(t_day), mean_shift, frac_pos, mean_a, mean_a * mean_shift)
    if shift_diag:
        _p = [v["frac_shift_positive"] for v in shift_diag.values()]
        _mag = [abs(v["mean_shift"]) for v in shift_diag.values()]
        if bool(getattr(model, "marker_sign_positive", False)):
            # The marker columns are softplus-constrained, so frac>0 == 1.0 BY CONSTRUCTION
            # and "INTENDED" here would be a tautology, not evidence -- this project has
            # already banked four false gate passes, do not add a fifth. The constraint
            # removes the INVERTED failure mode and leaves exactly one: the model can go
            # INERT, driving the marker shift to ~0 and keeping inflammation in basal.
            # So the discriminating quantity is now the shift MAGNITUDE, not its sign.
            verdict = ("INERT (marker shift ~0: constraint satisfied by switching the pert "
                       "path OFF; inflammation still lives in basal)" if max(_mag) < INERT_SHIFT_TOL else
                       "ACTIVE (marker shift > 0 with real magnitude; sign is constrained, "
                       "so this is NOT independent evidence the mechanism is right -- read the "
                       "trajectory gate alongside it)")
        else:
            verdict = ("INTENDED (shift>0: a(t) ADDS inflammation)" if min(_p) > 0.5 else
                       "INVERTED (shift<0: basal carries inflammation, pert term SUBTRACTS it)"
                       if max(_p) < 0.5 else "MIXED/AMBIGUOUS (shift sign not consistent)")
        logging.info("[shift] VERDICT: %s", verdict)
        shift_diag["verdict"] = verdict
        shift_diag["marker_sign_constrained"] = bool(getattr(model, "marker_sign_positive", False))

    day0 = np.isclose(src_day, 0.0, atol=1e-3)
    day9 = np.isclose(src_day, 9.0, atol=1e-3)

    a_by_day = {}
    for t_day, chunks in sorted(_a_by_t.items()):
        av = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        if av.shape[0] != N:
            logging.warning("[a-by-day] a(t=%s) length %d != N=%d; skipping breakdown "
                            "(batch/sample alignment assumption broke)", t_day, av.shape[0], N)
            continue
        for d_name, mask in (("day0", day0), ("day9", day9)):
            if int(mask.sum()) == 0:
                continue
            entry = a_by_day.setdefault(f"src_{d_name}", {})
            entry[f"a_at_t{int(t_day)}"] = float(av[mask].mean())
            entry["is_diagonal_at"] = f"t{0 if d_name == 'day0' else 9}"
    if a_by_day:
        logging.info("[a-by-day] === a(t) SPLIT BY SOURCE DAY (diagonal = cell's own day) ===")
        for k, v in a_by_day.items():
            logging.info("[a-by-day] %s: %s (diagonal at %s)", k,
                         {kk: round(vv, 4) for kk, vv in v.items() if kk.startswith("a_at")},
                         v.get("is_diagonal_at"))
        if infl_score_all is not None:
            arr = np.asarray(infl_score_all, dtype=np.float32).reshape(-1)
            if arr.shape[0] == N:
                for d_name, mask in (("day0", day0), ("day9", day9)):
                    if int(mask.sum()) > 0:
                        a_by_day.setdefault(f"src_{d_name}", {})["observed_infl_score_target"] = \
                            float(arr[mask].mean())
                logging.info("[a-by-day] observed infl target: day0=%.4f day9=%.4f",
                             float(arr[day0].mean()) if int(day0.sum()) else float("nan"),
                             float(arr[day9].mean()) if int(day9.sum()) else float("nan"))

    def _m(v, mask):
        return float(v[mask].mean()) if int(mask.sum()) > 0 else float("nan")

    res = {
        "shift_sign_diagnostic": shift_diag,
        "a_by_source_day": a_by_day,
        "n_day0_cells": int(day0.sum()), "n_day9_cells": int(day9.sum()),
        "day0_to_t0":  _m(score_t0,  day0),
        "day0_to_t9":  _m(score_t9,  day0),
        "day9_to_t9":  _m(score_t9,  day9),
        "day9_to_t21": _m(score_t21, day9),
    }
    res["forward_rises"]  = bool(res["day0_to_t9"]  > res["day0_to_t0"])   # inflammation onset
    res["backward_falls"] = bool(res["day9_to_t21"] < res["day9_to_t9"])   # resolution
    res["SUCCESS"] = bool(res["forward_rises"] and res["backward_falls"])
    res["metric"] = "mean log1p predicted DSS-marker expression (count space)"

    (outdir / "field_trajectory_test.json").write_text(json.dumps(res, indent=2))
    logging.info("[traj] === MAKE-OR-BREAK TEST ===")
    logging.info("[traj] day0->t0 =%.4f  day0->t9 =%.4f  (forward RISES: %s)",
                 res["day0_to_t0"], res["day0_to_t9"], res["forward_rises"])
    logging.info("[traj] day9->t9 =%.4f  day9->t21=%.4f  (backward FALLS: %s)",
                 res["day9_to_t9"], res["day9_to_t21"], res["backward_falls"])
    logging.info("[traj] SUCCESS = %s  -> inflammation %s with time under re-decode",
                 res["SUCCESS"], "MOVES" if res["SUCCESS"] else "did NOT move")
    return res


def run_case_study_deliverables(model, adata, cell_atts, pos_batched, sample_indices,
                                sample_id_all, multi_paths, gene_names, pert_map, cfg, outdir,
                                tier1_id=None, arch_desc=None) -> Dict:
    """Descriptive case-study deliverables C1-C3 on the FINAL fixed-t timekernel ckpt.

    All three are pure inference on the existing model.pt (no retrain). They reuse the SAME
    leakage-free re-decode as run_field_trajectory_test (basal from Tier1+arch codes only,
    never the cell's own expression; only time_t varies), so the predicted DSS-marker score
    at a target day is attributable purely to Phi(x,t).

      C1 per_mouse_trajectories.csv  -- each mouse's cells re-decoded at t=0/3/9/21 -> its own
                                       virtual longitudinal inflammation curve (impossible from
                                       raw data: each mouse is one terminal timepoint).
      C2 spatial_moran.csv           -- Moran's I of the predicted DIAGONAL field (each cell at
                                       its own day) per slice, vs observed marker score + label;
                                       expect the observed 0.06->0.3 rise reproduced.
      C3 heterogeneity.csv           -- every mouse projected to the peak day (t=9); cross-mouse
                                       spread of predicted inflammation (interp baseline = 0).

    NOT here: the held-out-interpolable-day check (deliverable 2) is a RETRAIN, run separately.
    """
    import torch, csv
    outdir = Path(outdir) / "case_study"
    outdir.mkdir(parents=True, exist_ok=True)
    ad = _import_ad()

    denom = float(getattr(cfg, "loto_day_denom", 21.0))
    # per-slice-file day + mouse (constant within a file: each slice = one mouse x one day)
    day_by_file = np.array([
        float(ad.read_h5ad(p, backed="r").obs[cfg.time_col].iloc[0]) for p in multi_paths],
        dtype=np.float32)

    def _mouse_of(p):
        o = ad.read_h5ad(p, backed="r").obs
        for c in ("Mouse_ID", "mouse_id", "Mouse", "mouse"):
            if c in o.columns:
                return str(o[c].iloc[0])
        return Path(p).stem
    mouse_by_file = np.array([_mouse_of(p) for p in multi_paths], dtype=object)

    sid = np.asarray(sample_id_all).astype(int)
    src_day = day_by_file[sid] * denom                    # actual day 0/3/9/21
    mouse = mouse_by_file[sid]                             # per-cell mouse id
    Y_expr = _dense(adata.X).astype(np.float32)
    # mu is size-factor-NORMALIZED (NB mean = mu * size_factor, VAE_utils.py:96-100).
    # sf is per-cell and time-INDEPENDENT, so it cannot change the shape of the day-profile
    # this function reports -- only its level, which is what makes it comparable to counts.
    sf_cell = adata.obs["size_factors"].to_numpy(dtype=np.float32)[:, None]
    X_full = np.asarray(pos_batched)
    coords = X_full[:, :2]
    N = X_full.shape[0]
    pert_code = int(pert_map.get("perturbed", 1))
    bs = int(getattr(cfg, "loto_batch_size", 512))

    # shared-basal re-decode needs per-cell Tier1 + arch codes (same contract as the gate)
    if bool(getattr(model, "use_shared_basal", False)):
        if tier1_id is None or arch_desc is None:
            raise ValueError("run_case_study_deliverables: use_shared_basal but tier1_id/arch_desc "
                             "not provided (leakage-free basal needs the codes).")
        tier1_codes = np.asarray(tier1_id)
        arch_codes = np.asarray(arch_desc, dtype=np.float32)
        if tier1_codes.shape[0] != N or arch_codes.shape[0] != N:
            raise ValueError(f"run_case_study_deliverables: code arrays misaligned "
                             f"(tier1={tier1_codes.shape[0]}, arch={arch_codes.shape[0]}, N={N}).")
    else:
        tier1_codes = arch_codes = None

    DSS_MARKERS = ['Isg15', 'Ifit1', 'Tnf', 'Cxcl1', 'Nos2', 'Il11', 'Il1b', 'Mmp3', 'Mmp10',
                   'Mmp13', 'Plau', 'Igfbp5', 'Col18a1', 'Mmp2', 'S100a8', 'S100a9', 'Osm',
                   'Cxcl10', 'Cxcl5', 'Ly6a']
    gnames = list(map(str, gene_names))
    mk_idx = [gnames.index(g) for g in DSS_MARKERS if g in gnames]
    logging.info("[case] %d/%d DSS markers present on the gene axis", len(mk_idx), len(DSS_MARKERS))

    def _decode(t_day: float) -> np.ndarray:
        tt = np.full(N, float(t_day) / denom, dtype=np.float32)
        mu, _ = model.counterfactualPrediction(
            X=X_full, sample_index=torch.as_tensor(np.asarray(sample_indices)),
            cell_atts=np.asarray(cell_atts).copy(), perturb_cell_id=torch.arange(N),
            target_cell_tissue=None, target_cell_perturbation=pert_code,
            target_has_guide=True, target_is_real_KO=True,
            n_samples=3, batch_size=bs, Y=Y_expr, time_t=tt,
            tier1_id=tier1_codes, arch_desc=arch_codes)
        mu = np.asarray(mu, dtype=np.float32) * sf_cell
        return np.log1p(mu[:, mk_idx]).mean(1) if mk_idx else np.zeros(N, np.float32)

    TARGET_DAYS = [0.0, 3.0, 9.0, 21.0]
    logging.info("[case] re-decoding all %d cells at target days %s ...", N, TARGET_DAYS)
    pred = {d: _decode(d) for d in TARGET_DAYS}

    # observed per-cell marker field + label (for calibration + Moran reference)
    obs_score = np.log1p(Y_expr[:, mk_idx]).mean(1) if mk_idx else np.zeros(N, np.float32)
    state = np.asarray(cell_atts)[:, 1].astype(int)
    y_obs = (state == pert_code).astype(float)

    # diagonal prediction: each cell decoded at ITS OWN observed day
    diag = np.zeros(N, np.float32)
    for d in TARGET_DAYS:
        m = np.isclose(src_day, d, atol=1e-3)
        diag[m] = pred[d][m]

    # C1: per-mouse virtual longitudinal trajectories
    mice = sorted(set(mouse.tolist()))
    traj_rows = []
    for mid in mice:
        mm = mouse == mid
        row = {"mouse": mid, "observed_day": float(np.round(src_day[mm][0], 3)),
               "n_cells": int(mm.sum()), "observed_marker_score": float(obs_score[mm].mean())}
        for d in TARGET_DAYS:
            row[f"pred_t{int(d)}"] = float(pred[d][mm].mean())
        traj_rows.append(row)
    with open(outdir / "per_mouse_trajectories.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(traj_rows[0].keys())); w.writeheader(); w.writerows(traj_rows)
    logging.info("[case] C1 wrote %d per-mouse trajectories", len(traj_rows))

    # C2: spatial Moran of the predicted diagonal field, per slice
    moran_rows = []
    for s in np.unique(sid):
        m = sid == s
        if int(m.sum()) < 20:
            continue
        moran_rows.append({"slice": int(s), "mouse": str(mouse[m][0]),
                           "day": float(src_day[m][0]), "n": int(m.sum()),
                           "moran_pred_diag": _moran_np(diag[m], coords[m]),
                           "moran_obs_score": _moran_np(obs_score[m], coords[m]),
                           "moran_obs_label": _moran_np(y_obs[m], coords[m])})
    with open(outdir / "spatial_moran.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(moran_rows[0].keys())); w.writeheader(); w.writerows(moran_rows)
    moran_by_day = {}
    for d in TARGET_DAYS:
        rs = [r for r in moran_rows if abs(r["day"] - d) < 1e-3]
        if rs:
            moran_by_day[int(d)] = {"moran_pred_diag": float(np.mean([r["moran_pred_diag"] for r in rs])),
                                    "moran_obs_score": float(np.mean([r["moran_obs_score"] for r in rs])),
                                    "moran_obs_label": float(np.mean([r["moran_obs_label"] for r in rs])),
                                    "n_slices": len(rs)}
    logging.info("[case] C2 Moran(pred_diag) by day: %s",
                 {d: round(v["moran_pred_diag"], 3) for d, v in moran_by_day.items()})

    # C3: per-mouse heterogeneity at the peak day (t=9)
    het_rows = []
    for mid in mice:
        mm = mouse == mid
        het_rows.append({"mouse": mid, "observed_day": float(np.round(src_day[mm][0], 3)),
                         "n_cells": int(mm.sum()),
                         "pred_marker_score_at_t9": float(pred[9.0][mm].mean())})
    with open(outdir / "heterogeneity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(het_rows[0].keys())); w.writeheader(); w.writerows(het_rows)
    het_spread = float(np.std([r["pred_marker_score_at_t9"] for r in het_rows]))
    logging.info("[case] C3 cross-mouse spread of predicted score @t9 = %.4f (interp baseline = 0)", het_spread)

    # population-mean trajectory (predicted, over all cells) + observed diagonal by day
    pop_pred = {int(d): float(pred[d].mean()) for d in TARGET_DAYS}
    obs_by_day = {int(d): float(obs_score[np.isclose(src_day, d, atol=1e-3)].mean())
                  for d in TARGET_DAYS if int(np.isclose(src_day, d, atol=1e-3).sum()) > 0}

    summary = {
        "n_cells": int(N), "n_mice": len(mice), "target_days": TARGET_DAYS,
        "C1_population_pred_trajectory": pop_pred,
        "C1_observed_diagonal_by_day": obs_by_day,
        "C2_moran_by_day": moran_by_day,
        "C3_cross_mouse_spread_at_t9": het_spread,
        "metric": "mean log1p predicted DSS-marker expression (count space)",
        "note": ("Held-out-interpolable-day check (deliverable 2) is a separate retrain, "
                 "not in this inference job."),
    }
    (outdir / "case_study_summary.json").write_text(json.dumps(summary, indent=2))
    _plot_case_study(outdir, traj_rows, moran_rows, moran_by_day, het_rows, TARGET_DAYS, obs_by_day, pop_pred)
    logging.info("[case] wrote case_study/ deliverables to %s", outdir)
    return summary


def _plot_case_study(outdir, traj_rows, moran_rows, moran_by_day, het_rows, target_days, obs_by_day, pop_pred):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("[case] no matplotlib: %s", e); return
    day_color = {0: "#4c78a8", 3: "#f58518", 9: "#e45756", 21: "#54a24b"}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # per-mouse virtual trajectories, colored by observed day; observed diagonal marked
    for r in traj_rows:
        od = int(round(r["observed_day"]))
        ys = [r[f"pred_t{int(d)}"] for d in target_days]
        ax[0].plot(target_days, ys, "-", color=day_color.get(od, "#888"), alpha=0.35, lw=1)
        ax[0].scatter([r["observed_day"]], [r["observed_marker_score"]],
                      color=day_color.get(od, "#888"), s=18, zorder=3, edgecolor="k", lw=0.3)
    ax[0].plot(target_days, [pop_pred[int(d)] for d in target_days], "k-o", lw=2.5, label="population mean (pred)")
    ax[0].set_xlabel("re-decode day t"); ax[0].set_ylabel("predicted DSS-marker score")
    ax[0].set_title("C1: per-mouse virtual longitudinal trajectories\n(line=predicted, dot=observed at own day)")
    ax[0].legend(frameon=False, fontsize=8)

    # Moran by day, predicted diagonal vs observed
    dd = sorted(moran_by_day.keys())
    ax[1].plot(dd, [moran_by_day[d]["moran_pred_diag"] for d in dd], "-o", label="predicted field")
    ax[1].plot(dd, [moran_by_day[d]["moran_obs_score"] for d in dd], "-s", label="observed marker score")
    ax[1].plot(dd, [moran_by_day[d]["moran_obs_label"] for d in dd], "-^", label="observed label", alpha=0.6)
    ax[1].set_xlabel("day"); ax[1].set_ylabel("Moran's I (mean over slices)")
    ax[1].set_title("C2: spatial contiguity over time"); ax[1].legend(frameon=False, fontsize=8)

    # per-mouse predicted score at peak, grouped by observed day
    for r in het_rows:
        od = int(round(r["observed_day"]))
        ax[2].scatter([od + np.random.uniform(-0.6, 0.6)], [r["pred_marker_score_at_t9"]],
                      color=day_color.get(od, "#888"), s=24, alpha=0.8)
    ax[2].set_xlabel("mouse observed day"); ax[2].set_ylabel("predicted score projected to t=9")
    ax[2].set_title("C3: per-mouse heterogeneity at peak\n(spread > 0; interp baseline = 0)")
    fig.tight_layout(); fig.savefig(outdir / "case_study_summary.png", dpi=130); plt.close(fig)


def run_heldout_mouse_eval(model, adata, cell_atts, pos_batched, sample_indices,
                           sample_id_all, multi_paths, gene_names, pert_map, cfg, outdir,
                           tier1_id=None, arch_desc=None) -> Dict:
    """Deliverable 2 -- leave-one-mouse-out GROUND-TRUTH reconstruction (interpolation rebuttal).

    The model is trained with whole mice HELD OUT (HOLDOUT_MICE). This eval is run on ONLY the
    held-out mice's slices (multi_sample_files = the held-out files); the ckpt loads with its
    basal / field / f_pert intact and slide_bias truncated to a reference batch (load_model's
    "target slide row 0" convention). Each held-out cell is re-decoded at ITS OWN (still-SEEN)
    day from its own Tier1 + architecture + niche geometry -- a genuine unseen-INDIVIDUAL
    reconstruction, with ground truth available because the day itself was in training.

    Because the day is seen, the sharpest "just interpolation" baseline is the population MEAN
    marker profile at that day (what any interpolation collapses to for an unseen individual).
    CONCERT must beat it by using the individual's spatial structure -- per-cell / spatial
    variation the population mean gives as a constant (per-cell corr 0 by construction).

    Metrics (per held-out mouse + pooled), written to <outdir>/heldout_eval/:
      * pseudobulk calibration -- predicted vs observed mouse-mean marker score (right level?)
      * per-cell spatial -- Pearson(pred, obs) marker score; population-mean baseline = 0
      * per-cell RMSE -- model vs day-mean-constant vs healthy-mean baselines (model lowest?)
      * spatial Moran -- predicted field vs observed field per slice
    """
    import torch, csv
    outdir = Path(outdir) / "heldout_eval"
    outdir.mkdir(parents=True, exist_ok=True)
    ad = _import_ad()

    denom = float(getattr(cfg, "loto_day_denom", 21.0))
    day_by_file = np.array([
        float(ad.read_h5ad(p, backed="r").obs[cfg.time_col].iloc[0]) for p in multi_paths],
        dtype=np.float32)

    def _mouse_of(p):
        o = ad.read_h5ad(p, backed="r").obs
        for c in ("Mouse_ID", "mouse_id", "Mouse", "mouse"):
            if c in o.columns:
                return str(o[c].iloc[0])
        return Path(p).stem
    mouse_by_file = np.array([_mouse_of(p) for p in multi_paths], dtype=object)

    sid = np.asarray(sample_id_all).astype(int)
    src_day = day_by_file[sid] * denom
    mouse = mouse_by_file[sid]
    X_full = np.asarray(pos_batched)
    coords = X_full[:, :2]
    N = X_full.shape[0]
    pert_code = int(pert_map.get("perturbed", 1))
    bs = int(getattr(cfg, "loto_batch_size", 512))

    DSS_MARKERS = ['Isg15', 'Ifit1', 'Tnf', 'Cxcl1', 'Nos2', 'Il11', 'Il1b', 'Mmp3', 'Mmp10',
                   'Mmp13', 'Plau', 'Igfbp5', 'Col18a1', 'Mmp2', 'S100a8', 'S100a9', 'Osm',
                   'Cxcl10', 'Cxcl5', 'Ly6a']
    gnames = list(map(str, gene_names))
    mk_idx = [gnames.index(g) for g in DSS_MARKERS if g in gnames]

    # observed ground truth in RAW-COUNT space
    # adata.X reaching this fn is normalized (negative after scaling), so re-read each held-out
    # file's raw counts and build the per-cell observed marker score in file/concat order (run()
    # concatenates multi_paths in order; cells with sample_id==i are contiguous in file-i order).
    obs_score = np.zeros(N, np.float32)
    for i, p in enumerate(multi_paths):
        m = sid == i
        if not m.any():
            continue
        Xi = ad.read_h5ad(p).X
        Xi = Xi.toarray() if sp.issparse(Xi) else np.asarray(Xi)
        if Xi.shape[0] != int(m.sum()):
            raise ValueError(f"heldout_eval: file {Path(p).name} has {Xi.shape[0]} cells but "
                             f"{int(m.sum())} in concat -- order/alignment broke.")
        obs_score[m] = np.log1p(Xi[:, mk_idx]).mean(1) if mk_idx else 0.0

    if bool(getattr(model, "use_shared_basal", False)):
        if tier1_id is None or arch_desc is None:
            raise ValueError("run_heldout_mouse_eval: use_shared_basal but tier1_id/arch_desc missing.")
        tier1_codes = np.asarray(tier1_id)
        arch_codes = np.asarray(arch_desc, dtype=np.float32)
    else:
        tier1_codes = arch_codes = None

    Y_expr = _dense(adata.X).astype(np.float32)   # model input (normalized) -- decoder consumes it

    # The decoder's mu is in SIZE-FACTOR-NORMALIZED space: the likelihood is
    # NB(x | mean = mu * size_factor) (VAE_utils.py:96-100). Comparing log1p(mu) against
    # log1p(RAW counts) therefore mis-scales every cell by its own library size -- and DSS
    # lowers library size (median counts/slice: day0 99 -> day9 61), so the mis-scaled cells
    # are exactly the late ones the calibration is judged on. Multiply mu back into COUNT
    # space before scoring so pred and obs are the same quantity.
    sf_cell = adata.obs["size_factors"].to_numpy(dtype=np.float32)[:, None]

    def _decode_score(t_day: float) -> np.ndarray:
        tt = np.full(N, float(t_day) / denom, dtype=np.float32)
        mu, _ = model.counterfactualPrediction(
            X=X_full, sample_index=torch.as_tensor(np.asarray(sample_indices)),
            cell_atts=np.asarray(cell_atts).copy(), perturb_cell_id=torch.arange(N),
            target_cell_tissue=None, target_cell_perturbation=pert_code,
            target_has_guide=True, target_is_real_KO=True,
            n_samples=3, batch_size=bs, Y=Y_expr, time_t=tt,
            tier1_id=tier1_codes, arch_desc=arch_codes)
        mu = np.asarray(mu, dtype=np.float32) * sf_cell
        return np.log1p(mu[:, mk_idx]).mean(1) if mk_idx else np.zeros(N, np.float32)

    # decode each held-out cell at its OWN day (the diagonal = the ground-truth-comparable point)
    uniq_days = sorted(set(np.round(src_day, 3).tolist()))
    pred_diag = np.zeros(N, np.float32)
    for d in uniq_days:
        m = np.isclose(src_day, d, atol=1e-3)
        pred_diag[m] = _decode_score(float(d))[m]

    # --- per-cell dump for spatial figures (viz 1-3): coords, obs, and the leakage-free
    # re-decode of EVERY held-out cell at t0/t3/t9/t21 (virtual longitudinal on unseen tissue).
    if bool(getattr(cfg, "dump_percell", False)):
        pc_days = [0.0, 3.0, 9.0, 21.0]
        pred_by_t = {t: _decode_score(float(t)) for t in pc_days}
        # per-GENE predicted mean expression on the DSS marker panel at each virtual time,
        # for the observed-vs-imputed marker bubble plot (Q4b). Also dump observed per-gene
        # log1p (raw counts) so Q4a/Q4b share one file.
        def _decode_marker_expr(t_day: float) -> np.ndarray:
            tt = np.full(N, float(t_day) / denom, dtype=np.float32)
            mu, _ = model.counterfactualPrediction(
                X=X_full, sample_index=torch.as_tensor(np.asarray(sample_indices)),
                cell_atts=np.asarray(cell_atts).copy(), perturb_cell_id=torch.arange(N),
                target_cell_tissue=None, target_cell_perturbation=pert_code,
                target_has_guide=True, target_is_real_KO=True,
                n_samples=3, batch_size=bs, Y=Y_expr, time_t=tt,
                tier1_id=tier1_codes, arch_desc=arch_codes)
            mu = np.asarray(mu, dtype=np.float32) * sf_cell
            return np.log1p(mu[:, mk_idx]) if mk_idx else np.zeros((N, 0), np.float32)
        marker_pred_by_t = {t: _decode_marker_expr(float(t)) for t in pc_days}
        obs_marker_expr = np.zeros((N, len(mk_idx)), np.float32)
        for i, p in enumerate(multi_paths):
            m = sid == i
            if not m.any():
                continue
            Xi = ad.read_h5ad(p).X
            Xi = Xi.toarray() if sp.issparse(Xi) else np.asarray(Xi)
            obs_marker_expr[m] = np.log1p(Xi[:, mk_idx]) if mk_idx else 0.0
        np.savez_compressed(
            outdir / "percell_dump.npz",
            coords=coords.astype(np.float32), sample_id=sid.astype(np.int32),
            mouse=mouse.astype(str), src_day=src_day.astype(np.float32),
            obs_score=obs_score.astype(np.float32),
            pred_t0=pred_by_t[0.0].astype(np.float32), pred_t3=pred_by_t[3.0].astype(np.float32),
            pred_t9=pred_by_t[9.0].astype(np.float32), pred_t21=pred_by_t[21.0].astype(np.float32),
            pred_diag=pred_diag.astype(np.float32),
            marker_names=np.array([DSS_MARKERS[j] for j, g in enumerate(DSS_MARKERS) if g in gnames], dtype=object),
            obs_marker_expr=obs_marker_expr.astype(np.float32),
            marker_pred_t0=marker_pred_by_t[0.0].astype(np.float32),
            marker_pred_t3=marker_pred_by_t[3.0].astype(np.float32),
            marker_pred_t9=marker_pred_by_t[9.0].astype(np.float32),
            marker_pred_t21=marker_pred_by_t[21.0].astype(np.float32))
        logging.info("[heldout] per-cell dump -> %s (N=%d, 4 decode times, +per-gene marker expr)",
                     outdir / "percell_dump.npz", N)

    if bool(getattr(cfg, "dump_pathways", False)):
        import json as _json
        pw_json = Path(__file__).resolve().parent / "pathway_gene_sets.json"
        if not pw_json.exists():
            raise FileNotFoundError(f"dump_pathways=True but {pw_json} missing "
                                    "(run pathway_panel_overlap.py first).")
        pw_sets = _json.loads(pw_json.read_text())
        pw_names = list(pw_sets.keys())
        pw_idx = {name: [gnames.index(g) for g in genes if g in gnames]
                  for name, genes in pw_sets.items()}
        logging.info("[pathway] %d pathways: %s", len(pw_names),
                     {k: len(v) for k, v in pw_idx.items()})
        pw_days = [0.0, 3.0, 9.0, 21.0]

        def _decode_full_logcount(t_day: float) -> np.ndarray:
            tt = np.full(N, float(t_day) / denom, dtype=np.float32)
            mu, _ = model.counterfactualPrediction(
                X=X_full, sample_index=torch.as_tensor(np.asarray(sample_indices)),
                cell_atts=np.asarray(cell_atts).copy(), perturb_cell_id=torch.arange(N),
                target_cell_tissue=None, target_cell_perturbation=pert_code,
                target_has_guide=True, target_is_real_KO=True,
                n_samples=3, batch_size=bs, Y=Y_expr, time_t=tt,
                tier1_id=tier1_codes, arch_desc=arch_codes)
            return np.log1p(np.asarray(mu, dtype=np.float32) * sf_cell)   # N x G, log1p count

        # Per-mouse per-GENE mean of the counterfactual log1p(count) decode, at each virtual
        # t, PLUS the observed-diagonal per-mouse per-gene mean. This is the future-proof
        # aggregate: any formal gene set (MSigDB/GO/Reactome) scores on CPU as the mean over
        # its panel genes of these means -- no GPU re-decode when the gene sets change.
        uniq_mice = sorted(set(mouse.tolist()))
        m_index = {mm: k for k, mm in enumerate(uniq_mice)}
        G = len(gnames)
        gmean_pred = np.zeros((len(uniq_mice), G, len(pw_days)), np.float32)  # (M, G, 4)
        gmean_obs = np.zeros((len(uniq_mice), G), np.float32)                 # (M, G) diagonal
        mouse_ncell = np.zeros(len(uniq_mice), np.int64)

        # predicted per-pathway score at each virtual t: name -> (N, 4); + per-mouse gene means
        pw_pred = {name: np.zeros((N, len(pw_days)), np.float32) for name in pw_names}
        for ti, t in enumerate(pw_days):
            lm = _decode_full_logcount(float(t))
            for name in pw_names:
                idx = pw_idx[name]
                pw_pred[name][:, ti] = lm[:, idx].mean(1) if idx else 0.0
            for mm in uniq_mice:
                msk = mouse == mm
                gmean_pred[m_index[mm], :, ti] = lm[msk].mean(0)
        # observed per-pathway score on the diagonal (each cell at its own SEEN day)
        pw_obs = {name: np.zeros(N, np.float32) for name in pw_names}
        for i, p in enumerate(multi_paths):
            m = sid == i
            if not m.any():
                continue
            Xi = ad.read_h5ad(p).X
            Xi = Xi.toarray() if sp.issparse(Xi) else np.asarray(Xi)
            lXi = np.log1p(Xi)
            for name in pw_names:
                idx = pw_idx[name]
                pw_obs[name][m] = lXi[:, idx].mean(1) if idx else 0.0
            k = m_index[str(mouse[m][0])]            # each file is one mouse's slice
            gmean_obs[k] += lXi.sum(0)
            mouse_ncell[k] += lXi.shape[0]
        gmean_obs /= np.maximum(mouse_ncell, 1)[:, None]
        save_kw = dict(
            coords=coords.astype(np.float32), sample_id=sid.astype(np.int32),
            mouse=mouse.astype(str), src_day=src_day.astype(np.float32),
            pathway_names=np.array(pw_names, dtype=object),
            virtual_days=np.array(pw_days, dtype=np.float32),
            n_genes_per_pathway=np.array([len(pw_idx[n]) for n in pw_names], dtype=np.int32),
            gene_names=np.array(gnames, dtype=object),
            mouse_order=np.array(uniq_mice, dtype=object),
            mouse_day=np.array([float(src_day[mouse == mm][0]) for mm in uniq_mice], dtype=np.float32),
            mouse_ncell=mouse_ncell.astype(np.int64),
            gmean_pred=gmean_pred, gmean_obs=gmean_obs)   # (M,G,4) and (M,G)
        for name in pw_names:
            save_kw[f"pred__{name}"] = pw_pred[name].astype(np.float32)   # (N,4) t0/t3/t9/t21
            save_kw[f"obs__{name}"] = pw_obs[name].astype(np.float32)     # (N,) diagonal
        np.savez_compressed(outdir / "pathway_dump.npz", **save_kw)
        logging.info("[heldout] pathway dump -> %s (N=%d, %d pathways, 4 virtual t, +per-mouse "
                     "per-gene means %s)", outdir / "pathway_dump.npz", N, len(pw_names),
                     gmean_pred.shape)

    def _pearson(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # per held-out mouse metrics
    rows = []
    for mid in sorted(set(mouse.tolist())):
        mm = mouse == mid
        d = float(np.round(src_day[mm][0], 3))
        o = obs_score[mm]; pr = pred_diag[mm]
        obs_mean = float(o.mean())
        # baselines: constant = this mouse's-day population mean is not knowable without leakage,
        # so the honest "mean" baseline is a CONSTANT prediction (best constant = obs mean) whose
        # per-cell corr is 0 and whose per-cell RMSE is the observed SD. Model must beat that SD.
        rmse_model = float(np.sqrt(np.mean((pr - o) ** 2)))
        rmse_constant = float(o.std())          # RMSE of the best constant predictor = SD(obs)
        rows.append({
            "mouse": mid, "day": d, "n_cells": int(mm.sum()),
            "obs_mouse_mean": obs_mean, "pred_mouse_mean": float(pr.mean()),
            "pseudobulk_ratio": float(pr.mean() / (obs_mean + 1e-9)),
            "percell_pearson_pred_vs_obs": _pearson(pr, o),
            "percell_rmse_model": rmse_model,
            "percell_rmse_best_constant": rmse_constant,
            "model_beats_constant": bool(rmse_model < rmse_constant),
        })
    with open(outdir / "heldout_per_mouse.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # spatial Moran per held-out slice: predicted field vs observed field
    srows = []
    for s in np.unique(sid):
        m = sid == s
        if int(m.sum()) < 20:
            continue
        srows.append({"slice": int(s), "mouse": str(mouse[m][0]), "day": float(src_day[m][0]),
                      "n": int(m.sum()),
                      "moran_pred": _moran_np(pred_diag[m], coords[m]),
                      "moran_obs": _moran_np(obs_score[m], coords[m])})
    with open(outdir / "heldout_spatial_moran.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys())); w.writeheader(); w.writerows(srows)

    pooled = {
        "n_heldout_cells": int(N), "n_heldout_mice": len(rows),
        "held_out_mice": [r["mouse"] for r in rows],
        "mean_percell_pearson": float(np.nanmean([r["percell_pearson_pred_vs_obs"] for r in rows])),
        "frac_mice_model_beats_constant": float(np.mean([r["model_beats_constant"] for r in rows])),
        "mean_pseudobulk_ratio": float(np.mean([r["pseudobulk_ratio"] for r in rows])),
        "metric": "mean log1p DSS-marker expression; obs in RAW counts, pred from re-decode",
        "interpretation": ("per-cell Pearson>0 and model_beats_constant => the held-out individual's "
                           "SPATIAL inflammation pattern is reconstructed, not just its mean; the "
                           "constant/population-mean baseline (interpolation) gives per-cell corr 0."),
    }
    (outdir / "heldout_summary.json").write_text(json.dumps({"per_mouse": rows, "pooled": pooled}, indent=2))
    logging.info("[heldout] mean per-cell Pearson=%.3f  frac mice beating constant=%.2f  pb ratio=%.2f",
                 pooled["mean_percell_pearson"], pooled["frac_mice_model_beats_constant"],
                 pooled["mean_pseudobulk_ratio"])
    _plot_heldout(outdir, rows, srows)
    return pooled


def _plot_heldout(outdir, rows, srows):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("[heldout] no matplotlib: %s", e); return
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    mids = [r["mouse"] for r in rows]
    x = np.arange(len(mids))
    ax[0].bar(x - 0.2, [r["obs_mouse_mean"] for r in rows], 0.4, label="observed")
    ax[0].bar(x + 0.2, [r["pred_mouse_mean"] for r in rows], 0.4, label="predicted")
    ax[0].set_xticks(x); ax[0].set_xticklabels(mids, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("mouse-mean marker score"); ax[0].set_title("Held-out pseudobulk calibration")
    ax[0].legend(frameon=False)
    ax[1].bar(x - 0.2, [r["percell_rmse_model"] for r in rows], 0.4, label="model")
    ax[1].bar(x + 0.2, [r["percell_rmse_best_constant"] for r in rows], 0.4, label="best constant (interp)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(mids, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("per-cell RMSE"); ax[1].set_title("Per-cell error: model vs interpolation")
    ax[1].legend(frameon=False)
    if srows:
        ax[2].scatter([r["moran_obs"] for r in srows], [r["moran_pred"] for r in srows], s=20)
        lim = [0, max(0.6, max([r["moran_obs"] for r in srows] + [0.1]))]
        ax[2].plot(lim, lim, "k--", alpha=0.5)
    ax[2].set_xlabel("Moran(observed)"); ax[2].set_ylabel("Moran(predicted)")
    ax[2].set_title("Held-out spatial contiguity")
    fig.tight_layout(); fig.savefig(outdir / "heldout_summary.png", dpi=130); plt.close(fig)


def _import_ad():
    import anndata as _ad
    return _ad


def _plot(outdir, rows, srows, hrows, spread):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("[comp] no matplotlib: %s", e); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    days = [r["day"] for r in rows]
    ax[0].plot(days, [r["obs_frac"] for r in rows], "-o", label="observed label fraction")
    ax[0].plot(days, [r["pred_pi_mean"] for r in rows], "-s", label="predicted π mean")
    ax[0].set_xlabel("day"); ax[0].set_ylabel("inflamed fraction"); ax[0].legend(frameon=False)
    ax[0].set_title("Composition calibration")
    if srows:
        ax[1].scatter([r["moran_obs_label"] for r in srows], [r["moran_pred_pi"] for r in srows], s=12)
        lim = [0, max(0.6, max([r["moran_obs_label"] for r in srows] + [0.1]))]
        ax[1].plot(lim, lim, "k--", alpha=0.5)
    ax[1].set_xlabel("Moran(observed label)"); ax[1].set_ylabel("Moran(predicted π)")
    ax[1].set_title("Spatial contiguity of π")
    fig.tight_layout(); fig.savefig(outdir / "composition_summary.png", dpi=130); plt.close(fig)
