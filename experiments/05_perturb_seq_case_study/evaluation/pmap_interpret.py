#!/usr/bin/env python3
"""Perturb-map interpretation: what the model learns per perturbation, and why an edited
neighbourhood does not move an unedited cell.

Part A -- niche-path diagnostic. z_niche DOES reach a cell's prediction, through gamma and
beta. So the null propagation result needs a mechanism. This walks the path for the core
cells with the neighbourhood at its observed state and after the ring's expression is
replaced by its perturbed prediction, and reports how far each stage actually moves.

Part B -- per-perturbation parameters. The predicted log-shift decomposes as

    shift = gamma(e_p, z_niche) * [ core(e_p) + field(e_p, coords) ] + beta(e_p, z_niche)

so each perturbation gets a niche-invariant core, a spatial residual, and a niche
modulation. Their relative sizes say what the model thinks is spatial about each knockout,
which is a statement about the model and needs no external validation to report.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CODEBASE / "benchmark_evaluation"))
import run_concert_pm_niche as R  # noqa: E402
import pmap_case_study as CS      # noqa: E402

OUT = Path(os.environ.get(
    "PMAP_INTERPRET_OUT",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/figures/pmap_interpret"))
PERTS = ["Tgfbr2", "Ifngr2", "Jak2"]


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def stage_tensors(model, ctx, rows, Y):
    """Run the encoder path once and return the pieces the shift is built from."""
    dev = model.device
    xs = torch.as_tensor(ctx["pos_batched"][rows], dtype=torch.get_default_dtype()).to(dev)
    cs = torch.as_tensor(np.asarray(ctx["cell_atts"])[rows], dtype=torch.int).to(dev)
    is_ = torch.as_tensor(ctx["sample_indices"][rows], dtype=torch.int).to(dev)
    ys = torch.as_tensor(Y, dtype=torch.get_default_dtype()).to(dev)

    lord = model.lord_encoder.predict(sample_indices=is_, batch_size=xs.shape[0], labels=cs,
                                      expression=ys if model.use_expr_encoder else None)
    q_mu, _ = model.encoder(model._encoder_input_from_lord(lord))
    gs_mu = q_mu[:, model.GP_dim:]
    z_niche = model.niche_descriptor(gs_mu.detach(), xs[:, :2])
    pert_emb = lord["perturbation"]

    core = model.f_pert(pert_emb)
    field = model.fpert_field(pert_emb, xs[:, :2]) if model.fpert_field is not None else torch.zeros_like(core)
    g_scale = torch.clamp(torch.nn.functional.softplus(model.g_scale_head(pert_emb)),
                          max=model.g_cap)
    h = model.niche_h(torch.cat([pert_emb, z_niche], dim=-1))
    gamma = 1.0 + g_scale * torch.tanh(model.gamma_head(h))
    beta = model.beta_out(model.beta_U(pert_emb) * model.beta_V(z_niche))
    shift = gamma * (core + field) + beta
    return dict(z_niche=z_niche, core=core, field=field, gamma=gamma, beta=beta,
                g_scale=g_scale, shift=shift, gs_mu=gs_mu)


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
    sel = {k: np.loadtxt(p, dtype=int) - 1 for k, p in CS.SELECTIONS.items()}
    core_idx, ring_idx = sel["tgfbr2_core"], sel["peri_ring"]

    Y_base = np.asarray(ctx["Y_expr"])[rows]
    results = {}

    # ---------- Part A: where the niche signal dies ----------------------------------
    log("== Part A: niche-path diagnostic ==")
    to_enc, rt_err = CS.encoder_space_transform(np.asarray(ctx["Y_obs"])[rows], Y_base)
    log(f"  encoder-space round-trip max err {rt_err:.4f}")

    pred_ring = CS.predict(ctx, args, rows, ring_idx, maps["tissue"]["tumor"],
                           maps["perturbation"]["Tgfbr2"], has_guide=True, is_real_ko=True)
    Y2 = Y_base.copy()
    Y2[ring_idx] = to_enc(pred_ring[ring_idx])

    A = stage_tensors(model, ctx, rows, Y_base)
    B = stage_tensors(model, ctx, rows, Y2)

    def relchange(t0, t1, idx):
        a, b = t0[idx].float(), t1[idx].float()
        denom = a.abs().mean().item()
        return {"abs": float((b - a).abs().mean().item()),
                "rel_to_own_scale": float((b - a).abs().mean().item() / max(denom, 1e-9))}

    stages = {}
    for nm in ("gs_mu", "z_niche", "gamma", "beta", "shift"):
        stages[f"core_cells__{nm}"] = relchange(A[nm], B[nm], core_idx)
        stages[f"ring_cells__{nm}"] = relchange(A[nm], B[nm], ring_idx)
        log(f"  {nm:8s} core {stages[f'core_cells__{nm}']['abs']:.6f} "
            f"(rel {stages[f'core_cells__{nm}']['rel_to_own_scale']:.4f}) | "
            f"ring {stages[f'ring_cells__{nm}']['abs']:.6f} "
            f"(rel {stages[f'ring_cells__{nm}']['rel_to_own_scale']:.4f})")
    # How much of the core's shift is even available to be modulated?
    stages["core_gamma_mean"] = float(A["gamma"][core_idx].mean().item())
    stages["core_g_scale_mean"] = float(A["g_scale"][core_idx].mean().item())
    stages["core_beta_over_shift"] = float(
        (A["beta"][core_idx].abs().mean() / A["shift"][core_idx].abs().mean()).item())
    log(f"  core gamma mean {stages['core_gamma_mean']:.4f}, "
        f"g_scale {stages['core_g_scale_mean']:.4f}, "
        f"|beta|/|shift| {stages['core_beta_over_shift']:.4f}")
    results["niche_path_diagnostic"] = stages

    # ---------- Part C: propagation vs neighbourhood overlap --------------------------
    # z_niche at the core did not move AT ALL, which is a geometry result, not an
    # architecture one: the descriptor takes k nearest neighbours, and a cell inside a
    # 75-spot patch has all k of them inside the patch. Only cells whose neighbourhood
    # actually CONTAINS an edited cell can possibly respond, so resolve the change by how
    # many of each cell's k neighbours were edited.
    log("== Part C: propagation vs neighbourhood overlap ==")
    k = int(getattr(model.niche_descriptor, "k", 25))
    pos_loc = np.asarray(ctx["pos_batched"])[rows][:, :2]
    from scipy.spatial import cKDTree
    nn_idx = cKDTree(pos_loc).query(pos_loc, k=k + 1)[1][:, 1:]
    edited_mask = np.zeros(len(rows), dtype=bool)
    edited_mask[ring_idx] = True
    n_edited_nb = edited_mask[nn_idx].sum(axis=1)

    dz = (B["z_niche"] - A["z_niche"]).abs().mean(dim=1).cpu().numpy()
    dshift = (B["shift"] - A["shift"]).abs().mean(dim=1).cpu().numpy()
    unedited = ~edited_mask
    overlap = {}
    for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10), (11, k)):
        m = unedited & (n_edited_nb >= lo) & (n_edited_nb <= hi)
        if m.sum() == 0:
            continue
        overlap[f"{lo}-{hi}_edited_neighbours"] = {
            "n_cells": int(m.sum()),
            "z_niche_change": float(dz[m].mean()),
            "shift_change": float(dshift[m].mean()),
        }
        log(f"  {lo}-{hi} edited neighbours: n={int(m.sum()):4d} "
            f"z_niche {dz[m].mean():.6f}  shift {dshift[m].mean():.6f}")
    core_nb = n_edited_nb[core_idx]
    log(f"  core cells: {int((core_nb > 0).sum())}/{len(core_idx)} have >=1 edited neighbour "
        f"(max {int(core_nb.max())} of k={k})")
    results["propagation_vs_overlap"] = {
        "k": k, "bins": overlap,
        "core_cells_with_any_edited_neighbour": int((core_nb > 0).sum()),
        "core_cells_total": int(len(core_idx)),
        "max_edited_neighbours_among_core": int(core_nb.max()),
    }

    # ---------- Part B: per-perturbation learned parameters ---------------------------
    log("== Part B: per-perturbation parameters ==")
    name2code = {str(k): int(v) for k, v in ctx["name2code"].items()}
    emb_tab = model.lord_encoder.s_encoder["perturbation"].weight
    ls_all = model.fpert_field.per_pert_lengthscale(emb_tab).detach().cpu().numpy().ravel()
    g_all = torch.clamp(torch.nn.functional.softplus(model.g_scale_head(emb_tab)),
                        max=model.g_cap).detach().cpu().numpy().ravel()

    per_pert = {}
    for p in PERTS:
        if p not in name2code:
            continue
        code = name2code[p]
        # Evaluate this perturbation's shift AT EVERY SPOT of the slide, so the spatial
        # spread is a property of the perturbation and not of where its cells happen to be.
        atts = np.asarray(ctx["cell_atts"])[rows].copy()
        atts[:, 1] = code
        atts[:, 0] = maps["tissue"]["tumor"]
        ctx_mod = dict(ctx)
        ctx_mod["cell_atts"] = np.asarray(ctx["cell_atts"]).copy()
        ctx_mod["cell_atts"][rows] = atts
        S = stage_tensors(model, ctx_mod, rows, Y_base)
        core_v = S["core"].float()
        field_v = S["field"].float()
        gam = S["gamma"].float()
        bet = S["beta"].float()
        tot = S["shift"].float()
        per_pert[p] = {
            "footprint_lengthscale_loc_units": float(ls_all[code] * 70.0),
            "niche_gain": float(g_all[code]),
            "core_magnitude": float(core_v.abs().mean().item()),
            "field_magnitude": float(field_v.abs().mean().item()),
            "field_spatial_sd": float(field_v.std(dim=0).mean().item()),
            "core_spatial_sd": float(core_v.std(dim=0).mean().item()),
            "gamma_mean": float(gam.mean().item()),
            "gamma_spatial_sd": float(gam.std().item()),
            "beta_magnitude": float(bet.abs().mean().item()),
            "total_shift_magnitude": float(tot.abs().mean().item()),
            "spatial_fraction_of_effect": float(
                (field_v.abs().mean() / (core_v.abs().mean() + field_v.abs().mean())).item()),
            "niche_fraction_of_effect": float(
                (bet.abs().mean() / tot.abs().mean()).item()),
        }
        log(f"  {p}: " + json.dumps({k: round(v, 4) for k, v in per_pert[p].items()}))
    results["per_perturbation"] = per_pert

    (OUT / "interpret_stats.json").write_text(json.dumps(results, indent=2))
    log("wrote " + str(OUT / "interpret_stats.json"))


if __name__ == "__main__":
    main()
