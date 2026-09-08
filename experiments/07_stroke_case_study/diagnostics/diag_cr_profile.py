#!/usr/bin/env python
"""Did giving the core and the rim their own perturbation label break the core-peaked lock-in?

Baseline (one `stroke` label): predicted depth Spearman was +0.89..+1.00 for ALL six markers --
every gene monotonically rising toward the core, while the observed profiles ranged -1.00..+0.94.
The claim under test is that two labels (ICA / PIA) let rim genes actually peak at the rim.

delta is assembled per compartment: core spots from the `oncore` (ICA) pass, rim spots from the
`onrim` (PIA) pass, both against the shared `off` control. Exact, not approximate -- the niche
code never reads neighbours' perturbation labels (concert_spac.py:413).

Marker panels are derived from region_markers.json rather than hand-picked: a gene joins the core
panel when delta_ICA - delta_PIA > MIN_SEP and the rim panel when the reverse. The old pooled
panel mixed the two, which is why the injury score was rim-weighted while the prediction was
core-peaked.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
SLIDES = NS / "stroke3d/slides_cr"
DEL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/bof695/CONCERT/delivery/stroke-registration-benchmark")
GENES = ["Gm42418", "Scgb1a1", "Vim", "Lgals1", "Timp1", "Spp1"]
MIN_SEP = 0.5
NB = 6

arm = sys.argv[1] if len(sys.argv) > 1 else "3d"
CRTAG = sys.argv[2] if len(sys.argv) > 2 else "_cr"
OUT = NS / f"outputs/stroke_{arm}{CRTAG}"


def log1p_cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def load_stack(stack):
    paths = sorted(SLIDES.glob(f"stroke_paste2_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in paths], label="sf", keys=[p.stem for p in paths],
                  index_unique=None)
    return A, np.asarray(A.obsm["array_coords"], np.float32)


def depth_of(xy, inside):
    d = np.zeros(len(xy))
    if inside.sum() and (~inside).sum():
        d[inside] = cKDTree(xy[~inside]).query(xy[inside])[0]
    return d


def profile(depth, vals, inside, nb=NB):
    d, v = depth[inside], vals[inside]
    qs = np.quantile(d, np.linspace(0, 1, nb + 1)); qs[-1] += 1e-6
    return [float(v[(d >= qs[i]) & (d < qs[i + 1])].mean()) for i in range(nb)]


def read_pass(tag):
    p = next(OUT.glob(f"stroke_{arm}{CRTAG}_{tag}_*_perturbed_counts.h5ad"))
    return log1p_cpm(np.asarray(ad.read_h5ad(p).X[rows], np.float32))


side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o + c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
L_core, L_rim, L_off = read_pass("oncore"), read_pass("onrim"), read_pass("off")
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)

A_sh, xy_sh = load_stack("Sham")
sd = OUT / "shape_sets"
mk = lambda f: np.isin(np.arange(len(xy_sh)), np.loadtxt(sd / f, dtype=np.int64))
pred_in, pred_core, pred_rim = mk("shape_lesion_local.txt"), mk("shape_core_local.txt"), mk("shape_rim_local.txt")
pred_core, pred_rim = pred_core & pred_in, pred_rim & pred_in

# compartment-assembled delta
D = np.zeros_like(L_off)
D[pred_core] = L_core[pred_core] - L_off[pred_core]
D[pred_rim] = L_rim[pred_rim] - L_off[pred_rim]

# --- observed side ---
A_pt, xy_pt = load_stack("PT")
Lpt = log1p_cpm(np.asarray(A_pt.X.todense(), np.float32))
gs = np.asarray(A_pt.var_names)
with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]])
    reg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
reg_pt = pd.Series(reg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
obs_core, obs_rim = reg_pt == "ICA", reg_pt == "PIA"
obs_in = obs_core | obs_rim
depth_pred, depth_obs = depth_of(xy_sh, pred_in), depth_of(xy_pt, obs_in)

res = {"arm": arm, "n_core": int(pred_core.sum()), "n_rim": int(pred_rim.sum()), "genes": {}}
for g in GENES:
    ip, io = int(np.where(gp == g)[0][0]), int(np.where(gs == g)[0][0])
    pv, ov = D[:, ip], Lpt[:, io]
    pc, pr = float(pv[pred_core].mean()), float(pv[pred_rim].mean())
    oc, orr = float(ov[obs_core].mean()), float(ov[obs_rim].mean())
    pp, po = profile(depth_pred, pv, pred_in), profile(depth_obs, ov, obs_in)
    res["genes"][g] = {
        "predicted_core": pc, "predicted_rim": pr, "predicted_core_minus_rim": pc - pr,
        "observed_core": oc, "observed_rim": orr, "observed_core_minus_rim": oc - orr,
        "sign_agrees": bool(np.sign(pc - pr) == np.sign(oc - orr)),
        "profile_predicted": pp, "profile_observed": po,
        "depth_rho_predicted": float(spearmanr(range(NB), pp).statistic),
        "depth_rho_observed": float(spearmanr(range(NB), po).statistic)}

# --- compartment-split marker panels, derived not hand-picked ---
rm = json.loads((NS / "stroke3d/region_markers.json").read_text())
allm = {e["gene"]: e for e in rm["ICA"] + rm["PIA"]}
core_panel = sorted([g for g, e in allm.items() if e["delta_ICA"] - e["delta_PIA"] > MIN_SEP])
rim_panel = sorted([g for g, e in allm.items() if e["delta_PIA"] - e["delta_ICA"] > MIN_SEP])
res["panels"] = {"min_sep": MIN_SEP, "core": core_panel, "rim": rim_panel}


def panel_score(M, names, panel):
    idx = [int(np.where(names == g)[0][0]) for g in panel if g in set(names.tolist())]
    return M[:, idx].mean(1) if idx else np.zeros(M.shape[0])


for pname, panel in (("core_panel", core_panel), ("rim_panel", rim_panel)):
    ps, os_ = panel_score(D, gp, panel), panel_score(Lpt, gs, panel)
    res.setdefault("panel_response", {})[pname] = {
        "n_genes_used": int(sum(g in set(gp.tolist()) for g in panel)),
        "predicted_core": float(ps[pred_core].mean()), "predicted_rim": float(ps[pred_rim].mean()),
        "observed_core": float(os_[obs_core].mean()), "observed_rim": float(os_[obs_rim].mean())}

(OUT / f"diag_cr_profile_{arm}{CRTAG}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
