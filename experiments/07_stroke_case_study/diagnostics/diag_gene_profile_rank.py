#!/usr/bin/env python
"""Why is every predicted marker core-peaked?

Two candidate explanations, both testable on the saved counterfactuals:

  H1 RANK-1 FIELD. The perturbation term is one scalar spatial field times a per-gene loading,
     so delta_g(x) = w_g * f(x). Then EVERY gene has the SAME spatial shape and can only differ
     in sign/scale -- a rim-peaking gene is unrepresentable. Test: SVD of the (spot x gene)
     delta matrix inside the lesion. If PC1 carries ~all the energy and the pairwise across-spot
     correlations of the marker genes are ~1, H1 holds.

  H2 GEOMETRIC DOSE. "Core" = deepest inside the patch = most perturbed neighbours, so the field
     accumulates there regardless of biology. Test: predicted delta vs depth-from-boundary should
     rise monotonically for all genes, while the observed profile does not.

Observed profiles are computed with the SAME depth binning on the real lesion, so the two rows
are comparable.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd
from scipy.spatial import cKDTree

NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
OUT = NS / "outputs/stroke_3d"
SLIDES = NS / "stroke3d/slides"
DEL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/bof695/CONCERT/delivery/stroke-registration-benchmark")
GENES = ["Gm42418", "Scgb1a1", "Vim", "Lgals1", "Timp1", "Spp1"]
POOLED = ("Lcn2", "Gfap", "Vim", "Timp1", "Cd68")


def log1p_cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def load_stack(prefix, stack):
    paths = sorted(SLIDES.glob(f"{prefix}_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in paths], label="sf", keys=[p.stem for p in paths],
                  index_unique=None)
    return A, np.asarray(A.obsm["array_coords"], np.float32)


def depth_of(xy, inside):
    d = np.zeros(len(xy))
    if inside.sum() and (~inside).sum():
        d[inside] = cKDTree(xy[~inside]).query(xy[inside])[0]
    return d


def profile(depth, vals, inside, nb=6):
    """Mean value in equal-count depth bins, shallow (rim) -> deep (core)."""
    d, v = depth[inside], vals[inside]
    qs = np.quantile(d, np.linspace(0, 1, nb + 1)); qs[-1] += 1e-6
    return [float(v[(d >= qs[i]) & (d < qs[i + 1])].mean()) for i in range(nb)]


P_on = ad.read_h5ad(next(OUT.glob("shape_on_*_perturbed_counts.h5ad")))
P_off = ad.read_h5ad(next(OUT.glob("stroke_3d_off_*_perturbed_counts.h5ad")))
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o + c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
D = log1p_cpm(np.asarray(P_on.X[rows], np.float32)) - log1p_cpm(np.asarray(P_off.X[rows], np.float32))

A_sh, xy_sh = load_stack("stroke_paste2", "Sham")
sd = OUT / "shape_sets"
mk = lambda f: np.isin(np.arange(len(xy_sh)), np.loadtxt(sd / f, dtype=np.int64))
pred_in, pred_core, pred_rim = mk("shape_lesion_local.txt"), mk("shape_core_local.txt"), mk("shape_rim_local.txt")
pred_core, pred_rim = pred_core & pred_in, pred_rim & pred_in

res = {"n_lesion_spots": int(pred_in.sum())}

# H1: is the predicted delta field rank-1 inside the lesion?
M = D[pred_in]                                    # spots x all HVGs
sv = np.linalg.svd(M, full_matrices=False)[1]
res["rank1"] = {
    "pc1_frac_energy_allHVG": float(sv[0] ** 2 / (sv ** 2).sum()),
    "pc1_pc2_ratio": float(sv[0] / sv[1]),
}
gi = [int(np.where(gp == g)[0][0]) for g in GENES]
C = np.corrcoef(M[:, gi].T)
iu = np.triu_indices(len(GENES), 1)
res["rank1"]["marker_pairwise_r_across_spots"] = {
    "min": float(C[iu].min()), "median": float(np.median(C[iu])), "max": float(C[iu].max())}
res["rank1"]["marker_corr_matrix"] = {g: {h: round(float(C[i, j]), 3) for j, h in enumerate(GENES)}
                                      for i, g in enumerate(GENES)}

# H2: depth profiles, predicted vs observed
depth_pred = depth_of(xy_sh, pred_in)
A_pt, xy_pt = load_stack("stroke_paste2", "PT")
Lpt = log1p_cpm(np.asarray(A_pt.X.todense(), np.float32))
gs = np.asarray(A_pt.var_names)
with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]])
    reg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
reg_pt = pd.Series(reg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
obs_in = (reg_pt == "ICA") | (reg_pt == "PIA")
depth_obs = depth_of(xy_pt, obs_in)

res["profiles_rim_to_core"] = {}
for g in GENES:
    res["profiles_rim_to_core"][g] = {
        "predicted": profile(depth_pred, D[:, int(np.where(gp == g)[0][0])], pred_in),
        "observed": profile(depth_obs, Lpt[:, int(np.where(gs == g)[0][0])], obs_in),
    }
# monotonicity: does predicted always increase rim->core?
from scipy.stats import spearmanr
res["depth_spearman"] = {g: {
    "predicted": float(spearmanr(range(6), v["predicted"]).statistic),
    "observed": float(spearmanr(range(6), v["observed"]).statistic)}
    for g, v in res["profiles_rim_to_core"].items()}

# how much of a gene's predicted map is explained by the pooled field?
pi = [int(np.where(gp == g)[0][0]) for g in POOLED if g in set(gp.tolist())]
f = D[:, pi].mean(1)
res["r_vs_pooled_field"] = {g: float(np.corrcoef(D[pred_in, int(np.where(gp == g)[0][0])], f[pred_in])[0, 1])
                            for g in GENES}

(OUT / "diag_gene_profile_rank.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
