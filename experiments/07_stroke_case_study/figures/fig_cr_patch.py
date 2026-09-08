#!/usr/bin/env python
"""3D compartment model on the imposed patch, against the observed lesion.

Same patch as the real lesion (shape, size and per-section shrinkage transferred from the stroke
mouse), so predicted and observed are the same object measured twice. delta is assembled per
compartment -- core spots from the ICA pass, rim spots from the PIA pass -- which is exact,
because the niche code never reads neighbours' perturbation labels.

Two outputs per run:
  fig_stroke_gene_<G>_3d_<tag>   spatial, one gene, observed row above predicted row
  fig_stroke_core_support_3d_<tag>  the support panel: depth profile (rim -> core) with the
                                    observed curve beside the predicted one, and core/rim bars

Rim markers are plotted but the honest readout is the core panel: no lengthscale setting recovered
a rim-high program in 3D (see diag_cr_profile_*). That is stated in the caption, not on the axes.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
DEL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/bof695/CONCERT/delivery/stroke-registration-benchmark")
CORE_C, RIM_C, OTHER_C = "#b2182b", "#ef8a62", "#dcdcdc"
LABEL_FS, TICK_FS, CBAR_FS = 17, 14, 13
NB = 6


def log1p_cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def load_stack(slides, stack):
    ps = sorted(Path(slides).glob(f"stroke_paste2_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in ps], label="sf", keys=[p.stem for p in ps],
                  index_unique=None)
    sec = np.unique(np.asarray(A.obsm["section_index"]).reshape(-1), return_inverse=True)[1]
    return A, np.asarray(A.obsm["array_coords"], np.float32), sec


def norm(xy):
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / np.maximum(hi - lo, 1e-8)


def depth_of(xy, inside):
    d = np.zeros(len(xy))
    if inside.sum() and (~inside).sum():
        d[inside] = cKDTree(xy[~inside]).query(xy[inside])[0]
    return d


def prof(depth, v, inside, nb=NB):
    d, x = depth[inside], v[inside]
    q = np.quantile(d, np.linspace(0, 1, nb + 1)); q[-1] += 1e-6
    m = [x[(d >= q[i]) & (d < q[i + 1])] for i in range(nb)]
    return ([float(b.mean()) for b in m],
            [float(b.std() / max(np.sqrt(len(b)), 1)) for b in m])


ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_crA")
ap.add_argument("--arm", default="3d")
ap.add_argument("--slides", default=str(NS / "stroke3d/slides_cr"))
ap.add_argument("--genes", default="Gm42418,Scgb1a1,Spp1,Vim,Lgals1,Timp1")
ap.add_argument("--core_genes", default="Gm42418,Scgb1a1")
ap.add_argument("--sections", default="",
                help="comma list of 1-based sections to draw (default all). The 2D arm sees each "
                     "section independently, so drawing four there invites a z reading it cannot "
                     "support.")
a = ap.parse_args()
OUT = NS / f"outputs/stroke_{a.arm}{a.tag}"
figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
paper_style.use()

side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o + c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
rd = lambda t: log1p_cpm(np.asarray(ad.read_h5ad(
    next(OUT.glob(f"stroke_{a.arm}{a.tag}_{t}_*_perturbed_counts.h5ad"))).X[rows], np.float32))
L_core, L_rim, L_off = rd("oncore"), rd("onrim"), rd("off")
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)

A_sh, xy_sh, sec_sh = load_stack(a.slides, "Sham")
sd = OUT / "shape_sets"
mk = lambda f: np.isin(np.arange(len(xy_sh)), np.loadtxt(sd / f, dtype=np.int64))
p_in, p_core, p_rim = mk("shape_lesion_local.txt"), mk("shape_core_local.txt"), mk("shape_rim_local.txt")
p_core, p_rim = p_core & p_in, p_rim & p_in
D = np.zeros_like(L_off)
D[p_core] = L_core[p_core] - L_off[p_core]
D[p_rim] = L_rim[p_rim] - L_off[p_rim]

A_pt, xy_pt, sec_pt = load_stack(a.slides, "PT")
Lpt = log1p_cpm(np.asarray(A_pt.X.todense(), np.float32))
Lsh_obs = log1p_cpm(np.asarray(A_sh.X.todense(), np.float32))   # healthy baseline, same places
gs = np.asarray(A_pt.var_names)
with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]])
    reg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
rp = pd.Series(reg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
o_core, o_rim = rp == "ICA", rp == "PIA"
o_in = o_core | o_rim
npt, nsh = norm(xy_pt), norm(xy_sh)
dep_p, dep_o = depth_of(xy_sh, p_in), depth_of(xy_pt, o_in)
nsec_all = int(sec_pt.max()) + 1
keep = ([int(v) - 1 for v in a.sections.split(",") if v.strip()] if a.sections
        else list(range(nsec_all)))
nsec = len(keep)
stats = {"tag": a.tag, "sections": [k + 1 for k in keep], "genes": {}}

for g in [x.strip() for x in a.genes.split(",") if x.strip()]:
    if g not in set(gs.tolist()) or g not in set(gp.tolist()):
        continue
    ov, pv = Lpt[:, int(np.where(gs == g)[0][0])], D[:, int(np.where(gp == g)[0][0])]
    fig, ax = plt.subplots(4, nsec, figsize=(2.45 * nsec + 1.9, 10.4), squeeze=False)
    vo, vp = np.quantile(ov, [.03, .97]), np.quantile(pv, [.03, .97])
    rc = ListedColormap([OTHER_C, RIM_C, CORE_C])
    oreg = np.where(o_core, 2, np.where(o_rim, 1, 0))
    preg = np.where(p_core, 2, np.where(p_rim, 1, 0))
    for ci, s in enumerate(keep):
        mp, ms = sec_pt == s, sec_sh == s
        h0 = ax[0, ci].scatter(npt[mp, 0], npt[mp, 1], c=ov[mp], s=2.6, lw=0,
                              cmap=paper_style.FIELD_CMAP, vmin=vo[0], vmax=vo[1], rasterized=True)
        h1 = ax[1, ci].scatter(nsh[ms, 0], nsh[ms, 1], c=pv[ms], s=2.6, lw=0,
                              cmap=paper_style.FIELD_CMAP, vmin=vp[0], vmax=vp[1], rasterized=True)
        ax[2, ci].scatter(npt[mp, 0], npt[mp, 1], c=oreg[mp], s=2.6, lw=0, cmap=rc, vmin=0, vmax=2, rasterized=True)
        ax[3, ci].scatter(nsh[ms, 0], nsh[ms, 1], c=preg[ms], s=2.6, lw=0, cmap=rc, vmin=0, vmax=2, rasterized=True)
        for r in range(4):
            ax[r, ci].set_xticks([]); ax[r, ci].set_yticks([]); ax[r, ci].set_aspect("equal"); ax[r, ci].grid(False)
            for sp in ax[r, ci].spines.values():
                sp.set_visible(False)
        ax[3, ci].set_xlabel(f"Section {s + 1}", fontsize=LABEL_FS)
    for r, lab in enumerate(["Observed\n(stroke mouse)", "Predicted\n(sham mouse)",
                             "Observed regions", "Imposed regions"]):
        ax[r, 0].set_ylabel(lab, fontsize=LABEL_FS - 2, labelpad=6)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.11, right=0.88, top=0.99, bottom=0.07)
    for h, arow, lb in ((h0, ax[0, -1], "Observed expression"), (h1, ax[1, -1], "Predicted induction")):
        po = arow.get_position(); pad = 0.12 * po.height
        cax = fig.add_axes([0.895, po.y0 + pad, 0.016, po.height - 2 * pad])
        cb = fig.colorbar(h, cax=cax); cb.set_label(lb, fontsize=CBAR_FS, labelpad=2)
        cb.ax.tick_params(labelsize=TICK_FS - 3); cb.set_ticks([cb.vmin, cb.vmax])
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig_stroke_gene_{g}_{a.arm}{a.tag}.{ext}", dpi=300)
    plt.close(fig)
    po_, eo = prof(dep_o, ov, o_in); pp_, ep = prof(dep_p, pv, p_in)
    ob = Lsh_obs[:, int(np.where(gs == g)[0][0])] if g in set(gs.tolist()) else None
    stats["genes"][g] = dict(observed_core=float(ov[o_core].mean() - ob[p_core].mean()),
                             observed_rim=float(ov[o_rim].mean() - ob[p_rim].mean()),
                             predicted_core=float(pv[p_core].mean()), predicted_rim=float(pv[p_rim].mean()),
                             observed_profile=po_, predicted_profile=pp_)

# ── support panel: depth profile + core/rim bars, core markers ─────────────
cg = [x.strip() for x in a.core_genes.split(",") if x.strip() and x.strip() in stats["genes"]]
fig, ax = plt.subplots(2, len(cg), figsize=(3.3 * len(cg) + 1.2, 6.4), squeeze=False)
xs = np.arange(NB)
for c, g in enumerate(cg):
    st = stats["genes"][g]
    po_, eo = prof(dep_o, Lpt[:, int(np.where(gs == g)[0][0])], o_in)
    pp_, ep = prof(dep_p, D[:, int(np.where(gp == g)[0][0])], p_in)
    gi_o = int(np.where(gs == g)[0][0])
    base_o = Lsh_obs[:, gi_o]
    od, eo2 = prof(dep_o, Lpt[:, gi_o], o_in)
    od = [v - float(base_o[p_in].mean()) for v in od]          # observed INDUCTION
    ax[0, c].errorbar(xs, od, yerr=eo2, color=paper_style.ACCENT, lw=2.2, marker="o", ms=5, label="Observed")
    ax[0, c].errorbar(xs, pp_, yerr=ep, color="#2166ac", lw=2.2, marker="s", ms=5, ls="--", label="CONCERT")
    if c == 0:
        ax[0, c].legend(fontsize=TICK_FS - 2, frameon=False)
    ax[0, c].set_xticks([0, NB - 1]); ax[0, c].set_xticklabels(["Rim", "Core"], fontsize=LABEL_FS - 2)
    ax[0, c].tick_params(labelsize=TICK_FS - 2)
    if c == 0:
        ax[0, c].set_ylabel("Induction over healthy", fontsize=LABEL_FS - 3)
    w = 0.36; x2 = np.arange(2)
    ax[1, c].bar(x2 - w / 2, [st["observed_core"], st["observed_rim"]], w,
                 color=paper_style.ACCENT, label="Observed")
    ax[1, c].bar(x2 + w / 2, [st["predicted_core"], st["predicted_rim"]], w,
                 color="#2166ac", label="CONCERT")
    ax[1, c].set_xticks(x2); ax[1, c].set_xticklabels(["Core", "Rim"], fontsize=LABEL_FS - 2)
    ax[1, c].tick_params(labelsize=TICK_FS - 2)
    ax[1, c].set_xlabel(g, fontsize=LABEL_FS)
    if c == 0:
        ax[1, c].set_ylabel("Mean induction", fontsize=LABEL_FS - 3)
        ax[1, c].legend(fontsize=TICK_FS - 2, frameon=False)
fig.subplots_adjust(wspace=0.42, hspace=0.26, left=0.11, right=0.90, top=0.98, bottom=0.10)
for ext in ("png", "pdf"):
    fig.savefig(figdir / f"fig_stroke_core_support_{a.arm}{a.tag}.{ext}", dpi=300)
plt.close(fig)
(OUT / f"figure_stats_patch_{a.arm}{a.tag}.json").write_text(json.dumps(stats, indent=2))
print(json.dumps({g: {k: round(v, 3) for k, v in s.items() if not isinstance(v, list)}
                  for g, s in stats["genes"].items()}, indent=2))
print(f"figures -> {figdir}")
