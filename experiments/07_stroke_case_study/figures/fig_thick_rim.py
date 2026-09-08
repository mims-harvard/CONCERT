#!/usr/bin/env python
"""Does a thicker imposed rim produce the ring-shaped marker field the observed data shows?

Per rim thickness, the counterfactual is assembled compartment by compartment -- core spots from
the ICA pass, rim spots from the PIA pass -- exactly as fig_cr_patch.py does. Reported per gene:
the core and rim means, and the RADIAL PROFILE outward from the core boundary, which is where a
ring shows up as a bump rather than a monotone decay.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
sys.path.insert(0, "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/CONCERT/codebase/"
                   "experiments/03_dss_gut_case_study/figures")
import paper_style

DEL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/bof695/CONCERT/delivery/stroke-registration-benchmark")
LABEL_FS, TICK_FS = 17, 13
NB = 7


def log1p_cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def load_stack(slide_dir, prefix, stack):
    ps = sorted(Path(slide_dir).glob(f"{prefix}_{stack}*.h5ad"))
    A = ad.concat([ad.read_h5ad(p) for p in ps], label="sf", keys=[p.stem for p in ps],
                  index_unique=None)
    sec = np.unique(np.asarray(A.obsm["section_index"]).reshape(-1), return_inverse=True)[1]
    return A, np.asarray(A.obsm["array_coords"], np.float32), sec


ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sets", required=True)
ap.add_argument("--slides", required=True)
ap.add_argument("--prefix", default="stroke_paste2")
ap.add_argument("--genes", default="Lcn2,Vim,Lgals1,Timp1,Gm42418")
ap.add_argument("--sections", default="")
a = ap.parse_args()
OUT = Path(a.out); figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
paper_style.use()

man = json.loads((Path(a.sets) / "thick_manifest.json").read_text())
side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o + c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
rd = lambda pat: log1p_cpm(np.asarray(
    ad.read_h5ad(next(OUT.glob(pat))).X[rows], np.float32))

L_core = rd(f"stroke_{a.arm}_cr_oncore_*_perturbed_counts.h5ad")
L_off = rd(f"stroke_{a.arm}_cr_off_*_perturbed_counts.h5ad")
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)

A_sh, xy_sh, sec_sh = load_stack(a.slides, a.prefix, "Sham")
A_pt, xy_pt, sec_pt = load_stack(a.slides, a.prefix, "PT")
Lpt = log1p_cpm(np.asarray(A_pt.X.todense(), np.float32))
gs = np.asarray(A_pt.var_names)
with h5py.File(DEL / "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]])
    reg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
rp = pd.Series(reg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
o_core, o_rim = rp == "ICA", rp == "PIA"

keep = ([int(v) - 1 for v in a.sections.split(",") if v.strip()]
        if a.sections else list(range(int(sec_sh.max()) + 1)))
in_sh = np.isin(sec_sh, keep); in_pt = np.isin(sec_pt, keep)

core = np.isin(np.arange(len(xy_sh)), np.loadtxt(Path(a.sets) / "thick_core_local.txt", dtype=np.int64))
spacing = man["spacing"]
d_core = cKDTree(xy_sh[core]).query(xy_sh)[0] / spacing          # depth outward, in rings
d_obs = cKDTree(xy_pt[o_core]).query(xy_pt)[0]
d_obs = d_obs / float(np.median(cKDTree(xy_pt).query(xy_pt, k=2)[0][:, 1]))

genes = [g.strip() for g in a.genes.split(",") if g.strip()]
edges = np.array([0.001, 1, 2, 3, 4, 6, 8, 12])


def profile(d, v, m):
    return [float(v[m & (d >= edges[i]) & (d < edges[i + 1])].mean())
            if (m & (d >= edges[i]) & (d < edges[i + 1])).sum() else float("nan")
            for i in range(len(edges) - 1)]


stats = {"arm": a.arm, "sections": [k + 1 for k in keep], "rims": {}, "observed": {}}
for g in genes:
    if g not in set(gs.tolist()):
        continue
    ov = Lpt[:, int(np.where(gs == g)[0][0])]
    stats["observed"][g] = dict(
        core=float(ov[o_core & in_pt].mean()), rim=float(ov[o_rim & in_pt].mean()),
        profile_outward=profile(d_obs, ov, in_pt))

for r in man["rims"]:
    tag = r["tag"]
    try:
        L_rim = rd(f"stroke_{a.arm}_cr_onrim_{tag}_*_perturbed_counts.h5ad")
    except StopIteration:
        print(f"[skip] rim {tag}: no eval output"); continue
    rim = np.isin(np.arange(len(xy_sh)),
                  np.loadtxt(Path(a.sets) / f"thick_rim_{tag}_local.txt", dtype=np.int64))
    rim = rim & ~core
    D = np.zeros_like(L_off)
    D[core] = L_core[core] - L_off[core]
    D[rim] = L_rim[rim] - L_off[rim]
    e = {}
    for g in genes:
        gi = np.where(gp == g)[0]
        if not len(gi):
            continue
        pv = D[:, int(gi[0])]
        e[g] = dict(core=float(pv[core & in_sh].mean()), rim=float(pv[rim & in_sh].mean()),
                    core_minus_rim=float(pv[core & in_sh].mean() - pv[rim & in_sh].mean()),
                    profile_outward=profile(d_core, pv, in_sh))
    stats["rims"][tag] = dict(n_spots=r["n_spots"], rim_core_ratio=r["rim_core_ratio"],
                              thickness_rings=r["thickness_rings"], genes=e)
    print(f"[{a.arm}] rim {tag:4s} n={r['n_spots']:5d} ratio={r['rim_core_ratio']:5.2f}  " +
          "  ".join(f"{g} core {e[g]['core']:+.3f} rim {e[g]['rim']:+.3f} "
                    f"(c-r {e[g]['core_minus_rim']:+.3f})" for g in ("Lcn2",) if g in e))

(OUT / f"figure_stats_thickrim_{a.arm}.json").write_text(json.dumps(stats, indent=2) + "\n")

# radial profiles: observed beside each rim thickness, one panel per gene
xs = 0.5 * (edges[:-1] + edges[1:])
plot_g = [g for g in genes if g in stats["observed"]]
fig, axes = plt.subplots(1, len(plot_g), figsize=(3.0 * len(plot_g) + 1.2, 3.5), squeeze=False)
tags = [t for t in stats["rims"]]
cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(tags)))
for ci, g in enumerate(plot_g):
    ax = axes[0, ci]
    for c, t in zip(cols, tags):
        ax.plot(xs, stats["rims"][t]["genes"][g]["profile_outward"], "-o", ms=3.4, lw=1.6,
                color=c, label=f"rim {t}")
    ax.axvline(1.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("Rings outward from core", fontsize=LABEL_FS - 4)
    if ci == 0:
        ax.set_ylabel("Predicted induction", fontsize=LABEL_FS - 4)
    ax.set_title(g, fontsize=LABEL_FS - 3)
    ax.tick_params(labelsize=TICK_FS - 2); ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0, -1].legend(fontsize=TICK_FS - 3, frameon=False)
fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.18, wspace=0.28)
for ext in ("png", "pdf"):
    fig.savefig(figdir / f"fig_thickrim_profiles_{a.arm}.{ext}", dpi=300)
plt.close(fig)
print(f"figures -> {figdir}")
