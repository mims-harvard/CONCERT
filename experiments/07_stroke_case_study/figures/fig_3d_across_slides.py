#!/usr/bin/env python
"""3D claim: the ICA (core) injury signal decays across sections, and CONCERT reproduces it.

MARKER CHOICE MATTERS HERE. The pooled panel (Lcn2, Gfap, Vim, Timp1, Cd68) is PIA-weighted --
Vim and Timp1 are rim markers (delta_PIA - delta_ICA = +1.10 and +1.02) -- and scored with it the
observed signal peaks at section 3, which contradicts section 1 being nearest the infarct. The
core panel is therefore derived from region_markers.json as the genes with
delta_ICA - delta_PIA > MIN_SEP, and used for BOTH rows so the comparison is like for like.

Layout: sections across; the trend across sections sits to the RIGHT as two stacked curves,
observed above predicted, each on its own scale because they are different quantities
(observed level vs predicted induction).
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import argparse, json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

NS = Path(f"{_rp.WORKDIR}")
DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")
LABEL_FS, TICK_FS, CBAR_FS = 19, 15, 14
PRED_C, MIN_SEP, SAT_Q = "#2166ac", 0.5, 0.97


def cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def stack(sl, p):
    ps = sorted(Path(sl).glob(f"stroke_paste2_{p}*.h5ad"))
    A = ad.concat([ad.read_h5ad(x) for x in ps], label="sf", keys=[x.stem for x in ps], index_unique=None)
    sec = np.unique(np.asarray(A.obsm["section_index"]).reshape(-1), return_inverse=True)[1]
    return A, np.asarray(A.obsm["array_coords"], np.float32), sec


def norm(xy):
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / np.maximum(hi - lo, 1e-8)


ap = argparse.ArgumentParser()
ap.add_argument("--tag", default=""); ap.add_argument("--arm", default="3d")
ap.add_argument("--slides", default=str(NS / "stroke3d/slides"))
a = ap.parse_args()
OUT = NS / f"outputs/stroke_{a.arm}{a.tag}"; figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
paper_style.use()

rm = json.loads((NS / "stroke3d/region_markers.json").read_text())
allm = {e["gene"]: e for e in rm["ICA"] + rm["PIA"]}
CORE = sorted([g for g, e in allm.items() if e["delta_ICA"] - e["delta_PIA"] > MIN_SEP])
print(f"core (ICA) panel: {CORE}")

side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o+c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
rd = lambda t: cpm(np.asarray(ad.read_h5ad(next(OUT.glob(
    f"stroke_{a.arm}{a.tag}_{t}_*_perturbed_counts.h5ad"))).X[rows], np.float32))
Lon, Lo = rd("on"), rd("off")
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
gi_p = [int(np.where(gp == m)[0][0]) for m in CORE if m in set(gp.tolist())]
pred = (Lon - Lo)[:, gi_p].mean(1)

A_sh, xy_sh, sec_sh = stack(a.slides, "Sham")
A_pt, xy_pt, sec_pt = stack(a.slides, "PT")
Lpt = cpm(np.asarray(A_pt.X.todense(), np.float32))
gs = np.asarray(A_pt.var_names)
gi_o = [int(np.where(gs == m)[0][0]) for m in CORE if m in set(gs.tolist())]
obs = Lpt[:, gi_o].mean(1)
with h5py.File(DEL/"benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]]); rg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
rp = pd.Series(rg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
o_core = rp == "ICA"                      # the ICA compartment, per the core-marker choice
o_any = o_core | (rp == "PIA")
npt, nsh = norm(xy_pt), norm(xy_sh)
nsec = int(sec_pt.max()) + 1

fig = plt.figure(figsize=(2.62 * nsec + 4.3, 6.4))
gs_ = fig.add_gridspec(2, nsec + 1, width_ratios=[1] * nsec + [0.85], wspace=0.05, hspace=0.06)
ax = np.array([[fig.add_subplot(gs_[r, c]) for c in range(nsec)] for r in range(2)])
vo = [float(np.quantile(obs, .03)), float(np.quantile(obs, SAT_Q))]
vp = [float(np.quantile(pred, .03)), float(np.quantile(pred, SAT_Q))]
so, sp_ = [], []
for s in range(nsec):
    mp, ms = sec_pt == s, sec_sh == s
    h0 = ax[0, s].scatter(npt[mp, 0], npt[mp, 1], c=obs[mp], s=3.2, lw=0,
                          cmap=paper_style.FIELD_CMAP, vmin=vo[0], vmax=vo[1], rasterized=True)
    h1 = ax[1, s].scatter(nsh[ms, 0], nsh[ms, 1], c=pred[ms], s=3.2, lw=0,
                          cmap=paper_style.FIELD_CMAP, vmin=vp[0], vmax=vp[1], rasterized=True)
    for r in (0, 1):
        ax[r, s].set_xticks([]); ax[r, s].set_yticks([]); ax[r, s].set_aspect("equal"); ax[r, s].grid(False)
        for spn in ax[r, s].spines.values():
            spn.set_visible(False)
    ax[1, s].set_xlabel(f"Section {s + 1}", fontsize=LABEL_FS)
    oc = o_core & mp
    so.append(float(obs[oc].mean()) if oc.sum() else np.nan)
    thr = np.quantile(pred, 1.0 - float(o_any.mean()))
    pm = (pred > thr) & ms
    sp_.append(float(pred[pm].mean()) if pm.sum() else np.nan)
ax[0, 0].set_ylabel("Observed\n(stroke mouse)", fontsize=LABEL_FS - 1)
ax[1, 0].set_ylabel("Predicted\n(sham mouse)", fontsize=LABEL_FS - 1)

x = np.arange(1, nsec + 1)
axo = fig.add_subplot(gs_[0, nsec]); axp = fig.add_subplot(gs_[1, nsec])
for axx, y, col, lab in ((axo, so, paper_style.ACCENT, "Observed core\nmarker level"),
                         (axp, sp_, PRED_C, "Predicted core\nmarker induction")):
    axx.plot(x, y, color=col, lw=2.6, marker="o", ms=7)
    axx.set_xticks(x); axx.tick_params(labelsize=TICK_FS - 2)
    axx.set_ylabel(lab, fontsize=LABEL_FS - 6)
    yy = np.array([v for v in y if not np.isnan(v)])
    sp2 = yy.max() - yy.min()
    axx.set_ylim(yy.min() - 0.30 * sp2, yy.max() + 0.18 * sp2)   # not zero-anchored
    axx.grid(False)
    for spn in ("top", "right"):
        axx.spines[spn].set_visible(False)
axo.set_xticklabels([])
axp.set_xlabel("Section", fontsize=LABEL_FS - 3)

fig.subplots_adjust(left=0.09, right=0.88, top=0.98, bottom=0.11)
for h, arow, lb in ((h0, ax[0, -1], "Core marker level"), (h1, ax[1, -1], "Predicted induction")):
    po = arow.get_position(); pad = 0.12 * po.height
    cax = fig.add_axes([0.893, po.y0 + pad, 0.013, po.height - 2 * pad])
    cb = fig.colorbar(h, cax=cax); cb.set_label(lb, fontsize=CBAR_FS - 3, labelpad=2)
    cb.ax.tick_params(labelsize=CBAR_FS - 5); cb.set_ticks([cb.vmin, cb.vmax])
for ext in ("png", "pdf"):
    fig.savefig(figdir / f"fig_stroke_across_slides_{a.arm}{a.tag}_core.{ext}", dpi=300)
plt.close(fig)
ok = ~(np.isnan(so) | np.isnan(sp_))
st = dict(core_panel=CORE, observed_core_level=so, predicted_core_induction=sp_,
          pearson=float(pearsonr(np.array(so)[ok], np.array(sp_)[ok])[0]),
          spearman=float(spearmanr(np.array(so)[ok], np.array(sp_)[ok])[0]),
          y_axes_truncated=True)
(OUT / f"figure_stats_across_slides_core_{a.arm}{a.tag}.json").write_text(json.dumps(st, indent=2))
print(json.dumps(st, indent=2)); print(f"figures -> {figdir}")
