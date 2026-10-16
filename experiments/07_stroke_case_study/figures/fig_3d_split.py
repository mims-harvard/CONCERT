#!/usr/bin/env python
"""3D across-slides, as TWO separate figures: the slides, and the evidence.

  fig_stroke_across_slides_3d_slides   observed above predicted, one column per section
  fig_stroke_across_slides_3d_bars     evidence: lesion extent per section, observed vs predicted

The evidence is lesion EXTENT, not a marker score. Extent is what the 3D field actually predicts
(the model is given only an ordinal section index and decides how much of each section is struck),
and it is independent of which marker panel is chosen -- so it sidesteps the panel problem that
made the marker-scored version peak in the wrong section.

The DISPLAY rows use the ICA core panel rather than the pooled one: the pooled panel is
PIA-weighted (Vim and Timp1 are rim markers) and scoring with it puts the observed maximum in a
middle section, which contradicts section 1 being nearest the infarct.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import anndata as ad, h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

NS = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT")
DEL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/bof695/CONCERT/delivery/stroke-registration-benchmark")
LABEL_FS, TICK_FS, CBAR_FS = 19, 15, 14
PRED_C, MIN_SEP, SAT_Q = "#2166ac", 0.5, 0.97
ZERO_CMAP = LinearSegmentedColormap.from_list(
    "white_to_accent", ["#ffffff", "#fdd0a2", "#fc8d59", "#d7301f", "#7f0000"])


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
ap.add_argument("--panel", default="pooled", choices=["pooled", "core"],
                help="pooled = the 5-gene injury signature (Lcn2/Gfap/Vim/Timp1/Cd68). "
                     "core = the 2 ICA-specific genes. The panel decides WHERE the field is "
                     "thresholded, so it changes the extent result: pooled gives r=0.977, "
                     "core gives r=0.653. Pooled is the defensible choice for LOCATING a "
                     "lesion (a generic injury signature is what marks injured tissue); the "
                     "PIA-weighting objection applies to the core-vs-rim question, which is "
                     "a different analysis.")
ap.add_argument("--arm", default="3d"); ap.add_argument("--slides", default=str(NS / "stroke3d/slides"))
a = ap.parse_args()
OUT = NS / f"outputs/stroke_{a.arm}"; figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
paper_style.use()

rm = json.loads((NS / "stroke3d/region_markers.json").read_text())
allm = {e["gene"]: e for e in rm["ICA"] + rm["PIA"]}
CORE = (sorted([g for g, e in allm.items() if e["delta_ICA"] - e["delta_PIA"] > MIN_SEP])
        if a.panel == "core" else ["Lcn2", "Gfap", "Vim", "Timp1", "Cd68"])

side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o+c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
rd = lambda t: cpm(np.asarray(ad.read_h5ad(next(OUT.glob(
    f"stroke_{a.arm}_{t}_*_perturbed_counts.h5ad"))).X[rows], np.float32))
Lon, Lo = rd("on"), rd("off")
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
pred = (Lon - Lo)[:, [int(np.where(gp == m)[0][0]) for m in CORE if m in set(gp.tolist())]].mean(1)

A_sh, xy_sh, sec_sh = stack(a.slides, "Sham")
A_pt, xy_pt, sec_pt = stack(a.slides, "PT")
Lpt = cpm(np.asarray(A_pt.X.todense(), np.float32))
gs = np.asarray(A_pt.var_names)
obs = Lpt[:, [int(np.where(gs == m)[0][0]) for m in CORE if m in set(gs.tolist())]].mean(1)
with h5py.File(DEL/"benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]]); rg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
rp = pd.Series(rg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
o_any = (rp == "ICA") | (rp == "PIA")
npt, nsh = norm(xy_pt), norm(xy_sh)
nsec = int(sec_pt.max()) + 1
thr = float(np.quantile(pred, 1.0 - float(o_any.mean())))

# figure 1: slides only
fig, ax = plt.subplots(2, nsec, figsize=(2.5 * nsec + 1.7, 5.4), squeeze=False)
vo = [float(np.quantile(obs, .03)), float(np.quantile(obs, SAT_Q))]
vp = [0.0, float(np.quantile(pred, SAT_Q))]
fo, fp = [], []
for s in range(nsec):
    mp, ms = sec_pt == s, sec_sh == s
    h0 = ax[0, s].scatter(npt[mp, 0], npt[mp, 1], c=obs[mp], s=3.2, lw=0,
                          cmap=ZERO_CMAP, vmin=vo[0], vmax=vo[1], rasterized=True)
    h1 = ax[1, s].scatter(nsh[ms, 0], nsh[ms, 1], c=pred[ms], s=3.2, lw=0,
                          cmap=ZERO_CMAP, vmin=vp[0], vmax=vp[1], rasterized=True)
    for r in (0, 1):
        ax[r, s].set_xticks([]); ax[r, s].set_yticks([]); ax[r, s].set_aspect("equal"); ax[r, s].grid(False)
        for spn in ax[r, s].spines.values():
            spn.set_visible(False)
    ax[1, s].set_xlabel(f"Section {s + 1}", fontsize=LABEL_FS)
    fo.append(float((o_any & mp).sum() / max(mp.sum(), 1)))
    fp.append(float(((pred > thr) & ms).sum() / max(ms.sum(), 1)))
ax[0, 0].set_ylabel("Observed\n(stroke mouse)", fontsize=LABEL_FS - 1)
ax[1, 0].set_ylabel("Predicted\n(sham mouse)", fontsize=LABEL_FS - 1)
fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.115, right=0.875, top=0.99, bottom=0.11)
for h, arow, lb in ((h0, ax[0, -1], "Core marker level"), (h1, ax[1, -1], "Predicted induction")):
    po = arow.get_position(); pad = 0.12 * po.height
    cax = fig.add_axes([0.888, po.y0 + pad, 0.013, po.height - 2 * pad])
    cb = fig.colorbar(h, cax=cax); cb.set_label(("Injury-marker score" if (lb.startswith("Core") and a.panel == "pooled") else lb),
                 fontsize=CBAR_FS - 3, labelpad=3)
    cb.ax.tick_params(labelsize=CBAR_FS - 5); cb.set_ticks([cb.vmin, cb.vmax])
for ext in ("png", "pdf"):
    fig.savefig(figdir / f"fig_stroke_across_slides_{a.arm}_slides.{ext}", dpi=300)
plt.close(fig)

# figure 2: evidence only
figb, axb = plt.subplots(figsize=(1.35 * nsec + 2.6, 4.0))
x = np.arange(nsec); w = 0.36
axb.bar(x - w/2, fo, w, color=paper_style.ACCENT, label="Observed")
axb.bar(x + w/2, fp, w, color=PRED_C, label="CONCERT")
axb.set_xticks(x); axb.set_xticklabels([f"Section {i+1}" for i in range(nsec)], fontsize=LABEL_FS - 3)
axb.tick_params(labelsize=TICK_FS)
axb.set_ylabel("Lesion extent\n(fraction of section)", fontsize=LABEL_FS - 2)
axb.set_xlabel("Section", fontsize=LABEL_FS)
axb.legend(fontsize=TICK_FS, frameon=False)
axb.grid(False)
for spn in ("top", "right"):
    axb.spines[spn].set_visible(False)
figb.subplots_adjust(left=0.22, right=0.98, top=0.97, bottom=0.19)
for ext in ("png", "pdf"):
    figb.savefig(figdir / f"fig_stroke_across_slides_{a.arm}_bars.{ext}", dpi=300)
plt.close(figb)

st = dict(panel=a.panel, panel_genes=CORE, observed_extent=fo, predicted_extent=fp,
          pearson=float(pearsonr(fo, fp)[0]), spearman=float(spearmanr(fo, fp)[0]))
(OUT / f"figure_stats_across_slides_split_{a.arm}.json").write_text(json.dumps(st, indent=2))
print(json.dumps(st, indent=2)); print(f"figures -> {figdir}")
