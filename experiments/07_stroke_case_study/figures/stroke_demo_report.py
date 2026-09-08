#!/usr/bin/env python
"""Read out the 2D capability demos: lesions no real experiment can deliver.

There is deliberately NO ground truth here -- no animal can be given four simultaneous infarcts,
or the same infarct at six sizes, or one placed in thalamus instead of cortex. So these panels are
not accuracy claims. What they test is whether the model's counterfactuals behave coherently:

  A. multi_region -- does striking several regions at once produce a REMOTE response (systemic),
     and is the joint response different from the sum of the single-region responses?
  B. size         -- does the response rise monotonically as the lesion is dialled up?
  C. depth        -- does the same-sized lesion read out differently at the surface than deep?

Every set is compared against ONE shared control decode, so differences between sets are the
perturbation and not baseline drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experiments" / "03_dss_gut_case_study" / "figures"))
import paper_style

MARKERS = ("Lcn2", "Gfap", "Vim", "Timp1", "Cd68")
LABEL_FS, TICK_FS, CBAR_FS = 18, 15, 14


def log1p_cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arm", default="2dflat")
    a = ap.parse_args()
    out = Path(a.outdir); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    paper_style.use()

    man = json.loads((out / "pert_sets" / "manifest.json").read_text())
    off, n_slide = man["offset"], man["n_slide"]
    rows = slice(off, off + n_slide)

    genes = np.loadtxt(out / "hvg_gene_names.txt", dtype=str)
    gidx = [int(np.where(genes == m)[0][0]) for m in MARKERS if m in set(genes.tolist())]
    if not gidx:
        raise SystemExit(f"none of {MARKERS} among the {len(genes)} modelled genes")

    def load(sample):
        hits = sorted(out.glob(f"{sample}_stroke_*_perturbed_counts.h5ad"))
        if not hits:
            raise SystemExit(f"missing counterfactual for {sample}")
        A = ad.read_h5ad(hits[-1])
        if A.n_obs != man["n_total"]:
            raise SystemExit(f"{hits[-1].name}: {A.n_obs} rows, expected {man['n_total']}")
        return log1p_cpm(np.asarray(A.X[rows], dtype=np.float32))[:, gidx].mean(1)

    base = load("demo_CONTROL")

    # target-slide coordinates and anatomy, for the spatial panels
    tgt = next(p for p in man["files"] if man["target_slide"] in Path(p).stem)
    with h5py.File(tgt, "r") as f:
        xy = np.asarray(f["obsm"]["array_coords"][:], dtype=np.float32)

    recs = {}
    for s in man["sets"]:
        d = load(f"demo_{s['name']}") - base
        idx = np.loadtxt(out / "pert_sets" / f"{s['name']}.txt", dtype=np.int64) - 1 - off
        m = np.zeros(n_slide, dtype=bool); m[idx] = True
        recs[s["name"]] = dict(meta=s, delta=d, mask=m,
                               inside=float(d[m].mean()), remote=float(d[~m].mean()),
                               centre=xy[m].mean(0))
        s.update(inside=recs[s["name"]]["inside"], remote=recs[s["name"]]["remote"])

    stats = {"arm": a.arm, "markers": [str(genes[i]) for i in gidx],
             "target_slide": man["target_slide"], "no_ground_truth": True}

    # ── A. multi-region: systemic + additivity ──────────────────────────────
    singles = [n for n, r in recs.items() if r["meta"]["scenario"] == "multi_region"
               and n != "region_ALL"]
    if singles and "region_ALL" in recs:
        joint = recs["region_ALL"]
        sum_single_remote = float(sum(recs[n]["remote"] for n in singles))
        stats["multi_region"] = dict(
            singles={n: dict(inside=recs[n]["inside"], remote=recs[n]["remote"]) for n in singles},
            joint=dict(inside=joint["inside"], remote=joint["remote"]),
            sum_of_single_remote=sum_single_remote,
            joint_over_sum=float(joint["remote"] / sum_single_remote) if sum_single_remote else None)

        fig, ax = plt.subplots(figsize=(1.25 * (len(singles) + 2) + 2.2, 3.8))
        names = singles + ["region_ALL"]
        x = np.arange(len(names) + 1)
        ins = [recs[n]["inside"] for n in names] + [np.nan]
        rem = [recs[n]["remote"] for n in names] + [sum_single_remote]
        w = 0.38
        ax.bar(x - w / 2, ins, w, color=paper_style.ACCENT, label="Lesion spots")
        ax.bar(x + w / 2, rem, w, color=paper_style.MUTED, label="Rest of section")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("region_", "").replace("ALL", "All together")
                            for n in names] + ["Sum of singles"],
                           fontsize=TICK_FS, rotation=30, ha="right")
        ax.set_ylabel("Predicted marker induction", fontsize=LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.legend(fontsize=13, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(pad=0.4)
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"fig_demo_multiregion_{a.arm}.{ext}", dpi=300)
        plt.close(fig)

    # ── B. dose-response ────────────────────────────────────────────────────
    sz = sorted([r for r in recs.values() if r["meta"]["scenario"] == "size"],
                key=lambda r: r["meta"]["requested_n"])
    if len(sz) > 2:
        n = np.array([r["meta"]["n_spots"] for r in sz], dtype=float)
        ins = np.array([r["inside"] for r in sz]); rem = np.array([r["remote"] for r in sz])
        stats["size"] = dict(n_spots=n.tolist(), inside=ins.tolist(), remote=rem.tolist(),
                             inside_spearman=float(spearmanr(n, ins)[0]),
                             remote_spearman=float(spearmanr(n, rem)[0]),
                             inside_pearson_logn=float(pearsonr(np.log(n), ins)[0]))
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        ax.plot(n, ins, "o-", color=paper_style.ACCENT, lw=2, ms=8, label="Lesion spots")
        ax.plot(n, rem, "s-", color=paper_style.MUTED, lw=2, ms=7, label="Rest of section")
        ax.set_xscale("log")
        ax.set_xlabel("Lesion size (spots)", fontsize=LABEL_FS)
        ax.set_ylabel("Predicted marker induction", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=13, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(pad=0.4)
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"fig_demo_size_{a.arm}.{ext}", dpi=300)
        plt.close(fig)

    # ── C. surface vs deep ──────────────────────────────────────────────────
    dep = [r for r in recs.values() if r["meta"]["scenario"] == "depth"]
    if len(dep) == 2:
        dep.sort(key=lambda r: r["meta"]["placement"] != "surface")
        stats["depth"] = {r["meta"]["placement"]: dict(region=r["meta"]["region"],
                          inside=r["inside"], remote=r["remote"]) for r in dep}
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9),
                                 gridspec_kw={"width_ratios": [1, 1, 0.85]})
        v = np.quantile(np.concatenate([r["delta"] for r in dep]), [0.02, 0.98])
        for ax, r in zip(axes[:2], dep):
            h = ax.scatter(xy[:, 0], xy[:, 1], c=r["delta"], s=4, lw=0,
                           cmap=paper_style.FIELD_CMAP, vmin=v[0], vmax=v[1], rasterized=True)
            ax.scatter(xy[r["mask"], 0], xy[r["mask"], 1], s=4, lw=0.35,
                       facecolors="none", edgecolors="#39ff14")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xlabel(f"{r['meta']['placement'].capitalize()} lesion ({r['meta']['region']})",
                          fontsize=LABEL_FS)
        cb = fig.colorbar(h, ax=axes[1], fraction=0.046, pad=0.02)
        cb.set_label("Predicted marker induction", fontsize=CBAR_FS)
        cb.ax.tick_params(labelsize=TICK_FS - 3); cb.set_ticks([cb.vmin, cb.vmax])
        x = np.arange(2); w = 0.38
        axes[2].bar(x - w / 2, [r["inside"] for r in dep], w,
                    color=paper_style.ACCENT, label="Lesion spots")
        axes[2].bar(x + w / 2, [r["remote"] for r in dep], w,
                    color=paper_style.MUTED, label="Rest of section")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([r["meta"]["placement"].capitalize() for r in dep],
                                fontsize=TICK_FS)
        axes[2].set_xlabel("Lesion placement", fontsize=LABEL_FS)
        axes[2].set_ylabel("Predicted marker induction", fontsize=LABEL_FS)
        axes[2].tick_params(axis="y", labelsize=TICK_FS)
        axes[2].legend(fontsize=12, frameon=False)
        for sp in ("top", "right"):
            axes[2].spines[sp].set_visible(False)
        fig.tight_layout(pad=0.5)
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"fig_demo_depth_{a.arm}.{ext}", dpi=300)
        plt.close(fig)

    pd.DataFrame([dict(name=n, scenario=r["meta"]["scenario"], n_spots=r["meta"]["n_spots"],
                       inside=r["inside"], remote=r["remote"]) for n, r in recs.items()]
                 ).to_csv(out / f"stroke_demo_{a.arm}.csv", index=False)
    (out / f"figure_stats_demo_{a.arm}.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k in ("multi_region", "size", "depth")},
                     indent=2))
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
