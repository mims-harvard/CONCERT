#!/usr/bin/env python3
"""Build the Perturb-map case-study panels, one file each (PNG + PDF), for assembly.

Reads the prediction bundle written by case_study.py. Pure matplotlib, runs on the
head node. Every panel is standalone and carries no title -- identity and numbers live in
the emitted legend file, which is generated from the same stats the panels are drawn from.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"
ACCENT = "#b2182b"
OBS_COL = "#4d4d4d"
NICHE_COL = {"Tumour core": "#3366BB", "Normal tissue": "#CC6677"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.9,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def lognorm(mat: np.ndarray) -> np.ndarray:
    lib = mat.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(mat / lib * 1e4)


def tidy(ax, ylabel=None, xlabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- spatial context maps
def panel_slide_map(bundle, out, name, highlights, legend_loc="upper right"):
    xy = bundle["pos_raw"]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(xy[:, 1], -xy[:, 0], s=13, c="#e8e8e8", edgecolor="none", zorder=1)
    handles = []
    for label, (idx, colour, marker) in highlights.items():
        ax.scatter(xy[idx, 1], -xy[idx, 0], s=58, c=colour, marker=marker,
                   edgecolor="white", linewidth=0.6, zorder=3)
        handles.append(Line2D([], [], marker=marker, ls="none", ms=9, color=colour,
                              markeredgecolor="white", label=label))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=handles, fontsize=12, frameon=False, loc=legend_loc,
              handletextpad=0.4, borderpad=0.2)
    fig.tight_layout(pad=0.2)
    save(fig, out, name)


# ---------------------------------------------------------------- transplantation
def panel_marker(bundle, preds, out, name, gene, groups, ylabel):
    """Observed value shown beside every prediction, per house rule."""
    gi = int(np.where(bundle["gene_names"] == gene)[0][0])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    labels, vals, cols, hatches = [], [], [], []
    for label, (mat, idx, colour, is_obs) in groups.items():
        labels.append(label)
        vals.append(float(lognorm(mat)[idx, gi].mean()))
        cols.append(colour)
        hatches.append("//" if is_obs else None)
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, 0.66, color=cols, edgecolor="white", linewidth=0.8, zorder=1)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
            # Black hatch: the GT bars are light grey, so a white hatch vanishes.
            b.set_edgecolor("black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, rotation=20, ha="right")
    tidy(ax, ylabel=ylabel)
    ax.legend(handles=[Patch(facecolor="#bdbdbd", hatch="//", edgecolor="black",
                             label="GT"),
                       Patch(facecolor="#bdbdbd", edgecolor="black", label="Predicted")],
              fontsize=12, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


def panel_profile_r2(stats, out, name):
    trans = stats["transplantation"]
    order = [("dest_core", "Tumour core"), ("dest_normal", "Normal tissue")]
    # Only two x-positions, so a 6-inch panel is mostly empty axes -- keep it narrow.
    fig, ax = plt.subplots(figsize=(4.0, 4.6))
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, [trans[k]["r2_baseline_to_observed_jak2"] for k, _ in order], w,
           color="#bdbdbd", edgecolor="white", linewidth=0.8, label="Before perturbation")
    ax.bar(x + w / 2, [trans[k]["r2_to_observed_jak2"] for k, _ in order], w,
           color=ACCENT, edgecolor="white", linewidth=0.8, label="After perturbation")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in order], fontsize=14)
    # Headroom for the legend: at 4 inches wide it would otherwise sit on the bars.
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    tidy(ax, ylabel="Agreement with GT\nJak2 knockout profile")
    ax.legend(fontsize=11, frameon=False, loc="upper center", ncol=1,
              handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


# ---------------------------------------------------------------- niche perturbation
def panel_propagation(stats, out, name):
    probe = stats["niche_propagation_probe"]
    order = [("ring_to_periphery_tissue", "Tissue\nrelabelled"),
             ("ring_to_tumour_tissue", "Tissue set\nto tumour"),
             ("ring_to_Jak2_KO", "Jak2\nknockout"),
             ("ring_to_Tgfbr2_KO", "Tgfbr2\nknockout"),
             ("distant_patch_to_Tgfbr2_KO", "Tgfbr2 knockout,\ndistant patch")]
    order = [o for o in order if o[0] in probe]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, [probe[k]["ring_over_noise"] for k, _ in order], w,
           color="#3366BB", edgecolor="white", linewidth=0.8,
           label="Edited surrounding spots", zorder=1)
    ax.bar(x + w / 2, [probe[k]["core_over_noise"] for k, _ in order], w,
           color=ACCENT, edgecolor="white", linewidth=0.8,
           label="Unedited core", zorder=1)
    ax.axhline(1.0, color=MUTED, lw=1.3, ls="--", zorder=2)
    ax.text(len(order) - 0.45, 1.35, "Sampling noise", fontsize=12, color=MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in order], fontsize=12)
    tidy(ax, ylabel="Change in predicted expression\n(multiples of sampling noise)")
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


def panel_distance(bundle, preds, out, name, ring_keys):
    xy = bundle["pos_raw"]
    core = bundle["sel_tgfbr2_core"]
    ring = bundle["sel_peri_ring"]
    base = preds["baseline"]
    noise = np.abs(preds["baseline_draw2"] - base).mean(axis=1)
    from scipy.spatial import cKDTree
    d = cKDTree(xy[ring]).query(xy[core], k=1)[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    edges = np.quantile(d, np.linspace(0, 1, 7))
    centres = 0.5 * (edges[:-1] + edges[1:])
    for label, key, colour in ring_keys:
        delta = np.abs(preds[key][core] - base[core]).mean(axis=1)
        m, e = [], []
        for b in range(len(edges) - 1):
            sel = (d >= edges[b]) & (d <= edges[b + 1])
            if sel.sum() < 2:
                m.append(np.nan); e.append(np.nan); continue
            m.append(delta[sel].mean())
            e.append(delta[sel].std(ddof=1) / np.sqrt(sel.sum()))
        m, e = np.array(m), np.array(e)
        ax.plot(centres, m, "-o", color=colour, lw=2.0, ms=7, label=label)
        ax.fill_between(centres, m - e, m + e, color=colour, alpha=0.18, lw=0)
    ax.axhline(noise[core].mean(), color=MUTED, lw=1.3, ls="--")
    ax.text(centres[-1], noise[core].mean() * 1.08, "Sampling noise",
            fontsize=12, color=MUTED, ha="right")
    tidy(ax, ylabel="Change in core expression",
         xlabel="Distance from the edited surrounding spots")
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


def panel_direction(bundle, preds, out, name):
    """Does the identity of the surrounding knockout set the direction of the core shift?"""
    core = bundle["sel_tgfbr2_core"]
    pert = bundle["pert_raw"]
    Y = lognorm(bundle["Y_obs"])
    base = lognorm(preds["baseline"])
    ntc = Y[pert == "KP"].mean(axis=0)
    sigs = {g: Y[pert == g].mean(axis=0) - ntc for g in ("Jak2", "Tgfbr2")}
    rings = [("Jak2", "ring_to_Jak2_KO"), ("Tgfbr2", "ring_to_Tgfbr2_KO")]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.arange(len(rings))
    w = 0.36
    for j, (sig_name, colour) in enumerate([("Jak2", "#228833"), ("Tgfbr2", "#3366BB")]):
        vals = []
        for _ring_name, key in rings:
            shift = lognorm(preds[key])[core].mean(axis=0) - base[core].mean(axis=0)
            vals.append(float(np.corrcoef(shift, sigs[sig_name])[0, 1]))
        ax.bar(x + (j - 0.5) * w, vals, w, color=colour, edgecolor="white",
               linewidth=0.8, label=f"{sig_name} knockout signature")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} ring" for n, _ in rings], fontsize=14)
    tidy(ax, ylabel="Correlation of the core's shift\nwith a knockout signature")
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


# ---------------------------------------------------------------- pathway enrichment
def panel_pathways(bundle, preds, out, name, top_n=8):
    try:
        import gseapy
    except Exception as e:                                   # pragma: no cover
        print("  pathway panel skipped (gseapy unavailable):", e)
        return None
    genes = np.char.upper(bundle["gene_names"].astype(str))
    pert = bundle["pert_raw"]
    normal = lognorm(bundle["Y_obs"])[pert == "None"].mean(axis=0)
    res = {}
    for key, label in (("transplant_dest_core", "Tumour core"),
                       ("transplant_dest_normal", "Normal tissue")):
        idx = bundle["sel_dest_core"] if "core" in key else bundle["sel_dest_normal"]
        rank = lognorm(preds[key])[idx].mean(axis=0) - normal
        df = (__import__("pandas").DataFrame({"gene": genes, "score": rank})
              .groupby("gene", as_index=False).mean()
              .sort_values("score", ascending=False))
        try:
            pre = gseapy.prerank(rnk=df, gene_sets="MSigDB_Hallmark_2020",
                                 permutation_num=200, seed=0, no_plot=True,
                                 min_size=5, max_size=1000, verbose=False)
            res[label] = pre.res2d.set_index("Term")["NES"].astype(float)
        except Exception as e:
            print(f"  prerank failed for {label}: {e}")
            return None
    common = res["Tumour core"].index.intersection(res["Normal tissue"].index)
    diff = (res["Normal tissue"][common] - res["Tumour core"][common]).sort_values()
    picks = list(diff.index[:top_n // 2]) + list(diff.index[-(top_n - top_n // 2):])
    fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(picks) + 1.6))
    y = np.arange(len(picks))
    h = 0.38
    ax.barh(y - h / 2, res["Tumour core"][picks], h, color=NICHE_COL["Tumour core"],
            edgecolor="white", linewidth=0.7, label="Tumour core")
    ax.barh(y + h / 2, res["Normal tissue"][picks], h, color=NICHE_COL["Normal tissue"],
            edgecolor="white", linewidth=0.7, label="Normal tissue")
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("_", " ").capitalize() for p in picks], fontsize=12)
    ax.set_xlabel("Normalized enrichment score", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=12, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    save(fig, out, name)
    return {k: {p: float(v[p]) for p in picks} for k, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path(f"{_rp.WORKDIR}/"
                                 "spac_seq_eval/figures/pmap_case_study"))
    ap.add_argument("--out", type=Path,
                    default=Path(f"{_rp.WORKDIR}/"
                                 "spaVAE/CONCERT-main/Outputs/case_study_pm"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    b = np.load(args.src / "case_study_bundle.npz", allow_pickle=False)
    bundle = {k: b[k] for k in b.files}
    stats = json.loads((args.src / "case_study_stats.json").read_text())
    preds = {}
    for f in sorted(args.src.glob("pred_*.npy")):
        preds[f.stem.replace("pred_", "")] = np.load(f)
    print("predictions loaded:", sorted(preds))

    pert = bundle["pert_raw"]
    obs_jak2 = np.where(pert == "Jak2")[0]

    print("building panels ->", args.out)
    panel_slide_map(bundle, args.out, "panel_transplant_map", {
        "GT Jak2 knockout": (obs_jak2, "#228833", "o"),
        "Destination: tumour core": (bundle["sel_dest_core"], NICHE_COL["Tumour core"], "s"),
        "Destination: normal tissue": (bundle["sel_dest_normal"], NICHE_COL["Normal tissue"], "^"),
    })
    panel_slide_map(bundle, args.out, "panel_niche_map", {
        "Core, never edited": (bundle["sel_tgfbr2_core"], ACCENT, "o"),
        "Surrounding spots, edited": (bundle["sel_peri_ring"], "#3366BB", "s"),
    })

    panel_marker(bundle, preds, args.out, "panel_transplant_marker", "Plac8", {
        "GT\nJak2 knockout": (bundle["Y_obs"], obs_jak2, OBS_COL, True),
        "Tumour core\nbefore": (preds["baseline"], bundle["sel_dest_core"], "#bdbdbd", False),
        "Tumour core\nafter": (preds["transplant_dest_core"], bundle["sel_dest_core"],
                               NICHE_COL["Tumour core"], False),
        "Normal tissue\nbefore": (preds["baseline"], bundle["sel_dest_normal"], "#bdbdbd", False),
        "Normal tissue\nafter": (preds["transplant_dest_normal"], bundle["sel_dest_normal"],
                                 NICHE_COL["Normal tissue"], False),
    }, "Plac8 expression")

    panel_profile_r2(stats, args.out, "panel_transplant_agreement")
    # Negative control panel: no edit to the surroundings reaches the unedited core.
    # The distance and direction panels are NOT built -- with the core at the noise floor
    # under every edit there is no propagation to resolve, and plotting a decay curve or a
    # response direction through pure sampling noise would invent structure that is absent.
    panel_propagation(stats, args.out, "panel_niche_propagation")
    for stale in ("panel_niche_distance", "panel_niche_direction"):
        for ext in ("png", "pdf"):
            p = args.out / f"{stale}.{ext}"
            if p.exists():
                p.unlink()
                print("  removed stale", p.name)
    nes = panel_pathways(bundle, preds, args.out, "panel_transplant_pathways")

    summary = {"stats_source": str(args.src), "pathway_nes": nes}
    (args.out / "panel_stats.json").write_text(json.dumps(summary, indent=2, default=float))
    print("done")


if __name__ == "__main__":
    main()
