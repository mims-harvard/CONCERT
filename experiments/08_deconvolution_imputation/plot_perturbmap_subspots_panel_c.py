#!/usr/bin/env python3
"""Rebuild panel c of resp_deconv_perturbmap.png from the current Task-3 run.

Panel c of the composite came from an older run (Jak2 CONCERT profile, but
empirical means for unperturbed tumor and normal). The current script
plot_perturbmap_subspots.py (panel code at lines 559-616) uses the
CONCERT counterfactual profile for all three populations. This script:

  1) recomputes the three bar pairs from the stored per-spot table
     peri_deconv_impute_scores.csv (written by the current script, line 550),
     using the same masks and means as lines 560-598, and asserts they equal
     panel_means in summary_deconv_impute.json (line 753);
  2) redraws the bars with the same colours, widths, figure size and dpi as
     lines 592-615, with revised titles, tick labels and y label;
  3) pastes the new panel into the composite in place of the old panel c,
     keeps panels a, b, d, e pixel-identical, flattens the transparent
     background to white and redraws bold lowercase panel letters a-e;
  4) writes resp_deconv_perturbmap_stats.json with the numbers and definitions.

The original script is not modified. Pure matplotlib/PIL, no model reload.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

RUN_DIR = Path(
    f"{_rp.WORKDIR}/paper/Outputs/"
    "spac_seq/pmap_s0_jak2_concert_amortized"
)
SCORES = RUN_DIR / "peri_deconv_impute_scores.csv"
SUMMARY = RUN_DIR / "summary_deconv_impute.json"
SUPP = Path(
    f"{_rp.WORKDIR}/paper/manuscript/MAIN/FIG/supp"
)
SRC_COMPOSITE = SUPP / "resp_deconv_perturbmap.png.bak_panelc"
OUT_COMPOSITE = SUPP / "resp_deconv_perturbmap.png"
OUT_COMPOSITE_PDF = SUPP / "resp_deconv_perturbmap.pdf"
OUT_STATS = SUPP / "resp_deconv_perturbmap_stats.json"
WORK = Path(f"{_rp.WORKDIR}/outputs/panelc_rebuild")
ORIG_SCRIPT = (
    f"{_rp.WORKDIR}/scoring/"
    "plot_perturbmap_subspots.py"
)
CACHE_SCRIPT = (
    f"{_rp.WORKDIR}/scoring/"
    "cache_perturbmap_signatures.py"
)

# Same colours as the original script (TYPE_COLORS lines 51-57, c_raw lines 566/576/586).
C_RAW = "#984ea3"
C_IMP = {"jak2": "#e31a1c", "unpert_tumor": "#33a02c", "normal": "#9e9e9e"}
TITLES = {"jak2": "Jak2-KO", "unpert_tumor": "Unperturbed tumor", "normal": "Normal"}
YLABEL = "Spearman $\\rho$ with local reference"
XTICKS = ["Observed spot", "CONCERT profile"]

# Old panel c occupies this opaque rectangle of the composite (x0, y0, x1, y1),
# measured from the alpha channel: columns 9..923 and rows 329..666.
PANEL_C_BOX = (9, 329, 924, 667)
# Search windows for the existing panel letters (x0, y0, x1, y1) and the letter to draw.
LETTER_WINDOWS = [
    ("a", (22, 8, 62, 48)),
    ("b", (398, 8, 432, 48)),
    ("c", (24, 316, 56, 344)),
    ("d", (24, 670, 56, 708)),   # was mislabelled "b" in the old composite
    ("e", (912, 316, 942, 344)),
]


def panel_means_from_scores(scores: pd.DataFrame) -> dict:
    masks = {
        "jak2": (scores.face_type == "jak2") & (scores.signature_source == "concert"),
        "unpert_tumor": scores.face_type == "unpert_tumor",
        "normal": scores.face_type == "normal",
    }
    out = {}
    for key, m in masks.items():
        s = scores.loc[m]
        out[key] = dict(
            n=int(len(s)),
            m_raw=float(s.rho_raw.mean()),
            m_imp=float(s.rho_imputed_jak2.mean()),
            sd_raw=float(s.rho_raw.std(ddof=1)),
            sd_imp=float(s.rho_imputed_jak2.std(ddof=1)),
            n_imp_gt_raw=int((s.rho_imputed_jak2 > s.rho_raw).sum()),
        )
    return out


def draw_panel_c(pm: dict, png_path: Path, pdf_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharey=False)
    for ax, key in zip(axes, ["jak2", "unpert_tumor", "normal"]):
        vals = [pm[key]["m_raw"], pm[key]["m_imp"]]
        ax.bar([0, 1], vals, color=[C_RAW, C_IMP[key]], width=0.42)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(XTICKS, fontsize=11)
        ax.set_ylabel(YLABEL, fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim(-0.55, 1.55)
        ymax = max(vals) * 1.35
        ax.set_ylim(0, ymax)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.03 * ymax, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{TITLES[key]} (n={pm[key]['n']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ink_bbox(rgb: np.ndarray, win) -> tuple:
    x0, y0, x1, y1 = win
    sub = rgb[y0:y1, x0:x1].max(-1) < 80
    ys, xs = np.where(sub)
    if ys.size == 0:
        raise RuntimeError(f"no letter ink in window {win}")
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)


def glyph_ink(font, ch):
    im = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(im).text((50, 50), ch, font=font, fill=255)
    a = np.array(im) > 127
    ys, xs = np.where(a)
    return xs.min() - 50, ys.min() - 50, xs.max() + 1 - 50, ys.max() + 1 - 50


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(SCORES)
    summary = json.loads(SUMMARY.read_text())
    pm = panel_means_from_scores(scores)
    for key in ["jak2", "unpert_tumor", "normal"]:
        ref = summary["panel_means"][key]
        assert pm[key]["n"] == ref["n"], (key, pm[key]["n"], ref["n"])
        assert abs(pm[key]["m_raw"] - ref["m_raw"]) < 1e-12, key
        assert abs(pm[key]["m_imp"] - ref["m_imp"]) < 1e-12, key
        print(key, pm[key], flush=True)
    assert (scores.signature_source == "concert").all()

    panel_png = WORK / "panel_c_observed_vs_concert.png"
    panel_pdf = WORK / "panel_c_observed_vs_concert.pdf"
    draw_panel_c(pm, panel_png, panel_pdf)

    # Composite: flatten transparent background to white.
    src = Image.open(SRC_COMPOSITE).convert("RGBA")
    white = Image.new("RGBA", src.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(white, src).convert("RGB")
    rgb = np.array(flat)

    # Measure and erase the existing letters.
    old_boxes = {}
    for ch, win in LETTER_WINDOWS:
        bb = ink_bbox(rgb, win)
        old_boxes[ch] = bb
        x0, y0, x1, y1 = bb
        rgb[y0 - 1:y1 + 1, x0 - 1:x1 + 1] = 255
    print("old letter ink boxes", old_boxes, flush=True)
    comp = Image.fromarray(rgb)

    # Paste the new panel c, aspect preserved, centred in the old rectangle.
    bx0, by0, bx1, by1 = PANEL_C_BOX
    bw, bh = bx1 - bx0, by1 - by0
    pc = Image.open(panel_png).convert("RGB")
    scale = min(bw / pc.width, bh / pc.height)
    nw, nh = int(round(pc.width * scale)), int(round(pc.height * scale))
    pc = pc.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (bw, bh), (255, 255, 255))
    canvas.paste(pc, ((bw - nw) // 2, (bh - nh) // 2))
    comp.paste(canvas, (bx0, by0))
    print(f"panel c src {Image.open(panel_png).size} -> {nw}x{nh} in box {bw}x{bh}", flush=True)

    # Draw bold lowercase letters; size matched to the old "a" x-height.
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    target_h = old_boxes["a"][3] - old_boxes["a"][1]
    best = None
    for size in range(10, 48):
        f = ImageFont.truetype(str(font_path), size)
        gx0, gy0, gx1, gy1 = glyph_ink(f, "a")
        err = abs((gy1 - gy0) - target_h)
        if best is None or err < best[0]:
            best = (err, size)
    font = ImageFont.truetype(str(font_path), best[1])
    print(f"letter font DejaVuSans-Bold size {best[1]} (old a ink height {target_h})", flush=True)
    draw = ImageDraw.Draw(comp)
    for ch, _ in LETTER_WINDOWS:
        ox0, oy0, ox1, oy1 = old_boxes[ch]
        gx0, gy0, gx1, gy1 = glyph_ink(font, ch)
        # align ink left edge and baseline (ink bottom) with the old letter
        draw.text((ox0 - gx0, oy1 - gy1), ch, font=font, fill=(0, 0, 0))

    comp.save(OUT_COMPOSITE, dpi=(150, 150))
    comp.save(OUT_COMPOSITE_PDF, "PDF", resolution=150.0)
    print(f"wrote {OUT_COMPOSITE} {comp.size}", flush=True)
    print(f"wrote {OUT_COMPOSITE_PDF}", flush=True)

    s = ORIG_SCRIPT
    stats = {
        "figure": str(OUT_COMPOSITE),
        "panel": "c",
        "built_by": f"{_rp.WORKDIR}/scoring/"
                    "plot_perturbmap_subspots_panel_c.py",
        "backup_of_previous_png": str(SRC_COMPOSITE),
        "source_table": str(SCORES),
        "source_summary": str(SUMMARY),
        "dataset": "Perturb-map slide 0 (perturbmap_s0_spacfmt.h5ad), Jak2-KO niche; "
                   f"{len(scores)} periphery/mixed spots scored, all with CONCERT signatures",
        "panel_c": {
            key: {
                "title": f"{TITLES[key]} (n={pm[key]['n']})",
                "n_spots": pm[key]["n"],
                "observed_spot_mean_rho": round(pm[key]["m_raw"], 6),
                "concert_profile_mean_rho": round(pm[key]["m_imp"], 6),
                "observed_spot_sd_rho": round(pm[key]["sd_raw"], 6),
                "concert_profile_sd_rho": round(pm[key]["sd_imp"], 6),
                "n_spots_concert_gt_observed": pm[key]["n_imp_gt_raw"],
            }
            for key in ["jak2", "unpert_tumor", "normal"]
        },
        "values_as_printed": {
            "observed_spot": [f"{pm[k]['m_raw']:.3f}" for k in ["jak2", "unpert_tumor", "normal"]],
            "concert_profile": [f"{pm[k]['m_imp']:.3f}" for k in ["jak2", "unpert_tumor", "normal"]],
            "order": ["Jak2-KO", "Unperturbed tumor", "Normal"],
        },
        "bar_height": "mean over the n spots of the per-spot Spearman rho "
                      f"({s}:594-598)",
        "definitions": {
            "population_assignment": (
                "Jak2-KO: spot has a Jak2-KO hex neighbour "
                f"({s}:444,450,481); otherwise unperturbed tumor or normal by neighbour "
                f"tissue, ties and no-neighbour cases by nearest pure spot ({s}:307-317,510-512). "
                f"Bar masks: {s}:565,575,585."
            ),
            "expression_scale": (
                "observed spots: log1p(counts / library size * 1e4), function named logcpm "
                f"({s}:64-67,397); note the scale factor is 1e4, i.e. log counts per 10,000"
            ),
            "genes": (
                "rho over 1,500 HVGs = top-1500 variance of the normalised observed matrix "
                f"across all spots, among genes shared with the CONCERT output ({s}:46,390-402)"
            ),
            "reference_profile": (
                "mean normalised expression of observed pure spots of that population within "
                "radius 15 (spatial units of obsm['spatial']) of the spot; if none lie within "
                f"radius 15, the mean over all spots of that population ({s}:294-300)"
            ),
            "reference_pools": {
                "jak2": f"all spots with perturbation == Jak2 ({s}:338,362,482)",
                "unpert_tumor": (
                    "unperturbed tumor spots outside the contiguous Jak2 patch, falling back to "
                    f"all unperturbed tumor spots if fewer than 10 ({s}:340,415-417,517)"
                ),
                "normal": f"all spots with tissue == normal ({s}:341,418,514)",
            },
            "rho": f"Spearman correlation via scipy.stats.spearmanr ({s}:70-75)",
            "observed_spot_arm": f"rho(observed spot profile, reference) ({s}:483,519)",
            "concert_profile_arm": (
                "rho(log1p CONCERT counterfactual mean for that population at the spot "
                f"location, reference) ({s}:484,515-520); counterfactuals loaded from "
                f"location_signatures.npz and log1p-transformed ({s}:373-388); generated by "
                f"cf_query at the spot coordinate and observed spot expression with the Jak2 KO "
                f"code and periphery-or-tumor tissue code (mu_ko), unperturbed code with tumor "
                f"tissue (mu_un) or unperturbed code with normal tissue (mu_normal) "
                f"({CACHE_SCRIPT}:129-143)"
            ),
        },
        "verification": "recomputed means equal summary_deconv_impute.json panel_means to 1e-12",
        "composite_edits": [
            "old panel c replaced in rectangle x 9-923, y 329-666 (px), aspect preserved",
            "panels a, b, d, e pixels unchanged apart from flattening transparency to white "
            "and redrawing the panel letters",
            "panel letters redrawn as bold lowercase DejaVu Sans; the panel d letter was "
            "previously 'b' and is now 'd'",
        ],
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_STATS}", flush=True)


if __name__ == "__main__":
    main()
