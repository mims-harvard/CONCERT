#!/usr/bin/env python3
"""Aggregate benchmark v3 results into CSV tables and markdown summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

METHODS = ["biolord", "gears", "scgen", "celcomen", "steamboat", "cellagentchat"]
METHOD_LABEL = {
    "biolord": "Biolord", "gears": "GEARS", "scgen": "scGEN",
    "celcomen": "CelComen", "steamboat": "Steamboat", "cellagentchat": "CellAgentChat",
}

SPAC_ROOT = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5_ho_v3")
PM_ROOT = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/results_panel1053ko_v3")
SPAC_PANEL = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5/prepared/benchmark_v3/eval_panel.json"
)
PM_PANEL = Path(
    "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/"
    "prepared_panel1053ko/benchmark_v3/eval_panel.json"
)
OUT = Path("/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main/Outputs/spac_seq")

SPAC_SEEN_SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]
PM_SEEN_SLIDES = ["0", "1", "2", "3"]
PM_SEEN_GENES = ["Jak2", "Ifngr2", "Tgfbr2"]

SLIDE_RE = re.compile(r"__slide_(.+)$")


def _stats(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(mean=np.nan, sd=np.nan, sem=np.nan, n=0)
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return dict(mean=float(a.mean()), sd=sd, sem=sd / np.sqrt(a.size), n=int(a.size))


def _load_panel(path: Path) -> list[tuple[str, str | None]]:
    data = json.loads(path.read_text())
    return [(str(u["gene"]), str(u["target_slide"]) if u.get("target_slide") else None)
            for u in data.get("eval_units", [])]


def _records_from_scores(path: Path, gene: str | None = None) -> tuple[list[float], list[float], list[float], list[float]]:
    if not path.is_file():
        return [], [], [], []
    data = json.loads(path.read_text()).get("L2", {})
    genes = [gene] if gene is not None else list(data.keys())
    sp, ed, sp_boot, ed_boot = [], [], [], []
    for g in genes:
        recs = data.get(g, [])
        if not isinstance(recs, list):
            continue
        for r in recs:
            v = r.get("spearman")
            if v is not None and np.isfinite(v):
                sp.append(float(v))
                b = r.get("spearman_boot_std")
                if b is not None and np.isfinite(b):
                    sp_boot.append(float(b))
            v = r.get("e_dist")
            if v is not None and np.isfinite(v):
                ed.append(float(v))
                b = r.get("e_dist_boot_std")
                if b is not None and np.isfinite(b):
                    ed_boot.append(float(b))
    return sp, ed, sp_boot, ed_boot


def _within_niche_err(boot_vals: list[float]) -> float:
    a = np.asarray(boot_vals, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else 0.0


def spac_seen_files(method: str, slide: str) -> Path:
    return SPAC_ROOT / method / "seen" / "patch" / f"slide_{slide}" / "lfc_scores.json"


def spac_unseen_unit_dir(method: str, unit: str) -> Path:
    boot = SPAC_ROOT / method / "unseen" / "unseen_pert" / unit
    if (boot / "lfc_scores.json").is_file():
        return boot
    return SPAC_ROOT / method / "unseen_pert" / unit


def spac_unseen_files(method: str, slide: str) -> list[Path]:
    out: dict[str, Path] = {}
    for base in (SPAC_ROOT / method / "unseen" / "unseen_pert", SPAC_ROOT / method / "unseen_pert"):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if not p.is_dir() or not p.name.endswith(f"__slide_{slide}"):
                continue
            lf = p / "lfc_scores.json"
            if lf.is_file():
                out[p.name] = lf
    return sorted(out.values())


def pm_seen_file(method: str, slide: str) -> Path:
    return PM_ROOT / method / "seen_pert" / f"slide_{slide}" / "lfc_scores.json"


def pm_unseen_files(method: str, slide: str) -> list[Path]:
    base = PM_ROOT / method / "unseen_pert"
    if not base.is_dir():
        return []
    return sorted(
        p / "lfc_scores.json"
        for p in base.iterdir()
        if p.is_dir() and p.name.endswith(f"__slide_{slide}")
    )


def collect(method: str, dataset: str, regime: str, slide: str) -> dict:
    sp, ed, sp_boot, ed_boot = [], [], [], []
    if dataset == "spacseq":
        if regime == "seen":
            sp, ed, sp_boot, ed_boot = _records_from_scores(spac_seen_files(method, slide))
        else:
            for f in spac_unseen_files(method, slide):
                s, e, sb, eb = _records_from_scores(f)
                sp += s
                ed += e
                sp_boot += sb
                ed_boot += eb
    else:
        if regime == "seen":
            sp, ed, sp_boot, ed_boot = _records_from_scores(pm_seen_file(method, slide))
        else:
            for f in pm_unseen_files(method, slide):
                s, e, sb, eb = _records_from_scores(f)
                sp += s
                ed += e
                sp_boot += sb
                ed_boot += eb
    lfc = _stats(sp)
    edist = _stats(ed)
    return {
        "dataset": dataset,
        "regime": regime,
        "method": method,
        "slide": slide,
        "n_records": lfc["n"],
        "lfc_spearman": lfc["mean"],
        "lfc_sd": lfc["sd"],
        "lfc_sem": lfc["sem"],
        "lfc_boot_err": _within_niche_err(sp_boot),
        "e_dist": edist["mean"],
        "e_dist_sd": edist["sd"],
        "e_dist_sem": edist["sem"],
        "e_dist_boot_err": _within_niche_err(ed_boot),
    }


def collect_gene(method: str, dataset: str, regime: str, gene: str, target_slide: str | None) -> dict:
    sp, ed, sp_boot, ed_boot = [], [], [], []
    if dataset == "spacseq":
        if regime == "seen":
            for slide in SPAC_SEEN_SLIDES:
                s, e, sb, eb = _records_from_scores(spac_seen_files(method, slide), gene)
                sp += s
                ed += e
                sp_boot += sb
                ed_boot += eb
        else:
            path = spac_unseen_unit_dir(method, f"{gene}__slide_{target_slide}") / "lfc_scores.json"
            sp, ed, sp_boot, ed_boot = _records_from_scores(path, gene)
    else:
        if regime == "seen":
            for slide in PM_SEEN_SLIDES:
                s, e, sb, eb = _records_from_scores(pm_seen_file(method, slide), gene)
                sp += s
                ed += e
                sp_boot += sb
                ed_boot += eb
        else:
            path = PM_ROOT / method / "unseen_pert" / f"{gene}__slide_{target_slide}" / "lfc_scores.json"
            sp, ed, sp_boot, ed_boot = _records_from_scores(path, gene)
    lfc = _stats(sp)
    edist = _stats(ed)
    return {
        "dataset": dataset,
        "regime": regime,
        "method": method,
        "gene": gene,
        "target_slide": target_slide or "",
        "n_records": lfc["n"],
        "lfc_spearman": lfc["mean"],
        "lfc_boot_err": _within_niche_err(sp_boot),
        "e_dist": edist["mean"],
        "e_dist_boot_err": _within_niche_err(ed_boot),
    }


def macro_rows(per_slide_rows: list[dict]) -> list[dict]:
    out = []
    keys = {(r["dataset"], r["regime"], r["method"]) for r in per_slide_rows}
    for dataset, regime, method in sorted(keys):
        sub = [r for r in per_slide_rows if r["dataset"] == dataset and r["regime"] == regime and r["method"] == method]
        lfc_vals = [r["lfc_spearman"] for r in sub if r["n_records"] > 0 and np.isfinite(r["lfc_spearman"])]
        ed_vals = [r["e_dist"] for r in sub if r["n_records"] > 0 and np.isfinite(r["e_dist"])]
        lfc = _stats(lfc_vals)
        ed = _stats(ed_vals)
        out.append({
            "dataset": dataset,
            "regime": regime,
            "method": method,
            "n_slides": sum(1 for r in sub if r["n_records"] > 0),
            "n_records": int(sum(r["n_records"] for r in sub)),
            "lfc_spearman": lfc["mean"],
            "lfc_sd": lfc["sd"],
            "lfc_sem": lfc["sem"],
            "e_dist": ed["mean"],
            "e_dist_sd": ed["sd"],
            "e_dist_sem": ed["sem"],
        })
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) and np.isfinite(r[k]) else r.get(k, "")) for k in fields})


def write_markdown(path: Path, macro: list[dict]) -> None:
    lines = [
        "# Benchmark v3 -- method comparison (seen / unseen)",
        "",
        "Metrics pooled over eval records (SPAC seen: per-niche patch holdout; "
        "SPAC/PM unseen: one eval unit per gene @ target slide). "
        "Plot error bars use within-niche bootstrap SD, not cross-niche SD.",
        "",
    ]
    for dataset in ("spacseq", "perturbmap"):
        dlabel = "SPAC-seq" if dataset == "spacseq" else "Perturb-map"
        for regime in ("seen", "unseen"):
            sub = [r for r in macro if r["dataset"] == dataset and r["regime"] == regime]
            if not sub:
                continue
            sub.sort(key=lambda r: r["lfc_spearman"], reverse=True)
            lines += [
                f"## {dlabel} -- {regime}",
                "",
                "| Method | N records | LFC ρ (mean±SEM) | e-dist (mean±SEM) |",
                "| --- | ---: | ---: | ---: |",
            ]
            for r in sub:
                if r["n_records"] == 0:
                    continue
                lines.append(
                    f"| {METHOD_LABEL[r['method']]} | {r['n_records']} | "
                    f"{r['lfc_spearman']:.3f}±{r['lfc_sem']:.3f} | "
                    f"{r['e_dist']:.3f}±{r['e_dist_sem']:.3f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUT)
    args = ap.parse_args()

    spac_slides = {"seen": SPAC_SEEN_SLIDES, "unseen": ["1-2", "1-3", "1-4", "1-5"]}
    pm_slides = {"seen": PM_SEEN_SLIDES, "unseen": ["1"]}

    spac_panel = _load_panel(SPAC_PANEL)
    pm_panel = _load_panel(PM_PANEL)

    per_slide = []
    per_pert = []
    for m in METHODS:
        for slide in spac_slides["seen"]:
            per_slide.append(collect(m, "spacseq", "seen", slide))
        for slide in spac_slides["unseen"]:
            per_slide.append(collect(m, "spacseq", "unseen", slide))
        for slide in pm_slides["seen"]:
            per_slide.append(collect(m, "perturbmap", "seen", slide))
        for slide in pm_slides["unseen"]:
            per_slide.append(collect(m, "perturbmap", "unseen", slide))
        for gene, target in spac_panel:
            per_pert.append(collect_gene(m, "spacseq", "seen", gene, target))
            per_pert.append(collect_gene(m, "spacseq", "unseen", gene, target))
        for gene in PM_SEEN_GENES:
            per_pert.append(collect_gene(m, "perturbmap", "seen", gene, None))
        for gene, target in pm_panel:
            per_pert.append(collect_gene(m, "perturbmap", "unseen", gene, target))

    macro = macro_rows(per_slide)
    out = args.outdir
    fields_slide = [
        "dataset", "regime", "method", "slide", "n_records",
        "lfc_spearman", "lfc_sd", "lfc_sem", "e_dist", "e_dist_sd", "e_dist_sem",
    ]
    fields_macro = [
        "dataset", "regime", "method", "n_slides", "n_records",
        "lfc_spearman", "lfc_sd", "lfc_sem", "e_dist", "e_dist_sd", "e_dist_sem",
    ]
    fields_pert = [
        "dataset", "regime", "method", "gene", "target_slide", "n_records",
        "lfc_spearman", "lfc_boot_err", "e_dist", "e_dist_boot_err",
    ]
    write_csv(out / "benchmark_v3_per_slide.csv", per_slide, fields_slide)
    write_csv(out / "benchmark_v3_per_perturbation.csv", per_pert, fields_pert)
    write_csv(out / "benchmark_v3_summary.csv", macro, fields_macro)
    write_markdown(out / "benchmark_v3_summary.md", macro)
    print(f"Wrote {out / 'benchmark_v3_summary.csv'}")
    print(f"Wrote {out / 'benchmark_v3_per_slide.csv'}")
    print(f"Wrote {out / 'benchmark_v3_per_perturbation.csv'}")
    print(f"Wrote {out / 'benchmark_v3_summary.md'}")


if __name__ == "__main__":
    main()
