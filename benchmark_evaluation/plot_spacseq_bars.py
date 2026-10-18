#!/usr/bin/env python3
"""SPAC-seq seen/unseen benchmark bars: LFC rho and LFC RMSE (CONCERT niche in terracotta)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import release_paths as rp  # noqa: E402

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# Every panel is also written as a vector PDF for Illustrator. Matplotlib defaults to Type 3
# font embedding, which Illustrator cannot re-edit as text (labels open as unselectable
# glyph outlines); fonttype 42 embeds TrueType instead, so every label stays live text.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SPAC_ROOT = rp.SPACSEQ_PREDICTIONS
CONCERT_ROOT = SPAC_ROOT / "concert"
DEFAULT_OUT = rp.WORKDIR / "spacseq_results"
SPAC_PANEL = rp.SPACSEQ_BENCHMARK / "eval_panel.json"

SPAC_SEEN_SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]
SPAC_SEEN_TOP_N = 8
SPATIAL_SUBDIRS = ("celcomen", "cellagentchat")

METHOD_SOURCES: list[tuple[str, Path, str]] = [
    ("Biolord", SPAC_ROOT, "biolord"),
    ("GEARS", SPAC_ROOT, "gears"),
    ("scGEN", SPAC_ROOT, "scgen"),
    ("CelComen", SPAC_ROOT, "celcomen"),
    ("Steamboat", SPAC_ROOT, "steamboat"),
    ("CellAgentChat", SPAC_ROOT, "cellagentchat"),
]
# Naive + STATE (from rescored / collab CSVs) sit with competing methods
EXTRA_COMPETITORS = [
    "Mean target",
    "Mean niche",
    "Local kNN",
    "GEX kNN",
    "STATE",
]
METHOD_ORDER = [m[0] for m in METHOD_SOURCES] + EXTRA_COMPETITORS + ["CONCERT"]
# Cool qualitative palette. Naive refs are a high-contrast grey ramp;
# CONCERT is the one warm colour (#B2182B).
COLORS = {
    # Cool teal / blue / sage / lilac palette
    "Biolord": "#2F6FA3",
    "GEARS": "#9CC7D8",
    "scGEN": "#4E5D9D",
    "CelComen": "#77B89C",
    "Steamboat": "#356B5B",
    "CellAgentChat": "#147D78",
    "STATE": "#9B8EC4",
    # Naive reference strategies -- high-contrast grey ramp
    "Mean target": "#3D3D3D",
    "Mean niche": "#6B6B6B",
    "Local kNN": "#9E9E9E",
    "GEX kNN": "#C8C8C8",
    # Ours - red, the only warm colour in the palette
    "CONCERT": "#B2182B",
}
# White edge on mid-tone fills. CONCERT carries the same edge as every other bar; its colour
# alone marks it.
BAR_EDGE = "#FFFFFF"
BAR_EDGE_LW = 0.9
CONCERT_EDGE = BAR_EDGE
CONCERT_EDGE_LW = BAR_EDGE_LW


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--concert-dir", type=Path, default=CONCERT_ROOT)
    p.add_argument("--top-n", type=int, default=SPAC_SEEN_TOP_N)
    p.add_argument("--regime", choices=("seen", "unseen"), default="seen")
    # Drop named knockouts from the panel selection. The next knockouts in the existing
    # coverage ranking move up to fill the freed slots, so the selection rule is unchanged
    # and the replacement is not chosen by its result. Overall (panel b) is recomputed over
    # whatever the eight panels end up being, so the figure stays internally consistent.
    p.add_argument("--drop-genes", nargs="*", default=[],
                   help="knockouts to exclude from the panels (e.g. --drop-genes Cd83)")
    # Explicit panel order, overriding the coverage ranking. Used for the cross-slide
    # figure, which shows every held-out knockout in a hand-set order.
    p.add_argument("--gene-order", nargs="*", default=[],
                   help="explicit knockout order for the panels; overrides the ranking")
    # With 11 panels on a 3-wide grid the twelfth slot is free, so the legend goes inside
    # it instead of under the figure, and the panels can be shorter.
    p.add_argument("--inset-legend", action="store_true",
                   help="draw the legend in the free grid slot instead of below the figure")
    p.add_argument("--panel-height", type=float, default=2.8)
    return p.parse_args()


def _safe_json_l2(path: Path) -> dict:
    try:
        return json.loads(path.read_text()).get("L2", {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _gene_records(path: Path, gene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    data = _safe_json_l2(path).get(gene, [])
    lfc, edist, lfc_boot, edist_boot = [], [], [], []
    dist = []
    for rec in data:
        lfc.append(float(rec["spearman"]))
        edist.append(float(rec.get("e_dist", np.nan)))
        if "lfc_dist" in rec:
            dist.append(float(rec["lfc_dist"]))
        lfc_boot.append(float(rec.get("spearman_boot_std", np.nan)))
        edist_boot.append(float(rec.get("e_dist_boot_std", np.nan)))
    return map(np.asarray, (lfc, edist, lfc_boot, edist_boot, np.asarray(dist)))


def summarize(values: np.ndarray, boot_stds: np.ndarray | None = None) -> tuple[float, float, int]:
    """Mean +/- error for bar plots: SD across records; bootstrap only if n=1 record."""
    mask = np.isfinite(values)
    v = values[mask]
    if len(v) == 0:
        return float("nan"), 0.0, 0
    err = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    if err == 0.0 and boot_stds is not None:
        bs = boot_stds[mask]
        bs = bs[np.isfinite(bs)]
        if len(bs):
            err = float(np.mean(bs))
    return float(np.mean(v)), err, len(v)



def _lfc_payload_nonempty(payload) -> bool:
    if payload is None:
        return False
    if isinstance(payload, list):
        return len(payload) > 0
    if isinstance(payload, dict):
        return any(isinstance(v, list) and len(v) > 0 for v in payload.values())
    return False


def _stat_valid(stat: dict, metric_key: str) -> bool:
    if metric_key not in stat:
        return False
    m, _, n = stat[metric_key]
    if n <= 0 or not np.isfinite(m):
        return False
    if metric_key == "lfc_dist":
        return bool(stat.get("lfc_dist_rescore", False))
    return stat.get("can_predict", False)


def _load_mlfc_l2(mlfc: Path) -> dict:
    try:
        return json.loads(mlfc.read_text()).get("L2", {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def method_can_predict_seen(root: Path, subdir: str, gene: str) -> bool:
    for slide in SPAC_SEEN_SLIDES:
        mlfc = root / subdir / "seen" / "patch" / f"slide_{slide}" / "model_lfc.json"
        if not mlfc.is_file():
            continue
        payload = _load_mlfc_l2(mlfc).get(gene)
        if _lfc_payload_nonempty(payload):
            return True
    return False


def method_can_predict_unseen(root: Path, subdir: str, gene: str, target: str) -> bool:
    if not target:
        return False
    unit = f"{gene}__slide_{target}"
    for base in (root / subdir / "unseen" / "unseen_pert" / unit,
                 root / subdir / "unseen_pert" / unit):
        mlfc = base / "model_lfc.json"
        if not mlfc.is_file():
            continue
        payload = _load_mlfc_l2(mlfc).get(gene)
        if _lfc_payload_nonempty(payload):
            return True
    return False


def load_eval_units() -> list[tuple[str, str]]:
    data = json.loads(SPAC_PANEL.read_text())
    return [(str(u["gene"]), str(u["target_slide"])) for u in data.get("eval_units", []) if u.get("target_slide")]


def _unseen_scores_path(root: Path, subdir: str, gene: str, target: str) -> Path | None:
    unit = f"{gene}__slide_{target}"
    for base in (root / subdir / "unseen" / "unseen_pert" / unit,
                 root / subdir / "unseen_pert" / unit):
        p = base / "lfc_scores.json"
        if p.is_file():
            return p
    return None


def load_concert_by_gene(concert_root: Path | None = None, task: str = "seen", eval_units=None) -> dict[str, dict]:
    """Load CONCERT benchmark results from spacseq_predictions/concert/."""
    root = concert_root or CONCERT_ROOT
    out: dict[str, dict] = {}

    def _add(gene: str, lfc: np.ndarray, dist: np.ndarray, boot: np.ndarray | None = None):
        g = str(gene)
        if g not in out:
            out[g] = {"lfc": [], "dist": [], "boot": []}
        out[g]["lfc"].extend(lfc.tolist())
        out[g]["dist"].extend(dist[np.isfinite(dist)].tolist())
        if boot is not None:
            out[g]["boot"].extend(boot[np.isfinite(boot)].tolist())

    if task == "unseen" and eval_units:
        for gene, slide in eval_units:
            path = root / "unseen" / "unseen_pert" / f"{gene}__slide_{slide}" / "lfc_scores.json"
            lfc, _, lb, _, ld = _gene_records(path, gene)
            if len(lfc):
                _add(gene, lfc, ld, lb)
    else:
        for slide in SPAC_SEEN_SLIDES:
            path = root / "seen" / "patch" / f"slide_{slide}" / "lfc_scores.json"
            if not path.is_file():
                continue
            data = _safe_json_l2(path)
            for gene in data:
                lfc, _, lb, _, ld = _gene_records(path, gene)
                if len(lfc):
                    _add(gene, lfc, ld, lb)

    result: dict[str, dict] = {}
    for gene, parts in out.items():
        lfc_a = np.asarray(parts["lfc"], dtype=float)
        dist_a = np.asarray(parts["dist"], dtype=float)
        boot_a = np.asarray(parts["boot"], dtype=float)
        boot_for = boot_a if boot_a.size == lfc_a.size else None
        rho = summarize(lfc_a, boot_for)
        dist_s = summarize(dist_a) if dist_a.size else (float("nan"), 0.0, 0)
        can = False
        if task == "seen":
            for slide in SPAC_SEEN_SLIDES:
                mlfc = root / "seen" / "patch" / f"slide_{slide}" / "model_lfc.json"
                if mlfc.is_file() and _lfc_payload_nonempty(_load_mlfc_l2(mlfc).get(gene)):
                    can = True
                    break
        else:
            can = bool(len(lfc_a))
        if task == "unseen":
            can = bool(len(lfc_a) and np.isfinite(rho[0]))
        result[gene] = {
            "can_predict": can and ((rho[2] > 0 and np.isfinite(rho[0])) or (dist_s[2] > 0 and np.isfinite(dist_s[0]))),
            "lfc_rho": rho,
            "lfc_dist": dist_s,
            "lfc_dist_rescore": dist_s[2] > 0 and np.isfinite(dist_s[0]),
        }
    return result



def concert_has_eval(concert: dict[str, dict], gene: str) -> bool:
    return gene in concert and concert[gene].get("can_predict", False)



def spatial_coverage_seen(root: Path, gene: str) -> int:
    return sum(1 for sub in SPATIAL_SUBDIRS if method_can_predict_seen(root, sub, gene))


def spatial_coverage_unseen(root: Path, gene: str, target: str) -> int:
    return sum(1 for sub in SPATIAL_SUBDIRS if method_can_predict_unseen(root, sub, gene, target))

def baseline_coverage_seen(root: Path, gene: str) -> int:
    return sum(1 for _, _, subdir in METHOD_SOURCES if method_can_predict_seen(root, subdir, gene))


def select_top_genes(concert: dict[str, dict], root: Path, top_n: int) -> list[str]:
    """Top genes: prefer CelComen/CellAgentChat, then # baselines; require CONCERT eval."""
    niche_counts: Counter[str] = Counter()
    all_genes: set[str] = set()
    for _, _, subdir in METHOD_SOURCES:
        for slide in SPAC_SEEN_SLIDES:
            path = root / subdir / "seen" / "patch" / f"slide_{slide}" / "lfc_scores.json"
            if not path.is_file():
                continue
            for gene, recs in _safe_json_l2(path).items():
                all_genes.add(gene)
                niche_counts[gene] += len(recs if isinstance(recs, list) else [])

    ranked = sorted(
        all_genes,
        key=lambda g: (
            -spatial_coverage_seen(root, g),
            -baseline_coverage_seen(root, g),
            -niche_counts.get(g, 0),
            g,
        ),
    )
    eligible = [
        g for g in ranked
        if baseline_coverage_seen(root, g) > 0 and concert_has_eval(concert, g)
    ]
    return eligible[:top_n]


def collect_lfc_stats(genes: list[str], concert: dict[str, dict]) -> dict:
    stats: dict[str, dict[str, dict]] = {g: {} for g in genes}
    for gene in genes:
        if gene in concert and concert[gene].get("can_predict", False):
            stats[gene]["CONCERT"] = {**concert[gene], "lfc_dist_rescore": True}
        else:
            stats[gene]["CONCERT"] = {"can_predict": False, "lfc_rho": (float("nan"), 0.0, 0),
                                      "lfc_dist": (float("nan"), 0.0, 0), "lfc_dist_rescore": False}
        for label, root, subdir in METHOD_SOURCES:
            parts_lfc, parts_ed, parts_lb, parts_eb, parts_ld = [], [], [], [], []
            for slide in SPAC_SEEN_SLIDES:
                path = root / subdir / "seen" / "patch" / f"slide_{slide}" / "lfc_scores.json"
                lfc, ed, lb, eb, ld = _gene_records(path, gene)
                parts_lfc.extend(lfc.tolist())
                parts_ed.extend(ed.tolist())
                parts_lb.extend(lb.tolist())
                parts_eb.extend(eb.tolist())
                parts_ld.extend(ld.tolist())
            lfc_a, ed_a, lb_a, eb_a, ld_a = map(np.asarray, (parts_lfc, parts_ed, parts_lb, parts_eb, parts_ld))
            can = method_can_predict_seen(root, subdir, gene)
            ld_fin = ld_a[np.isfinite(ld_a)]
            has_ld = len(ld_fin) > 0
            stats[gene][label] = {
                "can_predict": can,
                "lfc_rho": summarize(lfc_a, lb_a) if can else (float("nan"), 0.0, 0),
                "lfc_dist": summarize(ld_fin) if has_ld else (float("nan"), 0.0, 0),
                "lfc_dist_rescore": has_ld,
            }
    return stats



def collect_lfc_stats_unseen(genes: list[str], eval_units: list[tuple[str, str]], concert: dict) -> dict:
    unit_map = {g: t for g, t in eval_units}
    stats: dict[str, dict[str, dict]] = {g: {} for g in genes}
    for gene in genes:
        cg = concert.get(gene)
        if cg and cg.get("can_predict", False):
            stats[gene]["CONCERT"] = {**cg, "lfc_dist_rescore": True}
        else:
            stats[gene]["CONCERT"] = {"can_predict": False, "lfc_rho": (float("nan"), 0.0, 0),
                                      "lfc_dist": (float("nan"), 0.0, 0), "lfc_dist_rescore": False}
        target = unit_map.get(gene)
        for label, root, subdir in METHOD_SOURCES:
            path = _unseen_scores_path(root, subdir, gene, target) if target else None
            if path is None:
                stats[gene][label] = {"can_predict": False, "lfc_rho": (float("nan"), 0.0, 0),
                                      "lfc_dist": (float("nan"), 0.0, 0)}
                continue
            lfc, ed, lb, eb, ld = _gene_records(path, gene)
            can = method_can_predict_unseen(root, subdir, gene, target or "")
            ld_fin = ld[np.isfinite(ld)]
            has_ld = len(ld_fin) > 0
            stats[gene][label] = {
                "can_predict": can,
                "lfc_rho": summarize(lfc, lb) if can else (float("nan"), 0.0, 0),
                "lfc_dist": summarize(ld_fin) if has_ld else (float("nan"), 0.0, 0),
                "lfc_dist_rescore": has_ld,
            }
    return stats


def baseline_coverage_unseen(root: Path, gene: str, target: str) -> int:
    return sum(
        1 for _, _, subdir in METHOD_SOURCES
        if method_can_predict_unseen(root, subdir, gene, target)
    )


def select_unseen_genes(concert: dict, eval_units: list[tuple[str, str]], root: Path) -> list[str]:
    unit_map = dict(eval_units)

    def coverage(g: str) -> int:
        t = unit_map.get(g)
        return baseline_coverage_unseen(root, g, t) if t else 0

    def spatial(g: str) -> int:
        t = unit_map.get(g)
        return spatial_coverage_unseen(root, g, t) if t else 0

    genes = [g for g, _ in eval_units]
    ranked = sorted(genes, key=lambda g: (-spatial(g), -coverage(g), g))
    return [g for g in ranked if coverage(g) > 0 and concert_has_eval(concert, g)]


def _boot_col(rows: list[dict], key: str) -> float:
    """Mean of a bootstrap column over a gene's records; NaN if the column is absent.

    Older rescored CSVs predate the bootstrap columns, so a missing key must degrade to
    "no error bar" rather than raise.
    """
    vals = []
    for r in rows:
        raw = r.get(key)
        if raw in (None, ""):
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def load_lfc_dist_from_csv(csv_path: Path, genes: list[str]) -> dict:
    stats: dict[str, dict[str, dict]] = {g: {} for g in genes}
    if not csv_path.is_file():
        return stats
    rows = list(csv.DictReader(csv_path.open()))
    method_map = {
        "scgen": "scGEN", "biolord": "Biolord", "gears": "GEARS",
        "steamboat": "Steamboat", "celcomen": "CelComen",
        "cellagentchat": "CellAgentChat",
        "concert_niche_seen": "CONCERT",
        "concert_niche_unseen": "CONCERT",
        "mean_target": "Mean target",
        "mean_niche": "Mean niche",
        "local_knn": "Local kNN",
        "gex_knn": "GEX kNN",
        "state": "STATE",
    }
    for gene in genes:
        for method, label in method_map.items():
            sub = [r for r in rows if r["gene"] == gene and r["method"] == method]
            dist_vals = [float(r["lfc_dist"]) for r in sub if np.isfinite(float(r["lfc_dist"]))]
            rho_vals = [float(r["lfc_spearman"]) for r in sub if np.isfinite(float(r["lfc_spearman"]))]
            can_dist = len(dist_vals) > 0
            can_rho = len(rho_vals) > 0
            # Per-gene error = the gene-bootstrap SD written by score_lfc_agreement, which
            # every method now carries. An SEM across records is NOT interchangeable with
            # it: most genes have a single record, and the handful of methods that happen
            # to have several would then be the only ones drawing a bar.
            rho_boot = _boot_col(sub, "lfc_spearman_boot")
            dist_boot = _boot_col(sub, "lfc_dist_boot")
            dist_err = dist_boot if np.isfinite(dist_boot) else 0.0
            rho_err = rho_boot if np.isfinite(rho_boot) else 0.0
            entry: dict = {}
            if can_dist:
                entry["lfc_dist"] = (float(np.mean(dist_vals)), dist_err, len(dist_vals))
                entry["lfc_dist_rescore"] = True
            if can_rho:
                entry["lfc_rho"] = (float(np.mean(rho_vals)), rho_err, len(rho_vals))
                entry["lfc_rho_rescore"] = True
            if entry:
                stats[gene][label] = entry
    return stats


def merge_dist(stats: dict, dist_stats: dict, genes: list[str]) -> None:
    """Merge rescored LFC rho and RMSE (same niche records for all methods)."""
    for gene in genes:
        for label in METHOD_ORDER:
            ds = dist_stats.get(gene, {}).get(label)
            if not ds:
                continue
            entry = stats[gene].setdefault(
                label,
                {
                    "can_predict": False,
                    "lfc_rho": (float("nan"), 0.0, 0),
                    "lfc_dist": (float("nan"), 0.0, 0),
                    "lfc_dist_rescore": False,
                },
            )
            if "lfc_dist" in ds:
                m, e, n = ds["lfc_dist"]
                if n > 0 and np.isfinite(m):
                    entry["lfc_dist"] = (m, e, n)
                    entry["lfc_dist_rescore"] = True
                    entry["can_predict"] = True
            if "lfc_rho" in ds:
                m, e, n = ds["lfc_rho"]
                if n > 0 and np.isfinite(m):
                    entry["lfc_rho"] = (m, e, n)
                    entry["can_predict"] = True
                    stats[gene][label]["can_predict"] = True


def ensure_all_methods(stats: dict, genes: list[str]) -> None:
    """Guarantee every gene has an entry for every METHOD_ORDER label.

    collect_lfc_stats only fills CONCERT + METHOD_SOURCES; EXTRA_COMPETITORS (the four
    reference strategies + STATE) arrive from separate CSVs and are legitimately absent
    for some regimes -- STATE has no unseen rows, for instance. Half a dozen call sites
    then indexed stats[gene][label] directly and died on KeyError. Filling the gaps once,
    here, is the structural fix; the .get() guards downstream are belt-and-braces.
    A filled entry is can_predict=False, so it is skipped by _stat_valid and never plotted.
    """
    blank = {"can_predict": False, "lfc_rho": (float("nan"), 0.0, 0),
             "lfc_dist": (float("nan"), 0.0, 0), "lfc_dist_rescore": False}
    missing: set[str] = set()
    for gene in genes:
        for label in METHOD_ORDER:
            if label not in stats.setdefault(gene, {}):
                stats[gene][label] = dict(blank)
                missing.add(label)
    if missing:
        print(f"[info] no data for {sorted(missing)} -- omitted from panels")


def merge_extra_csvs(stats: dict, outdir: Path, genes: list[str], regime: str) -> None:
    """Merge the naive reference strategies + STATE from their own rescored CSVs.

    These are model-independent, so the same files serve any CONCERT arm. Prefer the
    our-rule STATE scoring over the collaborator's ntc20 metrics when both exist.
    """
    if regime == "seen":
        naive = outdir / "spac_seen_naive_baselines_our_rule.csv"
        state_our = outdir / "spac_seen_state_our_rule.csv"
        state_fb = outdir / "spac_seen_state_collab_metrics.csv"
    else:
        naive = outdir / "spac_unseen_naive_baselines_our_rule.csv"
        state_our = outdir / "spac_unseen_state_our_rule.csv"
        state_fb = outdir / "spac_unseen_state_collab_metrics.csv"
    for path in (naive, state_our if state_our.is_file() else state_fb):
        if path.is_file():
            merge_dist(stats, load_lfc_dist_from_csv(path, genes), genes)
        else:
            print(f"[warn] missing reference CSV, methods will be absent: {path}")


def macro_overall(per_gene_stats: dict, genes: list[str], metric_key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label in METHOD_ORDER:
        means, errs, ns = [], [], []
        for gene in genes:
            if metric_key == "lfc_dist":
                if not _stat_valid(per_gene_stats[gene].get("CONCERT", {}), metric_key):
                    continue
            # .get: collect_lfc_stats only fills CONCERT + METHOD_SOURCES; the
            # EXTRA_COMPETITORS (naive baselines, STATE) arrive via merge_dist and are
            # absent whenever their CSV is missing. Indexing [label] made that a crash.
            if not _stat_valid(per_gene_stats[gene].get(label, {}), metric_key):
                continue
            m, e, n = per_gene_stats[gene][label][metric_key]
            means.append(m)
            errs.append(e)
            ns.append(n)
        can = len(means) > 0
        if can:
            m = float(np.mean(means))
            # SEM across gene means (not mean of within-gene SEMs)
            e = float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0
            entry = {"can_predict": True, metric_key: (m, e, len(means))}
        else:
            entry = {
                "can_predict": False,
                metric_key: (float("nan"), 0.0, 0),
            }
        if metric_key == "lfc_dist":
            entry["lfc_dist_rescore"] = can
        out[label] = entry
    return out


def _rank_methods(gene_stats: dict, metric_key: str, higher_better: bool) -> list[str]:
    ranked = []
    for label in METHOD_ORDER:
        if metric_key == "lfc_dist":
            if not _stat_valid(gene_stats.get(label, {}), metric_key):
                continue
        elif not gene_stats[label].get("can_predict", False):
            continue
        m, _, n = gene_stats[label][metric_key]
        if n == 0 or not np.isfinite(m):
            ranked.append((label, float("-inf") if higher_better else float("inf")))
        else:
            ranked.append((label, m))
    ranked.sort(key=lambda x: x[1], reverse=higher_better)
    return [lb for lb, _ in ranked]


def _n_methods_plotted(gene_stats: dict, metric_key: str, higher_better: bool) -> int:
    order = _rank_methods(gene_stats, metric_key, higher_better)
    return sum(
        1 for lb in order
        if _stat_valid(gene_stats.get(lb, {}), metric_key)
    )


def _ylim(panels: list[dict], metric_key: str, higher_better: bool, pad=0.12):
    vals = []
    for gs in panels:
        for label in METHOD_ORDER:
            if metric_key != "lfc_dist" and not gs.get(label, {}).get("can_predict", False):
                continue
            if not _stat_valid(gs.get(label, {}), metric_key):
                continue
            m, e, n = gs[label][metric_key]
            vals.extend([m - e, m + e])
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if metric_key == "lfc_rho":
        lo = min(lo, -0.05)
        hi = max(hi, 0.05)
    span = hi - lo or 1.0
    return lo - pad * span, hi + pad * span


def _plot_panel(ax, gene_stats, metric_key, ylabel, higher_better, ylim, title,
                show_ylabel: bool = True):
    order = _rank_methods(gene_stats, metric_key, higher_better)
    means, stds, colors, labels = [], [], [], []
    for label in order:
        if metric_key == "lfc_dist":
            if not _stat_valid(gene_stats.get(label, {}), metric_key):
                continue
        elif not gene_stats[label].get("can_predict", False):
            continue
        m, s, n = gene_stats[label][metric_key]
        if n == 0 or not np.isfinite(m):
            continue
        means.append(m)
        stds.append(s)
        colors.append(COLORS[label])
        labels.append(label)
    if not means:
        ax.axis("off")
        return
    order = sorted(range(len(means)), key=lambda i: means[i], reverse=higher_better)
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    colors = [colors[i] for i in order]
    labels = [labels[i] for i in order]
    xp = np.arange(len(means))
    # A zero error is "no replicate and no bootstrap term", not "measured as exactly 0".
    # NaN makes matplotlib skip the bar entirely; a 0.0 would still draw a cap and read
    # as a precise measurement.
    yerr = np.array([s if (np.isfinite(s) and s > 0) else np.nan for s in stds], dtype=float)
    bars = ax.bar(
        xp, means, yerr=yerr, capsize=3.5, width=0.80, color=colors,
        edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW,
        error_kw={"elinewidth": 1.1, "ecolor": "#2B2B2B"},
    )
    for i, lb in enumerate(labels):
        if lb == "CONCERT":
            bars[i].set_edgecolor(CONCERT_EDGE)
            bars[i].set_linewidth(CONCERT_EDGE_LW)
            # NB: do NOT raise the bar's zorder. Error bars are drawn at zorder 2, so a
            # bar at zorder 3 paints over the lower half of its own error bar and the
            # series reads as having a one-sided (upper-only) error.
    if metric_key == "lfc_rho":
        ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    # Method identity comes from the shared legend under the figure, so the rotated
    # per-panel tick labels (which ate ~1/5 of every panel) are dropped.
    ax.set_xticks(xp)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="normal", pad=5)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return labels


def plot_grid(genes: list[str], per_gene: dict, overall: dict, metric_key: str,
              ylabel: str, higher_better: bool, suptitle: str, out_path: Path,
              inset_legend: bool = False, panel_height: float = 3.5):
    # Overall first; no figure title/subtitle. Panel titles plain (same size as axis text).
    panels = [overall] + [per_gene[g] for g in genes]
    titles = ["Overall"] + list(genes)

    n_panels = len(panels)
    ncols = 3
    nrows = -(-n_panels // ncols)
    ylim = _ylim(panels, metric_key, higher_better)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.9, nrows * panel_height),
                             squeeze=False)
    plotted: list[str] = []
    for idx, (gs, title) in enumerate(zip(panels, titles)):
        r, c = divmod(idx, ncols)
        # One y-axis label per ROW (leftmost column only) -- the axis means the same
        # thing across a row, so repeating it three times only costs width.
        used = _plot_panel(axes[r, c], gs, metric_key, ylabel, higher_better, ylim,
                           title, show_ylabel=(c == 0))
        for lb in used or []:
            if lb not in plotted:
                plotted.append(lb)
    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")
    # Shared legend under the grid, in the canonical method order, replacing the
    # per-panel x tick labels. Bars are re-sorted by value in every panel, so a legend
    # is the only way to name them once.
    handles = [Patch(facecolor=COLORS[lb],
                     edgecolor=CONCERT_EDGE if lb == "CONCERT" else BAR_EDGE,
                     linewidth=CONCERT_EDGE_LW if lb == "CONCERT" else BAR_EDGE_LW,
                     label=lb)
               for lb in METHOD_ORDER if lb in plotted]
    free_slots = nrows * ncols - n_panels
    if inset_legend and free_slots:
        # ncol=3 with 12 entries fills column-wise into 4 rows. The legend sits in the
        # first free slot, which is already blanked by the loop above.
        r, c = divmod(n_panels, ncols)
        fig.tight_layout()
        axes[r, c].legend(handles=handles, loc="center", ncol=3, frameon=False,
                          fontsize=13, handlelength=1.4, handleheight=1.1,
                          columnspacing=1.2, labelspacing=0.9, borderaxespad=0.0)
    else:
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
                   frameon=False, fontsize=14, handlelength=1.5, handleheight=1.1,
                   columnspacing=1.6, borderaxespad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    # Vector sibling for Illustrator. Same bbox as the raster so the two stay interchangeable.
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {pdf_path}")


def main():
    args = parse_args()
    drop = set(args.drop_genes)
    if args.regime == "seen":
        concert = load_concert_by_gene(args.concert_dir, "seen")
        # Drop before truncating to top-n, so the next-ranked knockouts fill the slots.
        genes = [g for g in select_top_genes(concert, SPAC_ROOT, args.top_n + len(drop))
                 if g not in drop][:args.top_n]
        lfc_stats = collect_lfc_stats(genes, concert)
        dist_csv = args.outdir / "spac_seen_lfc_dist_per_record.csv"
        prefix, title_tag = "spacseq_seen", "seen -- patch holdout"
    else:
        eval_units = load_eval_units()
        concert = load_concert_by_gene(args.concert_dir, "unseen", eval_units)
        genes = [g for g in select_unseen_genes(concert, eval_units, SPAC_ROOT)
                 if g not in drop]
        if args.gene_order:
            missing = [g for g in args.gene_order if g not in genes]
            if missing:
                raise SystemExit(f"--gene-order names knockouts with no CONCERT eval: {missing}")
            genes = list(args.gene_order)
        genes = genes[:args.top_n]
        lfc_stats = collect_lfc_stats_unseen(genes, eval_units, concert)
        dist_csv = args.outdir / "spac_unseen_lfc_dist_per_record.csv"
        prefix, title_tag = "spacseq_unseen", "unseen -- held-out gene @ target slide"
    if drop or args.gene_order:
        print(f"[info] panels = {genes}")

    dist_stats = load_lfc_dist_from_csv(dist_csv, genes)
    merge_dist(lfc_stats, dist_stats, genes)
    # The four reference strategies (mean-target/mean-niche/local-kNN/GEX-kNN) and STATE
    # live in separate rescored CSVs, NOT in the per-record CSV above. The dots figure has
    # always merged them; the bars figure never did, so every method in EXTRA_COMPETITORS
    # was silently missing from the bar panels (and crashed macro_overall). Same sources
    # and same our-rule-over-collab preference as plot_spacseq_paired_dots.py.
    merge_extra_csvs(lfc_stats, args.outdir, genes, args.regime)
    ensure_all_methods(lfc_stats, genes)
    # Overall keeps its own across-gene SEM (a different, clearly-labelled quantity);
    # the per-gene panels carry the per-evaluation gene bootstrap from the CSVs.
    lfc_overall = macro_overall(lfc_stats, genes, "lfc_rho")
    dist_overall = macro_overall(lfc_stats, genes, "lfc_dist")

    plot_grid(
        genes, lfc_stats, lfc_overall,
        "lfc_rho", "Log-fold-change\nrank correlation", True,
        f"SPAC-seq {title_tag} (CONCERT niche-FiLM)\n"
        "error bars = SD across CONCERT-eval niche records (rescored)",
        args.outdir / f"{prefix}_lfc_bars.png",
        inset_legend=args.inset_legend, panel_height=args.panel_height,
    )
    plot_grid(
        genes, lfc_stats, dist_overall,
        "lfc_dist", "Log-fold-change RMSE", False,
        f"SPAC-seq {title_tag} -- LFC RMSE\n"
        "LFC RMSE at CONCERT eval niches (rescored); competing methods only if rescore succeeded; lower is better",
        args.outdir / f"{prefix}_lfc_dist_bars.png",
        inset_legend=args.inset_legend, panel_height=args.panel_height,
    )


if __name__ == "__main__":
    main()
