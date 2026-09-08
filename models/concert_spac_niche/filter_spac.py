#!/usr/bin/env python3
"""
filter_spac.py — Add filtering obs columns to SPAC-seq h5ad files.

Adds the following obs columns in-place (no cells removed, just annotated):
  filter_pass       bool   True = included in CONCERT training
  filter_reason     str    "pass" / "multiple" / "rare_target" / "low_expr"
  target_n_spots    int32  spots sharing the same perturbation label in this sample
  is_baseline       bool   True for control + unassigned (both are clear unperturbed states)
  is_perturbed_pad  bool   True for spots that received a guide (PAD denoiser group)

Rules (see SPAC_FILTERING_RULES.md):
  1. Remove "multiple"           — UMI-tie noise, target is unresolvable
  2. Remove rare gene-KO targets — fewer than --min_spots_per_target spots
  3. Remove low-expression spots — fewer than --min_counts or --min_genes (optional)

  "unassigned" (no guide captured) is a CLEAR unperturbed state and PASSES the filter.
  It is treated identically to "control" as an unperturbed reference.

Usage (single file):
    python filter_spac.py --h5ad raw_output_1-1.h5ad [options]

Usage (all files in a directory):
    python filter_spac.py --h5ad_dir /path/to/spac_seq/ [options]

Options:
    --min_spots_per_target INT   Default: 25 (recommended for CONCERT inference)
    --min_counts           INT   Default: 200
    --min_genes            INT   Default: 100
    --dry_run                    Print summary without writing to disk
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ─── constants ────────────────────────────────────────────────────────────────
LABEL_MULTIPLE   = "multiple"
LABEL_UNASSIGNED = "unassigned"
LABEL_CONTROL    = "control"

REASON_PASS      = "pass"
REASON_MULTIPLE  = "multiple"
REASON_RARE      = "rare_target"
REASON_LOW_EXPR  = "low_expr"


# ─── core filter logic ────────────────────────────────────────────────────────
def compute_filter_vectors(
    perturbation: np.ndarray,      # (N,) str array of perturbation labels
    n_counts: np.ndarray,          # (N,) int/float total UMI per spot
    n_genes: np.ndarray,           # (N,) int number of genes detected
    min_spots_per_target: int = 25,
    min_counts: int = 200,
    min_genes: int = 100,
) -> pd.DataFrame:
    """
    Returns a DataFrame with obs-level filter columns.

    Parameters
    ----------
    perturbation : str array of perturbation labels (e.g. "Tgfbr2", "control", …)
    n_counts     : total UMI count per spot
    n_genes      : number of detected genes per spot
    min_spots_per_target : minimum spots to keep a gene-KO target (removes singletons)
    min_counts   : minimum total UMI to pass expression QC
    min_genes    : minimum genes detected to pass expression QC

    Filter rules:
      1. "multiple"   → always removed (ambiguous UMI tie, target unresolvable)
      2. gene-KO with < min_spots_per_target → removed (too rare for inference)
      3. n_counts < min_counts or n_genes < min_genes → removed (low expression QC)
      "unassigned" and "control" always pass — both are clear unperturbed states.
    """
    N = len(perturbation)
    assert len(n_counts) == N and len(n_genes) == N

    # ── count spots per target (gene-KO targets only) ─────────────────────────
    # "unassigned" is a valid clear label, not a gene-KO target, so exclude from count
    NON_GENE_KO = {LABEL_MULTIPLE, LABEL_UNASSIGNED, LABEL_CONTROL}
    gene_ko_mask = ~np.isin(perturbation, list(NON_GENE_KO))

    target_counts: dict[str, int] = {}
    for lbl in perturbation[gene_ko_mask]:
        target_counts[lbl] = target_counts.get(lbl, 0) + 1

    target_n_spots = np.zeros(N, dtype=np.int32)
    for i, lbl in enumerate(perturbation):
        target_n_spots[i] = target_counts.get(lbl, 0)

    # ── assign filter_reason (first failing rule wins) ────────────────────────
    filter_reason = np.full(N, REASON_PASS, dtype=object)

    # Rule 1: multiple — unclear state, always removed
    filter_reason[perturbation == LABEL_MULTIPLE] = REASON_MULTIPLE

    # Rule 2: rare gene-KO targets (only among gene-KO spots not already filtered)
    still_pass = filter_reason == REASON_PASS
    rare_mask = still_pass & gene_ko_mask & (target_n_spots < min_spots_per_target)
    filter_reason[rare_mask] = REASON_RARE

    # Rule 3: expression QC (applies to any remaining spots)
    still_pass = filter_reason == REASON_PASS
    low_expr = still_pass & ((n_counts < min_counts) | (n_genes < min_genes))
    filter_reason[low_expr] = REASON_LOW_EXPR

    # ── derived columns ───────────────────────────────────────────────────────
    filter_pass = (filter_reason == REASON_PASS).astype(bool)

    # is_baseline: clear unperturbed reference — both control and unassigned
    # unassigned = no guide captured = definitively unperturbed
    is_baseline = np.isin(perturbation, [LABEL_CONTROL, LABEL_UNASSIGNED])

    # is_perturbed_pad: spots that received a guide (for PAD denoiser)
    # = gene-KO + multiple (all received Cas9/guide delivery)
    is_perturbed_pad = ~is_baseline

    return pd.DataFrame({
        "filter_pass":       filter_pass,
        "filter_reason":     filter_reason.astype(str),
        "target_n_spots":    target_n_spots,
        "is_baseline":       is_baseline,
        "is_perturbed_pad":  is_perturbed_pad,
    })


def print_filter_summary(sample: str, perturbation: np.ndarray,
                          df: pd.DataFrame) -> None:
    N = len(df)
    n_pass  = df["filter_pass"].sum()
    n_fail  = N - n_pass

    # Per-reason counts
    reasons = df["filter_reason"].value_counts()

    # Per-category summary
    NON_GENE_KO = {LABEL_MULTIPLE, LABEL_UNASSIGNED, LABEL_CONTROL}
    gene_ko_pass = (~np.isin(perturbation, list(NON_GENE_KO)) &
                    df["filter_pass"].values)
    n_targets_pass = len(np.unique(perturbation[gene_ko_pass]))
    ctrl_n       = (perturbation == LABEL_CONTROL).sum()
    unassigned_n = (perturbation == LABEL_UNASSIGNED).sum()

    print(f"\n{'='*60}")
    print(f"  {sample}")
    print(f"{'='*60}")
    print(f"  Total spots       : {N:>10,}")
    print(f"  ✅ filter_pass    : {n_pass:>10,}  ({100*n_pass/N:.1f}%)")
    print(f"  ❌ filtered out   : {n_fail:>10,}  ({100*n_fail/N:.1f}%)")
    print()
    print(f"  Filter reasons (removed):")
    for reason, cnt in sorted(reasons.items()):
        if reason == REASON_PASS:
            continue
        print(f"    {reason:<20}: {cnt:>8,}  ({100*cnt/N:.1f}%)")
    print()
    # post-filter counts (filter_pass=True only)
    fp = df["filter_pass"].values.astype(bool)
    unassigned_pass = ((perturbation == LABEL_UNASSIGNED) & fp).sum()
    ctrl_pass       = ((perturbation == LABEL_CONTROL) & fp).sum()
    baseline_pass   = (df["is_baseline"].values.astype(bool) & fp).sum()

    print(f"  Training set (filter_pass=True):")
    print(f"    unassigned            : {unassigned_pass:>8,}  (clear unperturbed)")
    print(f"    control               : {ctrl_pass:>8,}  (clear unperturbed)")
    print(f"    gene-KO               : {gene_ko_pass.sum():>8,}  across {n_targets_pass} targets")
    print(f"  Baseline pool (PAD ref) : {baseline_pass:>8,}  (unassigned + control, pass only)")
    print()


# ─── h5ad I/O ─────────────────────────────────────────────────────────────────
def read_perturbation_and_qc(h5ad_path: Path):
    """Read perturbation labels, n_counts, n_genes from h5ad via h5py (no full X load)."""
    with h5py.File(h5ad_path, "r") as f:
        # Perturbation (categorical)
        codes = f["obs"]["perturbation"]["codes"][:]
        cats  = [c.decode() if isinstance(c, bytes) else c
                 for c in f["obs"]["perturbation"]["categories"][:]]
        perturbation = np.array([cats[c] for c in codes])

        n_counts = f["obs"]["n_counts"][:].astype(np.float32)
        n_genes  = f["obs"]["n_genes"][:].astype(np.float32)

    return perturbation, n_counts, n_genes


def write_filter_columns(h5ad_path: Path, df: pd.DataFrame) -> None:
    """Write filter obs columns into existing h5ad file (in-place via h5py)."""
    with h5py.File(h5ad_path, "a") as f:
        obs = f["obs"]

        # Remove existing columns if present (re-run safe)
        for col in ["filter_pass", "filter_reason", "target_n_spots",
                    "is_baseline", "is_perturbed_pad"]:
            if col in obs:
                del obs[col]

        obs.create_dataset("filter_pass",
                           data=df["filter_pass"].values.astype(np.uint8))
        obs.create_dataset("filter_reason",
                           data=np.array(df["filter_reason"].values,
                                        dtype=h5py.string_dtype()))
        obs.create_dataset("target_n_spots",
                           data=df["target_n_spots"].values.astype(np.int32))
        obs.create_dataset("is_baseline",
                           data=df["is_baseline"].values.astype(np.uint8))
        obs.create_dataset("is_perturbed_pad",
                           data=df["is_perturbed_pad"].values.astype(np.uint8))


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Add SPAC-seq filter columns to h5ad files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad",     type=Path,
                     help="Single h5ad file to filter.")
    src.add_argument("--h5ad_dir", type=Path,
                     help="Directory; all *.h5ad files will be processed.")

    p.add_argument("--min_spots_per_target", type=int, default=25,
                   help="Min spots to keep a gene-KO target (Rule 2). Default 25 is recommended for CONCERT inference.")
    p.add_argument("--min_counts",  type=int, default=200,
                   help="Min total UMI per spot (Rule 4).")
    p.add_argument("--min_genes",   type=int, default=100,
                   help="Min genes detected per spot (Rule 4).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print summaries without writing to disk.")
    return p.parse_args()


def process_file(h5ad_path: Path, args) -> None:
    sample = h5ad_path.stem
    print(f"\n[Processing] {h5ad_path.name}")

    perturbation, n_counts, n_genes = read_perturbation_and_qc(h5ad_path)
    df = compute_filter_vectors(
        perturbation=perturbation,
        n_counts=n_counts,
        n_genes=n_genes,
        min_spots_per_target=args.min_spots_per_target,
        min_counts=args.min_counts,
        min_genes=args.min_genes,
    )
    print_filter_summary(sample, perturbation, df)

    if args.dry_run:
        print("  [dry_run] Skipping write.")
    else:
        write_filter_columns(h5ad_path, df)
        print(f"  [OK] Written to {h5ad_path}")


def main():
    args = parse_args()

    files = []
    if args.h5ad:
        files = [args.h5ad]
    else:
        files = sorted(args.h5ad_dir.glob("*.h5ad"))
        if not files:
            print(f"No *.h5ad files found in {args.h5ad_dir}")
            sys.exit(1)

    print(f"SPAC-seq filter  |  min_spots_per_target={args.min_spots_per_target}"
          f"  min_counts={args.min_counts}  min_genes={args.min_genes}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")

    for f in files:
        try:
            process_file(f, args)
        except Exception as e:
            print(f"  ERROR processing {f.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
