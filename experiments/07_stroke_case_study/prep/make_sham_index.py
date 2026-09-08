#!/usr/bin/env python
"""Write the 1-based row indices of the Sham cells in loader-concatenation order.

`load_h5ad_multi` appends cells file by file in the order given on the command line, and this
dataset triggers none of the row filters in `run()` (no `multiple` labels, no holdout, both
perturbation classes far above --min_cells_per_pert). That is ASSERTED, not assumed: the total
row count is checked against the sum of per-file row counts before the indices are written.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="comma list, same order as --multi_sample_files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stack", default="Sham", help="stack whose cells get the counterfactual")
    a = ap.parse_args()

    paths = [p.strip() for p in a.files.split(",") if p.strip()]
    counts, is_target = [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            n = f["obs"]["_index"].shape[0]
            mouse = f["obs"]["mouse"]
            mouse = mouse["categories"][:][mouse["codes"][0]] if isinstance(mouse, h5py.Group) else mouse[0]
            mouse = mouse.decode() if isinstance(mouse, bytes) else str(mouse)
        counts.append(n)
        is_target.append(mouse == a.stack)
        print(f"  {Path(p).stem:24s} n={n:6d} mouse={mouse} target={is_target[-1]}")

    offsets = np.concatenate([[0], np.cumsum(counts)])
    idx = np.concatenate([np.arange(offsets[i], offsets[i + 1])
                          for i, t in enumerate(is_target) if t])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out, idx + 1, fmt="%d")          # runner subtracts 1
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"n_total_expected": int(offsets[-1]), "n_target": int(len(idx)),
         "files": [str(p) for p in paths], "counts": [int(c) for c in counts],
         "is_target": [bool(t) for t in is_target], "stack": a.stack}, indent=2) + "\n")
    print(f"[sham-index] {len(idx)} / {offsets[-1]} cells -> {a.out}")


if __name__ == "__main__":
    main()
