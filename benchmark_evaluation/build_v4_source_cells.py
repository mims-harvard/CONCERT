#!/usr/bin/env python3
"""Shared unseen prediction-source cells for benchmark v4 (SPAC-seq).

For each target slide T of benchmark_v4/eval_panel.json:
  pool   = benchmark_v4/holdout_unassigned/slide_T.json (patch role 1 unassigned cells)
  mapped = pool cells that CONCERT's multi-slide loader keeps (map_prep_to_model_rows); the loader
           only drops cells by label and coordinate matching, so this set does not depend on weights
  source = seed-42 subsample (np.random.default_rng(42).choice, no replacement) of N_SOURCE=5000
           cells from `mapped`, the same draw rule the v3 adapters apply to the pool
Writes benchmark_v4/source_cells/slide_T.json; every v4 unseen evaluator (CONCERT, competitors,
reference strategies) scores on exactly these cells.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EVAL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval")
sys.path.insert(0, str(EVAL))
N_SOURCE = 5000
CKPT = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/ms_spac1_niche_unseen_lsfix/model.pt"


def main():
    import run_concert_v4_benchmark_eval as rc
    from benchmark_v4_eval_lib import load_eval_panel, load_holdout_unassigned
    from spacseq_multi_common import load_prepared_slide

    prep = EVAL / "multi_1-5/prepared"
    bench = prep / "benchmark_v4"
    sys.argv = ["x", "--regime", "unseen", "--outdir", str(EVAL / "multi_1-5_ho_v4/_smoke/source_cells"),
                "--model_file", CKPT, "--ls_prior_min", "3", "--ls_prior_max", "18",
                "--fpert_ls_min", "0.15", "--fpert_ls_max", "0.90"]
    args = rc.parse_args()
    panel = load_eval_panel(bench)
    out_dir = bench / "source_cells"
    out_dir.mkdir(parents=True, exist_ok=True)
    for T in sorted({u["target_slide"] for u in panel["eval_units"]}):
        pool = load_holdout_unassigned(bench, T)
        adata = load_prepared_slide(prep, T)
        ctx = rc.build_concert_ctx(args, T)
        kept, _ = rc.map_prep_to_model_rows(T, np.asarray(adata.obs_names), pool, ctx)
        rng = np.random.default_rng(42)
        src = rng.choice(kept, N_SOURCE, replace=False) if len(kept) > N_SOURCE else kept
        src = np.asarray(src, dtype=int)
        pert = adata.obs["perturbation_canon"].astype(str).to_numpy()
        assert set(src.tolist()) <= set(pool.tolist())
        assert len(np.unique(src)) == len(src)
        payload = {
            "slide_id": T, "version": "v4", "n_source": int(len(src)), "source_idx": src.tolist(),
            "n_pool_holdout_unassigned": int(len(pool)), "n_pool_mappable_to_concert": int(len(kept)),
            "rule": f"rng=np.random.default_rng(42).choice(mappable_pool, {N_SOURCE}, replace=False)",
            "labels": sorted(set(pert[src].tolist())),
        }
        (out_dir / f"slide_{T}.json").write_text(json.dumps(payload, indent=2))
        print(f"slide {T}: pool={len(pool)} mappable={len(kept)} ({len(kept)/len(pool):.4f}) source={len(src)} "
              f"labels={payload['labels']}", flush=True)
        del ctx
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
