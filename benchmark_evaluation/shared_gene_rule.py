#!/usr/bin/env python3
"""Run an existing SPAC-seq benchmark eval script under the shared gene rule.

Shared gene rule (identical for every method): a gene is scored for a unit when
    hvg_rank <= n_hvgs on the target slide  AND  gene is in CONCERT's model gene list  AND  gene != knockout.
CONCERT's model gene list is the 2000-gene list saved at training
(checkpoints/spacseq_{cross_slide,within_niche}/hvg_gene_names.txt; the two files are identical).

Mechanism: spacseq_common.select_hvgs is wrapped so the HVG mask it returns is ANDed with the CONCERT
gene list. Every model eval script (evaluate_concert_spacseq.py, the competitor unseen evaluator,
run_multi_seen_eval.py) uses that mask only to build the scored-gene mask `keep` (LFC and the
source-cell bootstrap) and hvg_info.json, so predictions, source cells, reference cells, panel and
bootstrap settings are untouched. The KO gene is removed by the scripts themselves.

Options (before the script path):
  --concert_ctx_check   assert ctx["gene_names"] of every loaded CONCERT model equals the saved list
  --pred_bin_sums SLIDES comma list of target slides; for unseen units on these slides, save per-bin sums
                        of the method's normalised prediction (logcp10k, cp10k = expm1, raw = counts via
                        the source-cell library sizes, as in gene_align.log_cp10k_to_counts) over the CONCERT
                        genes to <outdir>/unseen_pert/<unit>/pred_bin_sums.npz (used by the NTC audit)
  --seen_pred_bin_sums SLIDES same for the seen patch loop of competitor runs (run_multi_seen_eval.py): per panel
                        gene, per-bin sums over the eval's patch source cells, saved to
                        <outdir>/patch/slide_<S>/pred_bin_sums/<gene>.npz

Usage: python shared_gene_rule.py [--concert_ctx_check] [--pred_bin_sums 1-5] <script.py> [script args...]
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import release_paths as rp  # noqa: E402

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

CHECKPOINTS = rp.CHECKPOINTS
GENE_FILES = [
    CHECKPOINTS / "spacseq_cross_slide" / "hvg_gene_names.txt",
    CHECKPOINTS / "spacseq_within_niche" / "hvg_gene_names.txt",
]
GENE_RULE = "hvg_rank<=n_hvgs AND CONCERT model gene (hvg_gene_names.txt) AND != KO"


def load_concert_genes() -> np.ndarray:
    lists = [[ln.strip() for ln in f.read_text().splitlines() if ln.strip()] for f in GENE_FILES]
    assert all(lst == lists[0] for lst in lists), "CONCERT seen/unseen gene lists differ"
    assert len(set(lists[0])) == len(lists[0]), "duplicate genes in CONCERT gene list"
    return np.asarray(lists[0], dtype=object)


def main():
    argv = sys.argv[1:]
    ctx_check = False
    bin_slides: set[str] = set()
    seen_bin_slides: set[str] = set()
    while argv and argv[0].startswith("--"):
        opt = argv.pop(0)
        if opt == "--concert_ctx_check":
            ctx_check = True
        elif opt == "--pred_bin_sums":
            bin_slides = {s for s in argv.pop(0).split(",") if s}
        elif opt == "--seen_pred_bin_sums":
            seen_bin_slides = {s for s in argv.pop(0).split(",") if s}
        else:
            raise SystemExit(f"unknown wrapper option {opt}")
    if not argv:
        raise SystemExit(__doc__)
    script = Path(argv[0]).resolve()
    genes = load_concert_genes()
    gene_set = set(genes.tolist())

    ecf = None
    if ctx_check:
        # Mirror evaluate_concert_spacseq.py import order (model code, then this directory, on
        # sys.path; eval_spatial_counterfactual imported before spacseq_common.import_eval_utils).
        sys.path.insert(0, str(EVAL_DIR.parent / "models" / "concert_benchmark"))
        sys.path.insert(0, str(EVAL_DIR))
        import eval_spatial_counterfactual as ecf

    import spacseq_common as sec

    eu, sa = sec.import_eval_utils()
    import inspect
    src = inspect.getsource(sa.open_raw_counts)
    print(f"[shared gene rule] gene_align = {sa.__file__}; eval_utils = {eu.__file__}", flush=True)
    assert "np.load(counts_path, mmap_mode=\"r\")" in src, "gene_align.open_raw_counts is not the fixed reader"

    orig_select = sec.select_hvgs
    orig_info = sa.hvg_info_dict

    def select_hvgs_shared(adata_all, n_top, sa_):
        mask = orig_select(adata_all, n_top, sa_)
        names = np.asarray(adata_all.var_names).astype(str)
        in_concert = np.fromiter((g in gene_set for g in names), dtype=bool, count=len(names))
        out = np.asarray(mask, dtype=bool) & in_concert
        print(f"[shared gene rule] hvg mask {int(mask.sum())} -> {int(out.sum())} genes after AND with CONCERT genes "
              f"({int(in_concert.sum())} CONCERT genes in var)", flush=True)
        return out

    def hvg_info_shared(hvg_mask):
        d = orig_info(hvg_mask)
        d["gene_rule"] = GENE_RULE
        d["concert_gene_file"] = str(GENE_FILES[0])
        return d

    sec.select_hvgs = select_hvgs_shared
    sa.hvg_info_dict = hvg_info_shared

    if ctx_check:
        orig_ctx = ecf.load_eval_context_multi

        def ctx_checked(fwd):
            ctx = orig_ctx(fwd)
            cg = np.asarray(ctx["gene_names"]).astype(str)
            assert cg.tolist() == genes.tolist(), "ctx gene_names differ from hvg_gene_names.txt"
            print(f"[shared gene rule] CONCERT ctx gene_names == hvg_gene_names.txt ({len(cg)} genes)", flush=True)
            return ctx

        ecf.load_eval_context_multi = ctx_checked

    if bin_slides:
        import spacseq_cross_slide_scoring as suel
        import spacseq_slides as smc

        orig_unseen = suel.unseen_spac

        def unseen_with_bins(adata_target, gene, global_src_idx, ko_idx, pred_g, hvg_mask, args, **kw):
            res = orig_unseen(adata_target, gene, global_src_idx, ko_idx, pred_g, hvg_mask, args, **kw)
            slide = str(args.slide_id)
            method = kw.get("scoring_method")
            if slide in bin_slides and pred_g is not None and not kw.get("unpredictable", False) \
                    and len(pred_g) == len(global_src_idx):
                p = smc.normalize_pred_cache(method, {gene: pred_g}).get(gene) if method else pred_g
                names = np.asarray(adata_target.var_names).astype(str)
                cols = np.array([i for i, g in enumerate(names) if g in gene_set], dtype=int)
                src = np.asarray(global_src_idx, dtype=int)
                cbin = adata_target.obs["bin_id"].to_numpy().astype(int)[src]
                lib = smc.lib_sizes_from_adata(adata_target, src)
                pc = np.asarray(p, dtype=np.float64)[:, cols]
                spaces = {"logcp10k": pc,
                          "cp10k": np.expm1(np.clip(pc, 0.0, 14.5)),
                          "raw": np.asarray(sa.log_cp10k_to_counts(np.asarray(p)[:, cols], lib), dtype=np.float64)}
                out = {"prep_gene_idx": cols, "genes": names[cols], "src_idx": src, "src_bin": cbin,
                       "src_count_per_bin": np.bincount(cbin, minlength=36), "method": str(method)}
                for k, v in spaces.items():
                    s = np.zeros((36, v.shape[1]))
                    np.add.at(s, cbin, v)
                    out[f"sum_{k}"] = s.astype(np.float32)
                udir = Path(args.outdir) / "unseen_pert" / f"{gene}__slide_{slide}"
                udir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(udir / "pred_bin_sums.npz", **out)
            return res

        suel.unseen_spac = unseen_with_bins

    if seen_bin_slides:
        import spacseq_slides as smc

        orig_patch_loop = smc.niche_lfc_patch_holdout_loop

        def patch_loop_with_bins(niche_dom, adata_target, src_idx, pred_g_cache, *a, **kw):
            args = a[3] if len(a) > 3 else kw["args"]
            slide = str(getattr(args, "slide_id", ""))
            method = kw.get("scoring_method")
            if slide in seen_bin_slides and kw.get("pred_basal") is None:
                norm = smc.normalize_pred_cache(method, pred_g_cache) if method else pred_g_cache
                names = np.asarray(adata_target.var_names).astype(str)
                cols = np.array([i for i, g in enumerate(names) if g in gene_set], dtype=int)
                src = np.asarray(src_idx, dtype=int)
                cbin = adata_target.obs["bin_id"].to_numpy().astype(int)[src]
                lib = smc.lib_sizes_from_adata(adata_target, src)
                bdir = Path(args.outdir) / "patch" / f"slide_{slide}" / "pred_bin_sums"
                bdir.mkdir(parents=True, exist_ok=True)
                for g, p in norm.items():
                    if p is None or len(p) != len(src):
                        continue
                    pc = np.asarray(p, dtype=np.float64)[:, cols]
                    spaces = {"logcp10k": pc, "cp10k": np.expm1(np.clip(pc, 0.0, 14.5)),
                              "raw": np.asarray(sa.log_cp10k_to_counts(pc, lib), dtype=np.float64)}
                    out = {"prep_gene_idx": cols, "genes": names[cols], "src_idx": src, "src_bin": cbin,
                           "src_count_per_bin": np.bincount(cbin, minlength=36), "method": str(method),
                           "niche_dom": np.array(sorted(niche_dom.items()), dtype=object)}
                    for k, v in spaces.items():
                        s_ = np.zeros((36, v.shape[1]))
                        np.add.at(s_, cbin, v)
                        out[f"sum_{k}"] = s_.astype(np.float32)
                    np.savez_compressed(bdir / f"{g}.npz", **out)
            return orig_patch_loop(niche_dom, adata_target, src_idx, pred_g_cache, *a, **kw)

        smc.niche_lfc_patch_holdout_loop = patch_loop_with_bins

    sys.argv = [str(script)] + argv[1:]
    sys.path.insert(0, str(script.parent))
    print(f"[shared gene rule] running {script} {' '.join(argv[1:])}", flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
