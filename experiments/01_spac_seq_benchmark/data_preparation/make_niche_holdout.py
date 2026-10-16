#!/usr/bin/env python
"""Generate targeted unassigned holdout indices for the multi-slide unseen eval.

Rule (mirrors eval_cf_una panel selection):
  1) Per slide, pick the top-KO panel (genes with >= panel_min_cells KO cells on
     that slide, top max_panel by count) and each gene's MOST ABUNDANT niche.
  2) Hold out the UNASSIGNED cells in that niche (source cells only -- never KO/NTC).
  3) Union over genes & slides -> indices into the post-exclusion concatenated
     training array (exactly what run_concert_spac.py --holdout_unassigned_file
     consumes: it applies the excluded-label filter, then removes these rows).

The coordinate frame replicates training (global MinMaxScaler * loc_range), so the
niche grid here matches what the eval sees after it reproduces the same scaling.
"""
import argparse, logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import eval_utils as eu
from run_concert_spac import load_h5ad_multi
from make_umap import UNASSIGNED_LABELS, CONTROL_LABELS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def classify(perturb_raw):
    out = np.empty(len(perturb_raw), dtype=object)
    for i, p in enumerate(perturb_raw):
        if p in UNASSIGNED_LABELS:
            out[i] = "unassigned"
        elif p in CONTROL_LABELS:
            out[i] = "control"
        else:
            out[i] = "gene_KO"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi_sample_files", required=True, help="comma-separated .h5ad paths")
    ap.add_argument("--out", required=True, help="output .npy of holdout indices")
    ap.add_argument("--hvg_max_rank", type=int, default=2000)
    ap.add_argument("--excluded_label", type=str, default="multiple,Multiple")
    ap.add_argument("--loc_range", type=float, default=20.0)
    ap.add_argument("--niche_grid", type=int, default=6)
    ap.add_argument("--panel_min_cells", type=int, default=200)
    ap.add_argument("--max_panel", type=int, default=30)
    ap.add_argument("--min_g_cells", type=int, default=20)
    ap.add_argument("--una_floor", type=int, default=50)
    ap.add_argument("--max_per_niche", type=int, default=100,
                    help="cap held-out unassigned per prediction niche (match eval max_src_per_niche); "
                         "hold out only the source cells the eval actually predicts on, not the whole niche")
    args = ap.parse_args()

    paths = [p.strip() for p in args.multi_sample_files.split(",") if p.strip()]
    log.info("Loading %d slides ...", len(paths))
    X, pos_raw, _, perturb_raw, has_guide, is_real_KO, gene_names, sample_id = load_h5ad_multi(
        paths, unassigned_labels=UNASSIGNED_LABELS, control_labels=CONTROL_LABELS,
        hvg_max_rank=args.hvg_max_rank)

    # Excluded-label filter -- IDENTICAL to run_concert_spac so indices align.
    excl = frozenset(l.strip() for l in args.excluded_label.split(",") if l.strip())
    if excl:
        keep = np.array([p not in excl for p in perturb_raw], dtype=bool)
        log.info("Excluded %d cells with labels %s", int((~keep).sum()), sorted(excl))
        X, pos_raw, perturb_raw, has_guide, is_real_KO, sample_id = (
            X[keep], pos_raw[keep], perturb_raw[keep], has_guide[keep], is_real_KO[keep], sample_id[keep])

    # Training coordinate frame: global MinMax -> [0, loc_range].
    pos_scaled = MinMaxScaler().fit_transform(pos_raw) * args.loc_range
    cls = classify(perturb_raw)

    holdout = []
    for s in np.unique(sample_id):
        s_idx = np.where(sample_id == s)[0]                 # global indices for this slide
        coords = pos_scaled[s_idx]
        cls_s = cls[s_idx]
        pert_s = perturb_raw[s_idx]
        bin_s, n_bins = eu.build_niche_grid(coords, args.niche_grid, args.loc_range)
        ko_local = np.where(cls_s == "gene_KO")[0]
        una_local = np.where(cls_s == "unassigned")[0]
        # panel: genes with >= panel_min_cells KO on this slide, top max_panel by count
        genes, counts = np.unique(pert_s[ko_local], return_counts=True)
        order = np.argsort(-counts)
        panel = [g for g, c in zip(genes[order], counts[order]) if c >= args.panel_min_cells][:args.max_panel]
        una_bin = bin_s[una_local]
        una_per_bin = np.bincount(una_bin, minlength=n_bins)
        n_slide = 0
        for g in panel:
            g_local = ko_local[pert_s[ko_local] == g]
            ko_counts = np.bincount(bin_s[g_local], minlength=n_bins)
            b = -1
            for cand in np.argsort(-ko_counts):
                if ko_counts[cand] < args.min_g_cells:
                    break
                if una_per_bin[cand] >= args.una_floor:
                    b = int(cand); break
            if b < 0:
                continue
            sel = una_local[una_bin == b]                   # unassigned cells in niche b
            # Cap: hold out only the source cells the eval predicts on (<= max_per_niche),
            # deterministic (seed 42+b, matching eval's subsample convention).
            if args.max_per_niche > 0 and len(sel) > args.max_per_niche:
                rng = np.random.default_rng(42 + int(b))
                sel = np.sort(rng.choice(sel, args.max_per_niche, replace=False))
            holdout.append(s_idx[sel])                      # -> global indices
            n_slide += len(sel)
        log.info("slide %s: panel=%d genes, held out %d unassigned (of %d) across niches",
                 s, len(panel), n_slide, len(una_local))

    holdout = np.unique(np.concatenate(holdout)) if holdout else np.array([], dtype=int)
    # sanity: every held-out cell must be genuine unassigned
    assert np.all(~has_guide[holdout]), "BUG: a held-out cell has a guide (KO/NTC)!"
    np.save(args.out, holdout)
    log.info("TOTAL held out %d unassigned cells -> %s", len(holdout), args.out)


if __name__ == "__main__":
    main()
