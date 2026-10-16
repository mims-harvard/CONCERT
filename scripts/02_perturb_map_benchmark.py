#!/usr/bin/env python
"""Spatial Perturb-map held-out perturbation benchmark, from a trained checkpoint.

The script form of `notebooks/02_perturb_map_benchmark.ipynb`. The question and the two
metrics are defined in `scripts/01_spac_seq_benchmark.py`. What differs here is the assay -- four sections of a lung tumour model carrying a small, balanced
knockout panel, with unambiguous guide assignment and explicit control spots -- and, in
consequence, three settings of the protocol:

  * `use_action_correction=False`: there is no non-targeting-control correction to make.
  * `min_cells_per_pert=0`: the panel is balanced, so no rare-perturbation filter.
  * scoring is restricted to the assay panel (`in_panel_1053`), not to all HVGs.

The reported within-niche numbers use the disjoint reference: the measured LFC of a record is
the knockout cells of the niche against the control cells of the niche outside every held-out
patch (patch role 0; the slide's role-0 controls when the niche has fewer than 5). The held-out
source spots are themselves labelled control on this assay, so the harness reference (every
control cell of the niche) can contain the cells being predicted. The rescore applies the
disjoint reference to the stored predicted LFCs of every method; nothing is re-run (benchmark_evaluation/rescore_perturbmap.py, tables in data/benchmark_assets/results/perturbmap_results/).

The method comparison rule is also the opposite of the SPAC-seq one: here the methods overlap
enough that the reported comparison is the intersection of the records **all** methods cover
(27), not a per-method pairing.

Steps
  weights      per-perturbation footprint lengthscale and niche gain, read out of the weights
  predict      one counterfactual prediction by hand, with a scatter against the measurement
  section      score one section with the function the SLURM job calls   (5-15 min, GPU)
  aggregate    the four sections together, harness scores and the disjoint-reference rescore
               checked against the reported table
  crossslide   the cross-slide panel and its (few) records
  compare      CONCERT next to the published methods and the reference strategies,
               on the all-method intersection

Examples
  python scripts/02_perturb_map_benchmark.py --steps weights aggregate compare
  python scripts/02_perturb_map_benchmark.py --steps section --section 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import concert_paths as cp

SECTIONS = ["0", "1", "2", "3"]

MODEL_SRC = cp.MODELS / "concert_benchmark"
PATCH_DIR = cp.rp.PERTURBMAP_PATCHES                  # niche patches and the index bridge
PM_PREP = cp.rp.PERTURBMAP_PREPARED       # eval panel + prepared cell sets
HOLDOUT_COORDS = cp.rp.HOLDOUT_COORDS
CKPT = {
    "seen": cp.CHECKPOINT_ROOT / "perturbmap_within_niche" / "model.pt",
    "unseen": cp.CHECKPOINT_ROOT / "perturbmap_cross_slide" / "model.pt",
}
SHIPPED = {
    "seen": cp.rp.PERTURBMAP_CONCERT / "within_niche",
    "unseen": cp.rp.PERTURBMAP_CONCERT / "cross_slide",
}
NUMBERS = cp.rp.PERTURBMAP_RESULTS         # the reported tables (disjoint reference)


def pmap_args(regime, outdir, slides=None, n_boot=10, masked_unseen=False):
    """Evaluation configuration for one Perturb-map run (the reported settings).

    As in the SPAC-seq script, every flag below the "model graph" line has to match training;
    the lengthscale band in particular is a plain float attribute, not a saved weight.
    """
    return SimpleNamespace(
        model_file=CKPT[regime],
        outdir=Path(outdir),
        patch_dir=PATCH_DIR,
        slides=list(slides or SECTIONS),
        # scoring protocol
        hvg_max_rank=2000,
        min_cells_per_pert=0,                  # the panel is balanced; no rare filter
        min_g_cells=20 if masked_unseen else 5,
        niche_grid=6, loc_range=20.0,
        n_samples=10, batch_size=512,
        max_src_cells=2000,
        n_boot=n_boot, boot_frac=0.5, boot_seed=1234,
        setup_only=False, fpert_off=False,
        # cross-slide: which cells were masked, and on which section
        unseen_ko_file=(HOLDOUT_COORDS / "holdout_pmap_unseen.json") if masked_unseen else None,
        # model graph: MUST match training
        use_action_correction=False,           # guide assignment is unambiguous here
        inducing_point_steps=4,
        niche_rank=4,
        use_expr_encoder=True, expr_hidden_dims=[512, 256], use_basal_encoder_only=True,
        use_nngp=False, nngp_lambda=0.0,
        use_paracrine=False, paracrine_l1=0.0,
        use_niche_interaction=False,           # FiLM head instead; auto-detected
        use_pert_gate=True, gate_hidden=64,
        spatial_basal_only=True,
        basal_field="rff", basal_bayes=True,
        # perturbation footprint field: the reported settings
        fpert_field="gp_pp",
        fpert_rff_features=128, fpert_rff_bandwidth=0.15,
        fpert_coord_scale=70.0, fpert_hidden=256,
        fpert_svgp_inducing=64,                # self-corrected from the checkpoint anyway
        fpert_svgp_ls_init=0.04,
        fpert_mb_bandwidths="", f_pert_clamp=5.0,
        ls_prior_min=3.0, ls_prior_max=18.0,
        fpert_ls_min=0.01, fpert_ls_max=0.60,
    )


# ----------------------------------------------------------------------------- steps

def step_weights(env, out):
    """Footprint lengthscale and niche gain per perturbation. See script 01 for the formulas."""
    import numpy as np
    import pandas as pd
    import torch

    args, ctx = env["args"], env["ctx"]
    sd = torch.load(str(args.model_file), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    emb = sd["lord_encoder.s_encoder.perturbation.weight"].float()
    ls_w, ls_b = sd["fpert_field.ls_head.weight"].float(), sd["fpert_field.ls_head.bias"].float()
    gs_w, gs_b = sd["g_scale_head.weight"].float(), sd["g_scale_head.bias"].float()

    lo, hi, g_cap, cs = args.fpert_ls_min, args.fpert_ls_max, 3.0, args.fpert_coord_scale
    ls = (lo + (hi - lo) * torch.sigmoid(emb @ ls_w.T + ls_b)).squeeze(1).numpy()
    gain = torch.clamp(torch.nn.functional.softplus(emb @ gs_w.T + gs_b), max=g_cap).squeeze(1).numpy()

    name = ctx["pert_name"]
    tbl = pd.DataFrame({
        "code": np.arange(len(ls)),
        "perturbation": [name.get(i, f"code{i}") for i in range(len(ls))],
        "lengthscale_norm": ls,
        "lengthscale_section_units": ls * cs,
        "niche_gain": gain,
    }).iloc[1:]
    print(f"band: [{lo * cs:.1f}, {hi * cs:.1f}] section units "
          f"-- none of these should be sitting on either end")
    print(tbl.round(3).to_string(index=False))
    tbl.to_csv(out / "per_perturbation_kernel.csv", index=False)
    print(f"\nwritten {out / 'per_perturbation_kernel.csv'}")


def step_predict(env, out):
    """One counterfactual prediction on the held-out patch of one section.

    A section-level illustration, not the reported per-niche score. Note that it rebuilds its
    own context through `_model_adata`, so the checkpoint is loaded a second time.
    """
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args, ctx, sect = env["args"], env["ctx"], env["section"]
    ecf, pm, pmeval, eu = env["ecf"], env["pm"], env["pmeval"], env["eu"]

    adata_t, _ctx2, gene_names, y_obs = pmeval._model_adata(args, sect)
    groups = pm.seen_niche_groups(adata_t, src_role=1)
    src_idx = pm.source_indices(adata_t, src_role=1)
    panel = sorted({g for (_b, g, _r) in groups})
    print(f"{len(groups)} niche groups, panel = {panel}, {len(src_idx)} source spots")

    gene = panel[0]
    pred_g = np.asarray(ecf._predict_swap(ctx, src_idx, ctx["name2code"][gene], args), np.float32)
    pred_basal = np.asarray(ecf._predict_swap(ctx, src_idx, 0, args), np.float32)
    model_lfc = eu.pseudobulk_lfc(pred_g, pred_basal)

    canon = adata_t.obs["perturbation_canon"].astype(str).to_numpy()
    role = adata_t.obs["role"].to_numpy().astype(int)
    ko = np.where(canon == gene)[0]
    ref = np.where((canon == pm.CTRL_CANON) & (role == 0))[0]   # controls outside every held-out patch
    assert not np.isin(ref, src_idx).any()
    obs_lfc = eu.pseudobulk_lfc(y_obs[ko], y_obs[ref])
    print(f"measured: {len(ko)} {gene} spots vs {len(ref)} control spots outside the held-out patches")

    # restrict to the assay panel, as the scored comparison does
    with h5py.File(str(cp.rp.PERTURBMAP_RAW / f"perturbmap_s{sect}_spacfmt.h5ad"), "r") as f:
        vnames = f["var"]["_index"][:].astype(str)
        inpanel = f["var"]["in_panel_1053"][:].astype(bool)
    panel_set = set(vnames[inpanel])
    mask = np.array([g in panel_set for g in gene_names], dtype=bool)
    print(f"scoring {int(mask.sum())} of {len(gene_names)} genes (assay-panel overlap)")

    scores = eu.score_lfc_agreement(model_lfc[mask], obs_lfc[mask])
    print(f"\nsection-level illustration for {gene} (NOT the reported per-niche score):")
    for k, v in scores.items():
        print(f"  {k:<16} {v:.4f}")

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(obs_lfc[mask], model_lfc[mask], s=6, lw=0, alpha=0.4, color="#2166ac",
               rasterized=True)
    lim = np.nanpercentile(np.abs(np.r_[obs_lfc[mask], model_lfc[mask]]), 99.5)
    ax.plot([-lim, lim], [-lim, lim], lw=1, color="0.6", zorder=0)
    ax.axhline(0, lw=0.6, color="0.85", zorder=0)
    ax.axvline(0, lw=0.6, color="0.85", zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"measured log fold change ({gene})", fontsize=13)
    ax.set_ylabel("predicted log fold change", fontsize=13)
    ax.tick_params(labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = out / f"one_prediction_{gene}_section_{sect}.pdf"
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"\nwritten {p}")


def step_section(env, out):
    """Score one section with the shipped harness (5-15 min on one H100).

    The harness writes its own scores with the harness reference and the predicted LFC vectors
    (model_lfc.json); `aggregate --result-root fresh` rescores those with the disjoint reference.
    """
    import pandas as pd

    args, sect = env["args"], env["section"]
    env["pmeval"].run_seen(args, sect)
    p = args.outdir / "seen_pert" / f"slide_{sect}" / "lfc_agg.json"
    agg = json.loads(p.read_text())["L2"]
    print("read", p, "\n")
    df = pd.DataFrame([{"perturbation": g, "lfc_spearman": v["spearman_mean"],
                        "lfc_rmse": v["lfc_dist_mean"], "n_niches": v["n_niches"],
                        "n_scored_genes": v["n_scored_genes"]}
                       for g, v in agg.items()]).sort_values("lfc_spearman", ascending=False)
    print(df.round(4).to_string(index=False))
    df.to_csv(out / f"section_{sect}_per_perturbation.csv", index=False)


def concert_rescored_records(env, result_root):
    """Rescore CONCERT's stored predicted LFCs with the disjoint reference (rescore_perturbmap.model_rows)."""
    import anndata as ad
    import pandas as pd

    rpmap, rpm = env["rpmap"], env["rpm"]
    rpmap.MODEL_METHODS = ["concert"]
    rpm.METHODS["concert"] = (Path(result_root), "CONCERT")
    rows = []
    for s in SECTIONS:
        if not (Path(result_root) / "seen_pert" / f"slide_{s}" / "model_lfc.json").is_file():
            continue
        rows += rpmap.model_rows(ad.read_h5ad(rpm.PREP / f"slide_{s}.h5ad"), s)
    df = pd.DataFrame(rows)
    df = df[df.scored]
    df["record"] = df.gene + "|" + df.slide.astype(str) + "|" + df.niche_key.astype(str)
    return df


def step_aggregate(env, out):
    """The four sections together: the harness's own scores (harness reference), then the same
    predicted LFCs rescored with the disjoint reference and checked against the reported table."""
    import numpy as np
    import pandas as pd

    root = env["result_root"]("seen")
    records = []
    for s in SECTIONS:
        p = root / "seen_pert" / f"slide_{s}" / "lfc_scores.json"
        if not p.exists():
            print(f"  section {s}: no result at {p}")
            continue
        for gene, recs in json.loads(p.read_text())["L2"].items():
            for r in recs:
                records.append({"section": s, "perturbation": gene, "niche": r.get("niche"),
                                "lfc_spearman": r["spearman"], "lfc_rmse": r["lfc_dist"],
                                "n_scored_genes": r.get("n_scored_genes")})
    df = pd.DataFrame(records)
    if df.empty:
        print("no per-section results found -- run the `section` step first, "
              "or drop --result-root fresh")
        return
    print(f"read {root}")
    print(f"{len(df)} records over {df.section.nunique()} sections\n")
    print("WITHIN-NICHE, harness scores (harness reference)")
    print(f"  LFC Spearman : {df.lfc_spearman.mean():.4f}")
    print(f"  LFC RMSE     : {df.lfc_rmse.mean():.4f}")
    print("\n" + df.groupby("perturbation").agg(n_records=("niche", "size"),
                                                lfc_spearman=("lfc_spearman", "mean"),
                                                lfc_rmse=("lfc_rmse", "mean")).round(4).to_string())
    df.to_csv(out / "within_niche_per_record_harness.csv", index=False)

    mine = concert_rescored_records(env, root)
    reported = pd.read_csv(NUMBERS / "pm_seen_lfc_dist_per_record.csv", dtype={"slide": str})
    reported = reported[reported.method == "concert"]
    reported["record"] = reported.gene + "|" + reported.slide + "|" + reported.niche_key.astype(str)
    both = mine.merge(reported, on="record", suffixes=("", "_reported"))
    print(f"\nrescore: {len(mine)} CONCERT records, {len(both)} matched to the reported table, "
          f"max |d rho| {np.abs(both.lfc_spearman - both.lfc_spearman_reported).max():.2e}, "
          f"max |d RMSE| {np.abs(both.lfc_dist - both.lfc_dist_reported).max():.2e}")
    mine.to_csv(out / "within_niche_per_record.csv", index=False)


def step_crossslide(env, out):
    """The cross-slide panel. There are few records here -- too few to read a trend into."""
    import pandas as pd

    panel = json.loads((PM_PREP / "benchmark" / "eval_panel.json").read_text())
    units = pd.DataFrame([{"gene": u["gene"], "target_section": u["target_slide"],
                           "donor_sections": ",".join(map(str, u.get("donor_slides", [])))}
                          for u in panel["eval_units"]])
    print("cross-slide units (gene masked on target section, kept on donors):")
    print(units.to_string(index=False))
    units.to_csv(out / "cross_slide_panel.csv", index=False)

    root = env["result_root"]("unseen")
    records = []
    for s in SECTIONS:
        p = root / "unseen_pert" / f"slide_{s}" / "lfc_scores.json"
        if not p.exists():
            continue
        for gene, recs in json.loads(p.read_text())["L2"].items():
            for r in recs:
                records.append({"section": s, "perturbation": gene,
                                "lfc_spearman": r["spearman"], "lfc_rmse": r["lfc_dist"]})
    df = pd.DataFrame(records)
    if df.empty:
        print(f"\nno cross-slide results under {root}")
        return
    print("\nCROSS-SLIDE (code name: unseen)")
    print(f"  {len(df)} records on {df.section.nunique()} section(s) "
          "-- too few to read a trend into")
    print(f"  LFC Spearman : {df.lfc_spearman.mean():.4f}")
    print(f"  LFC RMSE     : {df.lfc_rmse.mean():.4f}")
    print(df.round(4).to_string(index=False))
    df.to_csv(out / "cross_slide_per_record.csv", index=False)


def step_compare(env, out):
    """CONCERT next to the published methods and the reference strategies, on the records every
    method covers, all scored with the disjoint reference.

    Unlike the SPAC-seq benchmark, the methods here cover enough of the same records that the
    all-method intersection is non-empty: the model methods cover 31 records, the
    reference strategies and STATE 27, and the 27-record intersection is the reported set.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    parts = ["pm_seen_lfc_dist_per_record.csv", "pm_seen_naive_baselines_our_rule.csv",
             "pm_seen_state_our_rule.csv"]
    df = pd.concat([pd.read_csv(NUMBERS / f, dtype={"slide": str}) for f in parts], ignore_index=True)
    df["record"] = df.gene.astype(str) + "|" + df.slide + "|" + df.niche_key.astype(str)

    # concert is the reported model; concert_niche_seen is an alternative checkpoint
    sub = df[df.method != "concert_niche_seen"]
    covered = sub.groupby("method").record.apply(set)
    shared = set.intersection(*covered.tolist())
    print(f"{len(shared)} of {sub.record.nunique()} records are covered by every method")
    print("per method, before restriction:", {m: len(s) for m, s in covered.items()})

    sub = sub[sub.record.isin(shared)]
    overall = (sub.groupby(["method", "method_label"])
                 .agg(n_records=("record", "size"), lfc_spearman=("lfc_spearman", "mean"),
                      lfc_rmse=("lfc_dist", "mean"))
                 .reset_index().sort_values("lfc_spearman", ascending=False))
    print("\n" + overall.round(4).to_string(index=False))
    overall.to_csv(out / "within_niche_all_method_intersection.csv", index=False)

    stats = pd.read_csv(NUMBERS / "perturbmap_seen_concert_vs_competitor_stats.csv")
    print("\npaired Wilcoxon, CONCERT vs each method (BH-adjusted):")
    print(stats[["metric", "competitor_label", "n_pairs", "concert_mean", "competitor_mean",
                 "p_adj"]].to_string(index=False))

    per_gene = (sub.groupby(["method_label", "gene"])
                  .agg(n=("record", "size"), lfc_spearman=("lfc_spearman", "mean"),
                       lfc_rmse=("lfc_dist", "mean")).reset_index())
    pivot = per_gene.pivot(index="method_label", columns="gene", values="lfc_spearman").round(3)
    print("\nLFC Spearman per knockout:\n" + pivot.to_string())
    per_gene.to_csv(out / "within_niche_per_gene.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    d = overall.sort_values("lfc_spearman")
    axes[0].barh(d.method_label, d.lfc_spearman,
                 color=["#b2182b" if m == "concert" else "#9ab8d4" for m in d.method])
    axes[0].axvline(0, lw=0.8, color="0.4")
    axes[0].set_xlabel("LFC Spearman", fontsize=13)
    d2 = overall.sort_values("lfc_rmse", ascending=False)
    axes[1].barh(d2.method_label, d2.lfc_rmse,
                 color=["#b2182b" if m == "concert" else "#9ab8d4" for m in d2.method])
    axes[1].set_xlabel("LFC RMSE (lower is better)", fontsize=13)
    for ax in axes:
        ax.tick_params(labelsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "benchmark_comparison.pdf", dpi=300)
    plt.close(fig)
    print(f"\nwritten {out / 'benchmark_comparison.pdf'}")


STEPS = {"weights": step_weights, "predict": step_predict, "section": step_section,
         "aggregate": step_aggregate, "crossslide": step_crossslide, "compare": step_compare}
NEEDS_MODEL = {"weights", "predict", "section"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=list(STEPS),
                    default=["weights", "aggregate", "compare"])
    ap.add_argument("--all", action="store_true", help="run every step")
    ap.add_argument("--section", default="0", choices=SECTIONS,
                    help="section for `predict` and `section`")
    ap.add_argument("--result-root", choices=["shipped", "fresh"], default="shipped")
    ap.add_argument("--outdir", default=None, help="default: $CONCERT_WORKDIR/02_perturb_map")
    a = ap.parse_args()
    steps = list(STEPS) if a.all else a.steps

    out = Path(a.outdir) if a.outdir else cp.workdir("02_perturb_map")
    out.mkdir(parents=True, exist_ok=True)

    cp.section("0. Inputs")
    cp.check([
        ("model source", MODEL_SRC / "concert_spac.py"),
        ("scoring code", cp.EVAL_SRC / "run_concert_pm_niche.py"),
        ("section s0", cp.rp.PERTURBMAP_RAW / "perturbmap_s0_spacfmt.h5ad"),
        ("niche patches", PATCH_DIR / "slide0.npz"),
        ("eval panel", PM_PREP / "benchmark" / "eval_panel.json"),
        ("masked-cell coords", HOLDOUT_COORDS / "holdout_pmap_unseen.json"),
        ("seen checkpoint", CKPT["seen"]),
        ("seen vocabulary", CKPT["seen"].parent / "pert_map.json"),
        ("unseen checkpoint", CKPT["unseen"]),
        ("shipped seen result", SHIPPED["seen"] / "seen_pert" / "slide_0" / "lfc_scores.json"),
        ("reported table", NUMBERS / "pm_seen_lfc_dist_per_record.csv"),
    ])
    print(f"\noutput dir: {out}")

    cp.prepend_src(MODEL_SRC, cp.EVAL_SRC)
    import eval_spatial_counterfactual as ecf
    import spacseq_eval_common as sec
    import perturbmap_common as pm
    import run_concert_pm_niche as pmeval
    import perturbmap_reference as pmref
    import perturbmap_harness_rescore as rpm
    import rescore_perturbmap as rpmap
    eu, _sa = sec.import_eval_utils()
    for m in (ecf, sec, pm, pmeval, pmref, rpm, rpmap):
        assert Path(m.__file__).resolve().is_relative_to(cp.REPO_ROOT.resolve()), \
            f"{m.__name__} loaded from outside the release: {m.__file__}"
    print(f"\nmodel source      : {ecf.__file__}")
    print(f"scoring entrypoint: {pmeval.__file__}")
    cp.describe_device()

    args = pmap_args("seen", outdir=out / "seen_eval", slides=[a.section])
    env = {"args": args, "section": a.section, "ecf": ecf, "pm": pm, "pmeval": pmeval, "eu": eu,
           "rpmap": rpmap, "rpm": rpm,
           "result_root": lambda r: (SHIPPED[r] if a.result_root == "shipped"
                                     else out / f"{r}_eval")}

    if set(steps) & NEEDS_MODEL:
        cp.section(f"Loading the checkpoint on section s{a.section}")
        print(f"checkpoint  {args.model_file}")
        print(f"band        [{args.fpert_ls_min}, {args.fpert_ls_max}] normalized "
              f"= [{args.fpert_ls_min * args.fpert_coord_scale:.1f}, "
              f"{args.fpert_ls_max * args.fpert_coord_scale:.1f}] section units "
              f"(a section is {args.loc_range:.0f} units wide)")
        env["ctx"] = pmeval.build_pm_ctx(args, a.section)
        m = env["ctx"]["model"]
        print(f"\nsection s{a.section}: {env['ctx']['N']:,} spots x "
              f"{len(env['ctx']['gene_names'])} genes")
        print(f"perturbation vocabulary: {env['ctx']['n_batch']} codes (code 0 = basal)")
        print(f"parameters: {sum(p.numel() for p in m.parameters()):,}")
        print("perturbations:", sorted(k for k in env["ctx"]["name2code"]
                                       if env["ctx"]["name2code"][k] != 0))

    for name in STEPS:
        if name in steps:
            cp.section(f"Step: {name}")
            STEPS[name](env, out)

    print(f"\nDone. Everything written under {out}")


if __name__ == "__main__":
    main()
