#!/usr/bin/env python
"""SPAC-seq held-out perturbation benchmark, reproduced from a trained checkpoint.

The script form of `notebooks/01_spac_seq_benchmark.ipynb`. Nothing is retrained: every
step loads `model.pt` and runs inference.

Five sections (1-1 ... 1-5) carry a pooled CRISPR panel. A perturbation is held out and the
model, given where a cell sits and what it looked like unperturbed, predicts the perturbed
expression. The prediction is compared with the cells that really carry that perturbation,
through the log fold change over matched unperturbed cells.

Two settings, named `seen` / `unseen` on disk and *within-niche prediction* /
*cross-slide prediction* in the manuscript:

  within-niche  the perturbation is removed from a patch of one slide but is still visible
                elsewhere -- does its effect transfer to a new tissue context?
  cross-slide   the perturbation is removed from every slide -- can its effect be inferred
                from related perturbations?

Two metrics: LFC Spearman (do the right genes move, higher better) and LFC RMSE, called
`lfc_dist` in the tables (is the size of the response right, lower better). E-distance still
appears in some JSON files and is retired; it is not a reported metric.

Steps
  weights      per-perturbation footprint lengthscale and niche gain, read out of the weights
  predict      one counterfactual prediction by hand, with a scatter against the measurement
  slide        score one slide with the function the SLURM array job calls  (15-40 min, GPU)
  aggregate    the five slides together -- the reported within-niche number
  crossslide   the cross-slide panel, one unit run live plus the reported number
  compare      CONCERT next to the published methods, paired per competitor

Examples
  python scripts/01_spac_seq_benchmark.py --steps weights aggregate compare
  python scripts/01_spac_seq_benchmark.py --steps slide --slide 1-1
  python scripts/01_spac_seq_benchmark.py --all --result-root fresh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import concert_paths as cp

SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]

MODEL_SRC = cp.MODELS / "concert_benchmark"
SPAC_RAW = cp.DATA_ROOT / "newdata" / "spac_seq"
PREPARED = cp.SCORING_LIB / "multi_1-5" / "prepared"
CKPT = {
    "seen": cp.OUTPUT_ROOT / "ms_spac1_niche_seen_lsfix" / "model.pt",
    "unseen": cp.OUTPUT_ROOT / "ms_spac1_niche_unseen_lsfix" / "model.pt",
}
SHIPPED = {
    "seen": cp.SCORING_LIB / "multi_1-5_ho_v3" / "concert" / "seen_lsfix",
    "unseen": cp.SCORING_LIB / "multi_1-5_ho_v3" / "concert" / "unseen_lsfix",
}
NUMBERS = cp.PAPER_NUMBERS / "benchmark_spac_seq"


def spac_args(regime, outdir, target_slide=None, n_boot=10, force=True):
    """Evaluation configuration for one SPAC-seq run.

    Mirrors the CLI defaults of benchmark_evaluation/run_concert_v3_benchmark_eval.py with the
    values experiments/01_spac_seq_benchmark/train/train_spac_seq.sh trained with. Every flag
    below the "model graph" line changes the shape of the graph and must match training.

    Two of them fail silently rather than loudly when they are wrong:
      fpert_ls_min / fpert_ls_max  the footprint lengthscale band. It is a plain float
        attribute of the field, not a saved weight, so identical weights render a different
        spatial field if it is left at the default.
      fpert_svgp_inducing  the inducing-grid size. This one is a saved tensor, so a mismatch
        would fail the load; build_concert_ctx reads the true count out of the checkpoint.
    """
    return SimpleNamespace(
        # --- what to score ---------------------------------------------------------------
        regime=regime,
        target_slide=target_slide,
        prepared_dir=PREPARED,
        benchmark_v3_dir=None,                 # defaults to prepared_dir/benchmark_v3
        model_file=CKPT[regime],
        outdir=Path(outdir),
        force=force,
        # --- scoring protocol ------------------------------------------------------------
        n_hvgs=2000,                           # genes the metrics are computed over
        min_g_cells=20,                        # a record needs this many measured KO cells
        max_src_cells=500,                     # cross-slide: source cells per unit
        max_src_cells_per_niche=500,           # within-niche: source cells per niche
        patch_role=1,                          # 1 = the held-out patch
        n_samples=10,                          # decoder draws averaged per prediction
        batch_size=512,
        n_boot=n_boot, boot_frac=0.5, boot_seed=1234,
        # --- model graph: MUST match training --------------------------------------------
        hvg_max_rank=2000,
        min_cells_per_pert=25,
        use_expr_encoder=True, expr_hidden_dims=[512, 256], use_basal_encoder_only=True,
        use_nngp=False, nngp_lambda=0.0,
        use_paracrine=False, paracrine_l1=0.0,
        use_niche_interaction=False,           # replaced by the FiLM head, auto-detected
        niche_rank=8,
        use_pert_gate=True, gate_hidden=64,
        use_action_correction=True,            # SPAC-seq has non-targeting controls
        inducing_point_steps=6,
        spatial_basal_only=True,
        basal_field="rff", basal_bayes=True,
        # --- perturbation footprint field -------------------------------------------------
        fpert_field="gp_pp",
        fpert_rff_features=128, fpert_rff_bandwidth=0.15,
        fpert_coord_scale=20.0, fpert_hidden=256,
        fpert_svgp_inducing=16,
        fpert_svgp_ls_init=0.25,
        fpert_mb_bandwidths="", f_pert_clamp=5.0,
        ls_prior_min=3.0, ls_prior_max=18.0,   # legacy band, in loc units
        fpert_ls_min=0.15, fpert_ls_max=0.90,  # THE band that is used, in normalized units
    )


# ----------------------------------------------------------------------------- steps

def step_weights(env, out):
    """Footprint lengthscale and niche gain per perturbation -- arithmetic on the state dict.

    Both come from a small head on the 256-dimensional perturbation embedding:
        ls(p)      = ls_min + (ls_max - ls_min) * sigmoid(ls_head(e_p))
        g_scale(p) = min(softplus(g_scale_head(e_p)), g_cap)
    A lengthscale that sits inside the band rather than pinned against an end is what allows
    it to be called learned rather than bounded.
    """
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
    }).iloc[1:]                                # code 0 is the basal slot, not a perturbation

    pinned = int(((tbl.lengthscale_norm - lo).abs() < 1e-3).sum()
                 + ((tbl.lengthscale_norm - hi).abs() < 1e-3).sum())
    print(f"band in normalized units : [{lo}, {hi}]")
    print(f"learned lengthscales     : {tbl.lengthscale_norm.min():.3f} - {tbl.lengthscale_norm.max():.3f}")
    print(f"pinned at either end     : {pinned} of {len(tbl)}")
    print(tbl.sort_values("lengthscale_norm").head(10).round(4).to_string(index=False))
    tbl.to_csv(out / "per_perturbation_kernel.csv", index=False)
    print(f"\nwritten {out / 'per_perturbation_kernel.csv'}")


def step_predict(env, out):
    """One counterfactual prediction, by hand, on the held-out patch.

    The decoder is
        log mu = f_basal(x, expression) + 1[has_guide] * mu_action + 1[is_real_KO] * f_pert(g, x)
    so a decode to perturbation g and a decode to the basal slot (code 0) differ by exactly the
    perturbation terms -- the basal reconstruction cancels. Every score is computed on that
    difference rather than on one decode.

    This is an illustration, not the reported protocol: it pools the whole held-out patch,
    where the reported score is per niche. `step_slide` runs the real thing.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args, ctx, slide = env["args"], env["ctx"], env["slide"]
    ecf, smc, bench, eu = env["ecf"], env["smc"], env["bench"], env["eu"]

    adata_t = smc.load_prepared_slide(args.prepared_dir, slide)
    role = smc.load_patch_roles(args.prepared_dir, slide, adata_t)
    niche_dom = smc.patch_dominant_perts(adata_t, role, args.min_g_cells, src_role=args.patch_role)
    src_idx = smc.patch_source_indices(adata_t, role, niche_dom, args.max_src_cells_per_niche,
                                       seed=42, src_role=args.patch_role)
    _kept, model_rows = bench.map_prep_to_model_rows(slide, np.asarray(adata_t.obs_names), src_idx, ctx)
    print(f"{len(niche_dom)} evaluable niches, {len(model_rows)} source cells mapped to model rows")

    gene = sorted(set(niche_dom.values()))[0]
    code_g = ctx["name2code"][gene]
    print(f"predicting {gene} (code {code_g}) on {len(model_rows)} source cells")

    pred_g = ecf._predict_swap(ctx, model_rows, code_g, args)
    pred_basal = ecf._predict_swap(ctx, model_rows, 0, args)
    model_lfc = eu.pseudobulk_lfc(pred_g, pred_basal)

    ko = np.where(ctx["perturb_raw"] == gene)[0]
    ref = np.where(ctx["cls"] == "unassigned")[0]
    obs_lfc = eu.pseudobulk_lfc(ctx["Y_obs"][ko], ctx["Y_obs"][ref])
    print(f"measured: {len(ko)} {gene} cells vs {len(ref)} unassigned cells")

    scores = eu.score_lfc_agreement(model_lfc, obs_lfc)
    print("\nslide-level illustration (NOT the reported per-niche score):")
    for k, v in scores.items():
        print(f"  {k:<16} {v:.4f}")

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(obs_lfc, model_lfc, s=5, lw=0, alpha=0.35, color="#2166ac", rasterized=True)
    lim = np.nanpercentile(np.abs(np.r_[obs_lfc, model_lfc]), 99.5)
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
    p = out / f"one_prediction_{gene}_slide_{slide}.pdf"
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"\nwritten {p}")


def step_slide(env, out):
    """Score one slide with the shipped harness -- the slow step (15-40 min on one H100).

    eval_seen_slide is exactly what experiments/01_spac_seq_benchmark/evaluation/
    eval_spac_seq_benchmark.sh runs, one slide per array task: it finds the niches of the
    held-out patch and the perturbation dominating each, decodes every source cell to every
    perturbation in the panel plus the basal slot, scores each (niche, perturbation) record
    against the measured cells, and bootstraps the source cells within each niche.
    """
    import pandas as pd

    args, slide, bench = env["args"], env["slide"], env["bench"]
    bench.eval_seen_slide(args, slide)
    agg_path = args.outdir / "patch" / f"slide_{slide}" / "lfc_agg.json"
    agg = json.loads(agg_path.read_text())["L2"]
    print("read", agg_path, "\n")

    per_pert = pd.DataFrame([
        {"perturbation": g, "lfc_spearman": v["spearman_mean"], "lfc_rmse": v["lfc_dist_mean"],
         "n_niches": v["n_niches"], "n_scored_genes": v["n_scored_genes"]}
        for g, v in agg.items()]).sort_values("lfc_spearman", ascending=False)
    print(f"slide {slide}: {len(per_pert)} perturbations, "
          f"mean LFC Spearman {per_pert.lfc_spearman.mean():.4f}, "
          f"mean LFC RMSE {per_pert.lfc_rmse.mean():.4f}")
    print(per_pert.round(4).to_string(index=False))
    per_pert.to_csv(out / f"slide_{slide}_per_perturbation.csv", index=False)


def step_aggregate(env, out):
    """The five slides together -- the reported within-niche number.

    A record is one (perturbation, slide, niche) triple; the reported figure is the mean over
    records. Reads the shipped results by default, so it reproduces the published number
    exactly; --result-root fresh reads what `slide` wrote instead.
    """
    import numpy as np
    import pandas as pd

    root = env["result_root"]("seen")
    records = []
    for s in SLIDES:
        p = root / "patch" / f"slide_{s}" / "lfc_agg.json"
        if not p.exists():
            print(f"  slide {s}: no result at {p}")
            continue
        for gene, v in json.loads(p.read_text())["L2"].items():
            records.append({"slide": s, "perturbation": gene,
                            "lfc_spearman": v["spearman_mean"],
                            "lfc_rmse": v["lfc_dist_mean"],
                            "n_niches": v["n_niches"]})
    df = pd.DataFrame(records)
    if df.empty:
        print("no per-slide results found -- run the `slide` step first, or drop --result-root fresh")
        return

    n = len(df)
    print(f"read {root}")
    print(f"{n} perturbation x slide records over {df.slide.nunique()} slides\n")
    print("WITHIN-NICHE (code name: seen)")
    print(f"  LFC Spearman : {df.lfc_spearman.mean():.4f}  "
          f"(s.e.m. {df.lfc_spearman.std(ddof=1) / np.sqrt(n):.4f})")
    print(f"  LFC RMSE     : {df.lfc_rmse.mean():.4f}  "
          f"(s.e.m. {df.lfc_rmse.std(ddof=1) / np.sqrt(n):.4f})")
    by_slide = df.groupby("slide").agg(n_records=("perturbation", "size"),
                                       lfc_spearman=("lfc_spearman", "mean"),
                                       lfc_rmse=("lfc_rmse", "mean")).round(4)
    print("\n" + by_slide.to_string())
    df.to_csv(out / "within_niche_per_record.csv", index=False)


def step_crossslide(env, out):
    """The cross-slide setting: the panel, one unit run live, then the reported number.

    A perturbation is removed from every slide, so the evaluation walks a panel of
    (gene, target slide) units listed in benchmark_v3/eval_panel.json rather than the niches
    of a held-out patch. Each unit names the donor slides where the gene was still visible.
    """
    import pandas as pd

    panel = json.loads((PREPARED / "benchmark_v3" / "eval_panel.json").read_text())
    units = pd.DataFrame([{"gene": u["gene"], "target_slide": u["target_slide"],
                           "donor_slides": ",".join(map(str, u.get("donor_slides", [])))}
                          for u in panel["eval_units"]])
    print(f"{len(units)} cross-slide evaluation units")
    print(units.to_string(index=False))
    units.to_csv(out / "cross_slide_panel.csv", index=False)

    if env["run_unit"]:
        u_args = spac_args("unseen", outdir=out / "unseen_eval_demo")
        one = dict(panel)
        one["eval_units"] = panel["eval_units"][:1]
        demo = out / "demo_panel" / "benchmark_v3"
        demo.mkdir(parents=True, exist_ok=True)
        (demo / "eval_panel.json").write_text(json.dumps(one))
        # the holdout coordinates live beside the real panel; symlink the rest of the directory
        for f in (PREPARED / "benchmark_v3").glob("*"):
            tgt = demo / f.name
            if not tgt.exists():
                tgt.symlink_to(f)
        u_args.benchmark_v3_dir = demo
        env["bench"].eval_unseen_all(u_args)
        print("\ndemo unit written to", u_args.outdir)

    root = env["result_root"]("unseen")
    rows = []
    for d in sorted((root / "unseen_pert").glob("*__slide_*")):
        p = d / "lfc_scores.json"
        if not p.exists():
            continue
        gene, slide = d.name.split("__slide_")
        # lfc_scores.json is {"L2": {gene: [record]}}; a cross-slide unit has exactly one
        # record (niche = -1, the whole slide).
        rec = json.loads(p.read_text())["L2"][gene][0]
        rows.append({"gene": gene, "slide": slide, "n_src_cells": rec.get("n_src"),
                     "n_measured_ko": rec.get("n_real_g"),
                     "lfc_spearman": rec.get("spearman"), "lfc_rmse": rec.get("lfc_dist")})
    df = pd.DataFrame(rows).dropna(subset=["lfc_spearman"]) if rows else pd.DataFrame()
    if df.empty:
        print(f"\nno cross-slide results under {root}")
        return
    print("\nCROSS-SLIDE (code name: unseen)")
    print(f"  {len(df)} units")
    print(f"  LFC Spearman : {df.lfc_spearman.mean():.4f}")
    print(f"  LFC RMSE     : {df.lfc_rmse.mean():.4f}")
    print(df.sort_values("lfc_spearman", ascending=False).round(4).to_string(index=False))
    df.to_csv(out / "cross_slide_per_unit.csv", index=False)


def step_compare(env, out):
    """CONCERT next to the published methods, paired one competitor at a time.

    The methods cover very different subsets of the 129 within-niche records -- Biolord, scGEN
    and Steamboat all of them, GEARS 80, CelComen 15, CellAgentChat 11 -- so intersecting all
    of them at once leaves nothing to compare. The reported comparison instead takes, per
    competitor, the records that competitor and CONCERT both cover. Comparing unpaired means
    would instead reward a method for skipping the hard records.

    (The Perturb-map benchmark is the other way round -- there the methods overlap enough that
    an all-method intersection is the rule. See scripts/02_perturb_map_benchmark.py.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def paired(csv_path, concert_key):
        df = pd.read_csv(csv_path)
        df["record"] = df.gene.astype(str) + "|" + df.slide.astype(str) + "|" + df.niche_key.astype(str)
        con = df[df.method == concert_key].set_index("record")
        rows = []
        for _m, sub in df[df.method != concert_key].groupby("method"):
            sub = sub.set_index("record")
            shared = con.index.intersection(sub.index)
            if len(shared) == 0:
                continue
            rows.append({"competitor": sub.method_label.iloc[0], "n_pairs": len(shared),
                         "concert_rho": con.loc[shared, "lfc_spearman"].mean(),
                         "competitor_rho": sub.loc[shared, "lfc_spearman"].mean(),
                         "concert_rmse": con.loc[shared, "lfc_dist"].mean(),
                         "competitor_rmse": sub.loc[shared, "lfc_dist"].mean()})
        o = pd.DataFrame(rows)
        o["rho_advantage"] = o.concert_rho - o.competitor_rho
        o["rmse_advantage"] = o.competitor_rmse - o.concert_rmse   # positive = CONCERT lower
        return o.sort_values("n_pairs", ascending=False), len(con)

    seen_cmp, n_seen = paired(NUMBERS / "spac_seen_lfc_dist_per_record.csv", "concert_v3_seen")
    print(f"WITHIN-NICHE: CONCERT covers {n_seen} records; paired against each competitor:")
    print(seen_cmp.round(4).to_string(index=False))
    seen_cmp.to_csv(out / "within_niche_paired_comparison.csv", index=False)

    unseen_cmp, n_unseen = paired(NUMBERS / "spac_unseen_lfc_dist_per_record.csv", "concert_v3_unseen")
    print(f"\nCROSS-SLIDE: CONCERT covers {n_unseen} records; paired against each competitor:")
    print(unseen_cmp.round(4).to_string(index=False))
    unseen_cmp.to_csv(out / "cross_slide_paired_comparison.csv", index=False)

    # the shipped statistics table carries the same n_pairs -- matching it is the check that
    # the pairing above is the one the paper used
    stats = pd.read_csv(NUMBERS / "spacseq_seen_concert_vs_competitor_lfc_rho_stats_summary.csv")
    chk = (seen_cmp[["competitor", "n_pairs"]]
           .merge(stats[["method", "n_pairs"]].rename(columns={"method": "competitor",
                                                               "n_pairs": "n_pairs_shipped"}),
                  on="competitor", how="left"))
    chk["matches"] = chk.n_pairs == chk.n_pairs_shipped
    print("\n" + chk.to_string(index=False))
    print("all counts match:", bool(chk.matches.all()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (cmp_df, name) in zip(axes, [(seen_cmp, "within-niche"), (unseen_cmp, "cross-slide")]):
        d = cmp_df.sort_values("concert_rho")
        y = np.arange(len(d))
        ax.barh(y - 0.2, d.concert_rho, 0.4, color="#b2182b", label="CONCERT")
        ax.barh(y + 0.2, d.competitor_rho, 0.4, color="#9ab8d4", label="competitor")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(d.competitor, d.n_pairs)])
        ax.axvline(0, lw=0.8, color="0.4")
        ax.set_xlabel(f"LFC Spearman on paired records ({name})", fontsize=13)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "benchmark_comparison.pdf", dpi=300)
    plt.close(fig)
    print(f"\nwritten {out / 'benchmark_comparison.pdf'}")


STEPS = {"weights": step_weights, "predict": step_predict, "slide": step_slide,
         "aggregate": step_aggregate, "crossslide": step_crossslide, "compare": step_compare}
NEEDS_MODEL = {"weights", "predict", "slide"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=["weights", "aggregate", "compare"],
                    help="which steps to run (default: the ones that need no GPU inference)")
    ap.add_argument("--all", action="store_true", help="run every step")
    ap.add_argument("--slide", default="1-1", choices=SLIDES, help="slide for `predict` and `slide`")
    ap.add_argument("--result-root", choices=["shipped", "fresh"], default="shipped",
                    help="`aggregate` and `crossslide` read the results shipped with the paper, "
                         "or the ones this script wrote")
    ap.add_argument("--run-unit", action="store_true",
                    help="`crossslide`: also run one evaluation unit live (a few minutes, GPU)")
    ap.add_argument("--outdir", default=None, help="default: $CONCERT_WORKDIR/01_spac_seq")
    args_cli = ap.parse_args()
    steps = list(STEPS) if args_cli.all else args_cli.steps

    out = Path(args_cli.outdir) if args_cli.outdir else cp.workdir("01_spac_seq")
    out.mkdir(parents=True, exist_ok=True)

    cp.section("0. Inputs")
    cp.check([
        ("model source", MODEL_SRC / "concert_spac.py"),
        ("scoring code", cp.EVAL_SRC / "run_concert_v3_benchmark_eval.py"),
        ("metric helper lib", cp.HELPER_LIB / "spac_edist_align.py"),
        ("raw section 1-1", SPAC_RAW / "raw_output_1-1.h5ad"),
        ("prepared cell sets", PREPARED / "slide_1-1.h5ad"),
        ("eval panel", PREPARED / "benchmark_v3" / "eval_panel.json"),
        ("seen checkpoint", CKPT["seen"]),
        ("seen vocabulary", CKPT["seen"].parent / "pert_map.json"),
        ("unseen checkpoint", CKPT["unseen"]),
        ("unseen vocabulary", CKPT["unseen"].parent / "pert_map.json"),
        ("shipped seen result", SHIPPED["seen"] / "patch" / "slide_1-1" / "lfc_agg.json"),
        ("paper number table", NUMBERS / "spac_seen_lfc_dist_per_record.csv"),
    ])
    print(f"\noutput dir: {out}")

    # The released code has to win over the working trees; see concert_paths.prepend_src.
    cp.prepend_src(MODEL_SRC, cp.EVAL_SRC)
    import eval_spatial_counterfactual as ecf     # binds the released model copy first
    import spacseq_eval_common as sec
    import spacseq_multi_common as smc
    import run_concert_v3_benchmark_eval as bench
    eu, _sa = sec.import_eval_utils()
    print(f"\nmodel source      : {ecf.__file__}")
    print(f"scoring entrypoint: {bench.__file__}")
    print(f"metric helpers    : {eu.__file__}")
    cp.describe_device()

    args = spac_args("seen", outdir=out / "seen_eval", target_slide=args_cli.slide)
    env = {"args": args, "slide": args_cli.slide, "ecf": ecf, "smc": smc, "bench": bench,
           "eu": eu, "run_unit": args_cli.run_unit,
           "result_root": lambda r: (SHIPPED[r] if args_cli.result_root == "shipped"
                                     else out / f"{r}_eval")}

    if set(steps) & NEEDS_MODEL:
        cp.section(f"Loading the checkpoint on slide {args_cli.slide}")
        print(f"checkpoint  {args.model_file}")
        print(f"band        [{args.fpert_ls_min}, {args.fpert_ls_max}] normalized "
              f"= [{args.fpert_ls_min * args.fpert_coord_scale:.0f}, "
              f"{args.fpert_ls_max * args.fpert_coord_scale:.0f}] section units")
        env["ctx"] = bench.build_concert_ctx(args, args_cli.slide)
        m = env["ctx"]["model"]
        print(f"\nslide {args_cli.slide}: {env['ctx']['N']:,} cells x "
              f"{len(env['ctx']['gene_names'])} genes")
        print(f"perturbation vocabulary: {env['ctx']['n_batch']} codes (code 0 = basal)")
        print(f"parameters: {sum(p.numel() for p in m.parameters()):,}")

    for name in STEPS:
        if name in steps:
            cp.section(f"Step: {name}")
            STEPS[name](env, out)

    print(f"\nDone. Everything written under {out}")


if __name__ == "__main__":
    main()
