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
  cross-slide   the perturbation is removed from one target slide (85 knockout-target-slide
                pairs) and its effect is carried over from the donor slides

Scoring rule, identical for every method: each source spot's predicted profile
is turned into counts by one Poisson draw at the spot's measured library size, the predicted
LFC is taken against the measured source spots, and the scored genes are the slide's highly
variable genes inside the 2,000-gene CONCERT panel, minus the knocked-out gene.

Two metrics: LFC Spearman (do the right genes move, higher better) and LFC RMSE, called
`lfc_dist` in the tables (is the size of the response right, lower better). E-distance still
appears in some JSON files and is retired; it is not a reported metric.

Steps
  weights      per-perturbation footprint lengthscale and niche gain, read out of the weights
  predict      one counterfactual prediction by hand, with a scatter against the measurement
  slide        score one slide with the function the SLURM array job calls, checked against
               the shipped result  (15-40 min, GPU)
  aggregate    the five slides together -- the reported within-niche number
  crossslide   the cross-slide panel, one unit run live plus the reported number
  compare      CONCERT next to the published methods and the reference strategies, paired

Examples
  python reproduction/01_spac_seq_benchmark.py --steps weights aggregate compare
  python reproduction/01_spac_seq_benchmark.py --steps slide --slide 1-1
  python reproduction/01_spac_seq_benchmark.py --all --result-root fresh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import concert_paths as cp

SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]

MODEL_SRC = cp.MODELS / "concert_benchmark"
SPAC_RAW = cp.rp.SPACSEQ_RAW
PREPARED = cp.rp.SPACSEQ_PREPARED
BENCH = cp.rp.SPACSEQ_BENCHMARK
# clonality gate + slide guide shift
CKPT = {
    "seen": cp.CHECKPOINT_ROOT / "spacseq_within_niche" / "model.pt",
    "unseen": cp.CHECKPOINT_ROOT / "spacseq_cross_slide" / "model.pt",
}
SHIPPED = {
    "seen": cp.rp.SPACSEQ_PREDICTIONS / "concert" / "seen",
    "unseen": cp.rp.SPACSEQ_PREDICTIONS / "concert" / "unseen",
}
NUMBERS = cp.rp.SPACSEQ_RESULTS       # the reported per-record tables
CONCERT_KEY = {"seen": "concert_seen", "unseen": "concert_unseen"}


def spac_args(bench, regime, outdir, target_slide=None, units=None):
    """Evaluation configuration for one SPAC-seq run, parsed by the entry point itself.

    The flags are the ones experiments/01_spac_seq_benchmark/evaluation/
    evaluate.sh passes, so this script and the SLURM job cannot drift apart.

    Three of them fail silently rather than loudly when they are wrong:
      fpert_ls_min / fpert_ls_max  the footprint lengthscale band. It is a plain float
        attribute of the field, not a saved weight, so identical weights render a different
        spatial field if it is left at the default.
      cf_output poisson_depth  the scored prediction is one Poisson draw at each source spot's
        measured library size, seeded by the perturbation code.
      cf_seen_ref observed  the predicted LFC is taken against the measured source spots.
    """
    import sys

    argv = ["evaluate_concert_spacseq.py", "--regime", regime,
            "--prepared_dir", str(PREPARED), "--model_file", str(CKPT[regime]),
            "--outdir", str(outdir), "--n_boot", "10", "--min_g_cells", "20",
            "--ls_prior_min", "3", "--ls_prior_max", "18",
            "--fpert_ls_min", "0.15", "--fpert_ls_max", "0.90",
            "--cf_output", "poisson_depth", "--cf_seen_ref", "observed", "--force"]
    if regime == "unseen":
        argv += ["--benchmark_dir", str(BENCH), "--max_src_cells", "5000"]
        if units:
            argv += ["--units", *units]
    else:
        argv += ["--target_slide", target_slide, "--max_src_cells_per_niche", "500"]
    saved = sys.argv
    try:
        sys.argv = argv
        return bench.parse_args()
    finally:
        sys.argv = saved


def install_gene_rule(sec, shared_gene_rule):
    """The shared gene rule of shared_gene_rule.py: slide HVGs AND the 2,000 CONCERT genes."""
    import numpy as np

    genes = set(shared_gene_rule.load_concert_genes())
    orig = sec.select_hvgs

    def select_hvgs_shared(adata_all, n_top, sa_):
        mask = orig(adata_all, n_top, sa_)
        names = np.asarray(adata_all.var_names).astype(str)
        return np.asarray(mask, dtype=bool) & np.isin(names, list(genes))

    sec.select_hvgs = select_hvgs_shared
    print(f"gene rule        : slide HVGs AND {len(genes)} CONCERT genes, minus the knockout")


def read_seen_records(root, slides):
    """One row per (perturbation, slide, niche) record from lfc_scores.json; records whose LFC
    RMSE is undefined are dropped, as in the reported tables."""
    import numpy as np
    import pandas as pd

    rows = []
    for s in slides:
        p = Path(root) / "patch" / f"slide_{s}" / "lfc_scores.json"
        if not p.exists():
            print(f"  slide {s}: no result at {p}")
            continue
        for gene, recs in json.loads(p.read_text())["L2"].items():
            for r in recs:
                rows.append({"slide": s, "perturbation": gene, "niche": r["niche"],
                             "lfc_spearman": r.get("spearman"), "lfc_rmse": r.get("lfc_dist"),
                             "n_scored_genes": r.get("n_scored_genes")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[np.isfinite(df.lfc_rmse.astype(float))].reset_index(drop=True)


def read_unseen_records(root):
    """One row per cross-slide unit; lfc_scores.json is {"L2": {gene: [record]}} with exactly
    one record (niche = -1, the whole slide)."""
    import pandas as pd

    rows = []
    for d in sorted((Path(root) / "unseen_pert").glob("*__slide_*")):
        p = d / "lfc_scores.json"
        if not p.exists():
            continue
        gene, slide = d.name.split("__slide_")
        rec = json.loads(p.read_text())["L2"][gene][0]
        rows.append({"gene": gene, "slide": slide, "n_src_cells": rec.get("n_src"),
                     "n_measured_ko": rec.get("n_real_g"),
                     "lfc_spearman": rec.get("spearman"), "lfc_rmse": rec.get("lfc_dist")})
    return pd.DataFrame(rows).dropna(subset=["lfc_spearman"]) if rows else pd.DataFrame()


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
    The scored prediction is built as --cf_output poisson_depth builds it: each source spot's
    profile mu_i / sum(mu_i) is drawn as counts at the spot's measured library size, and the
    predicted LFC is taken against the measured counts of the same source spots.

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

    y_src = ctx["Y_obs"][model_rows]
    src_counts = np.asarray(y_src.toarray() if hasattr(y_src, "toarray") else y_src, dtype=np.float64)
    mu_g = np.asarray(ecf._predict_swap(ctx, model_rows, code_g, args), dtype=np.float64)
    p_g = mu_g / np.maximum(mu_g.sum(1, keepdims=True), 1e-12)
    pred_g = np.random.default_rng(code_g).poisson(p_g * src_counts.sum(1, keepdims=True))
    model_lfc = eu.pseudobulk_lfc(pred_g, src_counts)

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
    evaluate.sh runs, one slide per array task: it finds the niches of the
    held-out patch and the perturbation dominating each, decodes every source cell to every
    perturbation in the panel and draws counts at depth, scores each (niche, perturbation)
    record against the measured cells, and bootstraps the source cells within each niche.
    The draw is seeded, so the records match the shipped ones.
    """
    import numpy as np

    args, slide, bench = env["args"], env["slide"], env["bench"]
    bench.eval_seen_slide(args, slide)
    df = read_seen_records(args.outdir, [slide])
    print(f"slide {slide}: {len(df)} records, mean LFC Spearman {df.lfc_spearman.mean():.4f}, "
          f"mean LFC RMSE {df.lfc_rmse.mean():.4f}")
    ship = read_seen_records(SHIPPED["seen"], [slide])
    both = df.merge(ship, on=["slide", "perturbation", "niche"], suffixes=("", "_shipped"))
    print(f"vs shipped: {len(both)}/{len(ship)} records matched, "
          f"max |d rho| {np.abs(both.lfc_spearman - both.lfc_spearman_shipped).max():.2e}, "
          f"max |d RMSE| {np.abs(both.lfc_rmse - both.lfc_rmse_shipped).max():.2e}")
    print(df.sort_values("lfc_spearman", ascending=False).round(4).to_string(index=False))
    df.to_csv(out / f"slide_{slide}_per_record.csv", index=False)


def step_aggregate(env, out):
    """The five slides together -- the reported within-niche number.

    A record is one (perturbation, slide, niche) triple; the reported figure is the mean over
    records. Reads the shipped results by default, so it reproduces the published number
    exactly; --result-root fresh reads what `slide` wrote instead.
    """
    import numpy as np

    root = env["result_root"]("seen")
    df = read_seen_records(root, SLIDES)
    if df.empty:
        print("no per-slide results found -- run the `slide` step first, or drop --result-root fresh")
        return

    n = len(df)
    print(f"read {root}")
    print(f"{n} records ({df.perturbation.nunique()} knockouts) over {df.slide.nunique()} slides\n")
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

    A knockout is removed from one target slide and its effect is carried over from the donor
    slides. benchmark/eval_panel.json lists the 85 (knockout, target slide) units; every
    method is scored on the same 5,000 source cells per target slide.
    """
    import numpy as np

    panel = json.loads((BENCH / "eval_panel.json").read_text())
    import pandas as pd
    units = pd.DataFrame([{"gene": u["gene"], "target_slide": u["target_slide"],
                           "donor_slides": ",".join(map(str, u.get("donor_slides", []))),
                           "n_ko_target": u.get("n_ko_target")}
                          for u in panel["eval_units"]])
    print(f"{len(units)} cross-slide evaluation units "
          f"{units.target_slide.value_counts().sort_index().to_dict()}")
    units.to_csv(out / "cross_slide_panel.csv", index=False)

    if env["run_unit"]:
        u0 = panel["eval_units"][0]
        unit = f"{u0['gene']}@{u0['target_slide']}"
        u_args = spac_args(env["bench"], "unseen", outdir=out / "unseen_eval_demo", units=[unit])
        env["bench"].eval_unseen_all(u_args)
        demo = read_unseen_records(u_args.outdir).merge(
            read_unseen_records(SHIPPED["unseen"]), on=["gene", "slide"], suffixes=("", "_shipped"))
        print(f"\ndemo unit {unit} (written to {u_args.outdir}):")
        print(demo[["gene", "slide", "lfc_spearman", "lfc_spearman_shipped",
                    "lfc_rmse", "lfc_rmse_shipped"]].to_string(index=False))

    root = env["result_root"]("unseen")
    df = read_unseen_records(root)
    if df.empty:
        print(f"\nno cross-slide results under {root}")
        return
    print("\nCROSS-SLIDE (code name: unseen)")
    print(f"  {len(df)} units")
    print(f"  LFC Spearman : {df.lfc_spearman.mean():.4f}")
    print(f"  LFC RMSE     : {df.lfc_rmse.mean():.4f}")
    df.to_csv(out / "cross_slide_per_unit.csv", index=False)


def step_compare(env, out):
    """CONCERT next to the published methods and the reference strategies, paired per method.

    The reported tables (spacseq_results/) hold the published methods
    (spac_<setting>_lfc_dist_per_record.csv), the four reference strategies
    (spac_<setting>_naive_baselines_our_rule.csv; Mean target = 50 knockout cells per unit) and
    STATE (spac_<setting>_state_our_rule.csv). The methods cover very different subsets of the
    129 within-niche records -- GEARS 80, CelComen 15, CellAgentChat 11 -- so intersecting all of
    them at once leaves nothing to compare. The reported comparison takes, per method, the
    records that method and CONCERT both cover.

    (The Perturb-map benchmark is the other way round -- there the methods overlap enough that
    an all-method intersection is the rule. See reproduction/02_perturb_map_benchmark.py.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def paired(setting):
        parts = [NUMBERS / f"spac_{setting}_{n}.csv"
                 for n in ("lfc_dist_per_record", "naive_baselines_our_rule", "state_our_rule")]
        df = pd.concat([pd.read_csv(p, dtype={"slide": str, "niche_key": str}) for p in parts],
                       ignore_index=True)
        df["record"] = df.gene.astype(str) + "|" + df.slide + "|" + df.niche_key
        con = df[df.method == CONCERT_KEY[setting]].set_index("record")
        rows = []
        for _m, sub in df[df.method != CONCERT_KEY[setting]].groupby("method"):
            sub = sub.set_index("record")
            shared = con.index.intersection(sub.index)
            if len(shared) == 0:
                continue
            rows.append({"method": sub.method_label.iloc[0], "n_pairs": len(shared),
                         "concert_rho": con.loc[shared, "lfc_spearman"].mean(),
                         "method_rho": sub.loc[shared, "lfc_spearman"].mean(),
                         "concert_rmse": con.loc[shared, "lfc_dist"].mean(),
                         "method_rmse": sub.loc[shared, "lfc_dist"].mean()})
        o = pd.DataFrame(rows)
        o["rho_advantage"] = o.concert_rho - o.method_rho
        o["rmse_advantage"] = o.method_rmse - o.concert_rmse   # positive = CONCERT lower
        return o.sort_values("n_pairs", ascending=False).reset_index(drop=True), len(con)

    cmp = {}
    for setting, name in (("seen", "WITHIN-NICHE"), ("unseen", "CROSS-SLIDE")):
        cmp[setting], n_con = paired(setting)
        print(f"{name}: CONCERT covers {n_con} records; paired against each method:")
        print(cmp[setting].round(4).to_string(index=False))
        cmp[setting].to_csv(out / f"{setting}_paired_comparison.csv", index=False)
        # the shipped statistics table carries the same n_pairs -- matching it is the check
        # that the pairing above is the one the paper used
        stats = pd.read_csv(NUMBERS / f"spacseq_{setting}_concert_vs_competitor_lfc_rho_stats_summary.csv")
        chk = cmp[setting][["method", "n_pairs"]].merge(
            stats[["method", "n_pairs", "p_adj"]].rename(columns={"n_pairs": "n_pairs_shipped"}),
            on="method", how="left")
        chk["matches"] = chk.n_pairs == chk.n_pairs_shipped
        print("\n" + chk.to_string(index=False))
        print(f"all counts match: {bool(chk.matches.all())}\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (cmp_df, name) in zip(axes, [(cmp["seen"], "within-niche"), (cmp["unseen"], "cross-slide")]):
        d = cmp_df.sort_values("method_rho")
        y = np.arange(len(d))
        ax.barh(y - 0.2, d.concert_rho, 0.4, color="#b2182b", label="CONCERT")
        ax.barh(y + 0.2, d.method_rho, 0.4, color="#9ab8d4", label="other method")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(d.method, d.n_pairs)])
        ax.axvline(0, lw=0.8, color="0.4")
        ax.set_xlabel(f"LFC Spearman on paired records ({name})", fontsize=13)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "benchmark_comparison.pdf", dpi=300)
    plt.close(fig)
    print(f"written {out / 'benchmark_comparison.pdf'}")


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
        ("scoring code", cp.EVAL_SRC / "evaluate_concert_spacseq.py"),
        ("metric helpers", cp.EVAL_SRC / "scoring_helpers" / "spac_edist_align.py"),
        ("raw section 1-1", SPAC_RAW / "raw_output_1-1.h5ad"),
        ("prepared cell sets", PREPARED / "slide_1-1.h5ad"),
        ("cross-slide panel", BENCH / "eval_panel.json"),
        ("seen checkpoint", CKPT["seen"]),
        ("seen vocabulary", CKPT["seen"].parent / "pert_map.json"),
        ("unseen checkpoint", CKPT["unseen"]),
        ("unseen vocabulary", CKPT["unseen"].parent / "pert_map.json"),
        ("shipped seen result", SHIPPED["seen"] / "patch" / "slide_1-1" / "lfc_scores.json"),
        ("shipped unseen result", SHIPPED["unseen"] / "unseen_pert"),
        ("reported table", NUMBERS / "spac_seen_lfc_dist_per_record.csv"),
    ])
    print(f"\noutput dir: {out}")

    # The released code has to win over the working trees; see concert_paths.prepend_src.
    cp.prepend_src(MODEL_SRC, cp.EVAL_SRC)
    import eval_spatial_counterfactual as ecf     # binds the released model copy first
    import spacseq_common as sec
    import spacseq_slides as smc
    import spacseq_cross_slide_scoring as suel
    import shared_gene_rule
    import evaluate_concert_spacseq as bench
    eu, _sa = sec.import_eval_utils()
    for m in (ecf, sec, smc, suel, shared_gene_rule, bench):
        assert Path(m.__file__).resolve().is_relative_to(cp.REPO_ROOT.resolve()), \
            f"{m.__name__} loaded from outside the release: {m.__file__}"
    print(f"\nmodel source      : {ecf.__file__}")
    print(f"scoring entrypoint: {bench.__file__}")
    print(f"metric helpers    : {eu.__file__}")
    install_gene_rule(sec, shared_gene_rule)
    cp.describe_device()

    args = spac_args(bench, "seen", outdir=out / "seen_eval", target_slide=args_cli.slide)
    env = {"args": args, "slide": args_cli.slide, "ecf": ecf, "smc": smc, "bench": bench,
           "eu": eu, "run_unit": args_cli.run_unit,
           "result_root": lambda r: (SHIPPED[r] if args_cli.result_root == "shipped"
                                     else out / f"{r}_eval")}

    if set(steps) & NEEDS_MODEL:
        cp.section(f"Loading the checkpoint on slide {args_cli.slide}")
        print(f"checkpoint  {args.model_file}")
        print(f"output      {args.cf_output}, LFC reference = {args.cf_seen_ref}")
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
